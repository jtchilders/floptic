#include "floptic/kernel_base.hpp"
#include "floptic/kernel_registry.hpp"
#include "floptic/timer.hpp"
#include <cmath>
#include <vector>
#include <iostream>

#ifdef FLOPTIC_HAS_OPENMP
#include <omp.h>
#endif

namespace floptic {

extern volatile double g_validation_sink;

template <typename T>
static double run_throughput(int num_threads, int64_t iters_per_chain, int chains = 8) {
    // Each thread runs 'chains' independent FMA chains
    volatile T sink = 0;

    CpuTimer timer;
    timer.start();

    #ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel num_threads(num_threads)
    #endif
    {
        T a = static_cast<T>(1.0000001);
        T b = static_cast<T>(0.9999999);
        T r0 = 1, r1 = 2, r2 = 3, r3 = 4;
        T r4 = 5, r5 = 6, r6 = 7, r7 = 8;

        for (int64_t i = 0; i < iters_per_chain; i++) {
            r0 = std::fma(a, r0, b);
            r1 = std::fma(a, r1, b);
            r2 = std::fma(a, r2, b);
            r3 = std::fma(a, r3, b);
            r4 = std::fma(a, r4, b);
            r5 = std::fma(a, r5, b);
            r6 = std::fma(a, r6, b);
            r7 = std::fma(a, r7, b);
        }

        // Prevent DCE
        T total = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
        #ifdef FLOPTIC_HAS_OPENMP
        #pragma omp atomic
        #endif
        sink += static_cast<double>(total);
    }

    timer.stop();
    g_validation_sink = sink;
    return timer.elapsed_ms();
}

template <typename T>
static double run_latency(int num_threads, int64_t iters) {
    volatile T sink = 0;

    CpuTimer timer;
    timer.start();

    #ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel num_threads(num_threads)
    #endif
    {
        T a = static_cast<T>(1.0000001);
        T b = static_cast<T>(0.9999999);
        T r = static_cast<T>(1.0);

        for (int64_t i = 0; i < iters; i++) {
            r = std::fma(a, r, b);  // dependent chain
        }

        #ifdef FLOPTIC_HAS_OPENMP
        #pragma omp atomic
        #endif
        sink += static_cast<double>(r);
    }

    timer.stop();
    g_validation_sink = sink;
    return timer.elapsed_ms();
}

class ScalarFmaCpu : public KernelBase {
public:
    std::string name() const override { return "scalar_fma"; }
    std::string category() const override { return "scalar"; }
    std::string backend() const override { return "cpu"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP64, Precision::FP32};
    }

    std::vector<std::string> supported_modes() const override {
        return {"throughput", "latency"};
    }

    KernelResult run(const KernelConfig& config,
                     const DeviceInfo& device,
                     int measurement_trials) override {
        int num_threads = config.threads > 0 ? config.threads : device.compute_units;
        if (num_threads <= 0) num_threads = 1;

        int64_t iters = config.iterations;
        int chains = 8;

        // Total FLOPs per trial
        int64_t flops_per_trial;
        if (config.mode == "latency") {
            flops_per_trial = static_cast<int64_t>(num_threads) * iters * 2;  // 1 chain, 2 FLOPs/FMA
        } else {
            flops_per_trial = static_cast<int64_t>(num_threads) * chains * iters * 2;
        }

        std::cerr << "  Running scalar_fma [cpu/" << precision_to_string(config.precision)
                  << "/" << config.mode << "] threads=" << num_threads
                  << " iters=" << iters << std::endl;

        // Warmup
        for (int w = 0; w < 3; w++) {
            if (config.precision == Precision::FP32) {
                if (config.mode == "latency")
                    run_latency<float>(num_threads, iters);
                else
                    run_throughput<float>(num_threads, iters, chains);
            } else {
                if (config.mode == "latency")
                    run_latency<double>(num_threads, iters);
                else
                    run_throughput<double>(num_threads, iters, chains);
            }
        }

        // Measurement
        std::vector<double> times;
        times.reserve(measurement_trials);

        for (int t = 0; t < measurement_trials; t++) {
            double ms;
            if (config.precision == Precision::FP32) {
                if (config.mode == "latency")
                    ms = run_latency<float>(num_threads, iters);
                else
                    ms = run_throughput<float>(num_threads, iters, chains);
            } else {
                if (config.mode == "latency")
                    ms = run_latency<double>(num_threads, iters);
                else
                    ms = run_throughput<double>(num_threads, iters, chains);
            }
            times.push_back(ms);
        }

        auto stats = TimingStats::compute(times);

        KernelResult result;
        result.median_time_ms = stats.median_ms;
        result.min_time_ms = stats.min_ms;
        result.max_time_ms = stats.max_ms;
        result.total_flops = flops_per_trial;
        result.gflops = (flops_per_trial / 1e9) / (stats.median_ms / 1e3);
        result.effective_gflops = result.gflops;

        // Peak percent
        std::string prec_key = precision_to_string(config.precision);
        auto it = device.theoretical_peak_gflops.find(prec_key);
        if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
            result.peak_percent = (result.gflops / it->second) * 100.0;
        }

        return result;
    }
};

REGISTER_KERNEL(ScalarFmaCpu);

namespace force_link {
    void scalar_fma_cpu_link() {}  // referenced from main.cpp to prevent linker stripping
}

} // namespace floptic
