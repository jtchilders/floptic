#include "floptic/kernel_base.hpp"
#include "floptic/kernel_registry.hpp"
#include "floptic/timer.hpp"
#include "floptic/cpu_threads.hpp"
#include "floptic/axpy_kernel.hpp"
#include "floptic/aligned_buffer.hpp"
#include <cmath>
#include <vector>
#include <iostream>

#ifdef FLOPTIC_HAS_OPENMP
#include <omp.h>
#endif

namespace floptic {

extern volatile double g_validation_sink;

// ============================================================================
// Kernel class
// ============================================================================

class VectorAxpyCpu : public KernelBase {
public:
    std::string name() const override { return "vector_axpy"; }
    std::string category() const override { return "vector"; }
    std::string backend() const override { return "cpu"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP64, Precision::FP32};
    }

    std::vector<std::string> supported_modes() const override {
        return {"throughput"};
    }

    KernelResult run(const KernelConfig& config,
                     const DeviceInfo& device,
                     int measurement_trials) override {
        // Resolve requested threads against compile-time OpenMP support: a
        // serial (no-OpenMP) build always executes on exactly one thread
        // regardless of what was requested, and must account work that way.
        int num_threads = resolve_effective_cpu_threads(
            config.threads, device.compute_units, openmp_compiled_in());

        // Problem size
        int64_t n = static_cast<int64_t>(config.iterations) * 100;
        if (n < 1000000) n = 1000000;
        if (n > 100000000) n = 100000000;

        // Every one of the n elements is fma'd exactly once (vectorized main
        // loop plus scalar tail cleanup), so 2*n FLOPs is accurate for any n.
        int64_t flops_per_trial = n * 2;

        std::string simd_path = axpy_active_simd_path();

        size_t elem_bytes = (config.precision == Precision::FP64) ? 8 : 4;
        double bytes_per_trial = 3.0 * n * elem_bytes;

        std::cerr << "  Running vector_axpy [cpu/" << precision_to_string(config.precision)
                  << "/" << config.mode << "/" << simd_path
                  << "] n=" << n << " threads=" << num_threads << std::endl;

        // Allocate and initialize
        if (config.precision == Precision::FP64) {
            AlignedBuffer<double> xbuf(n);
            AlignedBuffer<double> ybuf(n);
            if (!xbuf.valid() || !ybuf.valid()) {
                std::cerr << "  ERROR: aligned allocation failed for n=" << n << std::endl;
                return KernelResult{};
            }
            double* x = xbuf.get();
            double* y = ybuf.get();
            double alpha = 1.5;

            #ifdef FLOPTIC_HAS_OPENMP
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            #endif
            for (int64_t i = 0; i < n; i++) {
                x[i] = 1.0 + 1e-8 * i;
                y[i] = 2.0 - 1e-8 * i;
            }

            auto run_fn = [&]() -> double {
                // Reset y before each trial
                #ifdef FLOPTIC_HAS_OPENMP
                #pragma omp parallel for num_threads(num_threads) schedule(static)
                #endif
                for (int64_t i = 0; i < n; i++) y[i] = 2.0 - 1e-8 * i;

                CpuTimer timer;
                timer.start();
                axpy_run(y, x, alpha, n, num_threads);
                timer.stop();
                return timer.elapsed_ms();
            };

            // Warmup
            for (int w = 0; w < config.warmup_trials; w++) run_fn();

            // Measurement
            std::vector<double> times;
            times.reserve(measurement_trials);
            for (int t = 0; t < measurement_trials; t++) {
                times.push_back(run_fn());
            }

            g_validation_sink = y[0];

            auto stats = TimingStats::compute(times);

            KernelResult result;
            result.median_time_ms = stats.median_ms;
            result.min_time_ms = stats.min_ms;
            result.max_time_ms = stats.max_ms;
            result.total_flops = flops_per_trial;
            result.gflops = (flops_per_trial / 1e9) / (stats.median_ms / 1e3);
            result.effective_gflops = result.gflops;

            double gbps = (bytes_per_trial / 1e9) / (stats.median_ms / 1e3);
            std::cerr << "  Bandwidth: " << gbps << " GB/s" << std::endl;

            std::string prec_key = precision_to_string(config.precision);
            auto it = device.theoretical_peak_gflops.find(prec_key);
            if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
                result.peak_percent = (result.gflops / it->second) * 100.0;
            }
            return result;

        } else {
            // FP32
            AlignedBuffer<float> xbuf(n);
            AlignedBuffer<float> ybuf(n);
            if (!xbuf.valid() || !ybuf.valid()) {
                std::cerr << "  ERROR: aligned allocation failed for n=" << n << std::endl;
                return KernelResult{};
            }
            float* x = xbuf.get();
            float* y = ybuf.get();
            float alpha = 1.5f;

            #ifdef FLOPTIC_HAS_OPENMP
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            #endif
            for (int64_t i = 0; i < n; i++) {
                x[i] = 1.0f + 1e-6f * i;
                y[i] = 2.0f - 1e-6f * i;
            }

            auto run_fn = [&]() -> double {
                #ifdef FLOPTIC_HAS_OPENMP
                #pragma omp parallel for num_threads(num_threads) schedule(static)
                #endif
                for (int64_t i = 0; i < n; i++) y[i] = 2.0f - 1e-6f * i;

                CpuTimer timer;
                timer.start();
                axpy_run(y, x, alpha, n, num_threads);
                timer.stop();
                return timer.elapsed_ms();
            };

            for (int w = 0; w < config.warmup_trials; w++) run_fn();

            std::vector<double> times;
            times.reserve(measurement_trials);
            for (int t = 0; t < measurement_trials; t++) {
                times.push_back(run_fn());
            }

            g_validation_sink = y[0];

            auto stats = TimingStats::compute(times);

            KernelResult result;
            result.median_time_ms = stats.median_ms;
            result.min_time_ms = stats.min_ms;
            result.max_time_ms = stats.max_ms;
            result.total_flops = flops_per_trial;
            result.gflops = (flops_per_trial / 1e9) / (stats.median_ms / 1e3);
            result.effective_gflops = result.gflops;

            double gbps = (bytes_per_trial / 1e9) / (stats.median_ms / 1e3);
            std::cerr << "  Bandwidth: " << gbps << " GB/s" << std::endl;

            std::string prec_key = precision_to_string(config.precision);
            auto it = device.theoretical_peak_gflops.find(prec_key);
            if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
                result.peak_percent = (result.gflops / it->second) * 100.0;
            }
            return result;
        }
    }
};

REGISTER_KERNEL(VectorAxpyCpu);

namespace force_link {
    void vector_axpy_cpu_link() {}
}

} // namespace floptic
