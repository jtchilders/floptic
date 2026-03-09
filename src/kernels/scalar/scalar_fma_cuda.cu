#ifdef FLOPTIC_HAS_CUDA

#include "floptic/kernel_base.hpp"
#include "floptic/kernel_registry.hpp"
#include "floptic/timer.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cstdint>

namespace floptic {

// ---- CUDA Kernels ----

template <typename T>
__global__ void scalar_fma_throughput_kernel(T* __restrict__ sink,
                                              T a, T b, int64_t iters) {
    T r0 = static_cast<T>(1.0);
    T r1 = static_cast<T>(2.0);
    T r2 = static_cast<T>(3.0);
    T r3 = static_cast<T>(4.0);
    T r4 = static_cast<T>(5.0);
    T r5 = static_cast<T>(6.0);
    T r6 = static_cast<T>(7.0);
    T r7 = static_cast<T>(8.0);

    for (int64_t i = 0; i < iters; i++) {
        r0 = fma(a, r0, b);
        r1 = fma(a, r1, b);
        r2 = fma(a, r2, b);
        r3 = fma(a, r3, b);
        r4 = fma(a, r4, b);
        r5 = fma(a, r5, b);
        r6 = fma(a, r6, b);
        r7 = fma(a, r7, b);
    }

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    sink[idx] = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
}

template <typename T>
__global__ void scalar_fma_latency_kernel(T* __restrict__ sink,
                                           T a, T b, int64_t iters) {
    T r = static_cast<T>(1.0);

    for (int64_t i = 0; i < iters; i++) {
        r = fma(a, r, b);  // dependent chain
    }

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    sink[idx] = r;
}

// ---- Kernel Runner ----

template <typename T>
static float run_cuda_benchmark(const std::string& mode,
                                 int blocks, int threads_per_block,
                                 int64_t iters) {
    int total_threads = blocks * threads_per_block;
    T* d_sink;
    cudaMalloc(&d_sink, total_threads * sizeof(T));

    T a = static_cast<T>(1.0000001);
    T b = static_cast<T>(0.9999999);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (mode == "latency") {
        scalar_fma_latency_kernel<<<blocks, threads_per_block>>>(d_sink, a, b, iters);
    } else {
        scalar_fma_throughput_kernel<<<blocks, threads_per_block>>>(d_sink, a, b, iters);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_sink);

    return ms;
}

// ---- Kernel Class ----

class ScalarFmaCuda : public KernelBase {
public:
    std::string name() const override { return "scalar_fma"; }
    std::string category() const override { return "scalar"; }
    std::string backend() const override { return "cuda"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP64, Precision::FP32};
    }

    std::vector<std::string> supported_modes() const override {
        return {"throughput", "latency"};
    }

    KernelResult run(const KernelConfig& config,
                     const DeviceInfo& device,
                     int measurement_trials) override {
        // Parse device index from id like "cuda:0"
        int dev_idx = 0;
        auto pos = device.id.find(':');
        if (pos != std::string::npos)
            dev_idx = std::stoi(device.id.substr(pos + 1));
        cudaSetDevice(dev_idx);

        // Launch configuration: fill all SMs
        int sms = device.compute_units;
        int threads_per_block = 256;
        int blocks_per_sm = 4;  // target reasonable occupancy
        int blocks = sms * blocks_per_sm;
        int total_threads = blocks * threads_per_block;

        int64_t iters = config.iterations;
        int chains = (config.mode == "latency") ? 1 : 8;
        int64_t flops_per_trial = static_cast<int64_t>(total_threads) * chains * iters * 2;

        std::cerr << "  Running scalar_fma [cuda/" << precision_to_string(config.precision)
                  << "/" << config.mode << "] blocks=" << blocks
                  << " threads=" << threads_per_block
                  << " iters=" << iters << std::endl;

        // Warmup
        for (int w = 0; w < 3; w++) {
            if (config.precision == Precision::FP32)
                run_cuda_benchmark<float>(config.mode, blocks, threads_per_block, iters);
            else
                run_cuda_benchmark<double>(config.mode, blocks, threads_per_block, iters);
        }

        // Measurement
        std::vector<double> times;
        times.reserve(measurement_trials);

        for (int t = 0; t < measurement_trials; t++) {
            float ms;
            if (config.precision == Precision::FP32)
                ms = run_cuda_benchmark<float>(config.mode, blocks, threads_per_block, iters);
            else
                ms = run_cuda_benchmark<double>(config.mode, blocks, threads_per_block, iters);
            times.push_back(static_cast<double>(ms));
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

        // Sample clock frequency
        result.clock_mhz = device.boost_clock_mhz;

        return result;
    }
};

REGISTER_KERNEL(ScalarFmaCuda);

} // namespace floptic

#endif // FLOPTIC_HAS_CUDA
