#ifdef FLOPTIC_HAS_HIP

#include "floptic/kernel_base.hpp"
#include "floptic/kernel_registry.hpp"
#include "floptic/timer.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <vector>
#include <iostream>
#include <cstdint>

namespace floptic {

// ============================================================================
// FMA dispatch
// ============================================================================

__device__ __forceinline__ double do_fma(double a, double b, double c) { return fma(a, b, c); }
__device__ __forceinline__ float  do_fma(float a, float b, float c)   { return fmaf(a, b, c); }
__device__ __forceinline__ _Float16 do_fma(_Float16 a, _Float16 b, _Float16 c) {
    return __hfma(a, b, c);
}
__device__ __forceinline__ hip_bfloat16 do_fma(hip_bfloat16 a, hip_bfloat16 b, hip_bfloat16 c) {
    // hip_bfloat16 doesn't have native FMA on all archs; use float path
    float fa = static_cast<float>(a);
    float fb = static_cast<float>(b);
    float fc = static_cast<float>(c);
    return hip_bfloat16(fmaf(fa, fb, fc));
}

// ============================================================================
// Type conversion helpers
// ============================================================================

template <typename T>
__device__ __forceinline__ T make_val(double v);

template <> __device__ __forceinline__ double       make_val<double>(double v)       { return v; }
template <> __device__ __forceinline__ float        make_val<float>(double v)        { return static_cast<float>(v); }
template <> __device__ __forceinline__ _Float16     make_val<_Float16>(double v)     { return static_cast<_Float16>(v); }
template <> __device__ __forceinline__ hip_bfloat16 make_val<hip_bfloat16>(double v) { return hip_bfloat16(static_cast<float>(v)); }

// ============================================================================
// HIP Kernels
// ============================================================================

template <typename T>
__global__ void scalar_fma_throughput_kernel(T* __restrict__ sink, int64_t iters) {
    T a = make_val<T>(1.0000001);
    T b = make_val<T>(0.9999999);
    T r0 = make_val<T>(1.0);
    T r1 = make_val<T>(2.0);
    T r2 = make_val<T>(3.0);
    T r3 = make_val<T>(4.0);
    T r4 = make_val<T>(5.0);
    T r5 = make_val<T>(6.0);
    T r6 = make_val<T>(7.0);
    T r7 = make_val<T>(8.0);

    for (int64_t i = 0; i < iters; i++) {
        r0 = do_fma(a, r0, b);
        r1 = do_fma(a, r1, b);
        r2 = do_fma(a, r2, b);
        r3 = do_fma(a, r3, b);
        r4 = do_fma(a, r4, b);
        r5 = do_fma(a, r5, b);
        r6 = do_fma(a, r6, b);
        r7 = do_fma(a, r7, b);
    }

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    T sum = do_fma(r0, make_val<T>(1.0), r1);
    sum = do_fma(sum, make_val<T>(1.0), r2);
    sum = do_fma(sum, make_val<T>(1.0), r3);
    sum = do_fma(sum, make_val<T>(1.0), r4);
    sum = do_fma(sum, make_val<T>(1.0), r5);
    sum = do_fma(sum, make_val<T>(1.0), r6);
    sum = do_fma(sum, make_val<T>(1.0), r7);
    sink[idx] = sum;
}

template <typename T>
__global__ void scalar_fma_latency_kernel(T* __restrict__ sink, int64_t iters) {
    T a = make_val<T>(1.0000001);
    T b = make_val<T>(0.9999999);
    T r = make_val<T>(1.0);

    for (int64_t i = 0; i < iters; i++) {
        r = do_fma(a, r, b);
    }

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    sink[idx] = r;
}

// ============================================================================
// Kernel Runner
// ============================================================================

template <typename T>
static float run_hip_benchmark(const std::string& mode,
                               int blocks, int threads_per_block,
                               int64_t iters) {
    int total_threads = blocks * threads_per_block;
    T* d_sink;
    hipMalloc(&d_sink, total_threads * sizeof(T));

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    if (mode == "latency") {
        scalar_fma_latency_kernel<<<blocks, threads_per_block>>>(d_sink, iters);
    } else {
        scalar_fma_throughput_kernel<<<blocks, threads_per_block>>>(d_sink, iters);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_sink);

    return ms;
}

static float dispatch_benchmark(Precision prec, const std::string& mode,
                                int blocks, int tpb, int64_t iters) {
    switch (prec) {
        case Precision::FP64: return run_hip_benchmark<double>(mode, blocks, tpb, iters);
        case Precision::FP32: return run_hip_benchmark<float>(mode, blocks, tpb, iters);
        case Precision::FP16: return run_hip_benchmark<_Float16>(mode, blocks, tpb, iters);
        case Precision::BF16: return run_hip_benchmark<hip_bfloat16>(mode, blocks, tpb, iters);
        default:
            std::cerr << "  ERROR: Unsupported precision for scalar_fma HIP" << std::endl;
            return 0.0f;
    }
}

// ============================================================================
// Kernel class
// ============================================================================

class ScalarFmaHip : public KernelBase {
public:
    std::string name() const override { return "scalar_fma"; }
    std::string category() const override { return "scalar"; }
    std::string backend() const override { return "hip"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP64, Precision::FP32, Precision::FP16, Precision::BF16};
    }

    std::vector<std::string> supported_modes() const override {
        return {"throughput", "latency"};
    }

    KernelResult run(const KernelConfig& config,
                     const DeviceInfo& device,
                     int measurement_trials) override {
        int dev_idx = 0;
        auto pos = device.id.find(':');
        if (pos != std::string::npos)
            dev_idx = std::stoi(device.id.substr(pos + 1));
        hipSetDevice(dev_idx);

        int cus = device.compute_units;
        int tpb = config.gpu_threads_per_block > 0 ? config.gpu_threads_per_block : 256;
        // AMD wavefront = 64 threads, so blocks_per_CU is different
        int bpcu = config.gpu_blocks_per_sm > 0 ? config.gpu_blocks_per_sm : 4;
        int blocks = config.gpu_blocks > 0 ? config.gpu_blocks : cus * bpcu;

        int64_t iters = config.iterations;
        int total_threads = blocks * tpb;

        // FLOPs per trial
        int fma_chains = (config.mode == "latency") ? 1 : 8;
        double flops_per_trial = (double)total_threads * iters * fma_chains * 2.0;

        // Warmup
        for (int w = 0; w < 10; w++) {
            dispatch_benchmark(config.precision, config.mode, blocks, tpb, iters);
        }
        hipDeviceSynchronize();

        // Measure
        std::vector<double> times;
        times.reserve(measurement_trials);
        for (int t = 0; t < measurement_trials; t++) {
            float ms = dispatch_benchmark(config.precision, config.mode, blocks, tpb, iters);
            times.push_back(ms);
        }

        std::sort(times.begin(), times.end());

        KernelResult result;
        result.min_time_ms = times.front();
        result.max_time_ms = times.back();
        result.median_time_ms = times[times.size() / 2];

        result.gflops = (flops_per_trial / (result.median_time_ms * 1e-3)) / 1e9;

        // Peak comparison
        std::string peak_key = precision_to_string(config.precision);
        auto it = device.theoretical_peak_gflops.find(peak_key);
        if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
            result.peak_percent = (result.gflops / it->second) * 100.0;
        }

        return result;
    }
};

REGISTER_KERNEL(ScalarFmaHip);

// Force-link
namespace force_link {
    void scalar_fma_hip_link() {
        volatile auto* p = &KernelRegistry::instance();
        (void)p;
    }
}

} // namespace floptic

#endif // FLOPTIC_HAS_HIP
