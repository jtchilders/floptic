#ifdef FLOPTIC_HAS_CUDA

#include "floptic/kernel_base.hpp"
#include "floptic/kernel_registry.hpp"
#include "floptic/timer.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <iostream>
#include <cstdint>

namespace floptic {

// ============================================================================
// FMA dispatch: each type needs its own fma call
// ============================================================================

__device__ __forceinline__ double  do_fma(double a, double b, double c)   { return fma(a, b, c); }
__device__ __forceinline__ float   do_fma(float a, float b, float c)      { return fmaf(a, b, c); }
__device__ __forceinline__ __half  do_fma(__half a, __half b, __half c)    { return __hfma(a, b, c); }
__device__ __forceinline__ __nv_bfloat16 do_fma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
    return __hfma(a, b, c);
}

// ============================================================================
// Type conversion helpers
// ============================================================================

template <typename T>
__device__ __forceinline__ T make_val(double v);

template <> __device__ __forceinline__ double        make_val<double>(double v)        { return v; }
template <> __device__ __forceinline__ float         make_val<float>(double v)         { return static_cast<float>(v); }
template <> __device__ __forceinline__ __half        make_val<__half>(double v)        { return __double2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 make_val<__nv_bfloat16>(double v) { return __double2bfloat16(v); }

template <typename T>
__device__ __forceinline__ double to_double(T v);

template <> __device__ __forceinline__ double to_double<double>(double v)               { return v; }
template <> __device__ __forceinline__ double to_double<float>(float v)                 { return static_cast<double>(v); }
template <> __device__ __forceinline__ double to_double<__half>(__half v)               { return __half2float(v); }
template <> __device__ __forceinline__ double to_double<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

// ============================================================================
// CUDA Kernels — generic via do_fma / make_val
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
    // Combine to prevent DCE — use addition chain
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
static float run_cuda_benchmark(const std::string& mode,
                                 int blocks, int threads_per_block,
                                 int64_t iters) {
    int total_threads = blocks * threads_per_block;
    T* d_sink;
    cudaMalloc(&d_sink, total_threads * sizeof(T));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (mode == "latency") {
        scalar_fma_latency_kernel<<<blocks, threads_per_block>>>(d_sink, iters);
    } else {
        scalar_fma_throughput_kernel<<<blocks, threads_per_block>>>(d_sink, iters);
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

// ============================================================================
// Dispatch helper — maps Precision enum to template instantiation
// ============================================================================

static float dispatch_benchmark(Precision prec, const std::string& mode,
                                int blocks, int tpb, int64_t iters) {
    switch (prec) {
        case Precision::FP64: return run_cuda_benchmark<double>(mode, blocks, tpb, iters);
        case Precision::FP32: return run_cuda_benchmark<float>(mode, blocks, tpb, iters);
        case Precision::FP16: return run_cuda_benchmark<__half>(mode, blocks, tpb, iters);
        case Precision::BF16: return run_cuda_benchmark<__nv_bfloat16>(mode, blocks, tpb, iters);
        default:
            std::cerr << "  ERROR: Unsupported precision for scalar_fma CUDA" << std::endl;
            return 0.0f;
    }
}

// ============================================================================
// Theoretical peak helpers for FP16/BF16 on CUDA cores
// ============================================================================

// FP16/BF16 ops per SM per clock (CUDA cores, not tensor cores)
static double fp16_ops_per_sm_per_clock(int major, int minor) {
    switch (major) {
        case 7:
            if (minor == 0) return 128 * 2;  // V100: 2x FP32 rate
            return 128 * 2;                   // Turing
        case 8:
            if (minor == 0) return 128 * 2;   // A100: 2x FP32
            return 256 * 2;                    // Ada
        case 9:  return 256 * 2;              // Hopper
        case 10: return 256 * 2;              // Blackwell est.
        default: return 128 * 2;
    }
}

// ============================================================================
// Kernel Class
// ============================================================================

class ScalarFmaCuda : public KernelBase {
public:
    std::string name() const override { return "scalar_fma"; }
    std::string category() const override { return "scalar"; }
    std::string backend() const override { return "cuda"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP64, Precision::FP32, Precision::FP16, Precision::BF16};
    }

    std::vector<std::string> supported_modes() const override {
        return {"throughput", "latency"};
    }

    KernelResult run(const KernelConfig& config,
                     const DeviceInfo& device,
                     int measurement_trials) override {
        // Parse device index
        int dev_idx = 0;
        auto pos = device.id.find(':');
        if (pos != std::string::npos)
            dev_idx = std::stoi(device.id.substr(pos + 1));
        cudaSetDevice(dev_idx);

        int sms = device.compute_units;
        int64_t iters = config.iterations;
        int chains = (config.mode == "latency") ? 1 : 8;

        // If user explicitly set blocks, use that config (no sweep)
        bool user_specified = (config.gpu_blocks > 0);

        if (user_specified) {
            int blocks = config.gpu_blocks;
            int tpb = config.gpu_threads_per_block;
            int total_threads = blocks * tpb;
            int64_t flops_per_trial = static_cast<int64_t>(total_threads) * chains * iters * 2;

            std::cerr << "  Running scalar_fma [cuda/" << precision_to_string(config.precision)
                      << "/" << config.mode << "] blocks=" << blocks
                      << " threads=" << tpb << " iters=" << iters << std::endl;

            for (int w = 0; w < 3; w++)
                dispatch_benchmark(config.precision, config.mode, blocks, tpb, iters);

            std::vector<double> times;
            times.reserve(measurement_trials);
            for (int t = 0; t < measurement_trials; t++)
                times.push_back(static_cast<double>(
                    dispatch_benchmark(config.precision, config.mode, blocks, tpb, iters)));

            auto stats = TimingStats::compute(times);
            KernelResult result;
            result.median_time_ms = stats.median_ms;
            result.min_time_ms = stats.min_ms;
            result.max_time_ms = stats.max_ms;
            result.total_flops = flops_per_trial;
            result.gflops = (flops_per_trial / 1e9) / (stats.median_ms / 1e3);
            result.effective_gflops = result.gflops;

            std::string prec_key = precision_to_string(config.precision);
            auto it = device.theoretical_peak_gflops.find(prec_key);
            if (it != device.theoretical_peak_gflops.end() && it->second > 0)
                result.peak_percent = (result.gflops / it->second) * 100.0;

            result.clock_mhz = device.boost_clock_mhz;
            return result;
        }

        // Auto-sweep: try multiple (blocks_per_sm, threads_per_block) configs
        int bpsm_values[] = {2, 4, 6, 8, 12, 16};
        int tpb_values[]  = {128, 256, 512, 1024};
        int n_bpsm = sizeof(bpsm_values) / sizeof(bpsm_values[0]);
        int n_tpb  = sizeof(tpb_values)  / sizeof(tpb_values[0]);

        double best_gflops = 0;
        int best_bpsm = 4, best_tpb = 256;

        std::cerr << "  Sweeping scalar_fma [cuda/" << precision_to_string(config.precision)
                  << "/" << config.mode << "] iters=" << iters << ":" << std::endl;

        for (int bi = 0; bi < n_bpsm; bi++) {
            for (int ti = 0; ti < n_tpb; ti++) {
                int bpsm = bpsm_values[bi];
                int tpb  = tpb_values[ti];
                int blocks = sms * bpsm;
                int total_threads = blocks * tpb;
                int64_t flops_per_trial = static_cast<int64_t>(total_threads) * chains * iters * 2;

                // Quick warmup + 2 trial measurement
                dispatch_benchmark(config.precision, config.mode, blocks, tpb, iters);
                float ms1 = dispatch_benchmark(config.precision, config.mode, blocks, tpb, iters);
                float ms2 = dispatch_benchmark(config.precision, config.mode, blocks, tpb, iters);
                double median_ms = std::min(ms1, ms2);
                double gflops = (flops_per_trial / 1e9) / (median_ms / 1e3);

                if (gflops > best_gflops) {
                    best_gflops = gflops;
                    best_bpsm = bpsm;
                    best_tpb = tpb;
                }
            }
        }

        int blocks = sms * best_bpsm;
        int tpb = best_tpb;
        int total_threads = blocks * tpb;
        int64_t flops_per_trial = static_cast<int64_t>(total_threads) * chains * iters * 2;

        std::cerr << "  Best config: bpsm=" << best_bpsm << " tpb=" << best_tpb
                  << " (" << best_gflops << " GFLOP/s) → full measurement ("
                  << measurement_trials << " trials)" << std::endl;

        // Full measurement at best config
        for (int w = 0; w < 3; w++)
            dispatch_benchmark(config.precision, config.mode, blocks, tpb, iters);

        std::vector<double> times;
        times.reserve(measurement_trials);
        for (int t = 0; t < measurement_trials; t++)
            times.push_back(static_cast<double>(
                dispatch_benchmark(config.precision, config.mode, blocks, tpb, iters)));

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

        result.clock_mhz = device.boost_clock_mhz;
        return result;
    }
};

REGISTER_KERNEL(ScalarFmaCuda);

namespace force_link {
    void scalar_fma_cuda_link() {}
}

} // namespace floptic

#endif // FLOPTIC_HAS_CUDA
