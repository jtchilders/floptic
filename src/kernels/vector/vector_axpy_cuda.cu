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
#include <cstdlib>
#include <cmath>

namespace floptic {

// ============================================================================
// Type conversion helpers (host-side)
// ============================================================================

template <typename T>
static T host_make_val(double v);

template <> double        host_make_val<double>(double v)        { return v; }
template <> float         host_make_val<float>(double v)         { return static_cast<float>(v); }
template <> __half        host_make_val<__half>(double v)        { return __float2half(static_cast<float>(v)); }
template <> __nv_bfloat16 host_make_val<__nv_bfloat16>(double v) { return __float2bfloat16(static_cast<float>(v)); }

// ============================================================================
// Device FMA dispatch
// ============================================================================

__device__ __forceinline__ double  do_fma(double a, double b, double c)   { return fma(a, b, c); }
__device__ __forceinline__ float   do_fma(float a, float b, float c)      { return fmaf(a, b, c); }
__device__ __forceinline__ __half  do_fma(__half a, __half b, __half c)    { return __hfma(a, b, c); }
__device__ __forceinline__ __nv_bfloat16 do_fma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
    return __hfma(a, b, c);
}

// ============================================================================
// AXPY kernel: y[i] = alpha * x[i] + y[i]
// Each element = 1 FMA = 2 FLOPs
// This is memory-bound: 3 memory ops (read x, read y, write y) per 2 FLOPs
// ============================================================================

template <typename T>
__global__ void vector_axpy_kernel(T* __restrict__ y,
                                    const T* __restrict__ x,
                                    T alpha, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        y[i] = do_fma(alpha, x[i], y[i]);
    }
}

// ============================================================================
// Benchmark runner
// ============================================================================

template <typename T>
static float run_axpy_benchmark(int64_t n, int blocks, int threads_per_block) {
    T* d_x;
    T* d_y;
    size_t bytes = n * sizeof(T);

    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    // Initialize with simple pattern (avoids host→device of large random arrays)
    // The kernel is memory-bound so data values don't matter much for timing
    cudaMemset(d_x, 1, bytes);
    cudaMemset(d_y, 0, bytes);

    T alpha = host_make_val<T>(1.5);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vector_axpy_kernel<<<blocks, threads_per_block>>>(d_y, d_x, alpha, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x);
    cudaFree(d_y);

    return ms;
}

static float dispatch_axpy(Precision prec, int64_t n, int blocks, int tpb) {
    switch (prec) {
        case Precision::FP64: return run_axpy_benchmark<double>(n, blocks, tpb);
        case Precision::FP32: return run_axpy_benchmark<float>(n, blocks, tpb);
        case Precision::FP16: return run_axpy_benchmark<__half>(n, blocks, tpb);
        case Precision::BF16: return run_axpy_benchmark<__nv_bfloat16>(n, blocks, tpb);
        default:
            std::cerr << "  ERROR: Unsupported precision for vector_axpy CUDA" << std::endl;
            return 0.0f;
    }
}

// ============================================================================
// Bytes per element for bandwidth calculation
// ============================================================================

static size_t precision_bytes(Precision p) {
    switch (p) {
        case Precision::FP64: return 8;
        case Precision::FP32: return 4;
        case Precision::FP16: return 2;
        case Precision::BF16: return 2;
        default: return 4;
    }
}

// ============================================================================
// Kernel class
// ============================================================================

class VectorAxpyCuda : public KernelBase {
public:
    std::string name() const override { return "vector_axpy"; }
    std::string category() const override { return "vector"; }
    std::string backend() const override { return "cuda"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP64, Precision::FP32, Precision::FP16, Precision::BF16};
    }

    std::vector<std::string> supported_modes() const override {
        return {"throughput"};  // memory-bound kernel, no latency mode
    }

    KernelResult run(const KernelConfig& config,
                     const DeviceInfo& device,
                     int measurement_trials) override {
        int dev_idx = 0;
        auto pos = device.id.find(':');
        if (pos != std::string::npos)
            dev_idx = std::stoi(device.id.substr(pos + 1));
        cudaSetDevice(dev_idx);

        // Problem size: use inner_iters as element count multiplier
        // Default: 10M elements (fits in GPU memory for all precisions)
        int64_t n = static_cast<int64_t>(config.iterations) * 100;  // 100K * 100 = 10M default
        if (n < 1000000) n = 1000000;   // minimum 1M elements
        if (n > 100000000) n = 100000000; // cap at 100M elements

        // Launch config
        int sms = device.compute_units;
        int tpb = config.gpu_threads_per_block;
        int blocks;
        if (config.gpu_blocks > 0) {
            blocks = config.gpu_blocks;
        } else {
            blocks = sms * config.gpu_blocks_per_sm;
        }

        // FLOPs: 2 per element (1 multiply + 1 add = 1 FMA)
        int64_t flops_per_trial = n * 2;

        // Bandwidth: read x, read y, write y = 3 * n * element_size
        size_t elem_bytes = precision_bytes(config.precision);
        double bytes_per_trial = 3.0 * n * elem_bytes;

        std::cerr << "  Running vector_axpy [cuda/" << precision_to_string(config.precision)
                  << "/" << config.mode << "] n=" << n
                  << " blocks=" << blocks << " threads=" << tpb << std::endl;

        // Warmup
        for (int w = 0; w < 3; w++) {
            dispatch_axpy(config.precision, n, blocks, tpb);
        }

        // Measurement
        std::vector<double> times;
        times.reserve(measurement_trials);

        for (int t = 0; t < measurement_trials; t++) {
            float ms = dispatch_axpy(config.precision, n, blocks, tpb);
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

        // Also report bandwidth (as GB/s via power_watts field... or just stderr)
        double gbps = (bytes_per_trial / 1e9) / (stats.median_ms / 1e3);
        std::cerr << "  Bandwidth: " << gbps << " GB/s" << std::endl;

        // Peak percent (against compute peak, will be low since this is memory-bound)
        std::string prec_key = precision_to_string(config.precision);
        auto it = device.theoretical_peak_gflops.find(prec_key);
        if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
            result.peak_percent = (result.gflops / it->second) * 100.0;
        }

        result.clock_mhz = device.boost_clock_mhz;
        return result;
    }
};

REGISTER_KERNEL(VectorAxpyCuda);

namespace force_link {
    void vector_axpy_cuda_link() {}
}

} // namespace floptic

#endif // FLOPTIC_HAS_CUDA
