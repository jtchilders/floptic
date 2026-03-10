#ifdef FLOPTIC_HAS_CUDA

#include "floptic/kernel_base.hpp"
#include "floptic/kernel_registry.hpp"
#include "floptic/timer.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cstdint>

namespace floptic {

// ============================================================================
// STREAM Triad: a[i] = b[i] + scalar * c[i]
//
// 3 memory operations per element (read b, read c, write a)
// 2 FLOPs per element (1 multiply + 1 add), but the metric is GB/s
//
// This is the standard HPC memory bandwidth benchmark (McCalpin STREAM).
// ============================================================================

__global__ void stream_triad_fp64_kernel(double* __restrict__ a,
                                          const double* __restrict__ b,
                                          const double* __restrict__ c,
                                          double scalar, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        a[i] = b[i] + scalar * c[i];
    }
}

__global__ void stream_triad_fp32_kernel(float* __restrict__ a,
                                          const float* __restrict__ b,
                                          const float* __restrict__ c,
                                          float scalar, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        a[i] = b[i] + scalar * c[i];
    }
}

// ============================================================================
// Copy kernel: a[i] = b[i]
// Pure memory bandwidth — 0 FLOPs, 2 memory ops per element
// ============================================================================

__global__ void stream_copy_fp64_kernel(double* __restrict__ a,
                                         const double* __restrict__ b,
                                         int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        a[i] = b[i];
    }
}

// ============================================================================
// Benchmark runners
// ============================================================================

static float run_stream_triad_fp64(int64_t n, int blocks, int tpb) {
    double *d_a, *d_b, *d_c;
    size_t bytes = n * sizeof(double);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemset(d_a, 0, bytes);
    cudaMemset(d_b, 0, bytes);
    cudaMemset(d_c, 0, bytes);

    double scalar = 3.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    stream_triad_fp64_kernel<<<blocks, tpb>>>(d_a, d_b, d_c, scalar, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return ms;
}

static float run_stream_triad_fp32(int64_t n, int blocks, int tpb) {
    float *d_a, *d_b, *d_c;
    size_t bytes = n * sizeof(float);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemset(d_a, 0, bytes);
    cudaMemset(d_b, 0, bytes);
    cudaMemset(d_c, 0, bytes);

    float scalar = 3.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    stream_triad_fp32_kernel<<<blocks, tpb>>>(d_a, d_b, d_c, scalar, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return ms;
}

static float run_stream_copy_fp64(int64_t n, int blocks, int tpb) {
    double *d_a, *d_b;
    size_t bytes = n * sizeof(double);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMemset(d_a, 0, bytes);
    cudaMemset(d_b, 0, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    stream_copy_fp64_kernel<<<blocks, tpb>>>(d_a, d_b, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    return ms;
}

// ============================================================================
// Kernel class — STREAM Triad
// ============================================================================

class StreamTriadCuda : public KernelBase {
public:
    std::string name() const override { return "stream_triad"; }
    std::string category() const override { return "memory"; }
    std::string backend() const override { return "cuda"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP64, Precision::FP32};
    }

    std::vector<std::string> supported_modes() const override {
        return {"throughput"};
    }

    KernelResult run(const KernelConfig& config,
                     const DeviceInfo& device,
                     int measurement_trials) override {
        int dev_idx = 0;
        auto pos = device.id.find(':');
        if (pos != std::string::npos)
            dev_idx = std::stoi(device.id.substr(pos + 1));
        cudaSetDevice(dev_idx);

        // Use a large array — need enough data to saturate HBM
        // Target: ~256 MB per array (3 arrays for triad)
        size_t elem_size = (config.precision == Precision::FP64) ? 8 : 4;
        int64_t n = 256 * 1024 * 1024 / elem_size;  // ~256 MB per array

        // Cap based on device memory (3 arrays)
        size_t total_bytes = 3 * n * elem_size;
        while (total_bytes > device.memory_bytes * 0.5 && n > 1000000) {
            n /= 2;
            total_bytes = 3 * n * elem_size;
        }

        int sms = device.compute_units;
        int tpb = config.gpu_threads_per_block;
        int blocks = (config.gpu_blocks > 0) ? config.gpu_blocks : sms * config.gpu_blocks_per_sm;

        // Triad: 3 memory ops per element × elem_size bytes
        double bytes_per_trial = 3.0 * n * elem_size;
        // FLOPs: 2 per element (multiply + add) — but bandwidth is the metric
        int64_t flops_per_trial = n * 2;

        std::cerr << "  Running stream_triad [cuda/" << precision_to_string(config.precision)
                  << "] n=" << n << " ("
                  << (n * elem_size / (1024*1024)) << " MB/array)" << std::endl;

        // Warmup
        for (int w = 0; w < 3; w++) {
            if (config.precision == Precision::FP64)
                run_stream_triad_fp64(n, blocks, tpb);
            else
                run_stream_triad_fp32(n, blocks, tpb);
        }

        // Measurement
        std::vector<double> times;
        times.reserve(measurement_trials);
        for (int t = 0; t < measurement_trials; t++) {
            float ms;
            if (config.precision == Precision::FP64)
                ms = run_stream_triad_fp64(n, blocks, tpb);
            else
                ms = run_stream_triad_fp32(n, blocks, tpb);
            times.push_back(static_cast<double>(ms));
        }

        auto stats = TimingStats::compute(times);

        KernelResult result;
        result.median_time_ms = stats.median_ms;
        result.min_time_ms = stats.min_ms;
        result.max_time_ms = stats.max_ms;
        result.total_flops = flops_per_trial;

        // Primary metric: bandwidth in GB/s — store in gflops field
        // We repurpose gflops to hold GB/s for memory kernels
        result.gflops = (bytes_per_trial / 1e9) / (stats.median_ms / 1e3);
        result.effective_gflops = result.gflops;

        // Peak percent vs theoretical HBM bandwidth
        // A100 PCIe: 1555 GB/s, A100 SXM: 2039 GB/s, H100 SXM: 3352 GB/s
        // We don't have this in DeviceInfo yet, so just report raw
        result.peak_percent = 0.0;  // TODO: add theoretical bandwidth to DeviceInfo

        result.clock_mhz = device.boost_clock_mhz;
        return result;
    }
};

REGISTER_KERNEL(StreamTriadCuda);

// ============================================================================
// Kernel class — STREAM Copy (pure bandwidth, no FLOPs)
// ============================================================================

class StreamCopyCuda : public KernelBase {
public:
    std::string name() const override { return "stream_copy"; }
    std::string category() const override { return "memory"; }
    std::string backend() const override { return "cuda"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP64};  // Copy is precision-agnostic; one size enough
    }

    std::vector<std::string> supported_modes() const override {
        return {"throughput"};
    }

    KernelResult run(const KernelConfig& config,
                     const DeviceInfo& device,
                     int measurement_trials) override {
        int dev_idx = 0;
        auto pos = device.id.find(':');
        if (pos != std::string::npos)
            dev_idx = std::stoi(device.id.substr(pos + 1));
        cudaSetDevice(dev_idx);

        int64_t n = 256 * 1024 * 1024 / 8;  // ~256 MB per array
        size_t total_bytes = 2 * n * 8;
        while (total_bytes > device.memory_bytes * 0.5 && n > 1000000) {
            n /= 2;
            total_bytes = 2 * n * 8;
        }

        int sms = device.compute_units;
        int tpb = config.gpu_threads_per_block;
        int blocks = (config.gpu_blocks > 0) ? config.gpu_blocks : sms * config.gpu_blocks_per_sm;

        // Copy: 2 memory ops per element (read + write) × 8 bytes
        double bytes_per_trial = 2.0 * n * 8;

        std::cerr << "  Running stream_copy [cuda/FP64] n=" << n
                  << " (" << (n * 8 / (1024*1024)) << " MB/array)" << std::endl;

        for (int w = 0; w < 3; w++) {
            run_stream_copy_fp64(n, blocks, tpb);
        }

        std::vector<double> times;
        times.reserve(measurement_trials);
        for (int t = 0; t < measurement_trials; t++) {
            times.push_back(static_cast<double>(run_stream_copy_fp64(n, blocks, tpb)));
        }

        auto stats = TimingStats::compute(times);

        KernelResult result;
        result.median_time_ms = stats.median_ms;
        result.min_time_ms = stats.min_ms;
        result.max_time_ms = stats.max_ms;
        result.total_flops = 0;  // no FLOPs in a copy

        // GB/s
        result.gflops = (bytes_per_trial / 1e9) / (stats.median_ms / 1e3);
        result.effective_gflops = result.gflops;
        result.peak_percent = 0.0;

        result.clock_mhz = device.boost_clock_mhz;
        return result;
    }
};

REGISTER_KERNEL(StreamCopyCuda);

namespace force_link {
    void stream_triad_cuda_link() {}
    void stream_copy_cuda_link() {}
}

} // namespace floptic

#endif // FLOPTIC_HAS_CUDA
