#ifdef FLOPTIC_HAS_HIP

#include "floptic/kernel_base.hpp"
#include "floptic/kernel_registry.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdint>

namespace floptic {

// ============================================================================
// STREAM Triad: a[i] = b[i] + scalar * c[i]
// Measures HBM bandwidth (3 arrays × N elements × sizeof(T))
// ============================================================================

template <typename T>
__global__ void stream_triad_kernel(T* __restrict__ a,
                                     const T* __restrict__ b,
                                     const T* __restrict__ c,
                                     T scalar, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < N; i += stride) {
        a[i] = b[i] + scalar * c[i];
    }
}

template <typename T>
__global__ void stream_copy_kernel(T* __restrict__ a,
                                    const T* __restrict__ b,
                                    int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < N; i += stride) {
        a[i] = b[i];
    }
}

template <typename T>
static float run_triad(int blocks, int tpb, T* a, T* b, T* c, T scalar, int64_t N) {
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    stream_triad_kernel<<<blocks, tpb>>>(a, b, c, scalar, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return ms;
}

template <typename T>
static float run_copy(int blocks, int tpb, T* a, T* b, int64_t N) {
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    stream_copy_kernel<<<blocks, tpb>>>(a, b, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return ms;
}

// ============================================================================
// Stream Triad kernel class
// ============================================================================

class StreamTriadHip : public KernelBase {
public:
    std::string name() const override { return "stream_triad"; }
    std::string category() const override { return "memory"; }
    std::string backend() const override { return "hip"; }

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
        hipSetDevice(dev_idx);

        int cus = device.compute_units;
        int tpb = config.gpu_threads_per_block > 0 ? config.gpu_threads_per_block : 256;
        int bpcu = config.gpu_blocks_per_sm > 0 ? config.gpu_blocks_per_sm : 4;
        int blocks = config.gpu_blocks > 0 ? config.gpu_blocks : cus * bpcu;

        // Use ~100 MB per array
        size_t elem_size = (config.precision == Precision::FP64) ? 8 : 4;
        int64_t N = (100 * 1024 * 1024) / elem_size;

        KernelResult result;

        if (config.precision == Precision::FP64) {
            double *a, *b, *c;
            hipMalloc(&a, N * sizeof(double));
            hipMalloc(&b, N * sizeof(double));
            hipMalloc(&c, N * sizeof(double));
            hipMemset(a, 0, N * sizeof(double));
            hipMemset(b, 0, N * sizeof(double));
            hipMemset(c, 0, N * sizeof(double));

            // Warmup
            for (int w = 0; w < config.warmup; w++)
                run_triad(blocks, tpb, a, b, c, 2.0, N);

            std::vector<double> times;
            for (int t = 0; t < measurement_trials; t++) {
                float ms = run_triad(blocks, tpb, a, b, c, 2.0, N);
                times.push_back(ms);
            }

            hipFree(a); hipFree(b); hipFree(c);

            std::sort(times.begin(), times.end());
            double median_ms = times[times.size() / 2];
            // Triad: 2 reads + 1 write = 3 * N * sizeof(double) bytes
            double bytes = 3.0 * N * sizeof(double);
            double gb_per_s = (bytes / (median_ms * 1e-3)) / 1e9;

            result.gflops = gb_per_s;  // reported as GB/s in the "gflops" field
            result.median_time_ms = median_ms;
            result.min_time_ms = times.front();
            result.max_time_ms = times.back();
            double sum = 0; for (auto t : times) sum += t;
            result.mean_time_ms = sum / times.size();
        } else {
            float *a, *b, *c;
            hipMalloc(&a, N * sizeof(float));
            hipMalloc(&b, N * sizeof(float));
            hipMalloc(&c, N * sizeof(float));
            hipMemset(a, 0, N * sizeof(float));
            hipMemset(b, 0, N * sizeof(float));
            hipMemset(c, 0, N * sizeof(float));

            for (int w = 0; w < config.warmup; w++)
                run_triad(blocks, tpb, a, b, c, 2.0f, N);

            std::vector<double> times;
            for (int t = 0; t < measurement_trials; t++) {
                float ms = run_triad(blocks, tpb, a, b, c, 2.0f, N);
                times.push_back(ms);
            }

            hipFree(a); hipFree(b); hipFree(c);

            std::sort(times.begin(), times.end());
            double median_ms = times[times.size() / 2];
            double bytes = 3.0 * N * sizeof(float);
            double gb_per_s = (bytes / (median_ms * 1e-3)) / 1e9;

            result.gflops = gb_per_s;
            result.median_time_ms = median_ms;
            result.min_time_ms = times.front();
            result.max_time_ms = times.back();
            double sum = 0; for (auto t : times) sum += t;
            result.mean_time_ms = sum / times.size();
        }

        return result;
    }
};

REGISTER_KERNEL(StreamTriadHip);

// ============================================================================
// Stream Copy kernel class
// ============================================================================

class StreamCopyHip : public KernelBase {
public:
    std::string name() const override { return "stream_copy"; }
    std::string category() const override { return "memory"; }
    std::string backend() const override { return "hip"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP64};
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
        hipSetDevice(dev_idx);

        int cus = device.compute_units;
        int tpb = config.gpu_threads_per_block > 0 ? config.gpu_threads_per_block : 256;
        int bpcu = config.gpu_blocks_per_sm > 0 ? config.gpu_blocks_per_sm : 4;
        int blocks = config.gpu_blocks > 0 ? config.gpu_blocks : cus * bpcu;

        int64_t N = (100 * 1024 * 1024) / sizeof(double);

        double *a, *b;
        hipMalloc(&a, N * sizeof(double));
        hipMalloc(&b, N * sizeof(double));
        hipMemset(a, 0, N * sizeof(double));
        hipMemset(b, 0, N * sizeof(double));

        for (int w = 0; w < config.warmup; w++)
            run_copy(blocks, tpb, a, b, N);

        std::vector<double> times;
        for (int t = 0; t < measurement_trials; t++) {
            float ms = run_copy(blocks, tpb, a, b, N);
            times.push_back(ms);
        }

        hipFree(a); hipFree(b);

        std::sort(times.begin(), times.end());
        double median_ms = times[times.size() / 2];
        double bytes = 2.0 * N * sizeof(double);
        double gb_per_s = (bytes / (median_ms * 1e-3)) / 1e9;

        KernelResult result;
        result.gflops = gb_per_s;
        result.median_time_ms = median_ms;
        result.min_time_ms = times.front();
        result.max_time_ms = times.back();
        double sum = 0; for (auto t : times) sum += t;
        result.mean_time_ms = sum / times.size();

        return result;
    }
};

REGISTER_KERNEL(StreamCopyHip);

namespace force_link {
    void stream_triad_hip_link() {
        volatile auto* p = &KernelRegistry::instance();
        (void)p;
    }
    void stream_copy_hip_link() {
        volatile auto* p = &KernelRegistry::instance();
        (void)p;
    }
}

} // namespace floptic

#endif // FLOPTIC_HAS_HIP
