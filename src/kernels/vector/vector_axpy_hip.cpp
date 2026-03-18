#ifdef FLOPTIC_HAS_HIP

#include "floptic/kernel_base.hpp"
#include "floptic/kernel_registry.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdint>

namespace floptic {

// ============================================================================
// AXPY: y[i] = alpha * x[i] + y[i]
// ============================================================================

template <typename T>
__global__ void axpy_kernel(T alpha, const T* __restrict__ x,
                            T* __restrict__ y, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < N; i += stride) {
        y[i] = alpha * x[i] + y[i];
    }
}

template <typename T>
static float run_axpy(int blocks, int tpb, T alpha, T* x, T* y, int64_t N) {
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    axpy_kernel<<<blocks, tpb>>>(alpha, x, y, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return ms;
}

class VectorAxpyHip : public KernelBase {
public:
    std::string name() const override { return "vector_axpy"; }
    std::string category() const override { return "vector"; }
    std::string backend() const override { return "hip"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP64, Precision::FP32, Precision::FP16, Precision::BF16};
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

        KernelResult result;

        auto measure = [&](auto alpha_val, auto* x_ptr, auto* y_ptr,
                          int64_t N, size_t elem_sz) {
            using T = std::remove_pointer_t<decltype(x_ptr)>;
            // Warmup
            for (int w = 0; w < 10; w++)
                run_axpy(blocks, tpb, alpha_val, x_ptr, y_ptr, N);

            std::vector<double> times;
            for (int t = 0; t < measurement_trials; t++) {
                float ms = run_axpy(blocks, tpb, alpha_val, x_ptr, y_ptr, N);
                times.push_back(ms);
            }

            std::sort(times.begin(), times.end());
            double median_ms = times[times.size() / 2];
            double flops = 2.0 * N;  // 1 mul + 1 add per element
            result.gflops = (flops / (median_ms * 1e-3)) / 1e9;
            result.median_time_ms = median_ms;
            result.min_time_ms = times.front();
            result.max_time_ms = times.back();
        };

        int64_t N = 10 * 1024 * 1024;  // 10M elements

        if (config.precision == Precision::FP64) {
            double *x, *y;
            hipMalloc(&x, N * sizeof(double));
            hipMalloc(&y, N * sizeof(double));
            hipMemset(x, 0, N * sizeof(double));
            hipMemset(y, 0, N * sizeof(double));
            measure(2.0, x, y, N, sizeof(double));
            hipFree(x); hipFree(y);
        } else if (config.precision == Precision::FP32) {
            float *x, *y;
            hipMalloc(&x, N * sizeof(float));
            hipMalloc(&y, N * sizeof(float));
            hipMemset(x, 0, N * sizeof(float));
            hipMemset(y, 0, N * sizeof(float));
            measure(2.0f, x, y, N, sizeof(float));
            hipFree(x); hipFree(y);
        } else if (config.precision == Precision::FP16) {
            _Float16 *x, *y;
            hipMalloc(&x, N * sizeof(_Float16));
            hipMalloc(&y, N * sizeof(_Float16));
            hipMemset(x, 0, N * sizeof(_Float16));
            hipMemset(y, 0, N * sizeof(_Float16));
            _Float16 alpha = static_cast<_Float16>(2.0f);
            measure(alpha, x, y, N, sizeof(_Float16));
            hipFree(x); hipFree(y);
        } else if (config.precision == Precision::BF16) {
            hip_bfloat16 *x, *y;
            hipMalloc(&x, N * sizeof(hip_bfloat16));
            hipMalloc(&y, N * sizeof(hip_bfloat16));
            hipMemset(x, 0, N * sizeof(hip_bfloat16));
            hipMemset(y, 0, N * sizeof(hip_bfloat16));
            hip_bfloat16 alpha(2.0f);
            measure(alpha, x, y, N, sizeof(hip_bfloat16));
            hipFree(x); hipFree(y);
        }

        // Peak comparison
        std::string peak_key = precision_to_string(config.precision);
        auto it = device.theoretical_peak_gflops.find(peak_key);
        if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
            result.peak_percent = (result.gflops / it->second) * 100.0;
        }

        return result;
    }
};

REGISTER_KERNEL(VectorAxpyHip);

namespace force_link {
    void vector_axpy_hip_link() {
        volatile auto* p = &KernelRegistry::instance();
        (void)p;
    }
}

} // namespace floptic

#endif // FLOPTIC_HAS_HIP
