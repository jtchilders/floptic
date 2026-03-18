#ifdef FLOPTIC_HAS_HIP

#include "floptic/kernel_base.hpp"
#include "floptic/kernel_registry.hpp"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace floptic {

// ============================================================================
// Helpers
// ============================================================================

static void check_rocblas(rocblas_status status, const char* msg) {
    if (status != rocblas_status_success) {
        std::cerr << "  rocBLAS error: " << msg << " (status=" << status << ")" << std::endl;
    }
}

static void check_hip(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "  HIP error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
    }
}

// ============================================================================
// GEMM runner for each precision
// ============================================================================

static float run_dgemm(rocblas_handle handle, int M, int N, int K,
                        double* A, double* B, double* C) {
    double alpha = 1.0, beta = 0.0;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    check_rocblas(rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                                M, N, K, &alpha, A, M, B, K, &beta, C, M),
                  "rocblas_dgemm");
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return ms;
}

static float run_sgemm(rocblas_handle handle, int M, int N, int K,
                        float* A, float* B, float* C) {
    float alpha = 1.0f, beta = 0.0f;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    check_rocblas(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                                M, N, K, &alpha, A, M, B, K, &beta, C, M),
                  "rocblas_sgemm");
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return ms;
}

static float run_hgemm(rocblas_handle handle, int M, int N, int K,
                        rocblas_half* A, rocblas_half* B, rocblas_half* C) {
    rocblas_half alpha, beta;
    // rocblas_half is typically __half or _Float16; use float conversion
    float alpha_f = 1.0f, beta_f = 0.0f;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    check_rocblas(rocblas_gemm_ex(handle,
                                  rocblas_operation_none, rocblas_operation_none,
                                  M, N, K,
                                  &alpha_f,
                                  A, rocblas_datatype_f16_r, M,
                                  B, rocblas_datatype_f16_r, K,
                                  &beta_f,
                                  C, rocblas_datatype_f16_r, M,
                                  C, rocblas_datatype_f16_r, M,
                                  rocblas_datatype_f32_r,  // compute in FP32
                                  rocblas_gemm_algo_standard,
                                  0, 0),
                  "rocblas_gemm_ex FP16");
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return ms;
}

static float run_bf16gemm(rocblas_handle handle, int M, int N, int K,
                           rocblas_bfloat16* A, rocblas_bfloat16* B, rocblas_bfloat16* C) {
    float alpha_f = 1.0f, beta_f = 0.0f;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    check_rocblas(rocblas_gemm_ex(handle,
                                  rocblas_operation_none, rocblas_operation_none,
                                  M, N, K,
                                  &alpha_f,
                                  A, rocblas_datatype_bf16_r, M,
                                  B, rocblas_datatype_bf16_r, K,
                                  &beta_f,
                                  C, rocblas_datatype_bf16_r, M,
                                  C, rocblas_datatype_bf16_r, M,
                                  rocblas_datatype_f32_r,
                                  rocblas_gemm_algo_standard,
                                  0, 0),
                  "rocblas_gemm_ex BF16");
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return ms;
}

static float run_int8gemm(rocblas_handle handle, int M, int N, int K,
                           int8_t* A, int8_t* B, int32_t* C) {
    int32_t alpha = 1, beta = 0;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    check_rocblas(rocblas_gemm_ex(handle,
                                  rocblas_operation_none, rocblas_operation_none,
                                  M, N, K,
                                  &alpha,
                                  A, rocblas_datatype_i8_r, M,
                                  B, rocblas_datatype_i8_r, K,
                                  &beta,
                                  C, rocblas_datatype_i32_r, M,
                                  C, rocblas_datatype_i32_r, M,
                                  rocblas_datatype_i32_r,
                                  rocblas_gemm_algo_standard,
                                  0, 0),
                  "rocblas_gemm_ex INT8");
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return ms;
}

// ============================================================================
// Auto-sweep + measure
// ============================================================================

struct SweepResult {
    int best_M;
    float best_ms;
    double best_gflops;
};

template <typename AllocT, typename RunFn>
static SweepResult sweep_and_measure(rocblas_handle handle,
                                      RunFn run_fn,
                                      int elem_size,
                                      int measurement_trials) {
    std::vector<int> sizes = {1024, 2048, 4096, 8192, 16384};
    SweepResult best = {0, 1e9f, 0.0};

    for (int M : sizes) {
        int64_t total_bytes = (int64_t)M * M * 3 * elem_size;
        // Skip if would use too much memory (leave 1 GB margin)
        if (total_bytes > 6LL * 1024 * 1024 * 1024) continue;

        AllocT* A = nullptr;
        AllocT* B = nullptr;
        AllocT* C = nullptr;
        hipError_t err = hipMalloc(&A, (size_t)M * M * elem_size);
        if (err != hipSuccess) continue;
        hipMalloc(&B, (size_t)M * M * elem_size);
        hipMalloc(&C, (size_t)M * M * elem_size);

        hipMemset(A, 0, (size_t)M * M * elem_size);
        hipMemset(B, 0, (size_t)M * M * elem_size);
        hipMemset(C, 0, (size_t)M * M * elem_size);

        // Warmup
        run_fn(handle, M, M, M, A, B, C);
        run_fn(handle, M, M, M, A, B, C);
        hipDeviceSynchronize();

        // Measure
        std::vector<float> times;
        for (int t = 0; t < measurement_trials; t++) {
            float ms = run_fn(handle, M, M, M, A, B, C);
            if (ms > 0) times.push_back(ms);
        }

        hipFree(A);
        hipFree(B);
        hipFree(C);

        if (times.empty()) continue;
        std::sort(times.begin(), times.end());
        float median_ms = times[times.size() / 2];
        double flops = 2.0 * (double)M * M * M;
        double gflops = (flops / (median_ms * 1e-3)) / 1e9;

        if (gflops > best.best_gflops) {
            best.best_M = M;
            best.best_ms = median_ms;
            best.best_gflops = gflops;
        }
    }

    return best;
}

// ============================================================================
// Kernel class
// ============================================================================

class GemmRocblas : public KernelBase {
public:
    std::string name() const override { return "gemm_rocblas"; }
    std::string category() const override { return "matrix"; }
    std::string backend() const override { return "hip"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP64, Precision::FP32, Precision::FP16, Precision::BF16, Precision::INT8};
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

        rocblas_handle handle;
        check_rocblas(rocblas_create_handle(&handle), "create handle");

        KernelResult result;
        SweepResult sr;
        std::string peak_key;

        switch (config.precision) {
            case Precision::FP64: {
                sr = sweep_and_measure<double>(handle, run_dgemm, sizeof(double), measurement_trials);
                // Use FP64_MFMA peak if available (rocBLAS uses matrix cores automatically)
                peak_key = device.theoretical_peak_gflops.count("FP64_MFMA") ? "FP64_MFMA" : "FP64";
                break;
            }
            case Precision::FP32: {
                sr = sweep_and_measure<float>(handle, run_sgemm, sizeof(float), measurement_trials);
                peak_key = device.theoretical_peak_gflops.count("FP32_MFMA") ? "FP32_MFMA" : "FP32";
                break;
            }
            case Precision::FP16: {
                sr = sweep_and_measure<rocblas_half>(handle, run_hgemm, sizeof(rocblas_half), measurement_trials);
                peak_key = device.theoretical_peak_gflops.count("FP16_MFMA") ? "FP16_MFMA" : "FP16";
                break;
            }
            case Precision::BF16: {
                sr = sweep_and_measure<rocblas_bfloat16>(handle, run_bf16gemm, sizeof(rocblas_bfloat16), measurement_trials);
                peak_key = device.theoretical_peak_gflops.count("BF16_MFMA") ? "BF16_MFMA" : "BF16";
                break;
            }
            case Precision::INT8: {
                // INT8 inputs, INT32 output — need separate allocation
                std::vector<int> sizes_i8 = {1024, 2048, 4096, 8192, 16384};
                sr = {0, 1e9f, 0.0};
                for (int M : sizes_i8) {
                    int64_t ab_bytes = (int64_t)M * M * sizeof(int8_t);
                    int64_t c_bytes  = (int64_t)M * M * sizeof(int32_t);
                    if (ab_bytes * 2 + c_bytes > 6LL * 1024 * 1024 * 1024) continue;

                    int8_t *A = nullptr, *B = nullptr;
                    int32_t *C = nullptr;
                    if (hipMalloc(&A, ab_bytes) != hipSuccess) continue;
                    hipMalloc(&B, ab_bytes);
                    hipMalloc(&C, c_bytes);
                    hipMemset(A, 0, ab_bytes);
                    hipMemset(B, 0, ab_bytes);
                    hipMemset(C, 0, c_bytes);

                    // Warmup
                    run_int8gemm(handle, M, M, M, A, B, C);
                    run_int8gemm(handle, M, M, M, A, B, C);
                    hipDeviceSynchronize();

                    std::vector<float> times;
                    for (int t = 0; t < measurement_trials; t++) {
                        float ms = run_int8gemm(handle, M, M, M, A, B, C);
                        if (ms > 0) times.push_back(ms);
                    }

                    hipFree(A); hipFree(B); hipFree(C);

                    if (times.empty()) continue;
                    std::sort(times.begin(), times.end());
                    float median_ms = times[times.size() / 2];
                    double flops = 2.0 * (double)M * M * M;
                    double gflops = (flops / (median_ms * 1e-3)) / 1e9;

                    if (gflops > sr.best_gflops) {
                        sr.best_M = M;
                        sr.best_ms = median_ms;
                        sr.best_gflops = gflops;
                    }
                }
                peak_key = device.theoretical_peak_gflops.count("INT8_MFMA") ? "INT8_MFMA" : "INT8";
                break;
            }
            default:
                std::cerr << "  Unsupported precision for gemm_rocblas" << std::endl;
                rocblas_destroy_handle(handle);
                return result;
        }

        result.gflops = sr.best_gflops;
        result.median_time_ms = sr.best_ms;
        result.min_time_ms = sr.best_ms;
        result.max_time_ms = sr.best_ms;

        auto it = device.theoretical_peak_gflops.find(peak_key);
        if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
            result.peak_percent = (result.gflops / it->second) * 100.0;
        }

        std::cerr << "  gemm_rocblas " << precision_to_string(config.precision)
                  << " best at M=N=K=" << sr.best_M
                  << ": " << sr.best_gflops << " GFLOP/s" << std::endl;

        rocblas_destroy_handle(handle);
        return result;
    }
};

REGISTER_KERNEL(GemmRocblas);

namespace force_link {
    void gemm_rocblas_link() {
        volatile auto* p = &KernelRegistry::instance();
        (void)p;
    }
}

} // namespace floptic

#endif // FLOPTIC_HAS_HIP
