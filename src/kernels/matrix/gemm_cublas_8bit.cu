#ifdef FLOPTIC_HAS_CUDA

#include "floptic/kernel_base.hpp"
#include "floptic/kernel_registry.hpp"
#include "floptic/timer.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstring>

namespace floptic {

// ============================================================================
// Helper: check cuBLAS errors
// ============================================================================

static void check_cublas_8(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "  cuBLAS ERROR: " << msg << " (status=" << status << ")" << std::endl;
    }
}

static void check_cuda_8(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "  CUDA ERROR: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
    }
}

// ============================================================================
// INT8 GEMM via cublasGemmEx
// Input: INT8, Output: INT32, Compute: INT32
// FLOPs convention: 2*M*N*K (same as float GEMM)
// ============================================================================

static float run_gemm_int8(cublasHandle_t handle, int M, int N, int K) {
    int8_t *d_A, *d_B;
    int32_t *d_C;

    check_cuda_8(cudaMalloc(&d_A, (size_t)M * K * sizeof(int8_t)), "alloc A");
    check_cuda_8(cudaMalloc(&d_B, (size_t)K * N * sizeof(int8_t)), "alloc B");
    check_cuda_8(cudaMalloc(&d_C, (size_t)M * N * sizeof(int32_t)), "alloc C");

    cudaMemset(d_A, 1, (size_t)M * K * sizeof(int8_t));
    cudaMemset(d_B, 1, (size_t)K * N * sizeof(int8_t));
    cudaMemset(d_C, 0, (size_t)M * N * sizeof(int32_t));

    int32_t alpha = 1, beta = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    check_cublas_8(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 M, N, K,
                                 &alpha,
                                 d_A, CUDA_R_8I, M,
                                 d_B, CUDA_R_8I, K,
                                 &beta,
                                 d_C, CUDA_R_32I, M,
                                 CUBLAS_COMPUTE_32I,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                   "cublasGemmEx INT8");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return ms;
}

// ============================================================================
// FP8 GEMM via cublasLtMatmul (Hopper+ only, sm_89/sm_90)
// Input: FP8 E4M3, Output: FP16 or FP32, Compute: FP32
//
// FP8 E4M3: 4-bit exponent, 3-bit mantissa (range-optimized)
// FP8 E5M2: 5-bit exponent, 2-bit mantissa (IEEE-like)
// ============================================================================

#if defined(__CUDA_FP8_TYPES_EXIST__) || (__CUDACC_VER_MAJOR__ >= 12)
#define FLOPTIC_HAS_FP8 1
#endif

#ifdef FLOPTIC_HAS_FP8

static float run_gemm_fp8_e4m3(cublasLtHandle_t ltHandle, int M, int N, int K) {
    // Allocate: A and B as FP8 E4M3, C and D as FP16
    void *d_A, *d_B;
    __half *d_C, *d_D;

    check_cuda_8(cudaMalloc(&d_A, (size_t)M * K), "alloc A fp8");
    check_cuda_8(cudaMalloc(&d_B, (size_t)K * N), "alloc B fp8");
    check_cuda_8(cudaMalloc(&d_C, (size_t)M * N * sizeof(__half)), "alloc C");
    check_cuda_8(cudaMalloc(&d_D, (size_t)M * N * sizeof(__half)), "alloc D");

    cudaMemset(d_A, 0, (size_t)M * K);
    cudaMemset(d_B, 0, (size_t)K * N);
    cudaMemset(d_C, 0, (size_t)M * N * sizeof(__half));
    cudaMemset(d_D, 0, (size_t)M * N * sizeof(__half));

    // Create operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // Create matrix layouts
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, M, K, M);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, N, K);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, M, N, M);
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16F, M, N, M);

    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cublasStatus_t status = cublasLtMatmul(ltHandle, operationDesc,
                                            &alpha,
                                            d_A, Adesc,
                                            d_B, Bdesc,
                                            &beta,
                                            d_C, Cdesc,
                                            d_D, Ddesc,
                                            nullptr, nullptr, 0, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "  cublasLtMatmul FP8 E4M3 failed (status=" << status << ")" << std::endl;
    }

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(operationDesc);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    return ms;
}

static float run_gemm_fp8_e5m2(cublasLtHandle_t ltHandle, int M, int N, int K) {
    void *d_A, *d_B;
    __half *d_C, *d_D;

    check_cuda_8(cudaMalloc(&d_A, (size_t)M * K), "alloc A fp8");
    check_cuda_8(cudaMalloc(&d_B, (size_t)K * N), "alloc B fp8");
    check_cuda_8(cudaMalloc(&d_C, (size_t)M * N * sizeof(__half)), "alloc C");
    check_cuda_8(cudaMalloc(&d_D, (size_t)M * N * sizeof(__half)), "alloc D");

    cudaMemset(d_A, 0, (size_t)M * K);
    cudaMemset(d_B, 0, (size_t)K * N);
    cudaMemset(d_C, 0, (size_t)M * N * sizeof(__half));
    cudaMemset(d_D, 0, (size_t)M * N * sizeof(__half));

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E5M2, M, K, M);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E5M2, K, N, K);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, M, N, M);
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16F, M, N, M);

    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cublasStatus_t status = cublasLtMatmul(ltHandle, operationDesc,
                                            &alpha,
                                            d_A, Adesc,
                                            d_B, Bdesc,
                                            &beta,
                                            d_C, Cdesc,
                                            d_D, Ddesc,
                                            nullptr, nullptr, 0, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "  cublasLtMatmul FP8 E5M2 failed (status=" << status << ")" << std::endl;
    }

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(operationDesc);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    return ms;
}

#endif // FLOPTIC_HAS_FP8

// ============================================================================
// Kernel class: INT8 GEMM
// ============================================================================

class GemmCublasInt8 : public KernelBase {
public:
    std::string name() const override { return "gemm_cublas_int8"; }
    std::string category() const override { return "matrix"; }
    std::string backend() const override { return "cuda"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::INT8};
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

        int M, N, K;
        if (config.iterations <= 1000) {
            M = N = K = 1024;
        } else if (config.iterations <= 10000) {
            M = N = K = 2048;
        } else if (config.iterations <= 100000) {
            M = N = K = 4096;
        } else {
            M = N = K = 8192;
        }

        int64_t flops_per_trial = 2LL * M * N * K;

        std::cerr << "  Running gemm_cublas_int8 [cuda/INT8/throughput (tensor cores)] M=N=K="
                  << M << std::endl;

        cublasHandle_t handle;
        check_cublas_8(cublasCreate(&handle), "cublasCreate");

        // Warmup
        for (int w = 0; w < 3; w++) {
            run_gemm_int8(handle, M, N, K);
        }

        std::vector<double> times;
        times.reserve(measurement_trials);
        for (int t = 0; t < measurement_trials; t++) {
            times.push_back(static_cast<double>(run_gemm_int8(handle, M, N, K)));
        }

        cublasDestroy(handle);

        auto stats = TimingStats::compute(times);

        KernelResult result;
        result.median_time_ms = stats.median_ms;
        result.min_time_ms = stats.min_ms;
        result.max_time_ms = stats.max_ms;
        result.total_flops = flops_per_trial;
        result.gflops = (flops_per_trial / 1e9) / (stats.median_ms / 1e3);
        result.effective_gflops = result.gflops;

        // INT8 TC peak: typically 2× FP16 TC rate
        // Use FP16_TC as reference × 2 if we don't have a specific INT8 peak
        auto it = device.theoretical_peak_gflops.find("FP16_TC");
        if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
            double int8_peak = it->second * 2.0;  // INT8 = 2× FP16 TC rate
            result.peak_percent = (result.gflops / int8_peak) * 100.0;
        }
        std::cerr << "  (peak% vs INT8_TC estimate = 2×FP16_TC)" << std::endl;

        result.clock_mhz = device.boost_clock_mhz;
        return result;
    }
};

REGISTER_KERNEL(GemmCublasInt8);

// ============================================================================
// Kernel class: FP8 GEMM (Hopper+ only)
// ============================================================================

#ifdef FLOPTIC_HAS_FP8

class GemmCublasFp8 : public KernelBase {
public:
    std::string name() const override { return "gemm_cublas_fp8"; }
    std::string category() const override { return "matrix"; }
    std::string backend() const override { return "cuda"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP8_E4M3, Precision::FP8_E5M2};
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

        int M, N, K;
        if (config.iterations <= 1000) {
            M = N = K = 1024;
        } else if (config.iterations <= 10000) {
            M = N = K = 2048;
        } else if (config.iterations <= 100000) {
            M = N = K = 4096;
        } else {
            M = N = K = 8192;
        }

        int64_t flops_per_trial = 2LL * M * N * K;

        std::string fp8_name = (config.precision == Precision::FP8_E4M3) ? "FP8_E4M3" : "FP8_E5M2";
        std::cerr << "  Running gemm_cublas_fp8 [cuda/" << fp8_name
                  << "/throughput (tensor cores)] M=N=K=" << M << std::endl;

        cublasLtHandle_t ltHandle;
        cublasLtCreate(&ltHandle);

        auto run_fn = [&]() -> float {
            if (config.precision == Precision::FP8_E4M3)
                return run_gemm_fp8_e4m3(ltHandle, M, N, K);
            else
                return run_gemm_fp8_e5m2(ltHandle, M, N, K);
        };

        // Warmup
        for (int w = 0; w < 3; w++) {
            run_fn();
        }

        std::vector<double> times;
        times.reserve(measurement_trials);
        for (int t = 0; t < measurement_trials; t++) {
            times.push_back(static_cast<double>(run_fn()));
        }

        cublasLtDestroy(ltHandle);

        auto stats = TimingStats::compute(times);

        KernelResult result;
        result.median_time_ms = stats.median_ms;
        result.min_time_ms = stats.min_ms;
        result.max_time_ms = stats.max_ms;
        result.total_flops = flops_per_trial;
        result.gflops = (flops_per_trial / 1e9) / (stats.median_ms / 1e3);
        result.effective_gflops = result.gflops;

        // FP8 TC peak: same as FP16 TC × 2 on Hopper
        auto it = device.theoretical_peak_gflops.find("FP16_TC");
        if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
            double fp8_peak = it->second * 2.0;
            result.peak_percent = (result.gflops / fp8_peak) * 100.0;
        }
        std::cerr << "  (peak% vs FP8_TC estimate = 2×FP16_TC)" << std::endl;

        result.clock_mhz = device.boost_clock_mhz;
        return result;
    }
};

REGISTER_KERNEL(GemmCublasFp8);

#endif // FLOPTIC_HAS_FP8

// ============================================================================
// Force link
// ============================================================================

namespace force_link {
    void gemm_cublas_int8_link() {}
    void gemm_cublas_fp8_link() {}
}

} // namespace floptic

#endif // FLOPTIC_HAS_CUDA
