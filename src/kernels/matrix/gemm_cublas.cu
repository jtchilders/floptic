#ifdef FLOPTIC_HAS_CUDA

#include "floptic/kernel_base.hpp"
#include "floptic/kernel_registry.hpp"
#include "floptic/timer.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cmath>

namespace floptic {

// ============================================================================
// cuBLAS GEMM wrapper per precision
//
// For each precision, we do C = alpha*A*B + beta*C with square matrices.
// FLOPs = 2 * M * N * K per call.
//
// Precision modes:
//   FP64: cublasDgemm (CUDA cores)
//   FP32: cublasSgemm (CUDA cores)
//   FP16: cublasGemmEx with CUBLAS_COMPUTE_16F (tensor cores if available)
//   BF16: cublasGemmEx with CUBLAS_COMPUTE_32F (tensor cores for BF16 input)
//   TF32: cublasSgemm with CUBLAS_COMPUTE_32F_FAST_TF32 (tensor cores)
// ============================================================================

static void check_cublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "  cuBLAS ERROR: " << msg << " (status=" << status << ")" << std::endl;
    }
}

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "  CUDA ERROR: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
    }
}

// ============================================================================
// FP64 GEMM: cublasDgemm
// ============================================================================

static float run_gemm_fp64(cublasHandle_t handle, int M, int N, int K) {
    double *d_A, *d_B, *d_C;
    check_cuda(cudaMalloc(&d_A, (size_t)M * K * sizeof(double)), "alloc A");
    check_cuda(cudaMalloc(&d_B, (size_t)K * N * sizeof(double)), "alloc B");
    check_cuda(cudaMalloc(&d_C, (size_t)M * N * sizeof(double)), "alloc C");

    // Initialize with curand-like pattern (just fill with constants for timing)
    double val = 1.0 / M;
    // Use cublasDscal-like approach: fill via kernel or memset
    // For simplicity, set to zero then rely on beta=0 for first call
    cudaMemset(d_A, 0, (size_t)M * K * sizeof(double));
    cudaMemset(d_B, 0, (size_t)K * N * sizeof(double));
    cudaMemset(d_C, 0, (size_t)M * N * sizeof(double));

    double alpha = 1.0, beta = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M),
                 "cublasDgemm");
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
// FP32 GEMM: cublasSgemm
// ============================================================================

static float run_gemm_fp32(cublasHandle_t handle, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    check_cuda(cudaMalloc(&d_A, (size_t)M * K * sizeof(float)), "alloc A");
    check_cuda(cudaMalloc(&d_B, (size_t)K * N * sizeof(float)), "alloc B");
    check_cuda(cudaMalloc(&d_C, (size_t)M * N * sizeof(float)), "alloc C");

    cudaMemset(d_A, 0, (size_t)M * K * sizeof(float));
    cudaMemset(d_B, 0, (size_t)K * N * sizeof(float));
    cudaMemset(d_C, 0, (size_t)M * N * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M),
                 "cublasSgemm");
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
// FP16 GEMM: cublasGemmEx with FP16 input, FP16 accumulator (tensor cores)
// ============================================================================

static float run_gemm_fp16(cublasHandle_t handle, int M, int N, int K) {
    __half *d_A, *d_B, *d_C;
    check_cuda(cudaMalloc(&d_A, (size_t)M * K * sizeof(__half)), "alloc A");
    check_cuda(cudaMalloc(&d_B, (size_t)K * N * sizeof(__half)), "alloc B");
    check_cuda(cudaMalloc(&d_C, (size_t)M * N * sizeof(__half)), "alloc C");

    cudaMemset(d_A, 0, (size_t)M * K * sizeof(__half));
    cudaMemset(d_B, 0, (size_t)K * N * sizeof(__half));
    cudaMemset(d_C, 0, (size_t)M * N * sizeof(__half));

    __half alpha = __float2half(1.0f);
    __half beta  = __float2half(0.0f);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    check_cublas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               M, N, K,
                               &alpha,
                               d_A, CUDA_R_16F, M,
                               d_B, CUDA_R_16F, K,
                               &beta,
                               d_C, CUDA_R_16F, M,
                               CUBLAS_COMPUTE_16F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                 "cublasGemmEx FP16");
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
// BF16 GEMM: cublasGemmEx with BF16 input, FP32 accumulator (tensor cores)
// ============================================================================

static float run_gemm_bf16(cublasHandle_t handle, int M, int N, int K) {
    __nv_bfloat16 *d_A, *d_B;
    float *d_C;  // accumulate in FP32

    check_cuda(cudaMalloc(&d_A, (size_t)M * K * sizeof(__nv_bfloat16)), "alloc A");
    check_cuda(cudaMalloc(&d_B, (size_t)K * N * sizeof(__nv_bfloat16)), "alloc B");
    check_cuda(cudaMalloc(&d_C, (size_t)M * N * sizeof(float)), "alloc C");

    cudaMemset(d_A, 0, (size_t)M * K * sizeof(__nv_bfloat16));
    cudaMemset(d_B, 0, (size_t)K * N * sizeof(__nv_bfloat16));
    cudaMemset(d_C, 0, (size_t)M * N * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    check_cublas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               M, N, K,
                               &alpha,
                               d_A, CUDA_R_16BF, M,
                               d_B, CUDA_R_16BF, K,
                               &beta,
                               d_C, CUDA_R_32F, M,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                 "cublasGemmEx BF16");
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
// TF32 GEMM: cublasSgemm with TF32 math mode (tensor cores, FP32 I/O)
// ============================================================================

static float run_gemm_tf32(cublasHandle_t handle, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    check_cuda(cudaMalloc(&d_A, (size_t)M * K * sizeof(float)), "alloc A");
    check_cuda(cudaMalloc(&d_B, (size_t)K * N * sizeof(float)), "alloc B");
    check_cuda(cudaMalloc(&d_C, (size_t)M * N * sizeof(float)), "alloc C");

    cudaMemset(d_A, 0, (size_t)M * K * sizeof(float));
    cudaMemset(d_B, 0, (size_t)K * N * sizeof(float));
    cudaMemset(d_C, 0, (size_t)M * N * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;

    // Enable TF32 tensor math
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M),
                 "cublasSgemm TF32");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Reset math mode
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

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
// Dispatch
// ============================================================================

static float dispatch_gemm(cublasHandle_t handle, Precision prec, int M, int N, int K) {
    switch (prec) {
        case Precision::FP64: return run_gemm_fp64(handle, M, N, K);
        case Precision::FP32: return run_gemm_fp32(handle, M, N, K);
        case Precision::FP16: return run_gemm_fp16(handle, M, N, K);
        case Precision::BF16: return run_gemm_bf16(handle, M, N, K);
        case Precision::TF32: return run_gemm_tf32(handle, M, N, K);
        default:
            std::cerr << "  ERROR: Unsupported precision for gemm_cublas" << std::endl;
            return 0.0f;
    }
}

// ============================================================================
// Choose peak key for each precision mode
// FP16/BF16/TF32 GEMM via cuBLAS will use tensor cores, so compare against
// tensor core peaks
// ============================================================================

static std::string peak_key_for(Precision p) {
    switch (p) {
        case Precision::FP64: return "FP64";      // CUDA core DGEMM
        case Precision::FP32: return "FP32";       // CUDA core SGEMM
        case Precision::FP16: return "FP16_TC";    // tensor core HGEMM
        case Precision::BF16: return "FP16_TC";    // tensor core (same rate as FP16)
        case Precision::TF32: return "FP16_TC";    // TF32 on tensor cores ~= FP16_TC/2
        default: return "FP32";
    }
}

// ============================================================================
// Kernel class
// ============================================================================

class GemmCublas : public KernelBase {
public:
    std::string name() const override { return "gemm_cublas"; }
    std::string category() const override { return "matrix"; }
    std::string backend() const override { return "cuda"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP64, Precision::FP32, Precision::FP16,
                Precision::BF16, Precision::TF32};
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

        // Matrix size: use inner_iters to scale
        // Default (100K inner_iters): M=N=K=4096
        // We use a fixed set of sizes based on the iterations
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

        // Check memory: need 3 matrices × M×K × elem_size
        size_t elem_size = 8; // default FP64
        switch (config.precision) {
            case Precision::FP16: elem_size = 2; break;
            case Precision::BF16: elem_size = 2; break;
            case Precision::FP32: elem_size = 4; break;
            case Precision::TF32: elem_size = 4; break;
            default: elem_size = 8;
        }
        size_t needed = 3ULL * M * K * elem_size;
        if (needed > device.memory_bytes * 0.8) {
            // Reduce matrix size
            while (needed > device.memory_bytes * 0.8 && M > 512) {
                M /= 2; N /= 2; K /= 2;
                needed = 3ULL * M * K * elem_size;
            }
        }

        // FLOPs per GEMM: 2 * M * N * K
        int64_t flops_per_trial = 2LL * M * N * K;

        std::string tc_note = "";
        if (config.precision == Precision::FP16 || config.precision == Precision::BF16 ||
            config.precision == Precision::TF32) {
            tc_note = " (tensor cores)";
        }

        std::cerr << "  Running gemm_cublas [cuda/" << precision_to_string(config.precision)
                  << "/" << config.mode << tc_note << "] M=N=K=" << M << std::endl;

        // Create cuBLAS handle
        cublasHandle_t handle;
        check_cublas(cublasCreate(&handle), "cublasCreate");

        // Warmup
        for (int w = 0; w < 3; w++) {
            dispatch_gemm(handle, config.precision, M, N, K);
        }

        // Measurement
        std::vector<double> times;
        times.reserve(measurement_trials);

        for (int t = 0; t < measurement_trials; t++) {
            float ms = dispatch_gemm(handle, config.precision, M, N, K);
            times.push_back(static_cast<double>(ms));
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

        // Peak percent — use tensor core peak for FP16/BF16/TF32
        std::string pk = peak_key_for(config.precision);
        auto it = device.theoretical_peak_gflops.find(pk);
        if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
            result.peak_percent = (result.gflops / it->second) * 100.0;
        }

        // Report which peak we compared against
        std::cerr << "  (peak% vs " << pk << " theoretical)" << std::endl;

        result.clock_mhz = device.boost_clock_mhz;
        return result;
    }
};

REGISTER_KERNEL(GemmCublas);

namespace force_link {
    void gemm_cublas_link() {}
}

} // namespace floptic

#endif // FLOPTIC_HAS_CUDA
