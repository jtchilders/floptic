#ifdef FLOPTIC_HAS_CUDA

#include "floptic/kernel_base.hpp"
#include "floptic/kernel_registry.hpp"
#include "floptic/timer.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstring>

namespace floptic {

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
// INT8 GEMM via cublasLtMatmul
// Input: INT8, Output: INT32, Compute: INT32
// ============================================================================

// Persistent handles to avoid create/destroy overhead per call
static cublasLtHandle_t g_ltHandle_int8 = nullptr;
static void* g_workspace_int8 = nullptr;
static size_t g_workspaceSize_int8 = 64 * 1024 * 1024;  // 64 MB

// Persistent device buffers for INT8
static int8_t* g_d_A_int8 = nullptr;
static int8_t* g_d_B_int8 = nullptr;
static int32_t* g_d_C_int8 = nullptr;
static int g_M_int8 = 0, g_N_int8 = 0, g_K_int8 = 0;

static void setup_int8(int M, int N, int K) {
    if (g_M_int8 == M && g_N_int8 == N && g_K_int8 == K && g_ltHandle_int8)
        return;

    // Cleanup old
    if (g_d_A_int8) cudaFree(g_d_A_int8);
    if (g_d_B_int8) cudaFree(g_d_B_int8);
    if (g_d_C_int8) cudaFree(g_d_C_int8);
    if (g_workspace_int8) cudaFree(g_workspace_int8);
    if (g_ltHandle_int8) cublasLtDestroy(g_ltHandle_int8);

    g_M_int8 = M; g_N_int8 = N; g_K_int8 = K;

    check_cuda_8(cudaMalloc(&g_d_A_int8, (size_t)M * K), "alloc A int8");
    check_cuda_8(cudaMalloc(&g_d_B_int8, (size_t)K * N), "alloc B int8");
    check_cuda_8(cudaMalloc(&g_d_C_int8, (size_t)M * N * sizeof(int32_t)), "alloc C int8");
    check_cuda_8(cudaMalloc(&g_workspace_int8, g_workspaceSize_int8), "alloc workspace");

    cudaMemset(g_d_A_int8, 1, (size_t)M * K);
    cudaMemset(g_d_B_int8, 1, (size_t)K * N);
    cudaMemset(g_d_C_int8, 0, (size_t)M * N * sizeof(int32_t));

    cublasLtCreate(&g_ltHandle_int8);
}

static float run_gemm_int8(int M, int N, int K) {
    setup_int8(M, N, K);

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I);

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, M, K, M);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, K, N, K);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, M, N, M);

    int32_t alpha = 1, beta = 0;

    // Find best algorithm
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &g_workspaceSize_int8, sizeof(g_workspaceSize_int8));

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult;
    cublasLtMatmulAlgoGetHeuristic(g_ltHandle_int8, operationDesc,
                                    Adesc, Bdesc, Cdesc, Cdesc,
                                    preference, 1, &heuristicResult, &returnedResults);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float ms = 0.0f;
    if (returnedResults > 0) {
        // Print algo info on first call
        static bool first_call = true;
        if (first_call) {
            int algoId = 0;
            cublasLtMatmulAlgoConfigGetAttribute(&heuristicResult.algo,
                CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr);
            std::cerr << "  INT8 algo id=" << algoId
                      << ", workspace=" << heuristicResult.workspaceSize << std::endl;
            first_call = false;
        }
        cudaEventRecord(start);
        cublasLtMatmul(g_ltHandle_int8, operationDesc,
                        &alpha, g_d_A_int8, Adesc, g_d_B_int8, Bdesc,
                        &beta, g_d_C_int8, Cdesc, g_d_C_int8, Cdesc,
                        &heuristicResult.algo,
                        g_workspace_int8, g_workspaceSize_int8, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
    } else {
        std::cerr << "  No algorithm found for INT8 GEMM" << std::endl;
    }

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(operationDesc);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

// ============================================================================
// FP8 GEMM via cublasLtMatmul (Hopper+ only)
// ============================================================================

// Check if FP8 types are available at compile time
#if (__CUDACC_VER_MAJOR__ > 11) || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 8)
// CUDA_R_8F_E4M3 and CUDA_R_8F_E5M2 available in CUDA 11.8+ headers
// but actual FP8 TC support requires sm_89+ hardware
#define FLOPTIC_HAS_FP8_TYPES 1
#endif

#ifdef FLOPTIC_HAS_FP8_TYPES

// FP8 GEMM via cublasLtMatmul — following NVIDIA's official LtFp8Matmul sample:
//   D (FP8 E4M3) = alpha * A (FP8) × B (FP8) + beta * C (BF16)
//   With per-tensor scaling factors (REQUIRED for FP8)
//   C and D are SEPARATE buffers with different types

static float run_gemm_fp8(cudaDataType_t fp8Type, int M, int N, int K) {
    void *d_A, *d_B, *d_D;  // FP8
    void *d_C;               // BF16 (bias)

    check_cuda_8(cudaMalloc(&d_A, (size_t)M * K), "alloc A fp8");
    check_cuda_8(cudaMalloc(&d_B, (size_t)K * N), "alloc B fp8");
    check_cuda_8(cudaMalloc(&d_C, (size_t)M * N * 2), "alloc C bf16");
    check_cuda_8(cudaMalloc(&d_D, (size_t)M * N), "alloc D fp8");

    cudaMemset(d_A, 0x38, (size_t)M * K);
    cudaMemset(d_B, 0x38, (size_t)K * N);
    cudaMemset(d_C, 0, (size_t)M * N * 2);
    cudaMemset(d_D, 0, (size_t)M * N);

    // Per-tensor scaling factors (device pointers) — all set to 1.0 for benchmarking
    float *d_a_scale, *d_b_scale, *d_c_scale, *d_d_scale, *d_amax_d;
    float h_one = 1.0f, h_zero = 0.0f;
    cudaMalloc(&d_a_scale, sizeof(float));
    cudaMalloc(&d_b_scale, sizeof(float));
    cudaMalloc(&d_c_scale, sizeof(float));
    cudaMalloc(&d_d_scale, sizeof(float));
    cudaMalloc(&d_amax_d, sizeof(float));
    cudaMemcpy(d_a_scale, &h_one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_scale, &h_one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_scale, &h_one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_scale, &h_one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_amax_d, &h_zero, sizeof(float), cudaMemcpyHostToDevice);

    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // Set per-tensor scaling attributes (required for FP8)
    cublasStatus_t s;
    s = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_a_scale, sizeof(d_a_scale));
    if (s != CUBLAS_STATUS_SUCCESS) std::cerr << "  SetAttribute A_SCALE failed: " << s << std::endl;
    s = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_b_scale, sizeof(d_b_scale));
    if (s != CUBLAS_STATUS_SUCCESS) std::cerr << "  SetAttribute B_SCALE failed: " << s << std::endl;
    s = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &d_c_scale, sizeof(d_c_scale));
    if (s != CUBLAS_STATUS_SUCCESS) std::cerr << "  SetAttribute C_SCALE failed: " << s << std::endl;
    s = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_d_scale, sizeof(d_d_scale));
    if (s != CUBLAS_STATUS_SUCCESS) std::cerr << "  SetAttribute D_SCALE failed: " << s << std::endl;
    s = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &d_amax_d, sizeof(d_amax_d));
    if (s != CUBLAS_STATUS_SUCCESS) std::cerr << "  SetAttribute AMAX_D failed: " << s << std::endl;

    // A=FP8(M,K), B=FP8(K,N), C=BF16(M,N), D=FP8(M,N)
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
    cublasLtMatrixLayoutCreate(&Adesc, fp8Type, M, K, M);
    cublasLtMatrixLayoutCreate(&Bdesc, fp8Type, K, N, K);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, M, N, M);
    cublasLtMatrixLayoutCreate(&Ddesc, fp8Type, M, N, M);

    float alpha = 1.0f, beta = 0.0f;

    size_t workspaceSize = 64 * 1024 * 1024;
    void* d_workspace;
    cudaMalloc(&d_workspace, workspaceSize);

    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

    // Print cuBLASLt version for debugging
    size_t cublasLtVer = cublasLtGetVersion();
    std::cerr << "  cublasLt version: " << cublasLtVer << std::endl;
    std::cerr << "  fp8Type enum value: " << (int)fp8Type << std::endl;

    // Try heuristic with up to 10 algorithms
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResults[10];
    cublasStatus_t heurStatus = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc,
        Adesc, Bdesc, Cdesc, Ddesc,
        preference, 10, heuristicResults, &returnedResults);

    std::cerr << "  Heuristic: status=" << heurStatus << ", results=" << returnedResults << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float ms = 0.0f;
    if (returnedResults > 0 && heurStatus == CUBLAS_STATUS_SUCCESS) {
        cudaEventRecord(start);
        cublasStatus_t status = cublasLtMatmul(ltHandle, operationDesc,
                                                &alpha,
                                                d_A, Adesc,
                                                d_B, Bdesc,
                                                &beta,
                                                d_C, Cdesc,
                                                d_D, Ddesc,
                                                &heuristicResults[0].algo,
                                                d_workspace, workspaceSize, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "  cublasLtMatmul FP8 exec failed (status=" << status << ")" << std::endl;
            ms = 0.0f;
        } else {
            cudaEventElapsedTime(&ms, start, stop);
        }
    } else {
        std::cerr << "  No algorithm found for FP8 GEMM (status=" << heurStatus << ")" << std::endl;
    }

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(ltHandle);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_workspace);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_a_scale);
    cudaFree(d_b_scale);
    cudaFree(d_c_scale);
    cudaFree(d_d_scale);
    cudaFree(d_amax_d);

    return ms;
}

#endif // FLOPTIC_HAS_FP8_TYPES

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
        if (config.iterations <= 1000)       { M = N = K = 1024; }
        else if (config.iterations <= 10000) { M = N = K = 2048; }
        else if (config.iterations <= 100000){ M = N = K = 4096; }
        else                                 { M = N = K = 8192; }

        int64_t flops_per_trial = 2LL * M * N * K;

        std::cerr << "  Running gemm_cublas_int8 [cuda/INT8/throughput (tensor cores)] M=N=K="
                  << M << std::endl;

        // Warmup
        for (int w = 0; w < 3; w++) {
            run_gemm_int8(M, N, K);
        }

        std::vector<double> times;
        times.reserve(measurement_trials);
        for (int t = 0; t < measurement_trials; t++) {
            float ms = run_gemm_int8(M, N, K);
            if (ms > 0) times.push_back(static_cast<double>(ms));
        }

        if (times.empty()) {
            std::cerr << "  INT8 GEMM failed — no valid timings" << std::endl;
            KernelResult result;
            return result;
        }

        auto stats = TimingStats::compute(times);

        KernelResult result;
        result.median_time_ms = stats.median_ms;
        result.min_time_ms = stats.min_ms;
        result.max_time_ms = stats.max_ms;
        result.total_flops = flops_per_trial;
        result.gflops = (flops_per_trial / 1e9) / (stats.median_ms / 1e3);
        result.effective_gflops = result.gflops;

        // INT8 TC peak: 2× FP16 TC
        auto it = device.theoretical_peak_gflops.find("FP16_TC");
        if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
            double int8_peak = it->second * 2.0;
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

#ifdef FLOPTIC_HAS_FP8_TYPES

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
        if (config.iterations <= 1000)       { M = N = K = 1024; }
        else if (config.iterations <= 10000) { M = N = K = 2048; }
        else if (config.iterations <= 100000){ M = N = K = 4096; }
        else                                 { M = N = K = 8192; }

        int64_t flops_per_trial = 2LL * M * N * K;

        cudaDataType_t fp8Type = (config.precision == Precision::FP8_E4M3)
                                  ? CUDA_R_8F_E4M3 : CUDA_R_8F_E5M2;
        std::string fp8Name = (config.precision == Precision::FP8_E4M3)
                               ? "FP8_E4M3" : "FP8_E5M2";

        std::cerr << "  Running gemm_cublas_fp8 [cuda/" << fp8Name
                  << "/throughput (tensor cores)] M=N=K=" << M << std::endl;

        // Test if this actually works before running trials
        float test_ms = run_gemm_fp8(fp8Type, M, N, K);
        if (test_ms <= 0) {
            std::cerr << "  " << fp8Name << " GEMM not supported on this device — skipping" << std::endl;
            KernelResult result;
            return result;
        }

        // Warmup
        for (int w = 0; w < 3; w++) {
            run_gemm_fp8(fp8Type, M, N, K);
        }

        std::vector<double> times;
        times.reserve(measurement_trials);
        for (int t = 0; t < measurement_trials; t++) {
            float ms = run_gemm_fp8(fp8Type, M, N, K);
            if (ms > 0) times.push_back(static_cast<double>(ms));
        }

        if (times.empty()) {
            KernelResult result;
            return result;
        }

        auto stats = TimingStats::compute(times);

        KernelResult result;
        result.median_time_ms = stats.median_ms;
        result.min_time_ms = stats.min_ms;
        result.max_time_ms = stats.max_ms;
        result.total_flops = flops_per_trial;
        result.gflops = (flops_per_trial / 1e9) / (stats.median_ms / 1e3);
        result.effective_gflops = result.gflops;

        // FP8 TC peak: 2× FP16 TC on Hopper
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

#endif // FLOPTIC_HAS_FP8_TYPES

namespace force_link {
    void gemm_cublas_int8_link() {}
    void gemm_cublas_fp8_link() {}
}

} // namespace floptic

#endif // FLOPTIC_HAS_CUDA
