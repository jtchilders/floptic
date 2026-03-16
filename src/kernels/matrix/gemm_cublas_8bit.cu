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

// Cached best algorithm after auto-tuning
static cublasLtMatmulAlgo_t g_bestAlgo_int8;
static bool g_hasBestAlgo_int8 = false;
static cublasLtMatmulDesc_t g_opDesc_int8 = nullptr;
static cublasLtMatrixLayout_t g_Adesc_int8 = nullptr;
static cublasLtMatrixLayout_t g_Bdesc_int8 = nullptr;
static cublasLtMatrixLayout_t g_Cdesc_int8 = nullptr;

static void autotune_int8(int M, int N, int K) {
    setup_int8(M, N, K);

    if (g_opDesc_int8) cublasLtMatmulDescDestroy(g_opDesc_int8);
    if (g_Adesc_int8) cublasLtMatrixLayoutDestroy(g_Adesc_int8);
    if (g_Bdesc_int8) cublasLtMatrixLayoutDestroy(g_Bdesc_int8);
    if (g_Cdesc_int8) cublasLtMatrixLayoutDestroy(g_Cdesc_int8);

    cublasLtMatmulDescCreate(&g_opDesc_int8, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    cublasLtMatrixLayoutCreate(&g_Adesc_int8, CUDA_R_8I, M, K, M);
    cublasLtMatrixLayoutCreate(&g_Bdesc_int8, CUDA_R_8I, K, N, K);
    cublasLtMatrixLayoutCreate(&g_Cdesc_int8, CUDA_R_32I, M, N, M);

    int32_t alpha = 1, beta = 0;

    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &g_workspaceSize_int8, sizeof(g_workspaceSize_int8));

    // Request many algorithms for auto-tuning
    const int maxAlgos = 32;
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResults[maxAlgos];
    cublasLtMatmulAlgoGetHeuristic(g_ltHandle_int8, g_opDesc_int8,
                                    g_Adesc_int8, g_Bdesc_int8,
                                    g_Cdesc_int8, g_Cdesc_int8,
                                    preference, maxAlgos,
                                    heuristicResults, &returnedResults);

    std::cerr << "  INT8 auto-tune: " << returnedResults << " algorithms found" << std::endl;

    if (returnedResults == 0) {
        std::cerr << "  No INT8 algorithms available!" << std::endl;
        g_hasBestAlgo_int8 = false;
        cublasLtMatmulPreferenceDestroy(preference);
        return;
    }

    // Time each algorithm, pick the fastest
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float bestTime = 1e30f;
    int bestIdx = 0;

    for (int i = 0; i < returnedResults; i++) {
        // Skip if workspace exceeds our allocation
        if (heuristicResults[i].workspaceSize > g_workspaceSize_int8) continue;

        // Warmup
        cublasStatus_t status = cublasLtMatmul(g_ltHandle_int8, g_opDesc_int8,
            &alpha, g_d_A_int8, g_Adesc_int8, g_d_B_int8, g_Bdesc_int8,
            &beta, g_d_C_int8, g_Cdesc_int8, g_d_C_int8, g_Cdesc_int8,
            &heuristicResults[i].algo,
            g_workspace_int8, g_workspaceSize_int8, 0);
        cudaDeviceSynchronize();

        if (status != CUBLAS_STATUS_SUCCESS) continue;

        // Time 3 runs
        float totalMs = 0;
        for (int r = 0; r < 3; r++) {
            cudaEventRecord(start);
            cublasLtMatmul(g_ltHandle_int8, g_opDesc_int8,
                &alpha, g_d_A_int8, g_Adesc_int8, g_d_B_int8, g_Bdesc_int8,
                &beta, g_d_C_int8, g_Cdesc_int8, g_d_C_int8, g_Cdesc_int8,
                &heuristicResults[i].algo,
                g_workspace_int8, g_workspaceSize_int8, 0);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            totalMs += ms;
        }
        float avgMs = totalMs / 3.0f;

        int algoId = 0;
        cublasLtMatmulAlgoConfigGetAttribute(&heuristicResults[i].algo,
            CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr);

        double gflops = (2.0 * M * N * K / 1e9) / (avgMs / 1e3);
        std::cerr << "    algo " << algoId << ": " << avgMs << " ms ("
                  << gflops << " GFLOP/s), ws=" << heuristicResults[i].workspaceSize << std::endl;

        if (avgMs < bestTime) {
            bestTime = avgMs;
            bestIdx = i;
        }
    }

    g_bestAlgo_int8 = heuristicResults[bestIdx].algo;
    g_hasBestAlgo_int8 = true;

    int bestAlgoId = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&g_bestAlgo_int8,
        CUBLASLT_ALGO_CONFIG_ID, &bestAlgoId, sizeof(bestAlgoId), nullptr);
    std::cerr << "  Best INT8 algo: " << bestAlgoId << " (" << bestTime << " ms)" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasLtMatmulPreferenceDestroy(preference);
}

static float run_gemm_int8(int M, int N, int K) {
    if (!g_hasBestAlgo_int8) return 0.0f;

    int32_t alpha = 1, beta = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cublasLtMatmul(g_ltHandle_int8, g_opDesc_int8,
                    &alpha, g_d_A_int8, g_Adesc_int8, g_d_B_int8, g_Bdesc_int8,
                    &beta, g_d_C_int8, g_Cdesc_int8, g_d_C_int8, g_Cdesc_int8,
                    &g_bestAlgo_int8,
                    g_workspace_int8, g_workspaceSize_int8, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

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

// FP8 GEMM using manual algorithm enumeration (bypasses heuristic)
// cublasLtMatmulAlgoGetHeuristic rejects our descriptors on some systems,
// but the algorithms exist — use AlgoGetIds + AlgoInit + AlgoCheck instead.

static cublasLtHandle_t g_ltHandle_fp8 = nullptr;
static void* g_workspace_fp8 = nullptr;
static size_t g_workspaceSize_fp8 = 64 * 1024 * 1024;
static void* g_d_A_fp8 = nullptr;
static void* g_d_B_fp8 = nullptr;
static void* g_d_C_fp8 = nullptr;  // BF16 bias
static void* g_d_D_fp8 = nullptr;  // FP8 output
static float* g_d_a_scale = nullptr;
static float* g_d_b_scale = nullptr;
static float* g_d_d_scale = nullptr;
static float* g_d_amax_d = nullptr;
static cublasLtMatmulAlgo_t g_bestAlgo_fp8;
static bool g_hasBestAlgo_fp8 = false;
static cublasLtMatmulDesc_t g_opDesc_fp8 = nullptr;
static cublasLtMatrixLayout_t g_Adesc_fp8 = nullptr;
static cublasLtMatrixLayout_t g_Bdesc_fp8 = nullptr;
static cublasLtMatrixLayout_t g_Cdesc_fp8 = nullptr;
static cublasLtMatrixLayout_t g_Ddesc_fp8 = nullptr;

static void autotune_fp8(cudaDataType_t fp8Type, int M, int N, int K) {
    // Cleanup old
    if (g_d_A_fp8) cudaFree(g_d_A_fp8);
    if (g_d_B_fp8) cudaFree(g_d_B_fp8);
    if (g_d_C_fp8) cudaFree(g_d_C_fp8);
    if (g_d_D_fp8) cudaFree(g_d_D_fp8);
    if (g_workspace_fp8) cudaFree(g_workspace_fp8);
    if (g_d_a_scale) cudaFree(g_d_a_scale);
    if (g_d_b_scale) cudaFree(g_d_b_scale);
    if (g_d_d_scale) cudaFree(g_d_d_scale);
    if (g_d_amax_d) cudaFree(g_d_amax_d);
    if (g_opDesc_fp8) cublasLtMatmulDescDestroy(g_opDesc_fp8);
    if (g_Adesc_fp8) cublasLtMatrixLayoutDestroy(g_Adesc_fp8);
    if (g_Bdesc_fp8) cublasLtMatrixLayoutDestroy(g_Bdesc_fp8);
    if (g_Cdesc_fp8) cublasLtMatrixLayoutDestroy(g_Cdesc_fp8);
    if (g_Ddesc_fp8) cublasLtMatrixLayoutDestroy(g_Ddesc_fp8);
    if (g_ltHandle_fp8) cublasLtDestroy(g_ltHandle_fp8);

    check_cuda_8(cudaMalloc(&g_d_A_fp8, (size_t)M * K), "alloc A fp8");
    check_cuda_8(cudaMalloc(&g_d_B_fp8, (size_t)K * N), "alloc B fp8");
    check_cuda_8(cudaMalloc(&g_d_C_fp8, (size_t)M * N * 2), "alloc C bf16");
    check_cuda_8(cudaMalloc(&g_d_D_fp8, (size_t)M * N), "alloc D fp8");
    check_cuda_8(cudaMalloc(&g_workspace_fp8, g_workspaceSize_fp8), "alloc workspace");

    cudaMemset(g_d_A_fp8, 0x38, (size_t)M * K);
    cudaMemset(g_d_B_fp8, 0x38, (size_t)K * N);
    cudaMemset(g_d_C_fp8, 0, (size_t)M * N * 2);
    cudaMemset(g_d_D_fp8, 0, (size_t)M * N);

    // Scaling factors
    float h_one = 1.0f, h_zero = 0.0f;
    cudaMalloc(&g_d_a_scale, sizeof(float));
    cudaMalloc(&g_d_b_scale, sizeof(float));
    cudaMalloc(&g_d_d_scale, sizeof(float));
    cudaMalloc(&g_d_amax_d, sizeof(float));
    cudaMemcpy(g_d_a_scale, &h_one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_b_scale, &h_one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_d_scale, &h_one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_amax_d, &h_zero, sizeof(float), cudaMemcpyHostToDevice);

    cublasLtCreate(&g_ltHandle_fp8);

    cublasLtMatmulDescCreate(&g_opDesc_fp8, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(g_opDesc_fp8, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &g_d_a_scale, sizeof(g_d_a_scale));
    cublasLtMatmulDescSetAttribute(g_opDesc_fp8, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &g_d_b_scale, sizeof(g_d_b_scale));
    cublasLtMatmulDescSetAttribute(g_opDesc_fp8, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &g_d_d_scale, sizeof(g_d_d_scale));
    cublasLtMatmulDescSetAttribute(g_opDesc_fp8, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &g_d_amax_d, sizeof(g_d_amax_d));

    cublasLtMatrixLayoutCreate(&g_Adesc_fp8, fp8Type, M, K, M);
    cublasLtMatrixLayoutCreate(&g_Bdesc_fp8, fp8Type, K, N, K);
    cublasLtMatrixLayoutCreate(&g_Cdesc_fp8, CUDA_R_16BF, M, N, M);
    cublasLtMatrixLayoutCreate(&g_Ddesc_fp8, fp8Type, M, N, M);

    // Get algorithm IDs (bypasses heuristic)
    int algoIds[100];
    int numAlgos = 0;
    cublasLtMatmulAlgoGetIds(g_ltHandle_fp8, CUBLAS_COMPUTE_32F, CUDA_R_32F,
        fp8Type, fp8Type, CUDA_R_16BF, fp8Type,
        100, algoIds, &numAlgos);

    std::cerr << "  FP8 AlgoGetIds: " << numAlgos << " algorithms" << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float bestTime = 1e30f;
    int bestIdx = -1;
    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < numAlgos; i++) {
        cublasLtMatmulAlgo_t algo;
        cublasLtMatmulAlgoInit(g_ltHandle_fp8, CUBLAS_COMPUTE_32F,
            CUDA_R_32F, fp8Type, fp8Type, CUDA_R_16BF, fp8Type,
            algoIds[i], &algo);

        // Check if algo is compatible with our problem
        cublasLtMatmulHeuristicResult_t checkResult;
        cublasStatus_t checkSt = cublasLtMatmulAlgoCheck(g_ltHandle_fp8,
            g_opDesc_fp8, g_Adesc_fp8, g_Bdesc_fp8, g_Cdesc_fp8, g_Ddesc_fp8,
            &algo, &checkResult);

        if (checkSt != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "    algo " << algoIds[i] << ": check failed (status=" << checkSt << ")" << std::endl;
            continue;
        }

        if (checkResult.workspaceSize > g_workspaceSize_fp8) {
            std::cerr << "    algo " << algoIds[i] << ": needs " << checkResult.workspaceSize << " B workspace, skipping" << std::endl;
            continue;
        }

        // Warmup
        cublasStatus_t execSt = cublasLtMatmul(g_ltHandle_fp8, g_opDesc_fp8,
            &alpha, g_d_A_fp8, g_Adesc_fp8, g_d_B_fp8, g_Bdesc_fp8,
            &beta, g_d_C_fp8, g_Cdesc_fp8, g_d_D_fp8, g_Ddesc_fp8,
            &algo, g_workspace_fp8, g_workspaceSize_fp8, 0);
        cudaDeviceSynchronize();

        if (execSt != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "    algo " << algoIds[i] << ": exec failed (status=" << execSt << ")" << std::endl;
            continue;
        }

        // Time 3 runs
        float totalMs = 0;
        for (int r = 0; r < 3; r++) {
            cudaEventRecord(start);
            cublasLtMatmul(g_ltHandle_fp8, g_opDesc_fp8,
                &alpha, g_d_A_fp8, g_Adesc_fp8, g_d_B_fp8, g_Bdesc_fp8,
                &beta, g_d_C_fp8, g_Cdesc_fp8, g_d_D_fp8, g_Ddesc_fp8,
                &algo, g_workspace_fp8, g_workspaceSize_fp8, 0);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            totalMs += ms;
        }
        float avgMs = totalMs / 3.0f;
        double gflops = (2.0 * M * N * K / 1e9) / (avgMs / 1e3);
        std::cerr << "    algo " << algoIds[i] << ": " << avgMs << " ms ("
                  << gflops << " GFLOP/s), ws=" << checkResult.workspaceSize << std::endl;

        if (avgMs < bestTime) {
            bestTime = avgMs;
            bestIdx = i;
            g_bestAlgo_fp8 = algo;
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (bestIdx >= 0) {
        g_hasBestAlgo_fp8 = true;
        std::cerr << "  Best FP8 algo: " << algoIds[bestIdx] << " (" << bestTime << " ms)" << std::endl;
    } else {
        g_hasBestAlgo_fp8 = false;
        std::cerr << "  No working FP8 algorithm found" << std::endl;
    }
}

static float run_gemm_fp8(cudaDataType_t /*fp8Type*/, int /*M*/, int /*N*/, int /*K*/) {
    if (!g_hasBestAlgo_fp8) return 0.0f;

    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cublasLtMatmul(g_ltHandle_fp8, g_opDesc_fp8,
                    &alpha, g_d_A_fp8, g_Adesc_fp8, g_d_B_fp8, g_Bdesc_fp8,
                    &beta, g_d_C_fp8, g_Cdesc_fp8, g_d_D_fp8, g_Ddesc_fp8,
                    &g_bestAlgo_fp8,
                    g_workspace_fp8, g_workspaceSize_fp8, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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

        // Auto-tune: find best algorithm
        autotune_int8(M, N, K);
        if (!g_hasBestAlgo_int8) {
            std::cerr << "  INT8 GEMM auto-tune failed" << std::endl;
            KernelResult result;
            return result;
        }

        // Warmup with best algo
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

        // INT8 TC peak from device tables
        auto it = device.theoretical_peak_gflops.find("INT8_TC");
        if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
            result.peak_percent = (result.gflops / it->second) * 100.0;
        }
        std::cerr << "  (peak% vs INT8_TC)" << std::endl;

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

        // Auto-tune: find best algorithm (bypasses heuristic)
        autotune_fp8(fp8Type, M, N, K);
        if (!g_hasBestAlgo_fp8) {
            std::cerr << "  " << fp8Name << " GEMM not supported on this device — skipping" << std::endl;
            KernelResult result;
            return result;
        }

        // Warmup with best algo
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

        // FP8 TC peak from device tables
        auto it = device.theoretical_peak_gflops.find("FP8_TC");
        if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
            result.peak_percent = (result.gflops / it->second) * 100.0;
        }
        std::cerr << "  (peak% vs FP8_TC)" << std::endl;

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
