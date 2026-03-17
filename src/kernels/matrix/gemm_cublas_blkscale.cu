// Block-scaled GEMM kernels for Blackwell (sm_100+)
//   gemm_cublas_mxfp8: MXFP8 (32-element 1D block scaling, E8M0 scales)
//   gemm_cublas_nvfp4: NVFP4 (16-element 1D block scaling, UE4M3 scales)
//
// Requires CUDA 12.8+ / cuBLAS 13+ with cublasLtMatmulMatrixScale_t support.
// Uses CUBLAS_OP_T for A (required for block-scaled matmul).

#ifdef FLOPTIC_HAS_CUDA

#include "floptic/kernel_base.hpp"
#include "floptic/kernel_registry.hpp"
#include "floptic/timer.hpp"
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>

// Block-scaling enums require CUDA 12.8+ / cuBLAS 13+.
// CUBLASLT_MATMUL_DESC_A_SCALE_MODE is an enum value (=31), not a preprocessor
// macro, so we can't use #if defined(). Instead, check CUDA version:
//   - cublasLtMatmulMatrixScale_t was added in cuBLAS 12.8
//   - CUDA_R_4F_E2M1 (FP4) was added in CUDA 12.8
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8)
#define FLOPTIC_HAS_BLKSCALE 1
#include <cuda_fp8.h>
#endif

namespace floptic {

#ifdef FLOPTIC_HAS_BLKSCALE

// ============================================================================
// Common infrastructure for block-scaled GEMMs
// ============================================================================

struct BlkScaleState {
    cublasLtHandle_t ltHandle = nullptr;
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr;
    cublasLtMatrixLayout_t Cdesc = nullptr, Ddesc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulAlgo_t bestAlgo;
    bool hasBestAlgo = false;

    void* d_A = nullptr;
    void* d_B = nullptr;
    void* d_C = nullptr;     // BF16 bias
    void* d_D = nullptr;     // output
    void* d_A_scale = nullptr;
    void* d_B_scale = nullptr;
    void* d_D_out_scale = nullptr;
    void* d_workspace = nullptr;
    size_t workspaceSize = 64 * 1024 * 1024;

    int M = 0, N = 0, K = 0;

    void cleanup() {
        if (d_A) cudaFree(d_A); d_A = nullptr;
        if (d_B) cudaFree(d_B); d_B = nullptr;
        if (d_C) cudaFree(d_C); d_C = nullptr;
        if (d_D) cudaFree(d_D); d_D = nullptr;
        if (d_A_scale) cudaFree(d_A_scale); d_A_scale = nullptr;
        if (d_B_scale) cudaFree(d_B_scale); d_B_scale = nullptr;
        if (d_D_out_scale) cudaFree(d_D_out_scale); d_D_out_scale = nullptr;
        if (d_workspace) cudaFree(d_workspace); d_workspace = nullptr;
        if (pref) cublasLtMatmulPreferenceDestroy(pref); pref = nullptr;
        if (Ddesc) cublasLtMatrixLayoutDestroy(Ddesc); Ddesc = nullptr;
        if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc); Cdesc = nullptr;
        if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc); Bdesc = nullptr;
        if (Adesc) cublasLtMatrixLayoutDestroy(Adesc); Adesc = nullptr;
        if (opDesc) cublasLtMatmulDescDestroy(opDesc); opDesc = nullptr;
        if (ltHandle) cublasLtDestroy(ltHandle); ltHandle = nullptr;
        hasBestAlgo = false;
        M = N = K = 0;
    }

    ~BlkScaleState() { cleanup(); }
};

// ============================================================================
// MXFP8: Block-scaled FP8 GEMM (32-element blocks, E8M0 scales)
// ============================================================================

static BlkScaleState g_mxfp8;

static void setup_mxfp8(int M, int N, int K) {
    if (g_mxfp8.M == M && g_mxfp8.N == N && g_mxfp8.K == K && g_mxfp8.ltHandle)
        return;

    g_mxfp8.cleanup();
    g_mxfp8.M = M; g_mxfp8.N = N; g_mxfp8.K = K;

    // Block scaling: 32-element blocks along K
    int scale_k = (K + 31) / 32;
    int scale_n = (N + 31) / 32;

    // Allocate: A(K×M), B(K×N), C(M×N BF16), D(M×N FP8)
    // transa=T: physical A is K×M, lda=K
    // transb=N: physical B is K×N, ldb=K
    cudaMalloc(&g_mxfp8.d_A, (size_t)K * M);
    cudaMalloc(&g_mxfp8.d_B, (size_t)K * N);
    cudaMalloc(&g_mxfp8.d_C, (size_t)M * N * 2);
    cudaMalloc(&g_mxfp8.d_D, (size_t)M * N);  // FP8 output

    // Scale arrays: E8M0 (1 byte each)
    cudaMalloc(&g_mxfp8.d_A_scale, (size_t)scale_k * M);
    cudaMalloc(&g_mxfp8.d_B_scale, (size_t)scale_k * N);
    cudaMalloc(&g_mxfp8.d_D_out_scale, (size_t)scale_n * M);

    cudaMalloc(&g_mxfp8.d_workspace, g_mxfp8.workspaceSize);

    // Initialize data
    cudaMemset(g_mxfp8.d_A, 0x38, (size_t)K * M);
    cudaMemset(g_mxfp8.d_B, 0x38, (size_t)K * N);
    cudaMemset(g_mxfp8.d_C, 0, (size_t)M * N * 2);
    cudaMemset(g_mxfp8.d_D, 0, (size_t)M * N);
    // E8M0: val=127 → scale = 2^(127-127) = 1.0
    cudaMemset(g_mxfp8.d_A_scale, 127, (size_t)scale_k * M);
    cudaMemset(g_mxfp8.d_B_scale, 127, (size_t)scale_k * N);
    cudaMemset(g_mxfp8.d_D_out_scale, 0, (size_t)scale_n * M);

    cublasLtCreate(&g_mxfp8.ltHandle);

    // Matmul descriptor: compute=FP32, scale=FP32
    cublasLtMatmulDescCreate(&g_mxfp8.opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(g_mxfp8.opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(g_mxfp8.opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    // Block scaling modes
    cublasLtMatmulMatrixScale_t scaleModeBlock = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    cublasLtMatmulDescSetAttribute(g_mxfp8.opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleModeBlock, sizeof(scaleModeBlock));
    cublasLtMatmulDescSetAttribute(g_mxfp8.opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleModeBlock, sizeof(scaleModeBlock));
    cublasLtMatmulDescSetAttribute(g_mxfp8.opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &scaleModeBlock, sizeof(scaleModeBlock));

    // Scale pointers
    cublasLtMatmulDescSetAttribute(g_mxfp8.opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &g_mxfp8.d_A_scale, sizeof(g_mxfp8.d_A_scale));
    cublasLtMatmulDescSetAttribute(g_mxfp8.opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &g_mxfp8.d_B_scale, sizeof(g_mxfp8.d_B_scale));
    cublasLtMatmulDescSetAttribute(g_mxfp8.opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &g_mxfp8.d_D_out_scale, sizeof(g_mxfp8.d_D_out_scale));

    // Matrix layouts (physical dims for T,N)
    cublasLtMatrixLayoutCreate(&g_mxfp8.Adesc, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&g_mxfp8.Bdesc, CUDA_R_8F_E4M3, K, N, K);
    cublasLtMatrixLayoutCreate(&g_mxfp8.Cdesc, CUDA_R_16BF, M, N, M);
    cublasLtMatrixLayoutCreate(&g_mxfp8.Ddesc, CUDA_R_8F_E4M3, M, N, M);

    // Preference
    cublasLtMatmulPreferenceCreate(&g_mxfp8.pref);
    cublasLtMatmulPreferenceSetAttribute(g_mxfp8.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &g_mxfp8.workspaceSize, sizeof(g_mxfp8.workspaceSize));

    // Get best heuristic
    int nResults = 0;
    cublasLtMatmulHeuristicResult_t hResult;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        g_mxfp8.ltHandle, g_mxfp8.opDesc, g_mxfp8.Adesc, g_mxfp8.Bdesc,
        g_mxfp8.Cdesc, g_mxfp8.Ddesc, g_mxfp8.pref, 1, &hResult, &nResults);

    if (st == CUBLAS_STATUS_SUCCESS && nResults > 0) {
        g_mxfp8.bestAlgo = hResult.algo;
        g_mxfp8.hasBestAlgo = true;
        std::cerr << "  MXFP8 heuristic: found algorithm (M=N=K=" << M << ")" << std::endl;
    } else {
        g_mxfp8.hasBestAlgo = false;
        std::cerr << "  MXFP8 heuristic: failed (status=" << st << ", results=" << nResults << ")" << std::endl;
    }
}

static float run_mxfp8(int M, int N, int K) {
    if (!g_mxfp8.hasBestAlgo) return 0.0f;

    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cublasLtMatmul(g_mxfp8.ltHandle, g_mxfp8.opDesc,
        &alpha, g_mxfp8.d_A, g_mxfp8.Adesc, g_mxfp8.d_B, g_mxfp8.Bdesc,
        &beta, g_mxfp8.d_C, g_mxfp8.Cdesc, g_mxfp8.d_D, g_mxfp8.Ddesc,
        &g_mxfp8.bestAlgo, g_mxfp8.d_workspace, g_mxfp8.workspaceSize, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// ============================================================================
// NVFP4: Block-scaled FP4 GEMM (16-element blocks, UE4M3 scales)
// ============================================================================

static BlkScaleState g_nvfp4;

static void setup_nvfp4(int M, int N, int K) {
    if (g_nvfp4.M == M && g_nvfp4.N == N && g_nvfp4.K == K && g_nvfp4.ltHandle)
        return;

    g_nvfp4.cleanup();
    g_nvfp4.M = M; g_nvfp4.N = N; g_nvfp4.K = K;

    // Block scaling: 16-element blocks along K
    int scale_k = (K + 15) / 16;
    int scale_n = (N + 15) / 16;

    // FP4 is packed: 2 elements per byte
    // A(K×M packed), B(K×N packed), C(M×N BF16), D(M×N FP4 packed)
    cudaMalloc(&g_nvfp4.d_A, (size_t)K * M / 2);
    cudaMalloc(&g_nvfp4.d_B, (size_t)K * N / 2);
    cudaMalloc(&g_nvfp4.d_C, (size_t)M * N * 2);
    cudaMalloc(&g_nvfp4.d_D, (size_t)M * N / 2);

    // Scale arrays: FP8 E4M3 (UE4M3, 1 byte each)
    cudaMalloc(&g_nvfp4.d_A_scale, (size_t)scale_k * M);
    cudaMalloc(&g_nvfp4.d_B_scale, (size_t)scale_k * N);
    cudaMalloc(&g_nvfp4.d_D_out_scale, (size_t)scale_n * M);

    // D global scale: single float
    void* d_D_scale = nullptr;
    cudaMalloc(&d_D_scale, sizeof(float));
    float one = 1.0f;
    cudaMemcpy(d_D_scale, &one, sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&g_nvfp4.d_workspace, g_nvfp4.workspaceSize);

    // Initialize
    cudaMemset(g_nvfp4.d_A, 0x33, (size_t)K * M / 2);
    cudaMemset(g_nvfp4.d_B, 0x33, (size_t)K * N / 2);
    cudaMemset(g_nvfp4.d_C, 0, (size_t)M * N * 2);
    cudaMemset(g_nvfp4.d_D, 0, (size_t)M * N / 2);
    // FP8 E4M3 scale = 1.0 → 0x38
    cudaMemset(g_nvfp4.d_A_scale, 0x38, (size_t)scale_k * M);
    cudaMemset(g_nvfp4.d_B_scale, 0x38, (size_t)scale_k * N);
    cudaMemset(g_nvfp4.d_D_out_scale, 0, (size_t)scale_n * M);

    cublasLtCreate(&g_nvfp4.ltHandle);

    cublasLtMatmulDescCreate(&g_nvfp4.opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(g_nvfp4.opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(g_nvfp4.opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    // Block scaling modes
    cublasLtMatmulMatrixScale_t scaleModeVec16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulMatrixScale_t scaleModeScalar = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;

    cublasLtMatmulDescSetAttribute(g_nvfp4.opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleModeVec16, sizeof(scaleModeVec16));
    cublasLtMatmulDescSetAttribute(g_nvfp4.opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleModeVec16, sizeof(scaleModeVec16));
    cublasLtMatmulDescSetAttribute(g_nvfp4.opDesc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &scaleModeScalar, sizeof(scaleModeScalar));
    cublasLtMatmulDescSetAttribute(g_nvfp4.opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &scaleModeVec16, sizeof(scaleModeVec16));

    // Scale pointers
    cublasLtMatmulDescSetAttribute(g_nvfp4.opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &g_nvfp4.d_A_scale, sizeof(g_nvfp4.d_A_scale));
    cublasLtMatmulDescSetAttribute(g_nvfp4.opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &g_nvfp4.d_B_scale, sizeof(g_nvfp4.d_B_scale));
    cublasLtMatmulDescSetAttribute(g_nvfp4.opDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_D_scale, sizeof(d_D_scale));
    cublasLtMatmulDescSetAttribute(g_nvfp4.opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &g_nvfp4.d_D_out_scale, sizeof(g_nvfp4.d_D_out_scale));

    // Matrix layouts (physical dims for T,N, FP4 packed type)
    cublasLtMatrixLayoutCreate(&g_nvfp4.Adesc, CUDA_R_4F_E2M1, K, M, K);
    cublasLtMatrixLayoutCreate(&g_nvfp4.Bdesc, CUDA_R_4F_E2M1, K, N, K);
    cublasLtMatrixLayoutCreate(&g_nvfp4.Cdesc, CUDA_R_16BF, M, N, M);
    cublasLtMatrixLayoutCreate(&g_nvfp4.Ddesc, CUDA_R_4F_E2M1, M, N, M);

    // Preference
    cublasLtMatmulPreferenceCreate(&g_nvfp4.pref);
    cublasLtMatmulPreferenceSetAttribute(g_nvfp4.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &g_nvfp4.workspaceSize, sizeof(g_nvfp4.workspaceSize));

    // Get heuristic
    int nResults = 0;
    cublasLtMatmulHeuristicResult_t hResult;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        g_nvfp4.ltHandle, g_nvfp4.opDesc, g_nvfp4.Adesc, g_nvfp4.Bdesc,
        g_nvfp4.Cdesc, g_nvfp4.Ddesc, g_nvfp4.pref, 1, &hResult, &nResults);

    if (st == CUBLAS_STATUS_SUCCESS && nResults > 0) {
        g_nvfp4.bestAlgo = hResult.algo;
        g_nvfp4.hasBestAlgo = true;
        std::cerr << "  NVFP4 heuristic: found algorithm (M=N=K=" << M << ")" << std::endl;
    } else {
        g_nvfp4.hasBestAlgo = false;
        std::cerr << "  NVFP4 heuristic: failed (status=" << st << ", results=" << nResults << ")" << std::endl;
    }
}

static float run_nvfp4(int M, int N, int K) {
    if (!g_nvfp4.hasBestAlgo) return 0.0f;

    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cublasLtMatmul(g_nvfp4.ltHandle, g_nvfp4.opDesc,
        &alpha, g_nvfp4.d_A, g_nvfp4.Adesc, g_nvfp4.d_B, g_nvfp4.Bdesc,
        &beta, g_nvfp4.d_C, g_nvfp4.Cdesc, g_nvfp4.d_D, g_nvfp4.Ddesc,
        &g_nvfp4.bestAlgo, g_nvfp4.d_workspace, g_nvfp4.workspaceSize, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// ============================================================================
// Common sweep+measure logic (shared by MXFP8 and NVFP4 kernels)
// ============================================================================

// Helper: sweep sizes, auto-tune, and run full measurement
template <typename SetupFn, typename RunFn>
static KernelResult sweep_and_measure(
    const DeviceInfo& device, int measurement_trials,
    const std::string& label, const std::string& peak_key,
    SetupFn setup_fn, RunFn run_fn)
{
    int sweep_sizes[] = {1024, 2048, 4096, 8192, 16384};
    int num_sweep = sizeof(sweep_sizes) / sizeof(sweep_sizes[0]);

    double best_gflops = 0;
    int best_size = 4096;

    std::cerr << "  Sweeping " << label << ":" << std::endl;

    for (int si = 0; si < num_sweep; si++) {
        int M = sweep_sizes[si], N = M, K = M;

        // Memory check (rough: A+B+C+D+scales)
        size_t needed = (size_t)M * K + (size_t)K * N + (size_t)M * N * 3;
        if (needed > device.memory_bytes * 0.8) {
            std::cerr << "    M=N=K=" << M << ": skipped (memory)" << std::endl;
            continue;
        }

        setup_fn(M, N, K);

        // Warmup
        for (int w = 0; w < 3; w++) {
            float ms = run_fn(M, N, K);
            if (ms <= 0) break;
        }

        // Quick trial
        std::vector<double> times;
        int sweep_trials = std::min(3, measurement_trials);
        for (int t = 0; t < sweep_trials; t++) {
            float ms = run_fn(M, N, K);
            if (ms > 0) times.push_back(static_cast<double>(ms));
        }
        if (times.empty()) {
            std::cerr << "    M=N=K=" << M << ": no valid results" << std::endl;
            continue;
        }
        std::sort(times.begin(), times.end());
        double median_ms = times[times.size() / 2];
        int64_t flops = 2LL * M * N * K;
        double gflops = (flops / 1e9) / (median_ms / 1e3);

        std::cerr << "    M=N=K=" << M << ": " << gflops << " GFLOP/s ("
                  << median_ms << " ms)" << std::endl;

        if (gflops > best_gflops) {
            best_gflops = gflops;
            best_size = M;
        }
    }

    if (best_gflops <= 0) {
        std::cerr << "  " << label << " — no valid configuration found" << std::endl;
        return KernelResult();
    }

    // Full measurement at best size
    int M = best_size, N = best_size, K = best_size;
    int64_t flops_per_trial = 2LL * M * N * K;

    std::cerr << "  Best size: M=N=K=" << best_size << " → full measurement ("
              << measurement_trials << " trials)" << std::endl;

    setup_fn(M, N, K);
    for (int w = 0; w < 3; w++) run_fn(M, N, K);

    std::vector<double> times;
    times.reserve(measurement_trials);
    for (int t = 0; t < measurement_trials; t++) {
        float ms = run_fn(M, N, K);
        if (ms > 0) times.push_back(static_cast<double>(ms));
    }

    if (times.empty()) return KernelResult();

    auto stats = TimingStats::compute(times);

    KernelResult result;
    result.median_time_ms = stats.median_ms;
    result.min_time_ms = stats.min_ms;
    result.max_time_ms = stats.max_ms;
    result.total_flops = flops_per_trial;
    result.gflops = (flops_per_trial / 1e9) / (stats.median_ms / 1e3);
    result.effective_gflops = result.gflops;

    auto it = device.theoretical_peak_gflops.find(peak_key);
    if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
        result.peak_percent = (result.gflops / it->second) * 100.0;
    }
    std::cerr << "  (peak% vs " << peak_key << ", M=N=K=" << best_size << ")" << std::endl;

    result.clock_mhz = device.boost_clock_mhz;
    return result;
}

// ============================================================================
// Kernel class: MXFP8 GEMM (Blackwell block-scaled FP8)
// ============================================================================

class GemmCublasMxfp8 : public KernelBase {
public:
    std::string name() const override { return "gemm_cublas_mxfp8"; }
    std::string category() const override { return "matrix"; }
    std::string backend() const override { return "cuda"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP8_E4M3};
    }

    std::vector<std::string> supported_modes() const override {
        return {"throughput"};
    }

    // MXFP8 block scaling only available on Blackwell (sm_100+)
    bool is_available(const DeviceInfo& device) const override {
        // Check arch string: "sm_100", "sm_120", etc.
        if (device.arch.size() >= 5 && device.arch.substr(0, 3) == "sm_") {
            int sm = std::stoi(device.arch.substr(3));
            return sm >= 100;
        }
        return false;
    }

    KernelResult run(const KernelConfig& config,
                     const DeviceInfo& device,
                     int measurement_trials) override {
        int dev_idx = 0;
        auto pos = device.id.find(':');
        if (pos != std::string::npos)
            dev_idx = std::stoi(device.id.substr(pos + 1));
        cudaSetDevice(dev_idx);

        return sweep_and_measure(device, measurement_trials,
            "gemm_cublas_mxfp8 [cuda/FP8_E4M3/throughput (block-scaled TC)]",
            "FP8_TC",
            setup_mxfp8, run_mxfp8);
    }
};

REGISTER_KERNEL(GemmCublasMxfp8);

// ============================================================================
// Kernel class: NVFP4 GEMM (Blackwell block-scaled FP4)
// ============================================================================

class GemmCublasNvfp4 : public KernelBase {
public:
    std::string name() const override { return "gemm_cublas_nvfp4"; }
    std::string category() const override { return "matrix"; }
    std::string backend() const override { return "cuda"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP4};
    }

    std::vector<std::string> supported_modes() const override {
        return {"throughput"};
    }

    // NVFP4 block scaling only available on Blackwell (sm_100+)
    bool is_available(const DeviceInfo& device) const override {
        if (device.arch.size() >= 5 && device.arch.substr(0, 3) == "sm_") {
            int sm = std::stoi(device.arch.substr(3));
            return sm >= 100;
        }
        return false;
    }

    KernelResult run(const KernelConfig& config,
                     const DeviceInfo& device,
                     int measurement_trials) override {
        int dev_idx = 0;
        auto pos = device.id.find(':');
        if (pos != std::string::npos)
            dev_idx = std::stoi(device.id.substr(pos + 1));
        cudaSetDevice(dev_idx);

        return sweep_and_measure(device, measurement_trials,
            "gemm_cublas_nvfp4 [cuda/FP4/throughput (block-scaled TC)]",
            "FP4_TC",
            setup_nvfp4, run_nvfp4);
    }
};

REGISTER_KERNEL(GemmCublasNvfp4);

#endif // FLOPTIC_HAS_BLKSCALE

namespace force_link {
    void gemm_cublas_mxfp8_link() {}
    void gemm_cublas_nvfp4_link() {}
}

} // namespace floptic

#endif // FLOPTIC_HAS_CUDA
