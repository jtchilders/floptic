// gemm_hipblaslt.cpp — hipBLASLt GEMM kernels for TF32, FP8, and improved FP16/BF16
//
// Requires hipBLASLt library (optional — gracefully skipped if not found)
// Supports: TF32, FP8_E4M3, FP8_E5M2 (CDNA3/gfx942 only)

#include "floptic/kernel_base.hpp"
#include "floptic/kernel_registry.hpp"
#include "floptic/precision.hpp"
#include "floptic/device_info.hpp"

#ifdef FLOPTIC_HAS_HIPBLASLT

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstring>

// Suppress nodiscard warnings from HIP runtime calls
#define HIP_CHECK(call) (void)(call)

namespace floptic {

// ============================================================================
// Error checking
// ============================================================================

static void check_hipblaslt(hipblasStatus_t status, const char* msg) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "hipBLASLt error in " << msg << ": " << (int)status << std::endl;
    }
}

// ============================================================================
// Helper: run a single hipBLASLt GEMM and return time in ms
// ============================================================================

struct HipBlasLtGemmConfig {
    hipDataType a_type;
    hipDataType b_type;
    hipDataType c_type;
    hipDataType d_type;
    hipblasComputeType_t compute_type;
    hipDataType scale_type;
    int elem_bytes;           // bytes per element for A/B
    int out_elem_bytes;       // bytes per element for C/D
    bool needs_scale_ptrs;    // FP8 needs scale pointers
};

static float run_hipblaslt_gemm(hipblasLtHandle_t handle,
                                 const HipBlasLtGemmConfig& cfg,
                                 int M, int N, int K,
                                 void* A, void* B, void* C, void* D,
                                 void* workspace, size_t workspace_size,
                                 void* d_a_scale, void* d_b_scale,
                                 void* d_d_scale) {
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    hipblasLtMatmulDesc_t matmul;
    hipblasLtMatmulPreference_t pref;

    check_hipblaslt(hipblasLtMatrixLayoutCreate(&matA, cfg.a_type, M, K, M), "layoutA");
    check_hipblaslt(hipblasLtMatrixLayoutCreate(&matB, cfg.b_type, K, N, K), "layoutB");
    check_hipblaslt(hipblasLtMatrixLayoutCreate(&matC, cfg.c_type, M, N, M), "layoutC");
    check_hipblaslt(hipblasLtMatrixLayoutCreate(&matD, cfg.d_type, M, N, M), "layoutD");

    check_hipblaslt(hipblasLtMatmulDescCreate(&matmul, cfg.compute_type, cfg.scale_type), "matmulDesc");

    hipblasOperation_t opN = HIPBLAS_OP_N;
    check_hipblaslt(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(int32_t)), "transA");
    check_hipblaslt(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(int32_t)), "transB");

    // FP8 requires scale pointers
    if (cfg.needs_scale_ptrs) {
        check_hipblaslt(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_a_scale, sizeof(void*)), "scaleA");
        check_hipblaslt(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_b_scale, sizeof(void*)), "scaleB");
        check_hipblaslt(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_d_scale, sizeof(void*)), "scaleD");
    }

    check_hipblaslt(hipblasLtMatmulPreferenceCreate(&pref), "prefCreate");
    int64_t max_ws = (int64_t)workspace_size;
    check_hipblaslt(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws)), "prefWS");

    // Request multiple solutions and pick the best
    const int request_solutions = 16;
    hipblasLtMatmulHeuristicResult_t heuristicResults[request_solutions];
    int returnedAlgoCount = 0;

    check_hipblaslt(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmul, matA, matB, matC, matD, pref,
        request_solutions, heuristicResults, &returnedAlgoCount), "heuristic");

    float best_ms = 1e9f;

    if (returnedAlgoCount == 0) {
        std::cerr << "  hipBLASLt: No valid solution found!" << std::endl;
        best_ms = -1.0f;
    } else {
        float alpha = 1.0f, beta = 0.0f;

        // Try all returned algorithms, pick fastest
        for (int ai = 0; ai < returnedAlgoCount; ai++) {
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            // Warmup
            hipblasLtMatmul(handle, matmul, &alpha, A, matA, B, matB, &beta, C, matC,
                            D, matD, &heuristicResults[ai].algo, workspace,
                            heuristicResults[ai].workspaceSize, 0);
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipEventRecord(start));
            hipblasLtMatmul(handle, matmul, &alpha, A, matA, B, matB, &beta, C, matC,
                            D, matD, &heuristicResults[ai].algo, workspace,
                            heuristicResults[ai].workspaceSize, 0);
            HIP_CHECK(hipEventRecord(stop));
            HIP_CHECK(hipEventSynchronize(stop));

            float ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
            HIP_CHECK(hipEventDestroy(start));
            HIP_CHECK(hipEventDestroy(stop));

            if (ms > 0 && ms < best_ms)
                best_ms = ms;
        }
    }

    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatmulDescDestroy(matmul);
    hipblasLtMatrixLayoutDestroy(matA);
    hipblasLtMatrixLayoutDestroy(matB);
    hipblasLtMatrixLayoutDestroy(matC);
    hipblasLtMatrixLayoutDestroy(matD);

    return best_ms;
}

// ============================================================================
// Sweep helper: tries multiple sizes, measures with best algo
// ============================================================================

struct SweepResult {
    int best_M;
    float best_ms;
    double best_gflops;
};

static SweepResult sweep_hipblaslt(hipblasLtHandle_t handle,
                                    const HipBlasLtGemmConfig& cfg,
                                    int measurement_trials) {
    std::vector<int> sizes = {1024, 2048, 4096, 8192, 16384, 32768};
    SweepResult best = {0, 1e9f, 0.0};

    size_t free_mem = 0, total_mem = 0;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    int64_t mem_limit = (int64_t)(free_mem * 0.70);  // 70% — need workspace too

    // Allocate workspace (32 MB)
    size_t workspace_size = 32 * 1024 * 1024;
    void* workspace = nullptr;
    HIP_CHECK(hipMalloc(&workspace, workspace_size));

    // Scale pointers for FP8
    void *d_a_scale = nullptr, *d_b_scale = nullptr, *d_d_scale = nullptr;
    if (cfg.needs_scale_ptrs) {
        float one = 1.0f;
        HIP_CHECK(hipMalloc(&d_a_scale, sizeof(float)));
        HIP_CHECK(hipMalloc(&d_b_scale, sizeof(float)));
        HIP_CHECK(hipMalloc(&d_d_scale, sizeof(float)));
        HIP_CHECK(hipMemcpy(d_a_scale, &one, sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b_scale, &one, sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_d_scale, &one, sizeof(float), hipMemcpyHostToDevice));
    }

    for (int M : sizes) {
        int64_t ab_bytes = (int64_t)M * M * cfg.elem_bytes;
        int64_t cd_bytes = (int64_t)M * M * cfg.out_elem_bytes;
        int64_t total_bytes = ab_bytes * 2 + cd_bytes * 2;  // A, B, C, D
        if (total_bytes > mem_limit) continue;

        void *A = nullptr, *B = nullptr, *C = nullptr, *D = nullptr;
        if (hipMalloc(&A, ab_bytes) != hipSuccess) continue;
        HIP_CHECK(hipMalloc(&B, ab_bytes));
        HIP_CHECK(hipMalloc(&C, cd_bytes));
        HIP_CHECK(hipMalloc(&D, cd_bytes));
        HIP_CHECK(hipMemset(A, 0, ab_bytes));
        HIP_CHECK(hipMemset(B, 0, ab_bytes));
        HIP_CHECK(hipMemset(C, 0, cd_bytes));
        HIP_CHECK(hipMemset(D, 0, cd_bytes));

        // Find best algo for this size
        float probe_ms = run_hipblaslt_gemm(handle, cfg, M, M, M,
                                             A, B, C, D, workspace, workspace_size,
                                             d_a_scale, d_b_scale, d_d_scale);
        if (probe_ms <= 0) {
            HIP_CHECK(hipFree(A)); HIP_CHECK(hipFree(B)); HIP_CHECK(hipFree(C)); HIP_CHECK(hipFree(D));
            continue;
        }

        // Measurement: re-run with timing (using the library's best algo from heuristic)
        std::vector<float> times;
        for (int t = 0; t < measurement_trials; t++) {
            // Create fresh matmul desc for each run to pick algo consistently
            hipblasLtMatrixLayout_t mA, mB, mC, mD;
            hipblasLtMatmulDesc_t mmul;
            hipblasLtMatmulPreference_t prf;

            hipblasLtMatrixLayoutCreate(&mA, cfg.a_type, M, M, M);
            hipblasLtMatrixLayoutCreate(&mB, cfg.b_type, M, M, M);
            hipblasLtMatrixLayoutCreate(&mC, cfg.c_type, M, M, M);
            hipblasLtMatrixLayoutCreate(&mD, cfg.d_type, M, M, M);
            hipblasLtMatmulDescCreate(&mmul, cfg.compute_type, cfg.scale_type);

            hipblasOperation_t opN = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(mmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(int32_t));
            hipblasLtMatmulDescSetAttribute(mmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(int32_t));

            if (cfg.needs_scale_ptrs) {
                hipblasLtMatmulDescSetAttribute(mmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_a_scale, sizeof(void*));
                hipblasLtMatmulDescSetAttribute(mmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_b_scale, sizeof(void*));
                hipblasLtMatmulDescSetAttribute(mmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_d_scale, sizeof(void*));
            }

            hipblasLtMatmulPreferenceCreate(&prf);
            int64_t max_ws = (int64_t)workspace_size;
            hipblasLtMatmulPreferenceSetAttribute(prf, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws));

            hipblasLtMatmulHeuristicResult_t hr[1];
            int count = 0;
            hipblasLtMatmulAlgoGetHeuristic(handle, mmul, mA, mB, mC, mD, prf, 1, hr, &count);

            if (count > 0) {
                float alpha = 1.0f, beta = 0.0f;
                hipEvent_t start, stop;
                HIP_CHECK(hipEventCreate(&start));
                HIP_CHECK(hipEventCreate(&stop));

                HIP_CHECK(hipEventRecord(start));
                hipblasLtMatmul(handle, mmul, &alpha, A, mA, B, mB, &beta, C, mC,
                                D, mD, &hr[0].algo, workspace, hr[0].workspaceSize, 0);
                HIP_CHECK(hipEventRecord(stop));
                HIP_CHECK(hipEventSynchronize(stop));

                float ms = 0.0f;
                HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
                if (ms > 0) times.push_back(ms);
                HIP_CHECK(hipEventDestroy(start));
                HIP_CHECK(hipEventDestroy(stop));
            }

            hipblasLtMatmulPreferenceDestroy(prf);
            hipblasLtMatmulDescDestroy(mmul);
            hipblasLtMatrixLayoutDestroy(mA);
            hipblasLtMatrixLayoutDestroy(mB);
            hipblasLtMatrixLayoutDestroy(mC);
            hipblasLtMatrixLayoutDestroy(mD);
        }

        HIP_CHECK(hipFree(A)); HIP_CHECK(hipFree(B)); HIP_CHECK(hipFree(C)); HIP_CHECK(hipFree(D));

        if (times.empty()) continue;
        std::sort(times.begin(), times.end());
        float median_ms = times[times.size() / 2];
        double flops = 2.0 * (double)M * M * M;
        double gflops = (flops / (median_ms * 1e-3)) / 1e9;

        std::cerr << "    M=" << M << ": " << gflops << " GFLOP/s"
                  << " (median " << median_ms << " ms)" << std::endl;

        if (gflops > best.best_gflops) {
            best.best_M = M;
            best.best_ms = median_ms;
            best.best_gflops = gflops;
        }
    }

    std::cerr << "  Best: M=" << best.best_M << " → " << best.best_gflops << " GFLOP/s" << std::endl;

    HIP_CHECK(hipFree(workspace));
    if (d_a_scale) HIP_CHECK(hipFree(d_a_scale));
    if (d_b_scale) HIP_CHECK(hipFree(d_b_scale));
    if (d_d_scale) HIP_CHECK(hipFree(d_d_scale));

    return best;
}

// ============================================================================
// Kernel class: gemm_hipblaslt
// Supports TF32, FP8_E4M3, FP8_E5M2 (and can also run FP16, BF16, FP32)
// ============================================================================

class GemmHipBlasLt : public KernelBase {
public:
    std::string name() const override { return "gemm_hipblaslt"; }
    std::string category() const override { return "matrix"; }
    std::string backend() const override { return "hip"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::TF32,
                Precision::FP8_E4M3, Precision::FP8_E5M2};
    }

    std::vector<std::string> supported_modes() const override {
        return {"throughput"};
    }

    bool is_available(const DeviceInfo& device) const override {
        // hipBLASLt with CDNA3 (gfx942) features
        // TF32 and FP8 require gfx940+
        return device.arch.find("gfx94") != std::string::npos;
    }

    KernelResult run(const KernelConfig& config,
                     const DeviceInfo& device,
                     int measurement_trials) override {
        int dev_idx = 0;
        auto pos = device.id.find(':');
        if (pos != std::string::npos)
            dev_idx = std::stoi(device.id.substr(pos + 1));
        HIP_CHECK(hipSetDevice(dev_idx));

        hipblasLtHandle_t handle;
        check_hipblaslt(hipblasLtCreate(&handle), "create handle");

        HipBlasLtGemmConfig cfg;
        std::string peak_key;

        switch (config.precision) {
            case Precision::TF32: {
                // FP32 inputs, TF32 compute (truncates to 19-bit mantissa)
                cfg.a_type = HIP_R_32F;
                cfg.b_type = HIP_R_32F;
                cfg.c_type = HIP_R_32F;
                cfg.d_type = HIP_R_32F;
                cfg.compute_type = HIPBLAS_COMPUTE_32F_FAST_TF32;
                cfg.scale_type = HIP_R_32F;
                cfg.elem_bytes = 4;
                cfg.out_elem_bytes = 4;
                cfg.needs_scale_ptrs = false;
                peak_key = "TF32_MFMA";
                break;
            }
            case Precision::FP8_E4M3: {
                // FP8 E4M3 FNUZ (MI300X/gfx942 specific)
                // AMD sample: A=FP8, B=FP8, C=FP8, D=FP8, compute=FP32
                cfg.a_type = HIP_R_8F_E4M3_FNUZ;
                cfg.b_type = HIP_R_8F_E4M3_FNUZ;
                cfg.c_type = HIP_R_8F_E4M3_FNUZ;
                cfg.d_type = HIP_R_8F_E4M3_FNUZ;
                cfg.compute_type = HIPBLAS_COMPUTE_32F;
                cfg.scale_type = HIP_R_32F;
                cfg.elem_bytes = 1;
                cfg.out_elem_bytes = 1;
                cfg.needs_scale_ptrs = true;
                peak_key = "FP8_MFMA";
                break;
            }
            case Precision::FP8_E5M2: {
                // FP8 E5M2 FNUZ (MI300X/gfx942 specific)
                cfg.a_type = HIP_R_8F_E5M2_FNUZ;
                cfg.b_type = HIP_R_8F_E5M2_FNUZ;
                cfg.c_type = HIP_R_8F_E5M2_FNUZ;
                cfg.d_type = HIP_R_8F_E5M2_FNUZ;
                cfg.compute_type = HIPBLAS_COMPUTE_32F;
                cfg.scale_type = HIP_R_32F;
                cfg.elem_bytes = 1;
                cfg.out_elem_bytes = 1;
                cfg.needs_scale_ptrs = true;
                peak_key = "FP8_MFMA";
                break;
            }
            default: {
                KernelResult r;
                r.gflops = 0;
                return r;
            }
        }

        auto sr = sweep_hipblaslt(handle, cfg, measurement_trials);

        KernelResult result;
        result.gflops = sr.best_gflops;
        result.median_time_ms = sr.best_ms;

        auto it = device.theoretical_peak_gflops.find(peak_key);
        if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
            result.peak_percent = (result.gflops / it->second) * 100.0;
        }

        check_hipblaslt(hipblasLtDestroy(handle), "destroy handle");

        return result;
    }
};

REGISTER_KERNEL(GemmHipBlasLt);

namespace force_link {
    void gemm_hipblaslt_link() {
        volatile auto* p = &KernelRegistry::instance();
        (void)p;
    }
}

} // namespace floptic

#else  // !FLOPTIC_HAS_HIPBLASLT

namespace floptic {
namespace force_link {
    void gemm_hipblaslt_link() {}
}
}

#endif // FLOPTIC_HAS_HIPBLASLT
