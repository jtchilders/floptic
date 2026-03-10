#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { auto s = (x); if(s) { printf("FAIL line %d: %d\n", __LINE__, s); exit(1); } } while(0)

int main() {
    int M=4096, N=4096, K=4096;

    // Alloc
    void *dA, *dB, *dC, *dD, *ws;
    float *dAs, *dBs, *dDs, *dAmax;
    cudaMalloc(&dA, M*K);
    cudaMalloc(&dB, K*N);
    cudaMalloc(&dC, M*N*2);  // BF16
    cudaMalloc(&dD, M*N);    // FP8
    cudaMalloc(&ws, 64*1024*1024);
    cudaMemset(dA, 0x38, M*K);
    cudaMemset(dB, 0x38, K*N);
    cudaMemset(dC, 0, M*N*2);
    cudaMemset(dD, 0, M*N);

    float one=1.0f, zero=0.0f;
    cudaMalloc(&dAs, 4); cudaMemcpy(dAs, &one, 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dBs, 4); cudaMemcpy(dBs, &one, 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dDs, 4); cudaMemcpy(dDs, &one, 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dAmax, 4); cudaMemcpy(dAmax, &zero, 4, cudaMemcpyHostToDevice);

    cublasLtHandle_t lt;
    cublasLtCreate(&lt);

    // Test 1: Minimal descriptor (no scale pointers)
    {
        printf("=== Test 1: No scale pointers ===\n");
        cublasLtMatmulDesc_t desc;
        CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

        cublasLtMatrixLayout_t Ad, Bd, Cd, Dd;
        CHECK(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8F_E4M3, M, K, M));
        CHECK(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8F_E4M3, K, N, K));
        CHECK(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF, M, N, M));
        CHECK(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_8F_E4M3, M, N, M));

        size_t wsz = 64*1024*1024;
        cublasLtMatmulPreference_t pref;
        CHECK(cublasLtMatmulPreferenceCreate(&pref));
        CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));

        int nr=0;
        cublasLtMatmulHeuristicResult_t hr;
        auto st = cublasLtMatmulAlgoGetHeuristic(lt, desc, Ad, Bd, Cd, Dd, pref, 1, &hr, &nr);
        printf("  Heuristic: status=%d, results=%d\n", st, nr);

        if (nr > 0) {
            float alpha=1, beta=0;
            auto ex = cublasLtMatmul(lt, desc, &alpha, dA, Ad, dB, Bd, &beta, dC, Cd, dD, Dd,
                                     &hr.algo, ws, wsz, 0);
            cudaDeviceSynchronize();
            printf("  Exec: %d\n", ex);
        }

        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(Dd); cublasLtMatrixLayoutDestroy(Cd);
        cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Ad);
        cublasLtMatmulDescDestroy(desc);
    }

    // Test 2: With scale pointers
    {
        printf("=== Test 2: With A/B/D scale + amax ===\n");
        cublasLtMatmulDesc_t desc;
        CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dAs, sizeof(dAs)));
        CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dBs, sizeof(dBs)));
        CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &dDs, sizeof(dDs)));
        CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &dAmax, sizeof(dAmax)));

        cublasLtMatrixLayout_t Ad, Bd, Cd, Dd;
        CHECK(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8F_E4M3, M, K, M));
        CHECK(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8F_E4M3, K, N, K));
        CHECK(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF, M, N, M));
        CHECK(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_8F_E4M3, M, N, M));

        size_t wsz = 64*1024*1024;
        cublasLtMatmulPreference_t pref;
        CHECK(cublasLtMatmulPreferenceCreate(&pref));
        CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));

        int nr=0;
        cublasLtMatmulHeuristicResult_t hr;
        auto st = cublasLtMatmulAlgoGetHeuristic(lt, desc, Ad, Bd, Cd, Dd, pref, 1, &hr, &nr);
        printf("  Heuristic: status=%d, results=%d\n", st, nr);

        if (nr > 0) {
            float alpha=1, beta=0;
            auto ex = cublasLtMatmul(lt, desc, &alpha, dA, Ad, dB, Bd, &beta, dC, Cd, dD, Dd,
                                     &hr.algo, ws, wsz, 0);
            cudaDeviceSynchronize();
            printf("  Exec: %d\n", ex);
        }

        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(Dd); cublasLtMatrixLayoutDestroy(Cd);
        cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Ad);
        cublasLtMatmulDescDestroy(desc);
    }

    // Test 3: C=D same type (both FP8)
    {
        printf("=== Test 3: C=D=FP8 (same buffer) ===\n");
        cublasLtMatmulDesc_t desc;
        CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dAs, sizeof(dAs)));
        CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dBs, sizeof(dBs)));
        CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &dDs, sizeof(dDs)));
        CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &dAmax, sizeof(dAmax)));

        cublasLtMatrixLayout_t Ad, Bd, Dd;
        CHECK(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8F_E4M3, M, K, M));
        CHECK(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8F_E4M3, K, N, K));
        CHECK(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_8F_E4M3, M, N, M));

        size_t wsz = 64*1024*1024;
        cublasLtMatmulPreference_t pref;
        CHECK(cublasLtMatmulPreferenceCreate(&pref));
        CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));

        int nr=0;
        cublasLtMatmulHeuristicResult_t hr;
        auto st = cublasLtMatmulAlgoGetHeuristic(lt, desc, Ad, Bd, Dd, Dd, pref, 1, &hr, &nr);
        printf("  Heuristic: status=%d, results=%d\n", st, nr);

        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(Dd);
        cublasLtMatrixLayoutDestroy(Bd);
        cublasLtMatrixLayoutDestroy(Ad);
        cublasLtMatmulDescDestroy(desc);
    }

    // Test 4: Smaller matrix 256x256
    {
        printf("=== Test 4: Small 256x256 ===\n");
        int m=256, n=256, k=256;
        cublasLtMatmulDesc_t desc;
        CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dAs, sizeof(dAs)));
        CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dBs, sizeof(dBs)));
        CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &dDs, sizeof(dDs)));

        cublasLtMatrixLayout_t Ad, Bd, Cd, Dd;
        CHECK(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8F_E4M3, m, k, m));
        CHECK(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8F_E4M3, k, n, k));
        CHECK(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF, m, n, m));
        CHECK(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_8F_E4M3, m, n, m));

        size_t wsz = 64*1024*1024;
        cublasLtMatmulPreference_t pref;
        CHECK(cublasLtMatmulPreferenceCreate(&pref));
        CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));

        int nr=0;
        cublasLtMatmulHeuristicResult_t hr;
        auto st = cublasLtMatmulAlgoGetHeuristic(lt, desc, Ad, Bd, Cd, Dd, pref, 1, &hr, &nr);
        printf("  Heuristic: status=%d, results=%d\n", st, nr);

        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(Dd); cublasLtMatrixLayoutDestroy(Cd);
        cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Ad);
        cublasLtMatmulDescDestroy(desc);
    }

    cublasLtDestroy(lt);
    printf("Done\n");
    return 0;
}
