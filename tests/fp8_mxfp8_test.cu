// Standalone test for Blackwell MXFP8 (block-scaled FP8) GEMM
// Build: nvcc -arch=sm_100 -o tests/fp8_mxfp8_test tests/fp8_mxfp8_test.cu -lcublasLt -lcublas
// Based on NVIDIA CUDALibrarySamples/cuBLASLt/LtMxfp8Matmul

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define CHECK_CUDA(x) do { auto e = (x); if(e) { printf("CUDA FAIL %d at line %d\n", e, __LINE__); exit(1); } } while(0)
#define CHECK_CUBLAS(x) do { auto s = (x); if(s) { printf("cuBLAS FAIL %d at line %d\n", s, __LINE__); exit(1); } } while(0)

int main() {
    // NVIDIA sample uses: CUBLAS_OP_T, CUBLAS_OP_N, m=64, n=128, k=256
    // cuBLAS column-major: C(m,n) = alpha * op(A)(m,k) * op(B)(k,n) + beta * C
    // With CUBLAS_OP_T for A: physical A is k×m, transposed to m×k
    // With CUBLAS_OP_N for B: physical B is k×n
    // lda = k (leading dim of physical A which is k×m)
    // ldb = k (leading dim of physical B which is k×n)

    int M = 4096, N = 4096, K = 4096;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    printf("MXFP8 Block-Scaled FP8 GEMM Test (M=%d, N=%d, K=%d)\n", M, N, K);
    printf("  transa=%s, transb=%s\n",
           transa == CUBLAS_OP_T ? "T" : "N",
           transb == CUBLAS_OP_T ? "T" : "N");
    printf("cuBLASLt version: %zu\n", cublasLtGetVersion());

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 10) {
        printf("MXFP8 requires sm_100+ (Blackwell). Skipping.\n");
        return 0;
    }

    // For MXFP8: 32-element 1D block scaling along K dimension
    // Scale type: __nv_fp8_e8m0 (unsigned exponent-only, 1 byte)
    // A physical: K×M (since transa=T), scale: ceil(K/32) × M  
    // B physical: K×N (since transb=N), scale: ceil(K/32) × N
    // D output:   M×N, scale: ceil(N/32) × M (output block scale)

    int scale_k = (K + 31) / 32;
    int scale_n = (N + 31) / 32;

    // Physical dimensions (column-major)
    // A: K×M (transposed), lda = K
    // B: K×N, ldb = K
    // C: M×N, ldc = M
    // D: M×N, ldd = M
    int lda = K, ldb = K, ldc = M, ldd = M;

    printf("Scale dims: scale_k=%d, scale_n=%d\n", scale_k, scale_n);

    // Allocate device memory
    void *d_A, *d_B, *d_C, *d_D;
    void *d_A_scale, *d_B_scale, *d_C_scale, *d_D_out_scale;
    void *d_workspace;
    size_t workspaceSize = 64 * 1024 * 1024;

    // A: K×M FP8 E4M3 (1 byte each)
    CHECK_CUDA(cudaMalloc(&d_A, (size_t)K * M));
    // B: K×N FP8 E4M3
    CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N));
    // C: M×N BF16 (2 bytes each)
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * 2));
    // D: M×N FP8 E4M3 (output)
    CHECK_CUDA(cudaMalloc(&d_D, (size_t)M * N));

    // Scale arrays: E8M0 type (1 byte each)
    // A scales: scale_k × M  
    CHECK_CUDA(cudaMalloc(&d_A_scale, (size_t)scale_k * M));
    // B scales: scale_k × N
    CHECK_CUDA(cudaMalloc(&d_B_scale, (size_t)scale_k * N));
    // C scale: single float (scalar mode)
    CHECK_CUDA(cudaMalloc(&d_C_scale, sizeof(float)));
    // D output scales: scale_n × M (block-scaled output)
    CHECK_CUDA(cudaMalloc(&d_D_out_scale, (size_t)scale_n * M));

    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));

    // Initialize
    cudaMemset(d_A, 0x38, (size_t)K * M);       // ~0.5 in E4M3
    cudaMemset(d_B, 0x38, (size_t)K * N);
    cudaMemset(d_C, 0, (size_t)M * N * 2);
    cudaMemset(d_D, 0, (size_t)M * N);
    // E8M0 scale = 2^(val-127), so val=127 → scale=1.0
    cudaMemset(d_A_scale, 127, (size_t)scale_k * M);
    cudaMemset(d_B_scale, 127, (size_t)scale_k * N);
    float c_scale_val = 1.0f;
    cudaMemcpy(d_C_scale, &c_scale_val, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_D_out_scale, 0, (size_t)scale_n * M);

    cublasLtHandle_t ltHandle;
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));

    // Create matmul descriptor
    cublasLtMatmulDesc_t opDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Set block scaling modes (matching NVIDIA sample exactly)
    cublasLtMatmulMatrixScale_t scaleModeBlock = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    cublasLtMatmulMatrixScale_t scaleModeScalar = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleModeBlock, sizeof(scaleModeBlock)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleModeBlock, sizeof(scaleModeBlock)));
    // C scale = scalar (per-tensor)
    // D out scale = block (VEC32_UE8M0)
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &scaleModeBlock, sizeof(scaleModeBlock)));

    // Set scale pointers
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_A_scale, sizeof(d_A_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_B_scale, sizeof(d_B_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &d_D_out_scale, sizeof(d_D_out_scale)));

    // Create matrix layouts
    // A physical: transa=T → physical is K×M, data type FP8 E4M3
    // B physical: transb=N → physical is K×N, data type FP8 E4M3
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, K, M, lda));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, N, ldb));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, M, N, ldc));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_8F_E4M3, M, N, ldd));

    // Create preference
    cublasLtMatmulPreference_t pref;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // Get heuristic
    int nResults = 0;
    cublasLtMatmulHeuristicResult_t hResult;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, opDesc, Adesc, Bdesc, Cdesc, Ddesc, pref, 1, &hResult, &nResults);

    printf("Heuristic: status=%d (%s), results=%d\n", st,
           st == 0 ? "SUCCESS" : (st == 15 ? "NOT_SUPPORTED" : "OTHER"), nResults);

    if (nResults > 0 && st == CUBLAS_STATUS_SUCCESS) {
        float alpha = 1.0f, beta = 0.0f;

        // Warmup
        CHECK_CUBLAS(cublasLtMatmul(ltHandle, opDesc,
            &alpha, d_A, Adesc, d_B, Bdesc,
            &beta, d_C, Cdesc, d_D, Ddesc,
            &hResult.algo, d_workspace, workspaceSize, 0));
        cudaDeviceSynchronize();

        // Time it
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int nruns = 10;
        float totalMs = 0;
        for (int i = 0; i < nruns; i++) {
            cudaEventRecord(start);
            cublasLtMatmul(ltHandle, opDesc,
                &alpha, d_A, Adesc, d_B, Bdesc,
                &beta, d_C, Cdesc, d_D, Ddesc,
                &hResult.algo, d_workspace, workspaceSize, 0);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            totalMs += ms;
            printf("  Run %d: %.3f ms\n", i, ms);
        }

        double avgMs = totalMs / nruns;
        double gflops = (2.0 * M * N * K / 1e9) / (avgMs / 1e3);
        printf("MXFP8: %.3f ms avg, %.0f GFLOP/s (%.2f TF/s)\n", avgMs, gflops, gflops/1e3);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        printf("MXFP8 GEMM not supported\n");

        // Try different combos
        printf("\nTrying alternative configurations...\n");
        
        // Try N,N
        cublasOperation_t opN = CUBLAS_OP_N;
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
        
        cublasLtMatrixLayoutDestroy(Adesc);
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, M, K, M));
        
        st = cublasLtMatmulAlgoGetHeuristic(ltHandle, opDesc, Adesc, Bdesc, Cdesc, Ddesc, pref, 1, &hResult, &nResults);
        printf("  N,N: status=%d, results=%d\n", st, nResults);

        // Try without D_OUT_SCALE (simpler: just A and B block scaled, output to BF16)
        cublasLtMatmulDescDestroy(opDesc);
        CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        
        cublasOperation_t opT = CUBLAS_OP_T;
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleModeBlock, sizeof(scaleModeBlock)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleModeBlock, sizeof(scaleModeBlock)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_A_scale, sizeof(d_A_scale)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_B_scale, sizeof(d_B_scale)));
        
        // D = BF16 (no FP8 output, no D_OUT_SCALE)
        cublasLtMatrixLayoutDestroy(Adesc);
        cublasLtMatrixLayoutDestroy(Ddesc);
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, K, M, K));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, M, N, M));
        
        st = cublasLtMatmulAlgoGetHeuristic(ltHandle, opDesc, Adesc, Bdesc, Cdesc, Ddesc, pref, 1, &hResult, &nResults);
        printf("  T,N with BF16 output (no D_OUT_SCALE): status=%d, results=%d\n", st, nResults);
        
        if (nResults > 0 && st == CUBLAS_STATUS_SUCCESS) {
            float alpha = 1.0f, beta = 0.0f;
            
            // D output buffer as BF16
            void *d_D_bf16;
            CHECK_CUDA(cudaMalloc(&d_D_bf16, (size_t)M * N * 2));
            cudaMemset(d_D_bf16, 0, (size_t)M * N * 2);
            
            CHECK_CUBLAS(cublasLtMatmul(ltHandle, opDesc,
                &alpha, d_A, Adesc, d_B, Bdesc,
                &beta, d_C, Cdesc, d_D_bf16, Ddesc,
                &hResult.algo, d_workspace, workspaceSize, 0));
            cudaDeviceSynchronize();

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            int nruns = 10;
            float totalMs = 0;
            for (int i = 0; i < nruns; i++) {
                cudaEventRecord(start);
                cublasLtMatmul(ltHandle, opDesc,
                    &alpha, d_A, Adesc, d_B, Bdesc,
                    &beta, d_C, Cdesc, d_D_bf16, Ddesc,
                    &hResult.algo, d_workspace, workspaceSize, 0);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms;
                cudaEventElapsedTime(&ms, start, stop);
                totalMs += ms;
                printf("    Run %d: %.3f ms\n", i, ms);
            }
            double avgMs = totalMs / nruns;
            double gflops = (2.0 * M * N * K / 1e9) / (avgMs / 1e3);
            printf("  MXFP8→BF16: %.3f ms avg, %.0f GFLOP/s (%.2f TF/s)\n", avgMs, gflops, gflops/1e3);
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(d_D_bf16);
        }
    }

    // Cleanup
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(opDesc);
    cublasLtDestroy(ltHandle);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    cudaFree(d_A_scale); cudaFree(d_B_scale); cudaFree(d_C_scale);
    cudaFree(d_D_out_scale); cudaFree(d_workspace);

    printf("Done.\n");
    return 0;
}
