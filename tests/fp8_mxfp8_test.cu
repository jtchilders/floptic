// Standalone test for Blackwell MXFP8 (block-scaled FP8) GEMM
// Build: nvcc -o tests/fp8_mxfp8_test tests/fp8_mxfp8_test.cu -lcublasLt -lcublas
// Based on NVIDIA CUDALibrarySamples/cuBLASLt/LtMxfp8Matmul

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(x) do { auto e = (x); if(e) { printf("CUDA FAIL %d at line %d\n", e, __LINE__); exit(1); } } while(0)
#define CHECK_CUBLAS(x) do { auto s = (x); if(s) { printf("cuBLAS FAIL %d at line %d\n", s, __LINE__); exit(1); } } while(0)

int main() {
    int M = 4096, N = 4096, K = 4096;

    printf("MXFP8 Block-Scaled FP8 GEMM Test (M=N=K=%d)\n", M);
    printf("cuBLASLt version: %zu\n", cublasLtGetVersion());

    // Check compute capability
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 10) {
        printf("MXFP8 requires sm_100+ (Blackwell). Skipping.\n");
        return 0;
    }

    // Check if required enums exist at compile time
#if !defined(CUBLASLT_MATMUL_DESC_A_SCALE_MODE)
    printf("ERROR: CUBLASLT_MATMUL_DESC_A_SCALE_MODE not defined. Need cuBLAS 12.9+\n");
    return 1;
#endif

    // For MXFP8: 32-element block scaling
    // Scale array size: ceil(K/32) per row for A, ceil(K/32) per column for B
    // A is M×K, scales are M × ceil(K/32)
    // B is K×N, scales are ceil(K/32) × N

    int scale_k = (K + 31) / 32;  // number of 32-element blocks in K dimension

    printf("Scale dimensions: scale_k=%d (K=%d, block=32)\n", scale_k, K);

    // Allocate device memory
    void *d_A, *d_B, *d_C, *d_D;
    void *d_A_scale, *d_B_scale, *d_D_out_scale;
    void *d_workspace;
    size_t workspaceSize = 64 * 1024 * 1024;

    // A: M×K FP8 E4M3 (1 byte each)
    CHECK_CUDA(cudaMalloc(&d_A, (size_t)M * K));
    // B: K×N FP8 E4M3
    CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N));
    // C: M×N BF16 (bias, 2 bytes)
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * 2));
    // D: M×N FP8 E4M3 (output)
    CHECK_CUDA(cudaMalloc(&d_D, (size_t)M * N));

    // Scale arrays: E8M0 type (1 byte each, unsigned exponent-only)
    // A scales: M × scale_k
    CHECK_CUDA(cudaMalloc(&d_A_scale, (size_t)M * scale_k));
    // B scales: scale_k × N
    CHECK_CUDA(cudaMalloc(&d_B_scale, (size_t)scale_k * N));
    // D output scales: M × ceil(N/32) (block-scaled output)
    int scale_n = (N + 31) / 32;
    CHECK_CUDA(cudaMalloc(&d_D_out_scale, (size_t)M * scale_n));

    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));

    // Initialize
    cudaMemset(d_A, 0x38, (size_t)M * K);       // ~0.5 in E4M3
    cudaMemset(d_B, 0x38, (size_t)K * N);
    cudaMemset(d_C, 0, (size_t)M * N * 2);
    cudaMemset(d_D, 0, (size_t)M * N);
    cudaMemset(d_A_scale, 127, (size_t)M * scale_k);   // E8M0 = 2^(val-127), so 127 → scale=1.0
    cudaMemset(d_B_scale, 127, (size_t)scale_k * N);
    cudaMemset(d_D_out_scale, 0, (size_t)M * scale_n);

    cublasLtHandle_t ltHandle;
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));

    // Create matmul descriptor
    cublasLtMatmulDesc_t opDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // Set block scaling modes
    // CUBLASLT_MATMUL_MATRIX_SCALE_1D_BLOCK_32 for FP8 (32-element blocks)
    cublasLtMatmulMatrixScale_t scaleMode1D = CUBLASLT_MATMUL_MATRIX_SCALE_1D_BLOCK_32;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode1D, sizeof(scaleMode1D)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode1D, sizeof(scaleMode1D)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &scaleMode1D, sizeof(scaleMode1D)));

    // Set scale pointers
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_A_scale, sizeof(d_A_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_B_scale, sizeof(d_B_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &d_D_out_scale, sizeof(d_D_out_scale)));

    // Create matrix layouts
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, M, K, M));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, N, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, M, N, M));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_8F_E4M3, M, N, M));

    // Create preference
    cublasLtMatmulPreference_t pref;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // Get heuristic
    int nResults = 0;
    cublasLtMatmulHeuristicResult_t hResult;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, opDesc, Adesc, Bdesc, Cdesc, Ddesc, pref, 1, &hResult, &nResults);

    printf("Heuristic: status=%d, results=%d\n", st, nResults);

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
        printf("MXFP8 GEMM not supported (status=%d, algos=%d)\n", st, nResults);

        // Try to enumerate algorithms directly
        int algoIds[100];
        int numAlgos = 0;
        cublasLtMatmulAlgoGetIds(ltHandle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
            CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, CUDA_R_8F_E4M3,
            100, algoIds, &numAlgos);
        printf("AlgoGetIds: %d algorithms found\n", numAlgos);
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
    cudaFree(d_A_scale); cudaFree(d_B_scale); cudaFree(d_D_out_scale);
    cudaFree(d_workspace);

    printf("Done.\n");
    return 0;
}
