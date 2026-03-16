// Standalone test for Blackwell NVFP4 (block-scaled FP4) GEMM
// Build: nvcc -arch=sm_100 -o tests/fp4_nvfp4_test tests/fp4_nvfp4_test.cu -lcublasLt -lcublas
// Based on NVIDIA CUDALibrarySamples/cuBLASLt/LtNvfp4Matmul

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(x) do { auto e = (x); if(e) { printf("CUDA FAIL %d at line %d\n", e, __LINE__); exit(1); } } while(0)
#define CHECK_CUBLAS(x) do { auto s = (x); if(s) { printf("cuBLAS FAIL %d at line %d\n", s, __LINE__); exit(1); } } while(0)

int main() {
    // NVIDIA sample: CUBLAS_OP_T, CUBLAS_OP_N, m=64, n=128, k=16
    // Use larger sizes for benchmarking
    int M = 4096, N = 4096, K = 4096;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    printf("NVFP4 Block-Scaled FP4 GEMM Test (M=%d, N=%d, K=%d)\n", M, N, K);
    printf("  transa=%s, transb=%s\n",
           transa == CUBLAS_OP_T ? "T" : "N",
           transb == CUBLAS_OP_T ? "T" : "N");
    printf("cuBLASLt version: %zu\n", cublasLtGetVersion());

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 10) {
        printf("NVFP4 requires sm_100+ (Blackwell). Skipping.\n");
        return 0;
    }

    // NVFP4: 16-element 1D block scaling
    // FP4 packed: 2 elements per byte (CUDA_R_4F_E2M1)
    // Scale type: UE4M3 (__nv_fp8_e4m3, unsigned), 1 byte each
    // With transa=T: physical A is K×M, scale_A = ceil(K/16) × M
    // With transb=N: physical B is K×N, scale_B = ceil(K/16) × N

    int scale_k = (K + 15) / 16;  // 16-element blocks for FP4

    printf("FP4 packed bytes per A: %zu (K×M/2)\n", (size_t)K * M / 2);
    printf("Scale dims: scale_k=%d (K=%d, block=16)\n", scale_k, K);

    int lda = K, ldb = K, ldc = M, ldd = M;

    // Allocate
    void *d_A, *d_B, *d_C, *d_D;
    void *d_A_scale, *d_B_scale, *d_D_scale, *d_D_out_scale;
    void *d_workspace;
    size_t workspaceSize = 64 * 1024 * 1024;

    // A: K×M FP4 packed (0.5 bytes each)
    CHECK_CUDA(cudaMalloc(&d_A, (size_t)K * M / 2));
    // B: K×N FP4 packed
    CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N / 2));
    // C: M×N BF16
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * 2));
    // D: M×N FP4 packed (output)
    CHECK_CUDA(cudaMalloc(&d_D, (size_t)M * N / 2));

    // Scale arrays: FP8 E4M3 for NVFP4
    CHECK_CUDA(cudaMalloc(&d_A_scale, (size_t)scale_k * M));
    CHECK_CUDA(cudaMalloc(&d_B_scale, (size_t)scale_k * N));
    // D global scale: single float
    float d_scale_val = 1.0f;
    CHECK_CUDA(cudaMalloc(&d_D_scale, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_D_scale, &d_scale_val, sizeof(float), cudaMemcpyHostToDevice));
    // D output block scales
    int scale_n = (N + 15) / 16;
    CHECK_CUDA(cudaMalloc(&d_D_out_scale, (size_t)scale_n * M));

    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));

    // Initialize
    cudaMemset(d_A, 0x33, (size_t)K * M / 2);
    cudaMemset(d_B, 0x33, (size_t)K * N / 2);
    cudaMemset(d_C, 0, (size_t)M * N * 2);
    cudaMemset(d_D, 0, (size_t)M * N / 2);
    // FP8 E4M3 scale = 1.0 → 0x38 (sign=0, exp=0111, mantissa=000)
    cudaMemset(d_A_scale, 0x38, (size_t)scale_k * M);
    cudaMemset(d_B_scale, 0x38, (size_t)scale_k * N);
    cudaMemset(d_D_out_scale, 0, (size_t)scale_n * M);

    cublasLtHandle_t ltHandle;
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));

    // Create matmul descriptor
    cublasLtMatmulDesc_t opDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Block scaling modes (matching NVIDIA NVFP4 sample)
    cublasLtMatmulMatrixScale_t scaleModeVec16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulMatrixScale_t scaleModeScalar = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleModeVec16, sizeof(scaleModeVec16)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleModeVec16, sizeof(scaleModeVec16)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &scaleModeScalar, sizeof(scaleModeScalar)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &scaleModeVec16, sizeof(scaleModeVec16)));

    // Scale pointers
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_A_scale, sizeof(d_A_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_B_scale, sizeof(d_B_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_D_scale, sizeof(d_D_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &d_D_out_scale, sizeof(d_D_out_scale)));

    // Matrix layouts — FP4 type (physical dims for T,N)
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, K, M, lda));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, K, N, ldb));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, M, N, ldc));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_4F_E2M1, M, N, ldd));

    // Preference
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
        printf("NVFP4: %.3f ms avg, %.0f GFLOP/s (%.2f TF/s)\n", avgMs, gflops, gflops/1e3);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        printf("NVFP4 GEMM not supported with FP4 output\n");
        
        // Try with BF16 output instead (simpler path, no D_OUT_SCALE)
        printf("\nTrying FP4 input → BF16 output (no quantized output)...\n");
        
        cublasLtMatmulDescDestroy(opDesc);
        CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleModeVec16, sizeof(scaleModeVec16)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleModeVec16, sizeof(scaleModeVec16)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_A_scale, sizeof(d_A_scale)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_B_scale, sizeof(d_B_scale)));
        
        // D = BF16
        cublasLtMatrixLayoutDestroy(Ddesc);
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, M, N, ldd));
        
        void *d_D_bf16;
        CHECK_CUDA(cudaMalloc(&d_D_bf16, (size_t)M * N * 2));
        cudaMemset(d_D_bf16, 0, (size_t)M * N * 2);
        
        st = cublasLtMatmulAlgoGetHeuristic(ltHandle, opDesc, Adesc, Bdesc, Cdesc, Ddesc, pref, 1, &hResult, &nResults);
        printf("  FP4→BF16: status=%d, results=%d\n", st, nResults);
        
        if (nResults > 0 && st == CUBLAS_STATUS_SUCCESS) {
            float alpha = 1.0f, beta = 0.0f;
            
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
            printf("  NVFP4→BF16: %.3f ms avg, %.0f GFLOP/s (%.2f TF/s)\n", avgMs, gflops, gflops/1e3);
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        
        cudaFree(d_D_bf16);
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
    cudaFree(d_A_scale); cudaFree(d_B_scale); cudaFree(d_D_scale);
    cudaFree(d_D_out_scale); cudaFree(d_workspace);

    printf("Done.\n");
    return 0;
}
