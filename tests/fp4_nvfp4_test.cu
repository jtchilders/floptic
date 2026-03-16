// Standalone test for Blackwell NVFP4 (block-scaled FP4) GEMM
// Build: nvcc -o tests/fp4_nvfp4_test tests/fp4_nvfp4_test.cu -lcublasLt -lcublas
// Based on NVIDIA CUDALibrarySamples/cuBLASLt/LtNvfp4Matmul

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(x) do { auto e = (x); if(e) { printf("CUDA FAIL %d at line %d\n", e, __LINE__); exit(1); } } while(0)
#define CHECK_CUBLAS(x) do { auto s = (x); if(s) { printf("cuBLAS FAIL %d at line %d\n", s, __LINE__); exit(1); } } while(0)

int main() {
    int M = 4096, N = 4096, K = 4096;

    printf("NVFP4 Block-Scaled FP4 GEMM Test (M=N=K=%d)\n", M);
    printf("cuBLASLt version: %zu\n", cublasLtGetVersion());

    // Check compute capability
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 10) {
        printf("NVFP4 requires sm_100+ (Blackwell). Skipping.\n");
        return 0;
    }

    // For NVFP4: 16-element block scaling
    // FP4 is packed: 2 elements per byte
    // Scale type: FP8 E4M3 (UE4M3 = unsigned, but stored as regular E4M3)
    // Scale array size: ceil(K/16) per row for A, ceil(K/16) per col for B

    int scale_k = (K + 15) / 16;  // number of 16-element blocks in K dimension

    printf("FP4 packed size: %zu bytes per matrix\n", (size_t)M * K / 2);
    printf("Scale dimensions: scale_k=%d (K=%d, block=16)\n", scale_k, K);

    // Allocate device memory
    void *d_A, *d_B, *d_C, *d_D;
    void *d_A_scale, *d_B_scale, *d_D_scale, *d_D_out_scale;
    void *d_workspace;
    size_t workspaceSize = 64 * 1024 * 1024;

    // A: M×K FP4 packed (0.5 bytes each = M*K/2 bytes)
    CHECK_CUDA(cudaMalloc(&d_A, (size_t)M * K / 2));
    // B: K×N FP4 packed
    CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N / 2));
    // C: M×N BF16 (bias, 2 bytes)
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * 2));
    // D: M×N FP4 packed (output)
    CHECK_CUDA(cudaMalloc(&d_D, (size_t)M * N / 2));

    // Scale arrays: FP8 E4M3 type for NVFP4 (1 byte each)
    // A scales: M × scale_k
    CHECK_CUDA(cudaMalloc(&d_A_scale, (size_t)M * scale_k));
    // B scales: scale_k × N
    CHECK_CUDA(cudaMalloc(&d_B_scale, (size_t)scale_k * N));
    // D global scale: single float
    float d_scale_val = 1.0f;
    CHECK_CUDA(cudaMalloc(&d_D_scale, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_D_scale, &d_scale_val, sizeof(float), cudaMemcpyHostToDevice));
    // D output block scales: M × ceil(N/16)
    int scale_n = (N + 15) / 16;
    CHECK_CUDA(cudaMalloc(&d_D_out_scale, (size_t)M * scale_n));

    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));

    // Initialize data
    cudaMemset(d_A, 0x33, (size_t)M * K / 2);
    cudaMemset(d_B, 0x33, (size_t)K * N / 2);
    cudaMemset(d_C, 0, (size_t)M * N * 2);
    cudaMemset(d_D, 0, (size_t)M * N / 2);
    // Scale = 1.0 in E4M3: bit pattern 0x38 (sign=0, exp=0111, mantissa=000 → 2^0 = 1.0)
    cudaMemset(d_A_scale, 0x38, (size_t)M * scale_k);
    cudaMemset(d_B_scale, 0x38, (size_t)scale_k * N);
    cudaMemset(d_D_out_scale, 0, (size_t)M * scale_n);

    cublasLtHandle_t ltHandle;
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));

    // Create matmul descriptor
    cublasLtMatmulDesc_t opDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // Set block scaling modes for NVFP4
    // CUBLASLT_MATMUL_MATRIX_SCALE_1D_BLOCK_16 for FP4 (16-element blocks)
    cublasLtMatmulMatrixScale_t scaleMode1D_16 = CUBLASLT_MATMUL_MATRIX_SCALE_1D_BLOCK_16;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode1D_16, sizeof(scaleMode1D_16)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode1D_16, sizeof(scaleMode1D_16)));

    // D has both a global float scale and per-block output scales
    cublasLtMatmulMatrixScale_t scaleModeTensor = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &scaleModeTensor, sizeof(scaleModeTensor)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &scaleMode1D_16, sizeof(scaleMode1D_16)));

    // Set scale pointers
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_A_scale, sizeof(d_A_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_B_scale, sizeof(d_B_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_D_scale, sizeof(d_D_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &d_D_out_scale, sizeof(d_D_out_scale)));

    // Create matrix layouts — FP4 type
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, M, K, M));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, K, N, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, M, N, M));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_4F_E2M1, M, N, M));

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
        printf("NVFP4: %.3f ms avg, %.0f GFLOP/s (%.2f TF/s)\n", avgMs, gflops, gflops/1e3);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        printf("NVFP4 GEMM not supported (status=%d, algos=%d)\n", st, nResults);
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
