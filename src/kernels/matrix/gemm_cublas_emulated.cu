// cuBLAS FP emulation GEMM kernels
//   gemm_cublas_emu_fp32: BF16x9 — emulates FP32 via BF16 tensor cores
//   gemm_cublas_emu_fp64: Ozaki  — emulates FP64 via INT8 tensor cores
//
// Uses standard cublasSgemm / cublasDgemm with emulation math modes.
// Requires cuBLAS 12.8+ for BF16x9, cuBLAS 13.0u2+ for Ozaki FP64.
//
// Architecture support (from NVIDIA docs):
//   BF16x9 FP32: Hopper (sm_90+) and Blackwell (sm_100+)
//   Ozaki FP64:  Blackwell (sm_100+) only

#ifdef FLOPTIC_HAS_CUDA

#include "floptic/kernel_base.hpp"
#include "floptic/kernel_registry.hpp"
#include "floptic/timer.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>
#include <algorithm>

// Guard: emulation API requires cuBLAS 12.8+.
// Check via CUDA version since cuBLAS version macros aren't reliable at compile time.
// CUBLAS_FP32_EMULATED_BF16X9_MATH was introduced in cuBLAS 12.8 (CUDA 12.8).
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8)
#define FLOPTIC_HAS_EMULATION 1
#endif

namespace floptic {

#ifdef FLOPTIC_HAS_EMULATION

static void check_cublas_emu(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "  cuBLAS ERROR: " << msg << " (status=" << status << ")" << std::endl;
    }
}

// ============================================================================
// BF16x9 emulated FP32 GEMM
// Uses cublasSgemm with CUBLAS_FP32_EMULATED_BF16X9_MATH
// ============================================================================

static float run_emu_sgemm(cublasHandle_t handle, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, (size_t)M * K * sizeof(float));
    cudaMalloc(&d_B, (size_t)K * N * sizeof(float));
    cudaMalloc(&d_C, (size_t)M * N * sizeof(float));

    cudaMemset(d_A, 0, (size_t)M * K * sizeof(float));
    cudaMemset(d_B, 0, (size_t)K * N * sizeof(float));
    cudaMemset(d_C, 0, (size_t)M * N * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                &alpha, d_A, M, d_B, K, &beta, d_C, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return ms;
}

// ============================================================================
// Ozaki emulated FP64 GEMM
// Uses cublasDgemm with CUBLAS_FP64_EMULATED_FIXED_POINT_MATH
// ============================================================================

static float run_emu_dgemm(cublasHandle_t handle, int M, int N, int K) {
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, (size_t)M * K * sizeof(double));
    cudaMalloc(&d_B, (size_t)K * N * sizeof(double));
    cudaMalloc(&d_C, (size_t)M * N * sizeof(double));

    cudaMemset(d_A, 0, (size_t)M * K * sizeof(double));
    cudaMemset(d_B, 0, (size_t)K * N * sizeof(double));
    cudaMemset(d_C, 0, (size_t)M * N * sizeof(double));

    double alpha = 1.0, beta = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                &alpha, d_A, M, d_B, K, &beta, d_C, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return ms;
}

// ============================================================================
// Common sweep+measure (reused from gemm_cublas pattern)
// ============================================================================

using RunFn = float (*)(cublasHandle_t, int, int, int);

static KernelResult sweep_and_measure_emu(
    cublasHandle_t handle, const DeviceInfo& device, int measurement_trials,
    const std::string& label, const std::string& peak_key,
    RunFn run_fn, size_t elem_size)
{
    int sweep_sizes[] = {1024, 2048, 4096, 8192, 16384};
    int num_sweep = sizeof(sweep_sizes) / sizeof(sweep_sizes[0]);

    double best_gflops = 0;
    int best_size = 4096;

    std::cerr << "  Sweeping " << label << ":" << std::endl;

    for (int si = 0; si < num_sweep; si++) {
        int M = sweep_sizes[si], N = M, K = M;

        // Memory check: 3 matrices + emulation workspace
        // Ozaki needs extra workspace (~4× for splits); be conservative
        size_t needed = 3ULL * M * N * elem_size * 5;
        if (needed > device.memory_bytes * 0.8) {
            std::cerr << "    M=N=K=" << M << ": skipped (memory)" << std::endl;
            continue;
        }

        // Warmup
        for (int w = 0; w < 3; w++) run_fn(handle, M, N, K);

        // Quick trial
        std::vector<double> times;
        int sweep_trials = std::min(3, measurement_trials);
        for (int t = 0; t < sweep_trials; t++) {
            float ms = run_fn(handle, M, N, K);
            if (ms > 0) times.push_back(static_cast<double>(ms));
        }
        if (times.empty()) continue;

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

    if (best_gflops <= 0) return KernelResult();

    // Full measurement at best size
    int M = best_size, N = best_size, K = best_size;
    int64_t flops_per_trial = 2LL * M * N * K;

    std::cerr << "  Best size: M=N=K=" << best_size << " → full measurement ("
              << measurement_trials << " trials)" << std::endl;

    for (int w = 0; w < 3; w++) run_fn(handle, M, N, K);

    std::vector<double> times;
    times.reserve(measurement_trials);
    for (int t = 0; t < measurement_trials; t++) {
        float ms = run_fn(handle, M, N, K);
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
// Kernel: BF16x9 Emulated FP32 GEMM
//
// Uses BF16 tensor cores to compute an FP32-accurate GEMM.
// BF16x9 decomposes each FP32 value into 3 BF16 values and performs 9 BF16
// GEMMs, giving exact FP32 results at near-BF16 TC speed.
//
// Supported: sm_90+ (Hopper), sm_100+ (Blackwell)
// ============================================================================

class GemmCublasEmuFP32 : public KernelBase {
public:
    std::string name() const override { return "gemm_cublas_emu_fp32"; }
    std::string category() const override { return "matrix"; }
    std::string backend() const override { return "cuda"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP32};
    }

    std::vector<std::string> supported_modes() const override {
        return {"throughput"};
    }

    // BF16x9 FP32 emulation: sm_90+ (Hopper) and sm_100+ (Blackwell)
    bool is_available(const DeviceInfo& device) const override {
        if (device.arch.size() >= 5 && device.arch.substr(0, 3) == "sm_") {
            int sm = std::stoi(device.arch.substr(3));
            return sm >= 90;
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

        cublasHandle_t handle;
        check_cublas_emu(cublasCreate(&handle), "cublasCreate");

        // Enable BF16x9 emulation mode
        check_cublas_emu(
            cublasSetMathMode(handle, CUBLAS_FP32_EMULATED_BF16X9_MATH),
            "set BF16x9 math mode");

        // Use EAGER strategy to always use emulation (even for small problems)
        // This ensures we benchmark the emulation path, not native fallback
        check_cublas_emu(
            cublasSetEmulationStrategy(handle, CUBLAS_EMULATION_STRATEGY_EAGER),
            "set emulation strategy eager");

        // Peak comparison: BF16x9 does 9 BF16 TC GEMMs internally.
        // Theoretical peak is BF16_TC_rate / 9 (since each FP32 op costs 9 BF16 ops).
        // But we report effective FP32 GFLOP/s and compare vs FP32 CUDA core peak.
        // This way we see the speedup vs native FP32.
        auto result = sweep_and_measure_emu(handle, device, measurement_trials,
            "gemm_cublas_emu_fp32 [BF16x9 emulated FP32]",
            "FP32", run_emu_sgemm, sizeof(float));

        cublasDestroy(handle);
        return result;
    }
};

REGISTER_KERNEL(GemmCublasEmuFP32);

// ============================================================================
// Kernel: Ozaki Emulated FP64 GEMM
//
// Uses INT8 tensor cores to compute an FP64-accurate GEMM via the Ozaki scheme.
// cuBLAS's Automatic Dynamic Precision (ADP) framework determines the number
// of mantissa bits needed to match or exceed native FP64 accuracy.
//
// Requires CUBLAS_FP64_EMULATED_FIXED_POINT_MATH (cuBLAS 13.1+ / CUDA 13.0u2+).
// Supported: sm_100+ (Blackwell) only
//
// The FP64 emulation API was added later than BF16x9; guard separately.
// ============================================================================

// Try to detect Ozaki API availability at compile time.
// CUBLAS_FP64_EMULATED_FIXED_POINT_MATH is a cublasMath_t enum value, not a macro.
// We check cuBLAS version at runtime instead, but still need the enum to compile.
// Use a conservative CUDA version gate: CUDA 13.0u2 shipped as CUDA 13.0.2.
// Since we can't distinguish 13.0.0 from 13.0.2 via __CUDACC_VER_MAJOR/MINOR__,
// we gate on CUDA >= 13.1 to be safe. If your CUDA 13.0u2 build fails,
// increase to >= 13.2.
// Ozaki FP64 emulation is available in CUDA 13.1+ (cuBLAS 13.2).
// Enum name: CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH (no underscore).
#if (__CUDACC_VER_MAJOR__ > 13) || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)
#define FLOPTIC_HAS_OZAKI 1
#endif

#ifdef FLOPTIC_HAS_OZAKI

class GemmCublasEmuFP64 : public KernelBase {
public:
    std::string name() const override { return "gemm_cublas_emu_fp64"; }
    std::string category() const override { return "matrix"; }
    std::string backend() const override { return "cuda"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP64};
    }

    std::vector<std::string> supported_modes() const override {
        return {"throughput"};
    }

    // Ozaki FP64 emulation: sm_100+ (Blackwell) only
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

        cublasHandle_t handle;
        check_cublas_emu(cublasCreate(&handle), "cublasCreate");

        // Enable fixed-point FP64 emulation (Ozaki scheme)
        // Note: enum is FIXEDPOINT (no underscore), not FIXED_POINT
        check_cublas_emu(
            cublasSetMathMode(handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH),
            "set FP64 fixed-point emulation math mode");

        // Dynamic mantissa control: cuBLAS automatically determines bits needed
        // for ≥ native FP64 accuracy (ADP framework)
        check_cublas_emu(
            cublasSetFixedPointEmulationMantissaControl(handle, CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC),
            "set dynamic mantissa control");

        // EAGER: always use emulation (benchmark the emulation path)
        check_cublas_emu(
            cublasSetEmulationStrategy(handle, CUBLAS_EMULATION_STRATEGY_EAGER),
            "set emulation strategy eager");

        // Compare vs native FP64 peak (CUDA cores)
        auto result = sweep_and_measure_emu(handle, device, measurement_trials,
            "gemm_cublas_emu_fp64 [Ozaki emulated FP64 via INT8 TC]",
            "FP64", run_emu_dgemm, sizeof(double));

        cublasDestroy(handle);
        return result;
    }
};

REGISTER_KERNEL(GemmCublasEmuFP64);

#endif // FLOPTIC_HAS_OZAKI

#endif // FLOPTIC_HAS_EMULATION

namespace force_link {
    void gemm_cublas_emu_fp32_link() {}
    void gemm_cublas_emu_fp64_link() {}
}

} // namespace floptic

#endif // FLOPTIC_HAS_CUDA
