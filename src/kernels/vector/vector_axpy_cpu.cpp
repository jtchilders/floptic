#include "floptic/kernel_base.hpp"
#include "floptic/kernel_registry.hpp"
#include "floptic/timer.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <cstdlib>

#ifdef FLOPTIC_HAS_OPENMP
#include <omp.h>
#endif

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#define FLOPTIC_AXPY_AVX512
#define FLOPTIC_AXPY_AVX2
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define FLOPTIC_AXPY_AVX2
#endif

namespace floptic {

extern volatile double g_validation_sink;

// ============================================================================
// Scalar fallback
// ============================================================================

template <typename T>
static double run_axpy_scalar(T* __restrict__ y, const T* __restrict__ x,
                               T alpha, int64_t n, int num_threads) {
    CpuTimer timer;
    timer.start();

    #ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    #endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = std::fma(alpha, x[i], y[i]);
    }

    timer.stop();
    return timer.elapsed_ms();
}

// ============================================================================
// AVX2 AXPY — FP64
// ============================================================================
#ifdef FLOPTIC_AXPY_AVX2

static double run_avx2_axpy_fp64(double* __restrict__ y, const double* __restrict__ x,
                                  double alpha, int64_t n, int num_threads) {
    __m256d va = _mm256_set1_pd(alpha);
    CpuTimer timer;
    timer.start();

    #ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    #endif
    for (int64_t i = 0; i < n - 3; i += 4) {
        __m256d vx = _mm256_loadu_pd(&x[i]);
        __m256d vy = _mm256_loadu_pd(&y[i]);
        vy = _mm256_fmadd_pd(va, vx, vy);
        _mm256_storeu_pd(&y[i], vy);
    }

    timer.stop();
    return timer.elapsed_ms();
}

static double run_avx2_axpy_fp32(float* __restrict__ y, const float* __restrict__ x,
                                  float alpha, int64_t n, int num_threads) {
    __m256 va = _mm256_set1_ps(alpha);
    CpuTimer timer;
    timer.start();

    #ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    #endif
    for (int64_t i = 0; i < n - 7; i += 8) {
        __m256 vx = _mm256_loadu_ps(&x[i]);
        __m256 vy = _mm256_loadu_ps(&y[i]);
        vy = _mm256_fmadd_ps(va, vx, vy);
        _mm256_storeu_ps(&y[i], vy);
    }

    timer.stop();
    return timer.elapsed_ms();
}

#endif // FLOPTIC_AXPY_AVX2

// ============================================================================
// AVX-512 AXPY — FP64 and FP32
// ============================================================================
#ifdef FLOPTIC_AXPY_AVX512

static double run_avx512_axpy_fp64(double* __restrict__ y, const double* __restrict__ x,
                                    double alpha, int64_t n, int num_threads) {
    __m512d va = _mm512_set1_pd(alpha);
    CpuTimer timer;
    timer.start();

    #ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    #endif
    for (int64_t i = 0; i < n - 7; i += 8) {
        __m512d vx = _mm512_loadu_pd(&x[i]);
        __m512d vy = _mm512_loadu_pd(&y[i]);
        vy = _mm512_fmadd_pd(va, vx, vy);
        _mm512_storeu_pd(&y[i], vy);
    }

    timer.stop();
    return timer.elapsed_ms();
}

static double run_avx512_axpy_fp32(float* __restrict__ y, const float* __restrict__ x,
                                    float alpha, int64_t n, int num_threads) {
    __m512 va = _mm512_set1_ps(alpha);
    CpuTimer timer;
    timer.start();

    #ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    #endif
    for (int64_t i = 0; i < n - 15; i += 16) {
        __m512 vx = _mm512_loadu_ps(&x[i]);
        __m512 vy = _mm512_loadu_ps(&y[i]);
        vy = _mm512_fmadd_ps(va, vx, vy);
        _mm512_storeu_ps(&y[i], vy);
    }

    timer.stop();
    return timer.elapsed_ms();
}

#endif // FLOPTIC_AXPY_AVX512

// ============================================================================
// Kernel class
// ============================================================================

class VectorAxpyCpu : public KernelBase {
public:
    std::string name() const override { return "vector_axpy"; }
    std::string category() const override { return "vector"; }
    std::string backend() const override { return "cpu"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP64, Precision::FP32};
    }

    std::vector<std::string> supported_modes() const override {
        return {"throughput"};
    }

    KernelResult run(const KernelConfig& config,
                     const DeviceInfo& device,
                     int measurement_trials) override {
        int num_threads = config.threads > 0 ? config.threads : device.compute_units;
        if (num_threads <= 0) num_threads = 1;

        // Problem size
        int64_t n = static_cast<int64_t>(config.iterations) * 100;
        if (n < 1000000) n = 1000000;
        if (n > 100000000) n = 100000000;

        int64_t flops_per_trial = n * 2;

        // Determine SIMD path
        std::string simd_path;
#ifdef FLOPTIC_AXPY_AVX512
        simd_path = "AVX-512";
#elif defined(FLOPTIC_AXPY_AVX2)
        simd_path = "AVX2";
#else
        simd_path = "scalar";
#endif

        size_t elem_bytes = (config.precision == Precision::FP64) ? 8 : 4;
        double bytes_per_trial = 3.0 * n * elem_bytes;

        std::cerr << "  Running vector_axpy [cpu/" << precision_to_string(config.precision)
                  << "/" << config.mode << "/" << simd_path
                  << "] n=" << n << " threads=" << num_threads << std::endl;

        // Allocate and initialize
        if (config.precision == Precision::FP64) {
            auto* x = static_cast<double*>(std::aligned_alloc(64, n * sizeof(double)));
            auto* y = static_cast<double*>(std::aligned_alloc(64, n * sizeof(double)));
            double alpha = 1.5;

            #ifdef FLOPTIC_HAS_OPENMP
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            #endif
            for (int64_t i = 0; i < n; i++) {
                x[i] = 1.0 + 1e-8 * i;
                y[i] = 2.0 - 1e-8 * i;
            }

            auto run_fn = [&]() -> double {
                // Reset y before each trial
                #ifdef FLOPTIC_HAS_OPENMP
                #pragma omp parallel for num_threads(num_threads) schedule(static)
                #endif
                for (int64_t i = 0; i < n; i++) y[i] = 2.0 - 1e-8 * i;

#ifdef FLOPTIC_AXPY_AVX512
                return run_avx512_axpy_fp64(y, x, alpha, n, num_threads);
#elif defined(FLOPTIC_AXPY_AVX2)
                return run_avx2_axpy_fp64(y, x, alpha, n, num_threads);
#else
                return run_axpy_scalar<double>(y, x, alpha, n, num_threads);
#endif
            };

            // Warmup
            for (int w = 0; w < 3; w++) run_fn();

            // Measurement
            std::vector<double> times;
            times.reserve(measurement_trials);
            for (int t = 0; t < measurement_trials; t++) {
                times.push_back(run_fn());
            }

            g_validation_sink = y[0];
            std::free(x);
            std::free(y);

            auto stats = TimingStats::compute(times);

            KernelResult result;
            result.median_time_ms = stats.median_ms;
            result.min_time_ms = stats.min_ms;
            result.max_time_ms = stats.max_ms;
            result.total_flops = flops_per_trial;
            result.gflops = (flops_per_trial / 1e9) / (stats.median_ms / 1e3);
            result.effective_gflops = result.gflops;

            double gbps = (bytes_per_trial / 1e9) / (stats.median_ms / 1e3);
            std::cerr << "  Bandwidth: " << gbps << " GB/s" << std::endl;

            std::string prec_key = precision_to_string(config.precision);
            auto it = device.theoretical_peak_gflops.find(prec_key);
            if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
                result.peak_percent = (result.gflops / it->second) * 100.0;
            }
            return result;

        } else {
            // FP32
            auto* x = static_cast<float*>(std::aligned_alloc(64, n * sizeof(float)));
            auto* y = static_cast<float*>(std::aligned_alloc(64, n * sizeof(float)));
            float alpha = 1.5f;

            #ifdef FLOPTIC_HAS_OPENMP
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            #endif
            for (int64_t i = 0; i < n; i++) {
                x[i] = 1.0f + 1e-6f * i;
                y[i] = 2.0f - 1e-6f * i;
            }

            auto run_fn = [&]() -> double {
                #ifdef FLOPTIC_HAS_OPENMP
                #pragma omp parallel for num_threads(num_threads) schedule(static)
                #endif
                for (int64_t i = 0; i < n; i++) y[i] = 2.0f - 1e-6f * i;

#ifdef FLOPTIC_AXPY_AVX512
                return run_avx512_axpy_fp32(y, x, alpha, n, num_threads);
#elif defined(FLOPTIC_AXPY_AVX2)
                return run_avx2_axpy_fp32(y, x, alpha, n, num_threads);
#else
                return run_axpy_scalar<float>(y, x, alpha, n, num_threads);
#endif
            };

            for (int w = 0; w < 3; w++) run_fn();

            std::vector<double> times;
            times.reserve(measurement_trials);
            for (int t = 0; t < measurement_trials; t++) {
                times.push_back(run_fn());
            }

            g_validation_sink = y[0];
            std::free(x);
            std::free(y);

            auto stats = TimingStats::compute(times);

            KernelResult result;
            result.median_time_ms = stats.median_ms;
            result.min_time_ms = stats.min_ms;
            result.max_time_ms = stats.max_ms;
            result.total_flops = flops_per_trial;
            result.gflops = (flops_per_trial / 1e9) / (stats.median_ms / 1e3);
            result.effective_gflops = result.gflops;

            double gbps = (bytes_per_trial / 1e9) / (stats.median_ms / 1e3);
            std::cerr << "  Bandwidth: " << gbps << " GB/s" << std::endl;

            std::string prec_key = precision_to_string(config.precision);
            auto it = device.theoretical_peak_gflops.find(prec_key);
            if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
                result.peak_percent = (result.gflops / it->second) * 100.0;
            }
            return result;
        }
    }
};

REGISTER_KERNEL(VectorAxpyCpu);

namespace force_link {
    void vector_axpy_cpu_link() {}
}

} // namespace floptic
