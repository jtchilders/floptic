#include "floptic/kernel_base.hpp"
#include "floptic/kernel_registry.hpp"
#include "floptic/timer.hpp"
#include <cmath>
#include <vector>
#include <iostream>

#ifdef FLOPTIC_HAS_OPENMP
#include <omp.h>
#endif

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#define FLOPTIC_HAS_AVX512_INTRINSICS
#define FLOPTIC_HAS_AVX2_INTRINSICS
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define FLOPTIC_HAS_AVX2_INTRINSICS
#endif

namespace floptic {

extern volatile double g_validation_sink;

// ============================================================================
// Scalar fallback (no SIMD) — always available
// ============================================================================

template <typename T>
static double run_scalar_throughput(int num_threads, int64_t iters) {
    volatile double sink = 0;
    CpuTimer timer;
    timer.start();

    #ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel num_threads(num_threads)
    #endif
    {
        T a = static_cast<T>(1.0000001);
        T b = static_cast<T>(0.9999999);
        T r0 = 1, r1 = 2, r2 = 3, r3 = 4;
        T r4 = 5, r5 = 6, r6 = 7, r7 = 8;

        for (int64_t i = 0; i < iters; i++) {
            r0 = std::fma(a, r0, b);
            r1 = std::fma(a, r1, b);
            r2 = std::fma(a, r2, b);
            r3 = std::fma(a, r3, b);
            r4 = std::fma(a, r4, b);
            r5 = std::fma(a, r5, b);
            r6 = std::fma(a, r6, b);
            r7 = std::fma(a, r7, b);
        }

        T total = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
        #ifdef FLOPTIC_HAS_OPENMP
        #pragma omp atomic
        #endif
        sink += static_cast<double>(total);
    }

    timer.stop();
    g_validation_sink = sink;
    return timer.elapsed_ms();
}

template <typename T>
static double run_scalar_latency(int num_threads, int64_t iters) {
    volatile double sink = 0;
    CpuTimer timer;
    timer.start();

    #ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel num_threads(num_threads)
    #endif
    {
        T a = static_cast<T>(1.0000001);
        T b = static_cast<T>(0.9999999);
        T r = static_cast<T>(1.0);

        for (int64_t i = 0; i < iters; i++) {
            r = std::fma(a, r, b);
        }

        #ifdef FLOPTIC_HAS_OPENMP
        #pragma omp atomic
        #endif
        sink += static_cast<double>(r);
    }

    timer.stop();
    g_validation_sink = sink;
    return timer.elapsed_ms();
}

// ============================================================================
// AVX2 intrinsics — FP64 (4 lanes × 256-bit)
// ============================================================================
#ifdef FLOPTIC_HAS_AVX2_INTRINSICS

// Throughput: 2 FMA units × 4 lanes × 8 register groups = 64 FP64 FMAs in flight
// We use 8 independent __m256d accumulators to saturate both FMA ports
static double run_avx2_fp64_throughput(int num_threads, int64_t iters) {
    volatile double sink = 0;
    CpuTimer timer;
    timer.start();

    #ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel num_threads(num_threads)
    #endif
    {
        __m256d a = _mm256_set1_pd(1.0000001);
        __m256d b = _mm256_set1_pd(0.9999999);
        // 8 independent accumulator registers
        __m256d r0 = _mm256_set1_pd(1.0);
        __m256d r1 = _mm256_set1_pd(2.0);
        __m256d r2 = _mm256_set1_pd(3.0);
        __m256d r3 = _mm256_set1_pd(4.0);
        __m256d r4 = _mm256_set1_pd(5.0);
        __m256d r5 = _mm256_set1_pd(6.0);
        __m256d r6 = _mm256_set1_pd(7.0);
        __m256d r7 = _mm256_set1_pd(8.0);

        for (int64_t i = 0; i < iters; i++) {
            r0 = _mm256_fmadd_pd(a, r0, b);
            r1 = _mm256_fmadd_pd(a, r1, b);
            r2 = _mm256_fmadd_pd(a, r2, b);
            r3 = _mm256_fmadd_pd(a, r3, b);
            r4 = _mm256_fmadd_pd(a, r4, b);
            r5 = _mm256_fmadd_pd(a, r5, b);
            r6 = _mm256_fmadd_pd(a, r6, b);
            r7 = _mm256_fmadd_pd(a, r7, b);
        }

        // Reduce to prevent DCE
        __m256d sum = _mm256_add_pd(
            _mm256_add_pd(_mm256_add_pd(r0, r1), _mm256_add_pd(r2, r3)),
            _mm256_add_pd(_mm256_add_pd(r4, r5), _mm256_add_pd(r6, r7)));

        // Horizontal sum of 4 doubles
        __m128d lo = _mm256_castpd256_pd128(sum);
        __m128d hi = _mm256_extractf128_pd(sum, 1);
        __m128d s = _mm_add_pd(lo, hi);
        double result = _mm_cvtsd_f64(s) + _mm_cvtsd_f64(_mm_unpackhi_pd(s, s));

        #ifdef FLOPTIC_HAS_OPENMP
        #pragma omp atomic
        #endif
        sink += result;
    }

    timer.stop();
    g_validation_sink = sink;
    return timer.elapsed_ms();
}

// Latency: single dependent chain of AVX2 FMAs
static double run_avx2_fp64_latency(int num_threads, int64_t iters) {
    volatile double sink = 0;
    CpuTimer timer;
    timer.start();

    #ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel num_threads(num_threads)
    #endif
    {
        __m256d a = _mm256_set1_pd(1.0000001);
        __m256d b = _mm256_set1_pd(0.9999999);
        __m256d r = _mm256_set1_pd(1.0);

        for (int64_t i = 0; i < iters; i++) {
            r = _mm256_fmadd_pd(a, r, b);
        }

        __m128d lo = _mm256_castpd256_pd128(r);
        double result = _mm_cvtsd_f64(lo);

        #ifdef FLOPTIC_HAS_OPENMP
        #pragma omp atomic
        #endif
        sink += result;
    }

    timer.stop();
    g_validation_sink = sink;
    return timer.elapsed_ms();
}

// ============================================================================
// AVX2 intrinsics — FP32 (8 lanes × 256-bit)
// ============================================================================

static double run_avx2_fp32_throughput(int num_threads, int64_t iters) {
    volatile double sink = 0;
    CpuTimer timer;
    timer.start();

    #ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel num_threads(num_threads)
    #endif
    {
        __m256 a = _mm256_set1_ps(1.0000001f);
        __m256 b = _mm256_set1_ps(0.9999999f);
        __m256 r0 = _mm256_set1_ps(1.0f);
        __m256 r1 = _mm256_set1_ps(2.0f);
        __m256 r2 = _mm256_set1_ps(3.0f);
        __m256 r3 = _mm256_set1_ps(4.0f);
        __m256 r4 = _mm256_set1_ps(5.0f);
        __m256 r5 = _mm256_set1_ps(6.0f);
        __m256 r6 = _mm256_set1_ps(7.0f);
        __m256 r7 = _mm256_set1_ps(8.0f);

        for (int64_t i = 0; i < iters; i++) {
            r0 = _mm256_fmadd_ps(a, r0, b);
            r1 = _mm256_fmadd_ps(a, r1, b);
            r2 = _mm256_fmadd_ps(a, r2, b);
            r3 = _mm256_fmadd_ps(a, r3, b);
            r4 = _mm256_fmadd_ps(a, r4, b);
            r5 = _mm256_fmadd_ps(a, r5, b);
            r6 = _mm256_fmadd_ps(a, r6, b);
            r7 = _mm256_fmadd_ps(a, r7, b);
        }

        // Reduce
        __m256 sum = _mm256_add_ps(
            _mm256_add_ps(_mm256_add_ps(r0, r1), _mm256_add_ps(r2, r3)),
            _mm256_add_ps(_mm256_add_ps(r4, r5), _mm256_add_ps(r6, r7)));

        // Horizontal sum of 8 floats
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        double result = static_cast<double>(_mm_cvtss_f32(s));

        #ifdef FLOPTIC_HAS_OPENMP
        #pragma omp atomic
        #endif
        sink += result;
    }

    timer.stop();
    g_validation_sink = sink;
    return timer.elapsed_ms();
}

static double run_avx2_fp32_latency(int num_threads, int64_t iters) {
    volatile double sink = 0;
    CpuTimer timer;
    timer.start();

    #ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel num_threads(num_threads)
    #endif
    {
        __m256 a = _mm256_set1_ps(1.0000001f);
        __m256 b = _mm256_set1_ps(0.9999999f);
        __m256 r = _mm256_set1_ps(1.0f);

        for (int64_t i = 0; i < iters; i++) {
            r = _mm256_fmadd_ps(a, r, b);
        }

        float result = _mm_cvtss_f32(_mm256_castps256_ps128(r));

        #ifdef FLOPTIC_HAS_OPENMP
        #pragma omp atomic
        #endif
        sink += static_cast<double>(result);
    }

    timer.stop();
    g_validation_sink = sink;
    return timer.elapsed_ms();
}

#endif // FLOPTIC_HAS_AVX2_INTRINSICS

// ============================================================================
// AVX-512 intrinsics — FP64 (8 lanes × 512-bit)
// ============================================================================
#ifdef FLOPTIC_HAS_AVX512_INTRINSICS

static double run_avx512_fp64_throughput(int num_threads, int64_t iters) {
    volatile double sink = 0;
    CpuTimer timer;
    timer.start();

    #ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel num_threads(num_threads)
    #endif
    {
        __m512d a = _mm512_set1_pd(1.0000001);
        __m512d b = _mm512_set1_pd(0.9999999);
        __m512d r0 = _mm512_set1_pd(1.0);
        __m512d r1 = _mm512_set1_pd(2.0);
        __m512d r2 = _mm512_set1_pd(3.0);
        __m512d r3 = _mm512_set1_pd(4.0);
        __m512d r4 = _mm512_set1_pd(5.0);
        __m512d r5 = _mm512_set1_pd(6.0);
        __m512d r6 = _mm512_set1_pd(7.0);
        __m512d r7 = _mm512_set1_pd(8.0);

        for (int64_t i = 0; i < iters; i++) {
            r0 = _mm512_fmadd_pd(a, r0, b);
            r1 = _mm512_fmadd_pd(a, r1, b);
            r2 = _mm512_fmadd_pd(a, r2, b);
            r3 = _mm512_fmadd_pd(a, r3, b);
            r4 = _mm512_fmadd_pd(a, r4, b);
            r5 = _mm512_fmadd_pd(a, r5, b);
            r6 = _mm512_fmadd_pd(a, r6, b);
            r7 = _mm512_fmadd_pd(a, r7, b);
        }

        __m512d sum = _mm512_add_pd(
            _mm512_add_pd(_mm512_add_pd(r0, r1), _mm512_add_pd(r2, r3)),
            _mm512_add_pd(_mm512_add_pd(r4, r5), _mm512_add_pd(r6, r7)));
        double result = _mm512_reduce_add_pd(sum);

        #ifdef FLOPTIC_HAS_OPENMP
        #pragma omp atomic
        #endif
        sink += result;
    }

    timer.stop();
    g_validation_sink = sink;
    return timer.elapsed_ms();
}

static double run_avx512_fp64_latency(int num_threads, int64_t iters) {
    volatile double sink = 0;
    CpuTimer timer;
    timer.start();

    #ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel num_threads(num_threads)
    #endif
    {
        __m512d a = _mm512_set1_pd(1.0000001);
        __m512d b = _mm512_set1_pd(0.9999999);
        __m512d r = _mm512_set1_pd(1.0);

        for (int64_t i = 0; i < iters; i++) {
            r = _mm512_fmadd_pd(a, r, b);
        }

        double result = _mm512_reduce_add_pd(r);

        #ifdef FLOPTIC_HAS_OPENMP
        #pragma omp atomic
        #endif
        sink += result;
    }

    timer.stop();
    g_validation_sink = sink;
    return timer.elapsed_ms();
}

static double run_avx512_fp32_throughput(int num_threads, int64_t iters) {
    volatile double sink = 0;
    CpuTimer timer;
    timer.start();

    #ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel num_threads(num_threads)
    #endif
    {
        __m512 a = _mm512_set1_ps(1.0000001f);
        __m512 b = _mm512_set1_ps(0.9999999f);
        __m512 r0 = _mm512_set1_ps(1.0f);
        __m512 r1 = _mm512_set1_ps(2.0f);
        __m512 r2 = _mm512_set1_ps(3.0f);
        __m512 r3 = _mm512_set1_ps(4.0f);
        __m512 r4 = _mm512_set1_ps(5.0f);
        __m512 r5 = _mm512_set1_ps(6.0f);
        __m512 r6 = _mm512_set1_ps(7.0f);
        __m512 r7 = _mm512_set1_ps(8.0f);

        for (int64_t i = 0; i < iters; i++) {
            r0 = _mm512_fmadd_ps(a, r0, b);
            r1 = _mm512_fmadd_ps(a, r1, b);
            r2 = _mm512_fmadd_ps(a, r2, b);
            r3 = _mm512_fmadd_ps(a, r3, b);
            r4 = _mm512_fmadd_ps(a, r4, b);
            r5 = _mm512_fmadd_ps(a, r5, b);
            r6 = _mm512_fmadd_ps(a, r6, b);
            r7 = _mm512_fmadd_ps(a, r7, b);
        }

        __m512 sum = _mm512_add_ps(
            _mm512_add_ps(_mm512_add_ps(r0, r1), _mm512_add_ps(r2, r3)),
            _mm512_add_ps(_mm512_add_ps(r4, r5), _mm512_add_ps(r6, r7)));
        double result = static_cast<double>(_mm512_reduce_add_ps(sum));

        #ifdef FLOPTIC_HAS_OPENMP
        #pragma omp atomic
        #endif
        sink += result;
    }

    timer.stop();
    g_validation_sink = sink;
    return timer.elapsed_ms();
}

static double run_avx512_fp32_latency(int num_threads, int64_t iters) {
    volatile double sink = 0;
    CpuTimer timer;
    timer.start();

    #ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel num_threads(num_threads)
    #endif
    {
        __m512 a = _mm512_set1_ps(1.0000001f);
        __m512 b = _mm512_set1_ps(0.9999999f);
        __m512 r = _mm512_set1_ps(1.0f);

        for (int64_t i = 0; i < iters; i++) {
            r = _mm512_fmadd_ps(a, r, b);
        }

        double result = static_cast<double>(_mm512_reduce_add_ps(r));

        #ifdef FLOPTIC_HAS_OPENMP
        #pragma omp atomic
        #endif
        sink += result;
    }

    timer.stop();
    g_validation_sink = sink;
    return timer.elapsed_ms();
}

#endif // FLOPTIC_HAS_AVX512_INTRINSICS

// ============================================================================
// Kernel class — dispatches to best available SIMD path
// ============================================================================

class ScalarFmaCpu : public KernelBase {
public:
    std::string name() const override { return "scalar_fma"; }
    std::string category() const override { return "scalar"; }
    std::string backend() const override { return "cpu"; }

    std::vector<Precision> supported_precisions() const override {
        return {Precision::FP64, Precision::FP32};
    }

    std::vector<std::string> supported_modes() const override {
        return {"throughput", "latency"};
    }

    KernelResult run(const KernelConfig& config,
                     const DeviceInfo& device,
                     int measurement_trials) override {
        int num_threads = config.threads > 0 ? config.threads : device.compute_units;
        if (num_threads <= 0) num_threads = 1;

        int64_t iters = config.iterations;

        // Determine SIMD path and FLOP count
        std::string simd_path;
        int64_t flops_per_trial;

        // SIMD lanes per register and chains determine total FLOPs
        // throughput: 8 independent chains × lanes × 2 FLOPs/FMA × iters × threads
        // latency:    1 chain × lanes × 2 FLOPs/FMA × iters × threads
        int lanes;
        int chains = 8;  // throughput uses 8 independent accumulators

#ifdef FLOPTIC_HAS_AVX512_INTRINSICS
        if (config.precision == Precision::FP64) {
            lanes = 8;   // 512-bit / 64-bit
            simd_path = "AVX-512";
        } else {
            lanes = 16;  // 512-bit / 32-bit
            simd_path = "AVX-512";
        }
#elif defined(FLOPTIC_HAS_AVX2_INTRINSICS)
        if (config.precision == Precision::FP64) {
            lanes = 4;   // 256-bit / 64-bit
            simd_path = "AVX2";
        } else {
            lanes = 8;   // 256-bit / 32-bit
            simd_path = "AVX2";
        }
#else
        lanes = 1;
        simd_path = "scalar";
#endif

        if (config.mode == "latency") {
            flops_per_trial = static_cast<int64_t>(num_threads) * lanes * iters * 2;
        } else {
            flops_per_trial = static_cast<int64_t>(num_threads) * chains * lanes * iters * 2;
        }

        std::cerr << "  Running scalar_fma [cpu/" << precision_to_string(config.precision)
                  << "/" << config.mode << "/" << simd_path
                  << "] threads=" << num_threads
                  << " iters=" << iters
                  << " lanes=" << lanes << std::endl;

        // Select run function
        auto run_fn = [&](void) -> double {
            bool is_fp32 = (config.precision == Precision::FP32);
            bool is_latency = (config.mode == "latency");

#ifdef FLOPTIC_HAS_AVX512_INTRINSICS
            if (is_fp32) {
                return is_latency ? run_avx512_fp32_latency(num_threads, iters)
                                  : run_avx512_fp32_throughput(num_threads, iters);
            } else {
                return is_latency ? run_avx512_fp64_latency(num_threads, iters)
                                  : run_avx512_fp64_throughput(num_threads, iters);
            }
#elif defined(FLOPTIC_HAS_AVX2_INTRINSICS)
            if (is_fp32) {
                return is_latency ? run_avx2_fp32_latency(num_threads, iters)
                                  : run_avx2_fp32_throughput(num_threads, iters);
            } else {
                return is_latency ? run_avx2_fp64_latency(num_threads, iters)
                                  : run_avx2_fp64_throughput(num_threads, iters);
            }
#else
            if (is_fp32) {
                return is_latency ? run_scalar_latency<float>(num_threads, iters)
                                  : run_scalar_throughput<float>(num_threads, iters);
            } else {
                return is_latency ? run_scalar_latency<double>(num_threads, iters)
                                  : run_scalar_throughput<double>(num_threads, iters);
            }
#endif
        };

        // Warmup
        for (int w = 0; w < 3; w++) {
            run_fn();
        }

        // Measurement
        std::vector<double> times;
        times.reserve(measurement_trials);
        for (int t = 0; t < measurement_trials; t++) {
            times.push_back(run_fn());
        }

        auto stats = TimingStats::compute(times);

        KernelResult result;
        result.median_time_ms = stats.median_ms;
        result.min_time_ms = stats.min_ms;
        result.max_time_ms = stats.max_ms;
        result.total_flops = flops_per_trial;
        result.gflops = (flops_per_trial / 1e9) / (stats.median_ms / 1e3);
        result.effective_gflops = result.gflops;

        // Peak percent
        std::string prec_key = precision_to_string(config.precision);
        auto it = device.theoretical_peak_gflops.find(prec_key);
        if (it != device.theoretical_peak_gflops.end() && it->second > 0) {
            result.peak_percent = (result.gflops / it->second) * 100.0;
        }

        return result;
    }
};

REGISTER_KERNEL(ScalarFmaCpu);

namespace force_link {
    void scalar_fma_cpu_link() {}
}

} // namespace floptic
