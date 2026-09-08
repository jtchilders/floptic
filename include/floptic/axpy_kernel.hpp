#pragma once
// AVX2/AVX-512/scalar AXPY (y = alpha*x + y) compute kernels, extracted from
// the vector_axpy CPU kernel class so they are:
//   1. Reusable — the kernel class and the correctness unit tests call the
//      exact same code, so a passing test really covers what runs at
//      benchmark time.
//   2. Independently testable for arbitrary n (including n smaller than a
//      SIMD width, and non-multiples of the width) without any device,
//      timer, or OpenMP dependency.
//
// Tail-correctness contract: every axpy_* function below updates all n
// elements of y for any n >= 0. Each vectorized path processes the largest
// prefix that is a multiple of its SIMD width with the vector loop, then
// hands the remaining `n % width` elements to a single serial scalar
// cleanup pass. The cleanup pass runs outside (after) any OpenMP-parallel
// vector loop, so there is no data race and no element is ever processed
// twice.
#include <cstdint>
#include <cmath>

#ifdef FLOPTIC_HAS_OPENMP
#include <omp.h>
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#define FLOPTIC_AXPY_HAS_AVX512
#define FLOPTIC_AXPY_HAS_AVX2
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define FLOPTIC_AXPY_HAS_AVX2
#endif

namespace floptic {

// ---------------------------------------------------------------------
// Scalar reference range: y[i] = fma(alpha, x[i], y[i]) for i in [begin, end).
// Used both as the no-SIMD fallback (whole range) and as the tail cleanup
// after a vectorized main loop (remainder range only).
// ---------------------------------------------------------------------
template <typename T>
inline void axpy_scalar_range(T* __restrict__ y, const T* __restrict__ x,
                               T alpha, int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
        y[i] = std::fma(alpha, x[i], y[i]);
    }
}

// Scalar fallback for the entire array, parallelized over num_threads when
// OpenMP is compiled in. Every element in [0, n) is processed exactly once.
template <typename T>
inline void axpy_scalar_parallel(T* __restrict__ y, const T* __restrict__ x,
                                  T alpha, int64_t n, int num_threads) {
#ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = std::fma(alpha, x[i], y[i]);
    }
}

#ifdef FLOPTIC_AXPY_HAS_AVX2

inline void axpy_avx2_fp64(double* __restrict__ y, const double* __restrict__ x,
                            double alpha, int64_t n, int num_threads) {
    constexpr int64_t width = 4;
    int64_t vec_n = n - (n % width);   // largest multiple of width <= n
    __m256d va = _mm256_set1_pd(alpha);

#ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < vec_n; i += width) {
        __m256d vx = _mm256_loadu_pd(&x[i]);
        __m256d vy = _mm256_loadu_pd(&y[i]);
        vy = _mm256_fmadd_pd(va, vx, vy);
        _mm256_storeu_pd(&y[i], vy);
    }
    // Single serial cleanup for the remainder — runs after the parallel
    // region above, so it neither races with it nor reprocesses any index.
    axpy_scalar_range(y, x, alpha, vec_n, n);
}

inline void axpy_avx2_fp32(float* __restrict__ y, const float* __restrict__ x,
                            float alpha, int64_t n, int num_threads) {
    constexpr int64_t width = 8;
    int64_t vec_n = n - (n % width);
    __m256 va = _mm256_set1_ps(alpha);

#ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < vec_n; i += width) {
        __m256 vx = _mm256_loadu_ps(&x[i]);
        __m256 vy = _mm256_loadu_ps(&y[i]);
        vy = _mm256_fmadd_ps(va, vx, vy);
        _mm256_storeu_ps(&y[i], vy);
    }
    axpy_scalar_range(y, x, alpha, vec_n, n);
}

#endif // FLOPTIC_AXPY_HAS_AVX2

#ifdef FLOPTIC_AXPY_HAS_AVX512

inline void axpy_avx512_fp64(double* __restrict__ y, const double* __restrict__ x,
                              double alpha, int64_t n, int num_threads) {
    constexpr int64_t width = 8;
    int64_t vec_n = n - (n % width);
    __m512d va = _mm512_set1_pd(alpha);

#ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < vec_n; i += width) {
        __m512d vx = _mm512_loadu_pd(&x[i]);
        __m512d vy = _mm512_loadu_pd(&y[i]);
        vy = _mm512_fmadd_pd(va, vx, vy);
        _mm512_storeu_pd(&y[i], vy);
    }
    axpy_scalar_range(y, x, alpha, vec_n, n);
}

inline void axpy_avx512_fp32(float* __restrict__ y, const float* __restrict__ x,
                              float alpha, int64_t n, int num_threads) {
    constexpr int64_t width = 16;
    int64_t vec_n = n - (n % width);
    __m512 va = _mm512_set1_ps(alpha);

#ifdef FLOPTIC_HAS_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
#endif
    for (int64_t i = 0; i < vec_n; i += width) {
        __m512 vx = _mm512_loadu_ps(&x[i]);
        __m512 vy = _mm512_loadu_ps(&y[i]);
        vy = _mm512_fmadd_ps(va, vx, vy);
        _mm512_storeu_ps(&y[i], vy);
    }
    axpy_scalar_range(y, x, alpha, vec_n, n);
}

#endif // FLOPTIC_AXPY_HAS_AVX512

// Name of the SIMD path that axpy_run() below will dispatch to, for
// logging/reporting purposes.
inline const char* axpy_active_simd_path() {
#if defined(FLOPTIC_AXPY_HAS_AVX512)
    return "AVX-512";
#elif defined(FLOPTIC_AXPY_HAS_AVX2)
    return "AVX2";
#else
    return "scalar";
#endif
}

// Dispatches to the best compiled-in path for the given precision. This is
// the single source of truth used by both the CPU AXPY kernel and its unit
// tests, so a test that exercises axpy_run() exercises exactly what the
// benchmark executes.
inline void axpy_run(double* __restrict__ y, const double* __restrict__ x,
                      double alpha, int64_t n, int num_threads) {
#if defined(FLOPTIC_AXPY_HAS_AVX512)
    axpy_avx512_fp64(y, x, alpha, n, num_threads);
#elif defined(FLOPTIC_AXPY_HAS_AVX2)
    axpy_avx2_fp64(y, x, alpha, n, num_threads);
#else
    axpy_scalar_parallel<double>(y, x, alpha, n, num_threads);
#endif
}

inline void axpy_run(float* __restrict__ y, const float* __restrict__ x,
                      float alpha, int64_t n, int num_threads) {
#if defined(FLOPTIC_AXPY_HAS_AVX512)
    axpy_avx512_fp32(y, x, alpha, n, num_threads);
#elif defined(FLOPTIC_AXPY_HAS_AVX2)
    axpy_avx2_fp32(y, x, alpha, n, num_threads);
#else
    axpy_scalar_parallel<float>(y, x, alpha, n, num_threads);
#endif
}

} // namespace floptic
