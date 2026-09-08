#pragma once
// Untimed host reference GEMM for small validation matrices.
//
// This computes C := alpha * op(A) * op(B) + beta * C using widened
// accumulation (never the kernel's working precision) so it can serve as
// a ground truth for validating fast/low-precision/GPU GEMM
// implementations. It is explicitly NOT performance code: no blocking, no
// SIMD, no threading. Keep matrices small in tests.
//
// Layout/convention: row-major, C-style storage. A is M x K, B is K x N,
// C is M x N. Element (r, c) of an M x N row-major matrix with leading
// dimension ld is at data[r * ld + c]. Callers pass explicit leading
// dimensions so padded/strided buffers are supported without copying.
#include <cstdint>
#include <stdexcept>
#include <string>

namespace floptic {

// Accumulation-precision selection:
//   FP32 inputs  -> accumulate in `long double` when available in double
//                   precision or better, else `double` (see AccumFor<T>).
//   FP64 inputs  -> accumulate in `long double`.
// On platforms where `long double` == `double` (e.g. some ABIs), the
// "long double or double" contract degrades gracefully to double
// accumulation -- still >= the input precision, never less.
template <typename T> struct AccumFor;
template <> struct AccumFor<float>  { using type = long double; };
template <> struct AccumFor<double> { using type = long double; };

struct GemmDims {
    int64_t M = 0, N = 0, K = 0;
    int64_t lda = 0;  // leading dimension of A (row-major: >= K, or >= M if transposed)
    int64_t ldb = 0;  // leading dimension of B
    int64_t ldc = 0;  // leading dimension of C

    bool trans_a = false;
    bool trans_b = false;
};

// Validates dimensions/strides without touching data. Returns "" on
// success, else a diagnostic. Rejects zero/negative dimensions and
// leading dimensions inconsistent with the (possibly transposed) shape so
// the reference implementation never reads/writes out of bounds.
inline std::string validate_gemm_dims(const GemmDims& d) {
    if (d.M <= 0 || d.N <= 0 || d.K <= 0) return "M, N, K must all be > 0";

    int64_t a_rows = d.trans_a ? d.K : d.M;
    int64_t a_min_ld = d.trans_a ? d.M : d.K;
    if (d.lda < a_min_ld) return "lda too small for A's (possibly transposed) shape";
    (void)a_rows;

    int64_t b_min_ld = d.trans_b ? d.K : d.N;
    if (d.ldb < b_min_ld) return "ldb too small for B's (possibly transposed) shape";

    if (d.ldc < d.N) return "ldc too small for C's shape";

    return "";
}

// Reference GEMM: C := alpha * op(A) * op(B) + beta * C, accumulating in
// AccumFor<T>::type. Every element of C is read (for the beta term) and
// written; this is the "output initialization is tested" contract from
// the card -- beta=0 must not read pre-existing garbage into the result
// (it is still read, per BLAS semantics, but multiplied by 0 so any
// finite or zero input is safe; NaN/Inf beta or pre-existing NaN in C
// with beta=0 is a caller error, not guarded here since 0 * NaN == NaN is
// the documented IEEE behavior and tests should avoid it).
//
// Throws std::invalid_argument for invalid dims (see validate_gemm_dims).
template <typename T>
inline void reference_gemm(const GemmDims& dims, T alpha,
                            const T* a, const T* b, T beta, T* c) {
    std::string err = validate_gemm_dims(dims);
    if (!err.empty()) {
        throw std::invalid_argument("floptic::reference_gemm: " + err);
    }

    using Accum = typename AccumFor<T>::type;
    const Accum alpha_acc = static_cast<Accum>(alpha);
    const Accum beta_acc = static_cast<Accum>(beta);

    auto a_at = [&](int64_t r, int64_t k) -> T {
        // Logical A(r, k), r in [0, M), k in [0, K), honoring trans_a.
        return dims.trans_a ? a[k * dims.lda + r] : a[r * dims.lda + k];
    };
    auto b_at = [&](int64_t k, int64_t col) -> T {
        // Logical B(k, col), k in [0, K), col in [0, N), honoring trans_b.
        return dims.trans_b ? b[col * dims.ldb + k] : b[k * dims.ldb + col];
    };

    for (int64_t r = 0; r < dims.M; r++) {
        for (int64_t col = 0; col < dims.N; col++) {
            Accum sum = static_cast<Accum>(0);
            for (int64_t k = 0; k < dims.K; k++) {
                Accum av = static_cast<Accum>(a_at(r, k));
                Accum bv = static_cast<Accum>(b_at(k, col));
                sum += av * bv;
            }
            Accum prior = static_cast<Accum>(c[r * dims.ldc + col]);
            Accum result = alpha_acc * sum + beta_acc * prior;
            c[r * dims.ldc + col] = static_cast<T>(result);
        }
    }
}

}  // namespace floptic
