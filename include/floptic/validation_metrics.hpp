#pragma once
// Validation metrics and pass/fail policy primitives.
//
// This header defines:
//   - ValidationResult: the fixed set of scalar diagnostics every
//     validation must report (element count, finiteness mismatches, error
//     norms, pass/fail + reason).
//   - compute_validation_metrics<T>(): computes a ValidationResult from an
//     actual/expected pair of buffers.
//   - ValidationPolicy + named_policy(): precision- and
//     accumulation-depth-aware tolerance policies. These are INITIAL
//     ACCEPTANCE POLICIES meant to catch gross correctness regressions
//     (garbage data, wrong reduction, wrong precision cast). They are not
//     a substitute for vendor-specific numerical characterization and must
//     be revisited once real backend error distributions are measured.
//   - Integer exactness + overflow-bound checking via widened arithmetic.
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "floptic/precision.hpp"

namespace floptic {

struct ValidationResult {
    int64_t element_count = 0;
    int64_t finite_mismatch_count = 0;  // exactly one of {actual, expected} is non-finite
    int64_t nonfinite_count = 0;        // either side non-finite (superset of mismatches)
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;         // see near-zero denominator rule below
    double frobenius_rel_error = 0.0;   // ||actual - expected||_F / ||expected||_F
    double rmse = 0.0;
    bool passed = false;
    std::string diagnostic;
};

// Near-zero denominator rule for relative error: when |expected| is below
// `floor`, the denominator used is `floor` itself instead of |expected|,
// preventing division-by-near-zero from producing a spuriously enormous
// (or infinite) relative error for elements that are legitimately supposed
// to be ~0. `floor` is policy-specific (see ValidationPolicy::near_zero_floor)
// because "near zero" means something different at FP8 scale than at FP64
// scale.
inline double relative_error_with_floor(double abs_error, double expected_abs, double floor) {
    double denom = expected_abs > floor ? expected_abs : floor;
    return abs_error / denom;
}

// Computes ValidationResult for actual vs. expected, both length `count`.
// near_zero_floor feeds the denominator rule above. This function never
// itself decides pass/fail -- see apply_policy() -- so the same metrics
// can be judged under different policies without recomputing.
template <typename T>
inline ValidationResult compute_validation_metrics(const T* actual, const T* expected,
                                                     int64_t count, double near_zero_floor) {
    ValidationResult r;
    r.element_count = count;

    double sum_sq_error = 0.0;
    double sum_sq_expected = 0.0;

    for (int64_t i = 0; i < count; i++) {
        double a = static_cast<double>(actual[i]);
        double e = static_cast<double>(expected[i]);

        bool a_finite = std::isfinite(a);
        bool e_finite = std::isfinite(e);

        if (!a_finite || !e_finite) {
            r.nonfinite_count++;
            if (a_finite != e_finite) {
                r.finite_mismatch_count++;
            }
            // Non-finite entries do not contribute finite error statistics;
            // they are visible via finite_mismatch_count / nonfinite_count
            // and, per the acceptance criteria, must fail visibly (handled
            // in apply_policy()).
            continue;
        }

        double abs_error = std::fabs(a - e);
        double rel_error = relative_error_with_floor(abs_error, std::fabs(e), near_zero_floor);

        if (abs_error > r.max_abs_error) r.max_abs_error = abs_error;
        if (rel_error > r.max_rel_error) r.max_rel_error = rel_error;

        sum_sq_error += abs_error * abs_error;
        sum_sq_expected += e * e;
    }

    int64_t finite_count = count - r.nonfinite_count;
    r.rmse = finite_count > 0 ? std::sqrt(sum_sq_error / static_cast<double>(finite_count)) : 0.0;
    r.frobenius_rel_error =
        sum_sq_expected > 0.0 ? std::sqrt(sum_sq_error) / std::sqrt(sum_sq_expected) : 0.0;

    return r;
}

// A named, precision-specific pass/fail policy. These are deliberately NOT
// one universal epsilon: tolerance scales with the accumulation depth K
// (random rounding error grows roughly with sqrt(K) for a K-term
// reduction) and with the working precision's unit round-off.
struct ValidationPolicy {
    std::string name;
    double max_abs_error = std::numeric_limits<double>::infinity();  // infinity = not checked
    double max_rel_error = std::numeric_limits<double>::infinity();
    double max_frobenius_rel_error = std::numeric_limits<double>::infinity();
    double near_zero_floor = 1e-30;
    bool require_exact = false;  // integer policies: bitwise/value equality, no tolerance

    // Applies this policy's thresholds to an already-computed
    // ValidationResult, filling in passed + diagnostic. Any non-finite
    // mismatch always fails, regardless of policy, per the "failures must
    // remain visible" requirement.
    ValidationResult apply(ValidationResult r) const {
        if (r.finite_mismatch_count > 0) {
            r.passed = false;
            r.diagnostic = "finite/non-finite mismatch in " +
                            std::to_string(r.finite_mismatch_count) + " element(s)";
            return r;
        }
        if (require_exact) {
            r.passed = (r.max_abs_error == 0.0);
            r.diagnostic = r.passed ? "exact match" : "exact-match policy violated: max_abs_error > 0";
            return r;
        }
        if (r.max_abs_error > max_abs_error) {
            r.passed = false;
            r.diagnostic = "max_abs_error " + std::to_string(r.max_abs_error) +
                            " exceeds policy limit " + std::to_string(max_abs_error);
            return r;
        }
        if (r.max_rel_error > max_rel_error) {
            r.passed = false;
            r.diagnostic = "max_rel_error " + std::to_string(r.max_rel_error) +
                            " exceeds policy limit " + std::to_string(max_rel_error);
            return r;
        }
        if (r.frobenius_rel_error > max_frobenius_rel_error) {
            r.passed = false;
            r.diagnostic = "frobenius_rel_error " + std::to_string(r.frobenius_rel_error) +
                            " exceeds policy limit " + std::to_string(max_frobenius_rel_error);
            return r;
        }
        r.passed = true;
        r.diagnostic = "within tolerance";
        return r;
    }
};

namespace detail {

// Unit round-off (machine epsilon) for a p-bit-precision floating format
// (p = explicit significand bits + 1 implicit bit), 2^-p.
constexpr double unit_roundoff(int precision_bits) {
    double v = 1.0;
    for (int i = 0; i < precision_bits; i++) v *= 0.5;
    return v;
}

}  // namespace detail

// Precision-and-depth-aware policy factory. `k_depth` is the accumulation
// depth (e.g. GEMM K dimension, or reduction length) driving the sqrt(K)
// error-growth scaling; pass 1 for non-reduction contexts.
//
// The multiplicative constant (10.0) is an explicit, documented initial
// acceptance margin -- generous enough to avoid false failures from
// legitimate rounding-order differences between this reference
// implementation and a backend's, tight enough to catch wrong-precision or
// garbage-data regressions. It is NOT a vendor accuracy claim.
inline ValidationPolicy named_policy(Precision precision, int64_t k_depth) {
    if (k_depth < 1) k_depth = 1;
    const double sqrt_k = std::sqrt(static_cast<double>(k_depth));
    constexpr double kMargin = 10.0;

    ValidationPolicy p;
    p.near_zero_floor = 1e-30;

    switch (precision) {
        case Precision::FP64: {
            double eps = detail::unit_roundoff(53);  // 2^-53
            p.name = "fp64_initial";
            p.max_rel_error = kMargin * eps * sqrt_k;
            p.max_frobenius_rel_error = p.max_rel_error;
            p.near_zero_floor = 1e-300;
            break;
        }
        case Precision::FP32: {
            double eps = detail::unit_roundoff(24);  // 2^-24
            p.name = "fp32_initial";
            p.max_rel_error = kMargin * eps * sqrt_k;
            p.max_frobenius_rel_error = p.max_rel_error;
            p.near_zero_floor = 1e-30;
            break;
        }
        case Precision::TF32: {
            double eps = detail::unit_roundoff(11);  // 10 explicit mantissa bits
            p.name = "tf32_initial";
            p.max_rel_error = kMargin * eps * sqrt_k;
            p.max_frobenius_rel_error = p.max_rel_error;
            p.near_zero_floor = 1e-12;
            break;
        }
        case Precision::FP16: {
            double eps = detail::unit_roundoff(11);  // 10 explicit mantissa bits
            p.name = "fp16_initial";
            p.max_rel_error = kMargin * eps * sqrt_k;
            p.max_frobenius_rel_error = p.max_rel_error;
            p.near_zero_floor = 1e-8;
            break;
        }
        case Precision::BF16: {
            double eps = detail::unit_roundoff(8);  // 7 explicit mantissa bits
            p.name = "bf16_initial";
            p.max_rel_error = kMargin * eps * sqrt_k;
            p.max_frobenius_rel_error = p.max_rel_error;
            p.near_zero_floor = 1e-6;
            break;
        }
        case Precision::FP8_E4M3: {
            double eps = detail::unit_roundoff(4);  // 3 explicit mantissa bits
            p.name = "fp8_e4m3_initial";
            p.max_rel_error = kMargin * eps * sqrt_k;
            p.max_frobenius_rel_error = p.max_rel_error;
            p.near_zero_floor = 1e-3;
            break;
        }
        case Precision::FP8_E5M2: {
            double eps = detail::unit_roundoff(3);  // 2 explicit mantissa bits
            p.name = "fp8_e5m2_initial";
            p.max_rel_error = kMargin * eps * sqrt_k;
            p.max_frobenius_rel_error = p.max_rel_error;
            p.near_zero_floor = 1e-3;
            break;
        }
        case Precision::FP4: {
            double eps = detail::unit_roundoff(2);  // 1 explicit mantissa bit
            p.name = "fp4_initial";
            p.max_rel_error = kMargin * eps * sqrt_k;
            p.max_frobenius_rel_error = p.max_rel_error;
            p.near_zero_floor = 1e-2;
            break;
        }
        case Precision::INT8:
        case Precision::INT4: {
            p.name = (precision == Precision::INT8) ? "int8_exact" : "int4_exact";
            p.require_exact = true;
            break;
        }
    }
    return p;
}

// ---------------------------------------------------------------------
// Integer exactness / overflow-bound checking.
// ---------------------------------------------------------------------

struct IntegerOverflowCheck {
    long double worst_case_magnitude = 0.0L;  // k * max|a| * max|b|, widened
    long double accumulator_max = 0.0L;       // max representable magnitude in AccumT
    bool overflow_possible = false;
};

// Computes whether a K-term dot product of values bounded by
// [max_abs_operand_a, max_abs_operand_b] could overflow the given signed
// accumulator type, using `long double` (widened relative to any of the
// realistic accumulator types this project uses) so the bound check itself
// cannot silently overflow.
template <typename AccumT>
inline IntegerOverflowCheck check_integer_overflow_bound(int64_t k,
                                                          int64_t max_abs_operand_a,
                                                          int64_t max_abs_operand_b) {
    IntegerOverflowCheck result;
    result.worst_case_magnitude = static_cast<long double>(k) *
                                   static_cast<long double>(max_abs_operand_a) *
                                   static_cast<long double>(max_abs_operand_b);
    result.accumulator_max = static_cast<long double>(std::numeric_limits<AccumT>::max());
    result.overflow_possible = result.worst_case_magnitude > result.accumulator_max;
    return result;
}

// Exact-equality check for integer buffers (INT8/INT4 policy): every
// element must match exactly. Returns a ValidationResult with
// require_exact semantics already applied.
template <typename T>
inline ValidationResult validate_integer_exact(const T* actual, const T* expected,
                                                 int64_t count) {
    ValidationResult r;
    r.element_count = count;
    for (int64_t i = 0; i < count; i++) {
        if (actual[i] != expected[i]) {
            double diff = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
            double abs_diff = std::fabs(diff);
            if (abs_diff > r.max_abs_error) r.max_abs_error = abs_diff;
        }
    }
    r.passed = (r.max_abs_error == 0.0);
    r.diagnostic = r.passed ? "exact match" : "integer exactness violated";
    return r;
}

}  // namespace floptic
