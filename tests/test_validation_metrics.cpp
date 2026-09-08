// Validation-metrics and policy tests: exact match, controlled
// perturbations, NaN/Inf visibility, near-zero relative-error stability,
// policy ordering, integer exactness + overflow-bound checks.
#include "floptic/validation_metrics.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using floptic::check_integer_overflow_bound;
using floptic::compute_validation_metrics;
using floptic::named_policy;
using floptic::Precision;
using floptic::validate_integer_exact;
using floptic::ValidationPolicy;
using floptic::ValidationResult;

namespace {

void test_exact_match_passes_with_zero_errors() {
    std::vector<double> a = {1.0, 2.0, 3.0, -4.5};
    std::vector<double> b = {1.0, 2.0, 3.0, -4.5};

    ValidationResult r = compute_validation_metrics(a.data(), b.data(),
                                                      static_cast<int64_t>(a.size()), 1e-30);
    CHECK_EQ(r.max_abs_error, 0.0);
    CHECK_EQ(r.max_rel_error, 0.0);
    CHECK_EQ(r.frobenius_rel_error, 0.0);
    CHECK_EQ(r.rmse, 0.0);
    CHECK_EQ(r.finite_mismatch_count, static_cast<int64_t>(0));

    ValidationPolicy policy = named_policy(Precision::FP64, /*k_depth=*/1);
    ValidationResult judged = policy.apply(r);
    CHECK_TRUE(judged.passed);
}

void test_controlled_perturbation_produces_expected_abs_and_rel_error() {
    std::vector<double> expected = {10.0, 100.0};
    std::vector<double> actual = {10.1, 100.0};  // abs error 0.1 on element 0

    ValidationResult r = compute_validation_metrics(actual.data(), expected.data(), 2, 1e-30);
    CHECK_TRUE(std::fabs(r.max_abs_error - 0.1) < 1e-9);
    CHECK_TRUE(std::fabs(r.max_rel_error - (0.1 / 10.0)) < 1e-9);
    CHECK_TRUE(r.rmse > 0.0);
    CHECK_TRUE(r.frobenius_rel_error > 0.0);
}

void test_nan_inf_mismatches_fail_visibly_even_with_zero_finite_error() {
    std::vector<double> expected = {1.0, 2.0, 3.0};
    std::vector<double> actual = {1.0, 2.0, std::numeric_limits<double>::quiet_NaN()};

    ValidationResult r = compute_validation_metrics(actual.data(), expected.data(), 3, 1e-30);
    CHECK_EQ(r.finite_mismatch_count, static_cast<int64_t>(1));

    ValidationPolicy policy = named_policy(Precision::FP64, 1);
    // Even a maximally permissive hand-built policy must still fail on a
    // finite/non-finite mismatch -- apply() checks that before any
    // tolerance comparison.
    policy.max_abs_error = std::numeric_limits<double>::infinity();
    policy.max_rel_error = std::numeric_limits<double>::infinity();
    policy.max_frobenius_rel_error = std::numeric_limits<double>::infinity();
    ValidationResult judged = policy.apply(r);
    CHECK_TRUE(!judged.passed);
}

void test_both_sides_nonfinite_is_not_counted_as_mismatch_but_is_nonfinite() {
    double nan_v = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> expected = {nan_v};
    std::vector<double> actual = {nan_v};

    ValidationResult r = compute_validation_metrics(actual.data(), expected.data(), 1, 1e-30);
    CHECK_EQ(r.finite_mismatch_count, static_cast<int64_t>(0));
    CHECK_EQ(r.nonfinite_count, static_cast<int64_t>(1));
}

void test_near_zero_relative_error_uses_floor_not_raw_denominator() {
    // expected ~ 0, actual has a small absolute error: without a floor,
    // relative error would explode toward infinity for a legitimately
    // near-zero target.
    double expected_val = 1e-20;
    double actual_val = 1e-20 + 1e-10;
    double abs_error = std::fabs(actual_val - expected_val);

    double floor = 1e-6;
    double rel_with_floor = floptic::relative_error_with_floor(abs_error, std::fabs(expected_val), floor);
    // Denominator should be the floor (1e-6), not |expected| (1e-20).
    CHECK_TRUE(std::fabs(rel_with_floor - (abs_error / floor)) < 1e-15);
    CHECK_TRUE(rel_with_floor < 1.0);  // stays bounded/reasonable, not exploding
}

void test_policy_ordering_lower_precision_no_stricter_than_higher_for_same_k() {
    int64_t k = 128;
    ValidationPolicy fp64 = named_policy(Precision::FP64, k);
    ValidationPolicy fp32 = named_policy(Precision::FP32, k);
    ValidationPolicy tf32 = named_policy(Precision::TF32, k);
    ValidationPolicy fp16 = named_policy(Precision::FP16, k);
    ValidationPolicy bf16 = named_policy(Precision::BF16, k);
    ValidationPolicy fp8e4m3 = named_policy(Precision::FP8_E4M3, k);
    ValidationPolicy fp8e5m2 = named_policy(Precision::FP8_E5M2, k);
    ValidationPolicy fp4 = named_policy(Precision::FP4, k);

    // "No stricter" means max_rel_error must be monotonically
    // non-decreasing as precision decreases.
    CHECK_TRUE(fp32.max_rel_error >= fp64.max_rel_error);
    CHECK_TRUE(tf32.max_rel_error >= fp32.max_rel_error);
    CHECK_TRUE(fp16.max_rel_error >= tf32.max_rel_error);
    CHECK_TRUE(bf16.max_rel_error >= fp16.max_rel_error);
    CHECK_TRUE(fp8e4m3.max_rel_error >= bf16.max_rel_error);
    CHECK_TRUE(fp8e5m2.max_rel_error >= fp8e4m3.max_rel_error);
    CHECK_TRUE(fp4.max_rel_error >= fp8e5m2.max_rel_error);
}

void test_policy_tolerance_grows_with_accumulation_depth() {
    ValidationPolicy shallow = named_policy(Precision::FP32, /*k_depth=*/1);
    ValidationPolicy deep = named_policy(Precision::FP32, /*k_depth=*/10000);
    CHECK_TRUE(deep.max_rel_error > shallow.max_rel_error);
}

void test_integer_policies_require_exact() {
    ValidationPolicy int8_policy = named_policy(Precision::INT8, 1);
    ValidationPolicy int4_policy = named_policy(Precision::INT4, 1);
    CHECK_TRUE(int8_policy.require_exact);
    CHECK_TRUE(int4_policy.require_exact);
}

void test_integer_exact_validation_passes_on_match_fails_on_mismatch() {
    std::vector<int32_t> expected = {1, 2, 3, 4};
    std::vector<int32_t> actual_match = {1, 2, 3, 4};
    std::vector<int32_t> actual_mismatch = {1, 2, 3, 5};

    ValidationResult ok = validate_integer_exact(actual_match.data(), expected.data(), 4);
    CHECK_TRUE(ok.passed);

    ValidationResult bad = validate_integer_exact(actual_mismatch.data(), expected.data(), 4);
    CHECK_TRUE(!bad.passed);
}

void test_integer_overflow_bound_detects_overflow_and_safe_case() {
    // int32_t accumulator, K=1000, operands up to |1000|: worst case
    // 1000*1000*1000 = 1e9, within int32 max (~2.147e9) -- should be safe.
    auto safe = check_integer_overflow_bound<int32_t>(1000, 1000, 1000);
    CHECK_TRUE(!safe.overflow_possible);

    // K=10000, operands up to |10000|: worst case 10000*10000*10000 = 1e12,
    // far beyond int32 max -- should be flagged.
    auto unsafe = check_integer_overflow_bound<int32_t>(10000, 10000, 10000);
    CHECK_TRUE(unsafe.overflow_possible);
}

}  // namespace

int main() {
    test_exact_match_passes_with_zero_errors();
    test_controlled_perturbation_produces_expected_abs_and_rel_error();
    test_nan_inf_mismatches_fail_visibly_even_with_zero_finite_error();
    test_both_sides_nonfinite_is_not_counted_as_mismatch_but_is_nonfinite();
    test_near_zero_relative_error_uses_floor_not_raw_denominator();
    test_policy_ordering_lower_precision_no_stricter_than_higher_for_same_k();
    test_policy_tolerance_grows_with_accumulation_depth();
    test_integer_policies_require_exact();
    test_integer_exact_validation_passes_on_match_fails_on_mismatch();
    test_integer_overflow_bound_detects_overflow_and_safe_case();
    return floptic_test::finish();
}
