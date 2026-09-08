// Hardware-independent typed-metric tests.
//
// These check pure logic in floptic/typed_metric.hpp: enum string mapping,
// legacy-to-typed normalization, the centralized kernel-semantics mapping
// function, and human-readable rate formatting. Must not touch device
// discovery, CUDA/HIP, or kernel execution.
#include "floptic/typed_metric.hpp"
#include "floptic/precision.hpp"
#include "floptic/benchmark_status.hpp"
#include "check.hpp"

#include <cmath>
#include <string>

using floptic::MetricKind;
using floptic::Precision;
using floptic::TypedMetric;

namespace {

// ---------------------------------------------------------------------
// Enum string values
// ---------------------------------------------------------------------

void test_metric_kind_to_string_produces_stable_spellings() {
    CHECK_EQ(floptic::metric_kind_to_string(MetricKind::FLOATING_POINT_OPERATIONS),
             std::string("floating_point_operations"));
    CHECK_EQ(floptic::metric_kind_to_string(MetricKind::INTEGER_OPERATIONS),
             std::string("integer_operations"));
    CHECK_EQ(floptic::metric_kind_to_string(MetricKind::TRANSFERRED_BYTES),
             std::string("transferred_bytes"));
}

void test_metric_unit_for_kind_produces_base_si_units() {
    CHECK_EQ(floptic::metric_unit_for_kind(MetricKind::FLOATING_POINT_OPERATIONS),
             std::string("FLOP/s"));
    CHECK_EQ(floptic::metric_unit_for_kind(MetricKind::INTEGER_OPERATIONS),
             std::string("OP/s"));
    CHECK_EQ(floptic::metric_unit_for_kind(MetricKind::TRANSFERRED_BYTES),
             std::string("B/s"));
}

// ---------------------------------------------------------------------
// Serialization / formatting for FLOP/s, OP/s, and B/s
// ---------------------------------------------------------------------

void test_format_metric_rate_for_floating_point_operations() {
    TypedMetric m;
    m.kind = MetricKind::FLOATING_POINT_OPERATIONS;
    m.rate_per_second = 244.2e12;  // 244.2 TFLOP/s
    CHECK_EQ(floptic::format_metric_rate(m), std::string("244.2 TFLOP/s"));
}

void test_format_metric_rate_for_integer_operations() {
    TypedMetric m;
    m.kind = MetricKind::INTEGER_OPERATIONS;
    m.rate_per_second = 5.0e8;  // 500.0 MOP/s
    CHECK_EQ(floptic::format_metric_rate(m), std::string("500.0 MOP/s"));
}

void test_format_metric_rate_for_transferred_bytes() {
    TypedMetric m;
    m.kind = MetricKind::TRANSFERRED_BYTES;
    m.rate_per_second = 1.23e10;  // 12.3 GB/s
    CHECK_EQ(floptic::format_metric_rate(m), std::string("12.3 GB/s"));
}

// ---------------------------------------------------------------------
// Status serialization: failed/validation_failed retained explicitly.
// (BenchmarkStatus lives in benchmark_status.hpp; verified here as it is
// the sibling type used alongside TypedMetric in KernelResult.)
// ---------------------------------------------------------------------

void test_benchmark_status_to_string_covers_every_state() {
    using floptic::BenchmarkStatus;
    CHECK_EQ(floptic::benchmark_status_to_string(BenchmarkStatus::OK), std::string("ok"));
    CHECK_EQ(floptic::benchmark_status_to_string(BenchmarkStatus::UNSUPPORTED), std::string("unsupported"));
    CHECK_EQ(floptic::benchmark_status_to_string(BenchmarkStatus::FAILED), std::string("failed"));
    CHECK_EQ(floptic::benchmark_status_to_string(BenchmarkStatus::NOT_REQUESTED), std::string("not_requested"));
    CHECK_EQ(floptic::benchmark_status_to_string(BenchmarkStatus::VALIDATION_FAILED), std::string("validation_failed"));
}

// ---------------------------------------------------------------------
// Legacy normalization behavior
// ---------------------------------------------------------------------

void test_normalize_legacy_metric_for_flops() {
    TypedMetric m = floptic::normalize_legacy_metric(
        MetricKind::FLOATING_POINT_OPERATIONS, /*legacy_rate_giga=*/244.2,
        /*count=*/1000000, "FMA counts as two floating-point operations");
    CHECK_TRUE(m.kind == MetricKind::FLOATING_POINT_OPERATIONS);
    CHECK_TRUE(std::abs(m.rate_per_second - 244.2e9) < 1.0);
    CHECK_EQ(m.operation_count, static_cast<int64_t>(1000000));
    CHECK_EQ(m.byte_count, static_cast<int64_t>(0));
    CHECK_EQ(m.arithmetic_convention, std::string("FMA counts as two floating-point operations"));
}

void test_normalize_legacy_metric_for_bytes() {
    TypedMetric m = floptic::normalize_legacy_metric(
        MetricKind::TRANSFERRED_BYTES, /*legacy_rate_giga=*/12.3,
        /*count=*/5000000000LL);
    CHECK_TRUE(m.kind == MetricKind::TRANSFERRED_BYTES);
    CHECK_TRUE(std::abs(m.rate_per_second - 12.3e9) < 1.0);
    CHECK_EQ(m.byte_count, static_cast<int64_t>(5000000000LL));
    CHECK_EQ(m.operation_count, static_cast<int64_t>(0));
}

// ---------------------------------------------------------------------
// Centralized kernel-semantics mapping used at the dispatch boundary.
// Memory kernels are bytes, INT kernels are integer operations, other
// compute kernels are floating operations.
// ---------------------------------------------------------------------

void test_infer_metric_kind_maps_memory_category_to_bytes() {
    CHECK_TRUE(floptic::infer_metric_kind_for_kernel("memory", Precision::FP32)
               == MetricKind::TRANSFERRED_BYTES);
    CHECK_TRUE(floptic::infer_metric_kind_for_kernel("memory", Precision::FP64)
               == MetricKind::TRANSFERRED_BYTES);
}

void test_infer_metric_kind_maps_int_precisions_to_integer_ops() {
    CHECK_TRUE(floptic::infer_metric_kind_for_kernel("scalar", Precision::INT8)
               == MetricKind::INTEGER_OPERATIONS);
    CHECK_TRUE(floptic::infer_metric_kind_for_kernel("matrix", Precision::INT4)
               == MetricKind::INTEGER_OPERATIONS);
}

void test_infer_metric_kind_defaults_other_compute_kernels_to_floating() {
    CHECK_TRUE(floptic::infer_metric_kind_for_kernel("scalar", Precision::FP64)
               == MetricKind::FLOATING_POINT_OPERATIONS);
    CHECK_TRUE(floptic::infer_metric_kind_for_kernel("vector", Precision::FP32)
               == MetricKind::FLOATING_POINT_OPERATIONS);
    CHECK_TRUE(floptic::infer_metric_kind_for_kernel("matrix", Precision::TF32)
               == MetricKind::FLOATING_POINT_OPERATIONS);
}

// ---------------------------------------------------------------------
// Zero / failed behavior: a default-constructed TypedMetric must format
// without inventing nonzero performance.
// ---------------------------------------------------------------------

void test_default_typed_metric_is_zero_rate_floating_point() {
    TypedMetric m;
    CHECK_TRUE(m.kind == MetricKind::FLOATING_POINT_OPERATIONS);
    CHECK_EQ(m.rate_per_second, 0.0);
    CHECK_EQ(m.operation_count, static_cast<int64_t>(0));
    CHECK_EQ(m.byte_count, static_cast<int64_t>(0));
    CHECK_EQ(floptic::format_metric_rate(m), std::string("0.0 FLOP/s"));
}

} // namespace
int main() {
    test_metric_kind_to_string_produces_stable_spellings();
    test_metric_unit_for_kind_produces_base_si_units();
    test_format_metric_rate_for_floating_point_operations();
    test_format_metric_rate_for_integer_operations();
    test_format_metric_rate_for_transferred_bytes();
    test_benchmark_status_to_string_covers_every_state();
    test_normalize_legacy_metric_for_flops();
    test_normalize_legacy_metric_for_bytes();
    test_infer_metric_kind_maps_memory_category_to_bytes();
    test_infer_metric_kind_maps_int_precisions_to_integer_ops();
    test_infer_metric_kind_defaults_other_compute_kernels_to_floating();
    test_default_typed_metric_is_zero_rate_floating_point();
    return floptic_test::finish();
}
