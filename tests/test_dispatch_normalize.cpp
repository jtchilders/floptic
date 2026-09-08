// Dispatch-boundary normalization tests (floptic/dispatch_normalize.hpp).
//
// These exercise normalize_dispatch_result() directly, hardware-independent
// (no device discovery, no kernel execution) — mirroring exactly what
// main.cpp calls once per executed benchmark.
#include "floptic/dispatch_normalize.hpp"
#include "floptic/kernel_base.hpp"
#include "floptic/benchmark_status.hpp"
#include "floptic/typed_metric.hpp"
#include "floptic/precision.hpp"
#include "check.hpp"

using floptic::BenchmarkStatus;
using floptic::KernelResult;
using floptic::MetricKind;
using floptic::Precision;

namespace {

// ---------------------------------------------------------------------
// Memory kernels: actual byte counts, not the unrelated FLOP count.
// ---------------------------------------------------------------------

void test_memory_kernel_uses_total_bytes_not_total_flops() {
    KernelResult r;
    r.gflops = 12.3;          // legacy GB/s
    r.total_flops = 999999;   // present but must NOT be used as byte_count
    r.total_bytes = 5000000000LL;

    floptic::normalize_dispatch_result(r, "memory", Precision::FP64);

    CHECK_TRUE(r.metric.kind == MetricKind::TRANSFERRED_BYTES);
    CHECK_EQ(r.metric.byte_count, static_cast<int64_t>(5000000000LL));
    CHECK_TRUE(r.metric.byte_count != r.total_flops);
}

void test_memory_kernel_with_zero_total_bytes_reports_zero_not_flops() {
    // Reproduces the reviewer-reported HIP memory-kernel defect: a kernel
    // that never populates total_bytes must report byte_count 0, not
    // silently substitute total_flops.
    KernelResult r;
    r.gflops = 500.0;
    r.total_flops = 42;  // stray/unrelated value; must not leak into bytes
    r.total_bytes = 0;

    floptic::normalize_dispatch_result(r, "memory", Precision::FP64);

    CHECK_EQ(r.metric.byte_count, static_cast<int64_t>(0));
}

// ---------------------------------------------------------------------
// Status is authoritative: never inferred from a nonpositive legacy rate.
// ---------------------------------------------------------------------

void test_default_result_status_is_failed_not_ok() {
    // KernelResult's own default must be a non-success sentinel, not OK —
    // otherwise a kernel that forgets to report anything silently looks
    // like a successful zero-performance result.
    KernelResult r;
    CHECK_TRUE(r.status == BenchmarkStatus::FAILED);
}

void test_valid_measurement_promotes_default_failed_to_ok() {
    KernelResult r;
    r.gflops = 244.2;
    r.total_flops = 1000;

    floptic::normalize_dispatch_result(r, "scalar", Precision::FP64);

    CHECK_TRUE(r.status == BenchmarkStatus::OK);
}

void test_zero_rate_result_status_stays_failed_not_inferred_from_rate() {
    // A kernel that produced no valid measurement (gflops == 0, metric
    // untouched) must remain FAILED — this is a status-driven decision,
    // not a "gflops <= 0" inference recomputed here.
    KernelResult r;
    r.gflops = 0.0;
    r.total_flops = 0;

    floptic::normalize_dispatch_result(r, "scalar", Precision::FP64);

    CHECK_TRUE(r.status == BenchmarkStatus::FAILED);
    CHECK_EQ(r.metric.rate_per_second, 0.0);
}

void test_explicit_non_failed_status_is_left_untouched() {
    // A kernel that already set an explicit status (e.g. validation
    // failure) must not be overwritten by the dispatch boundary even if
    // its legacy rate happens to be positive.
    KernelResult r;
    r.status = BenchmarkStatus::VALIDATION_FAILED;
    r.gflops = 244.2;
    r.total_flops = 1000;

    floptic::normalize_dispatch_result(r, "scalar", Precision::FP64);

    CHECK_TRUE(r.status == BenchmarkStatus::VALIDATION_FAILED);
}

void test_kernel_that_already_set_ok_and_metric_is_left_untouched() {
    // A migrated kernel that already populated result.metric directly
    // (rate_per_second != 0) must not be re-normalized from its legacy
    // fields at all.
    KernelResult r;
    r.status = BenchmarkStatus::OK;
    r.metric.kind = MetricKind::INTEGER_OPERATIONS;
    r.metric.rate_per_second = 42.0;
    r.metric.operation_count = 7;
    r.gflops = 0.0;  // legacy fields intentionally left at zero
    r.total_flops = 0;

    floptic::normalize_dispatch_result(r, "matrix", Precision::INT8);

    CHECK_TRUE(r.metric.kind == MetricKind::INTEGER_OPERATIONS);
    CHECK_EQ(r.metric.rate_per_second, 42.0);
    CHECK_EQ(r.metric.operation_count, static_cast<int64_t>(7));
    CHECK_TRUE(r.status == BenchmarkStatus::OK);
}

// ---------------------------------------------------------------------
// Arithmetic convention populated for applicable kernel semantics.
// ---------------------------------------------------------------------

void test_floating_point_kernel_gets_fma_arithmetic_convention() {
    KernelResult r;
    r.gflops = 244.2;
    r.total_flops = 1000;

    floptic::normalize_dispatch_result(r, "scalar", Precision::FP64);

    CHECK_TRUE(!r.metric.arithmetic_convention.empty());
}

void test_integer_kernel_gets_mac_arithmetic_convention() {
    KernelResult r;
    r.gflops = 500.0;
    r.total_flops = 2000;

    floptic::normalize_dispatch_result(r, "matrix", Precision::INT8);

    CHECK_TRUE(!r.metric.arithmetic_convention.empty());
}

void test_memory_kernel_has_no_arithmetic_convention() {
    KernelResult r;
    r.gflops = 12.3;
    r.total_bytes = 5000000000LL;

    floptic::normalize_dispatch_result(r, "memory", Precision::FP64);

    CHECK_TRUE(r.metric.arithmetic_convention.empty());
}

} // namespace

int main() {
    test_memory_kernel_uses_total_bytes_not_total_flops();
    test_memory_kernel_with_zero_total_bytes_reports_zero_not_flops();
    test_default_result_status_is_failed_not_ok();
    test_valid_measurement_promotes_default_failed_to_ok();
    test_zero_rate_result_status_stays_failed_not_inferred_from_rate();
    test_explicit_non_failed_status_is_left_untouched();
    test_kernel_that_already_set_ok_and_metric_is_left_untouched();
    test_floating_point_kernel_gets_fma_arithmetic_convention();
    test_integer_kernel_gets_mac_arithmetic_convention();
    test_memory_kernel_has_no_arithmetic_convention();
    return floptic_test::finish();
}
