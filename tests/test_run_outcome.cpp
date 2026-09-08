// Aggregate run-exit-code decision tests (floptic/run_outcome.hpp).
//
// main.cpp must exit nonzero after report output whenever any requested
// benchmark ended FAILED/VALIDATION_FAILED, or when nothing succeeded at
// all (e.g. every requested combination was unsupported) — but a mix of
// OK and UNSUPPORTED results must still exit zero. Header-only, hardware-
// independent. See TODO.md "Return a nonzero process status when a
// requested benchmark has no valid result."
#include "floptic/run_outcome.hpp"
#include "floptic/benchmark_status.hpp"
#include "check.hpp"

#include <vector>

using floptic::BenchmarkStatus;

namespace {

void test_all_ok_is_success() {
    std::vector<BenchmarkStatus> statuses = {
        BenchmarkStatus::OK, BenchmarkStatus::OK, BenchmarkStatus::OK};
    CHECK_TRUE(!floptic::run_has_failure_exit(statuses));
}

void test_any_failed_is_failure() {
    std::vector<BenchmarkStatus> statuses = {
        BenchmarkStatus::OK, BenchmarkStatus::FAILED, BenchmarkStatus::OK};
    CHECK_TRUE(floptic::run_has_failure_exit(statuses));
}

void test_any_validation_failed_is_failure() {
    std::vector<BenchmarkStatus> statuses = {
        BenchmarkStatus::OK, BenchmarkStatus::VALIDATION_FAILED};
    CHECK_TRUE(floptic::run_has_failure_exit(statuses));
}

void test_unsupported_only_with_no_success_is_failure() {
    std::vector<BenchmarkStatus> statuses = {
        BenchmarkStatus::UNSUPPORTED, BenchmarkStatus::UNSUPPORTED};
    CHECK_TRUE(floptic::run_has_failure_exit(statuses));
}

void test_no_success_at_all_is_failure_even_without_explicit_failed() {
    std::vector<BenchmarkStatus> statuses = {
        BenchmarkStatus::NOT_REQUESTED};
    CHECK_TRUE(floptic::run_has_failure_exit(statuses));
}

void test_mixed_ok_and_unsupported_is_success() {
    std::vector<BenchmarkStatus> statuses = {
        BenchmarkStatus::OK, BenchmarkStatus::UNSUPPORTED};
    CHECK_TRUE(!floptic::run_has_failure_exit(statuses));
}

void test_empty_statuses_is_failure() {
    // No requested benchmark recorded at all: no success and no explicit
    // failure signal either — must not silently claim success.
    std::vector<BenchmarkStatus> statuses;
    CHECK_TRUE(floptic::run_has_failure_exit(statuses));
}

} // namespace

int main() {
    test_all_ok_is_success();
    test_any_failed_is_failure();
    test_any_validation_failed_is_failure();
    test_unsupported_only_with_no_success_is_failure();
    test_no_success_at_all_is_failure_even_without_explicit_failed();
    test_mixed_ok_and_unsupported_is_success();
    test_empty_statuses_is_failure();
    return floptic_test::finish();
}
