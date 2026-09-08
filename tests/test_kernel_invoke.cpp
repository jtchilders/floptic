// Kernel invocation boundary tests (floptic/kernel_invoke.hpp).
//
// Exercises invoke_kernel_safely() directly: it is the single place a
// thrown std::exception (or any other exception) from a kernel/callable is
// converted into an explicit FAILED KernelResult with a diagnostic instead
// of aborting the whole benchmark run. Hardware-independent (no device
// discovery, no kernel execution) — mirrors exactly what main.cpp will
// call once per requested benchmark combination. See TODO.md "Make GPU/
// runtime/library errors invalidate a trial" (this is the CPU-testable
// boundary that the CUDA/HIP API-error propagation itself will plug into;
// that per-API-call replacement is out of scope for this card).
#include "floptic/kernel_invoke.hpp"
#include "floptic/kernel_base.hpp"
#include "floptic/benchmark_status.hpp"
#include "check.hpp"

#include <stdexcept>
#include <string>

using floptic::BenchmarkStatus;
using floptic::KernelResult;

namespace {

void test_std_exception_becomes_failed_with_exact_diagnostic_substring() {
    auto result = floptic::invoke_kernel_safely([]() -> KernelResult {
        throw std::runtime_error("cuBLAS launch failed: CUBLAS_STATUS_EXECUTION_FAILED");
    });

    CHECK_TRUE(result.status == BenchmarkStatus::FAILED);
    CHECK_TRUE(result.diagnostic.find("cuBLAS launch failed: CUBLAS_STATUS_EXECUTION_FAILED")
               != std::string::npos);
}

// A non-std::exception throw (e.g. a raw int, or a third-party type that
// doesn't derive from std::exception) must still be caught and produce a
// stable, generic diagnostic — never propagate and crash the whole run.
void test_unknown_exception_becomes_failed_with_stable_generic_diagnostic() {
    auto result = floptic::invoke_kernel_safely([]() -> KernelResult {
        throw 42; // deliberately not a std::exception
    });

    CHECK_TRUE(result.status == BenchmarkStatus::FAILED);
    CHECK_TRUE(!result.diagnostic.empty());
    CHECK_EQ(result.diagnostic, std::string(floptic::kUnknownExceptionDiagnostic));
}

// A successful callable's result must pass through completely unmodified —
// invoke_kernel_safely does not normalize or otherwise alter it; normal
// dispatch-boundary normalization (dispatch_normalize.hpp) still runs on
// the result afterward exactly as before.
void test_successful_callable_result_passes_through_unmodified() {
    auto result = floptic::invoke_kernel_safely([]() -> KernelResult {
        KernelResult r;
        r.status = BenchmarkStatus::OK;
        r.gflops = 123.4;
        r.total_flops = 5000;
        return r;
    });

    CHECK_TRUE(result.status == BenchmarkStatus::OK);
    CHECK_EQ(result.gflops, 123.4);
    CHECK_EQ(result.total_flops, static_cast<int64_t>(5000));
    CHECK_TRUE(result.diagnostic.empty());
}

} // namespace

int main() {
    test_std_exception_becomes_failed_with_exact_diagnostic_substring();
    test_unknown_exception_becomes_failed_with_stable_generic_diagnostic();
    test_successful_callable_result_passes_through_unmodified();
    return floptic_test::finish();
}
