#pragma once
#include <exception>
#include <functional>
#include <string>
#include "floptic/kernel_base.hpp"
#include "floptic/benchmark_status.hpp"

namespace floptic {

// Stable, generic diagnostic used when a caught exception is not a
// std::exception (so no ".what()" message is available). Kept as a named
// constant rather than an inline literal so tests assert against the same
// symbol the implementation uses, and so any future wording change stays
// in exactly one place.
inline constexpr const char* kUnknownExceptionDiagnostic =
    "unknown exception during kernel execution";

// Executes `fn` (a kernel/callable that produces a KernelResult) and
// converts any thrown exception into an explicit FAILED KernelResult with
// a nonempty diagnostic, instead of letting it propagate out of main.cpp
// and abort the entire benchmark run. This is the CPU-testable failure-
// propagation boundary called around each requested kernel run (see
// TODO.md "Make GPU/runtime/library errors invalidate a trial" — the
// per-API-call CUDA/HIP replacement itself is a separate, later card;
// this establishes the boundary those checks will eventually report
// through).
//
//   - std::exception: diagnostic is exactly e.what() (callers that want a
//     prefix/context should include it in the thrown message).
//   - any other thrown value: diagnostic is the stable
//     kUnknownExceptionDiagnostic string.
//   - no exception: `fn`'s result is returned completely unmodified — no
//     normalization happens here (dispatch_normalize.hpp still runs
//     afterward exactly as before).
inline KernelResult invoke_kernel_safely(const std::function<KernelResult()>& fn) {
    try {
        return fn();
    } catch (const std::exception& e) {
        KernelResult r;
        r.status = BenchmarkStatus::FAILED;
        r.diagnostic = e.what();
        return r;
    } catch (...) {
        KernelResult r;
        r.status = BenchmarkStatus::FAILED;
        r.diagnostic = kUnknownExceptionDiagnostic;
        return r;
    }
}

} // namespace floptic
