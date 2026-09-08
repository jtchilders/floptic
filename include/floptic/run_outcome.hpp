#pragma once
#include <vector>
#include "floptic/benchmark_status.hpp"

namespace floptic {

// Aggregate exit-code decision for a completed run, applied once after all
// requested benchmark combinations have executed and reports have been
// written (see main.cpp and TODO.md "Return a nonzero process status when
// a requested benchmark has no valid result"). Takes the resolved
// (serialization-ready; see resolve_status_for_serialization) status of
// every requested benchmark entry.
//
// Decision:
//   - any FAILED or VALIDATION_FAILED present            -> failure
//   - no OK present at all (including unsupported-only,
//     not_requested-only, or an empty list)               -> failure
//   - otherwise (at least one OK, no FAILED/               -> success
//     VALIDATION_FAILED — e.g. all OK, or a mix of OK and
//     UNSUPPORTED)
inline bool run_has_failure_exit(const std::vector<BenchmarkStatus>& statuses) {
    bool any_failed = false;
    bool any_ok = false;

    for (auto s : statuses) {
        if (s == BenchmarkStatus::FAILED || s == BenchmarkStatus::VALIDATION_FAILED) {
            any_failed = true;
        } else if (s == BenchmarkStatus::OK) {
            any_ok = true;
        }
    }

    if (any_failed) return true;
    if (!any_ok) return true;
    return false;
}

} // namespace floptic
