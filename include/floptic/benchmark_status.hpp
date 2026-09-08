#pragma once
#include <string>

namespace floptic {

// Explicit benchmark outcome states (schema v2). Every serialized benchmark
// entry carries exactly one of these instead of being silently omitted or
// represented as zero performance when something other than a clean
// measurement happened. See TODO.md "Distinguish benchmark states".
enum class BenchmarkStatus {
    OK,                 // requested kernel ran and produced a valid result
    UNSUPPORTED,        // combination not supported by this device/kernel
    FAILED,             // requested but did not produce a valid result
                         // (allocation/API/launch failure, etc.)
    NOT_REQUESTED,      // present for completeness; not selected to run
    VALIDATION_FAILED   // ran, but failed numerical validation against a
                         // reference result
};

inline std::string benchmark_status_to_string(BenchmarkStatus s) {
    switch (s) {
        case BenchmarkStatus::OK:                return "ok";
        case BenchmarkStatus::UNSUPPORTED:       return "unsupported";
        case BenchmarkStatus::FAILED:            return "failed";
        case BenchmarkStatus::NOT_REQUESTED:     return "not_requested";
        case BenchmarkStatus::VALIDATION_FAILED: return "validation_failed";
    }
    return "unknown";
}

} // namespace floptic
