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
    VALIDATION_FAILED,  // ran, but failed numerical validation against a
                         // reference result

    // Internal-only sentinel — NOT a public status. Distinct from an
    // explicit FAILED: KernelResult defaults to UNSET so
    // dispatch_normalize.hpp can tell "kernel never reported anything" (may
    // still be promoted to OK/FAILED once a measurement is inspected) apart
    // from "kernel explicitly reported FAILED" (must never be overwritten).
    // Every result must be resolved to one of the five statuses above
    // before it reaches a serializer (JSON/Markdown); see
    // benchmark_status_is_public() below. Never returned by
    // benchmark_status_to_string() as anything other than the placeholder
    // "unset" string, which must never appear in a report.
    UNSET
};

// True for the five statuses that are allowed to reach a serialized report.
// UNSET must be resolved (see dispatch_normalize.hpp) before this is
// checked at the JSON/Markdown boundary.
inline bool benchmark_status_is_public(BenchmarkStatus s) {
    return s != BenchmarkStatus::UNSET;
}

// Defense-in-depth for serializers (json_writer.cpp, md_writer.cpp): the
// dispatch boundary (floptic/dispatch_normalize.hpp) is the only place
// that is supposed to resolve UNSET, but a serializer must never emit the
// internal "unset" string even if that invariant is ever violated
// upstream. Resolves UNSET to FAILED (a benchmark nothing was reported for
// is not a success) and leaves every other status untouched.
inline BenchmarkStatus resolve_status_for_serialization(BenchmarkStatus s) {
    return (s == BenchmarkStatus::UNSET) ? BenchmarkStatus::FAILED : s;
}

inline std::string benchmark_status_to_string(BenchmarkStatus s) {
    switch (s) {
        case BenchmarkStatus::OK:                return "ok";
        case BenchmarkStatus::UNSUPPORTED:       return "unsupported";
        case BenchmarkStatus::FAILED:            return "failed";
        case BenchmarkStatus::NOT_REQUESTED:     return "not_requested";
        case BenchmarkStatus::VALIDATION_FAILED: return "validation_failed";
        case BenchmarkStatus::UNSET:             return "unset";
    }
    return "unknown";
}

} // namespace floptic
