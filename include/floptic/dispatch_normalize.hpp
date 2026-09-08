#pragma once
#include <string>
#include "floptic/precision.hpp"
#include "floptic/benchmark_status.hpp"
#include "floptic/typed_metric.hpp"
#include "floptic/kernel_base.hpp"

namespace floptic {

// Centralized dispatch-boundary normalization (main.cpp calls this exactly
// once per executed benchmark). Kept here, rather than inline in main.cpp,
// so it is directly unit-testable without a live device — see
// tests/test_dispatch_normalize.cpp for the required coverage:
//   - actual byte counts for memory kernels (not the FLOP count reused as
//     a byte count)
//   - status left authoritative: FAILED is the KernelResult default, and
//     this function only ever promotes FAILED -> OK on evidence of a valid
//     measurement; it never infers failure from a nonpositive rate, and it
//     never touches an already-explicit non-FAILED status (UNSUPPORTED,
//     NOT_REQUESTED, VALIDATION_FAILED, or an already-OK result).
//   - arithmetic convention populated for applicable kernel semantics
//     (FMA/MAC-based compute; empty for byte-transfer kernels).
inline void normalize_dispatch_result(KernelResult& result,
                                       const std::string& category,
                                       Precision precision) {
    // Kernels that already populate result.metric directly (rate_per_second
    // != 0) are left untouched — this only covers kernels that have not
    // migrated to the typed metric and still report through the legacy
    // gflops/effective_gflops/total_flops/total_bytes fields.
    if (result.metric.rate_per_second == 0.0 && result.gflops > 0) {
        MetricKind inferred_kind = infer_metric_kind_for_kernel(category, precision);

        // Explicit per-kind count: memory kernels report actual transferred
        // bytes (result.total_bytes), never the unrelated FLOP count.
        // Everything else (floating-point or integer operations) uses the
        // operation count.
        int64_t count = (inferred_kind == MetricKind::TRANSFERRED_BYTES)
                             ? result.total_bytes
                             : result.total_flops;

        result.metric = normalize_legacy_metric(
            inferred_kind, result.gflops, count,
            arithmetic_convention_for_kernel(inferred_kind));
    }

    // Status is authoritative and must never be derived from a nonpositive
    // legacy rate. KernelResult::status defaults to FAILED (a non-success
    // sentinel); the only transition this boundary performs is promoting
    // that default to OK once a valid, nonzero measurement exists. Any
    // status other than the default FAILED (OK already, UNSUPPORTED,
    // NOT_REQUESTED, VALIDATION_FAILED) was set explicitly by the kernel
    // and is left exactly as-is.
    if (result.status == BenchmarkStatus::FAILED && result.metric.rate_per_second > 0.0) {
        result.status = BenchmarkStatus::OK;
    }
}

} // namespace floptic
