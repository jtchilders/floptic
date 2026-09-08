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
//   - status left authoritative: KernelResult defaults to the internal-only
//     UNSET sentinel (never FAILED), and this function only ever resolves
//     UNSET -> OK (valid measurement present) or UNSET -> FAILED (no
//     measurement present). It never touches an already-explicit status —
//     OK, FAILED, UNSUPPORTED, NOT_REQUESTED, or VALIDATION_FAILED are all
//     left exactly as the kernel set them, even if the legacy rate happens
//     to be positive. UNSET must never reach a serializer.
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
    // legacy rate. KernelResult::status defaults to the internal-only UNSET
    // sentinel — distinct from an explicit FAILED — so this is the only
    // place that resolves "kernel reported nothing" into a public status:
    // UNSET is promoted to OK once a valid, nonzero measurement exists, and
    // otherwise resolved to FAILED. Any status other than UNSET (OK,
    // FAILED, UNSUPPORTED, NOT_REQUESTED, VALIDATION_FAILED) was set
    // explicitly by the kernel and is left exactly as-is, even if the
    // legacy rate is positive.
    if (result.status == BenchmarkStatus::UNSET) {
        result.status = (result.metric.rate_per_second > 0.0)
                             ? BenchmarkStatus::OK
                             : BenchmarkStatus::FAILED;
    }
}

} // namespace floptic
