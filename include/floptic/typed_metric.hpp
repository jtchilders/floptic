#pragma once
#include <string>
#include <cstdint>
#include <cstdio>
#include "floptic/precision.hpp"

namespace floptic {

// Typed primary-metric kind. Every KernelResult carries exactly one of
// these instead of an ambiguous "gflops" field that silently overloaded
// floating-point rate and memory bandwidth. See TODO.md P1 "Replace the
// overloaded gflops field with a typed metric representation."
enum class MetricKind {
    FLOATING_POINT_OPERATIONS,
    INTEGER_OPERATIONS,
    TRANSFERRED_BYTES
};

inline std::string metric_kind_to_string(MetricKind k) {
    switch (k) {
        case MetricKind::FLOATING_POINT_OPERATIONS: return "floating_point_operations";
        case MetricKind::INTEGER_OPERATIONS:        return "integer_operations";
        case MetricKind::TRANSFERRED_BYTES:         return "transferred_bytes";
    }
    return "unknown";
}

// Base SI unit for a metric kind's rate. Bandwidth is never stored/labeled
// as "GFLOP/s" and FLOP/s is never conflated with plain OP/s.
inline std::string metric_unit_for_kind(MetricKind k) {
    switch (k) {
        case MetricKind::FLOATING_POINT_OPERATIONS: return "FLOP/s";
        case MetricKind::INTEGER_OPERATIONS:        return "OP/s";
        case MetricKind::TRANSFERRED_BYTES:         return "B/s";
    }
    return "unknown";
}

// A single, unambiguous primary-metric measurement: kind, base-SI rate,
// and the explicit operation/byte count that produced it. Exactly one of
// operation_count / byte_count is meaningful depending on `kind` (bytes
// for TRANSFERRED_BYTES, operations otherwise) — the other stays zero.
struct TypedMetric {
    MetricKind kind = MetricKind::FLOATING_POINT_OPERATIONS;
    double rate_per_second = 0.0;
    int64_t operation_count = 0;
    int64_t byte_count = 0;
    // Optional free-text note on arithmetic convention, e.g. "FMA counts
    // as two floating-point operations." Empty when not applicable.
    std::string arithmetic_convention;
};

// Deterministic normalization from a legacy "gflops-shaped" rate (a value
// expressed in units of 1e9 per second, matching the old KernelResult::
// gflops/effective_gflops fields) into a typed metric, given a
// caller-provided metric kind. This never infers the kind from category —
// the caller (kernel dispatch boundary) must say what was measured.
inline TypedMetric normalize_legacy_metric(MetricKind kind,
                                            double legacy_rate_giga,
                                            int64_t count,
                                            const std::string& arithmetic_convention = "") {
    TypedMetric m;
    m.kind = kind;
    m.rate_per_second = legacy_rate_giga * 1e9;
    if (kind == MetricKind::TRANSFERRED_BYTES) {
        m.byte_count = count;
    } else {
        m.operation_count = count;
    }
    m.arithmetic_convention = arithmetic_convention;
    return m;
}

// Centralized kernel-semantics mapping used at the dispatch boundary
// (main.cpp) to normalize legacy per-kernel gflops/effective_gflops
// fields into a typed metric without per-kernel migration in this card.
// Explicit mapping by kernel semantics: memory kernels are bytes, INT
// kernels are integer operations, other compute kernels are floating
// operations. Kept centralized and tested here rather than scattered
// per-kernel guesses.
inline MetricKind infer_metric_kind_for_kernel(const std::string& category,
                                                Precision precision) {
    if (category == "memory") {
        return MetricKind::TRANSFERRED_BYTES;
    }
    if (precision == Precision::INT8 || precision == Precision::INT4) {
        return MetricKind::INTEGER_OPERATIONS;
    }
    return MetricKind::FLOATING_POINT_OPERATIONS;
}

// Human-readable rate formatting with SI prefix, e.g. "244.2 TFLOP/s",
// "500.0 MOP/s", "12.3 GB/s". Driven entirely by the typed metric — never
// by category-based unit guessing.
inline std::string format_metric_rate(const TypedMetric& m) {
    double val = m.rate_per_second;
    std::string unit = metric_unit_for_kind(m.kind);

    double scaled;
    const char* prefix;
    if (val >= 1e12) {
        scaled = val / 1e12; prefix = "T";
    } else if (val >= 1e9) {
        scaled = val / 1e9; prefix = "G";
    } else if (val >= 1e6) {
        scaled = val / 1e6; prefix = "M";
    } else if (val >= 1e3) {
        scaled = val / 1e3; prefix = "K";
    } else {
        scaled = val; prefix = "";
    }

    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f %s%s", scaled, prefix, unit.c_str());
    return std::string(buf);
}

} // namespace floptic
