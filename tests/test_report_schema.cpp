// Report-schema (v2) tests: JSON schema_version, typed metric
// serialization, explicit status retention (including failed/
// validation_failed rather than omission), legacy compatibility field
// naming, and Markdown rendering that uses the typed metric instead of
// category-based unit guessing.
//
// Builds Report/BenchmarkEntry structs directly (floptic_core, no device
// discovery) and calls report_to_json / write_markdown_report exactly as
// main.cpp does.
#include "floptic/report.hpp"
#include "floptic/benchmark_status.hpp"
#include "floptic/typed_metric.hpp"
#include "check.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using floptic::BenchmarkEntry;
using floptic::BenchmarkStatus;
using floptic::KernelResult;
using floptic::MetricKind;
using floptic::Report;
using floptic::TypedMetric;

namespace {

BenchmarkEntry make_entry(const std::string& kernel_name, const std::string& category,
                           BenchmarkStatus status, MetricKind kind,
                           double rate_per_second, int64_t count) {
    BenchmarkEntry e;
    e.device_id = "cpu:0";
    e.kernel_name = kernel_name;
    e.category = category;
    e.precision = "FP64";
    e.mode = "throughput";

    KernelResult r;
    r.status = status;
    r.metric.kind = kind;
    if (status == BenchmarkStatus::OK) {
        r.metric.rate_per_second = rate_per_second;
        if (kind == MetricKind::TRANSFERRED_BYTES) {
            r.metric.byte_count = count;
        } else {
            r.metric.operation_count = count;
        }
        r.gflops = rate_per_second / 1e9;
        r.effective_gflops = r.gflops;
    }
    r.median_time_ms = 1.0;
    e.result = r;
    return e;
}

// ---------------------------------------------------------------------
// JSON schema_version == 2
// ---------------------------------------------------------------------

void test_json_report_declares_schema_version_2() {
    Report report;
    report.benchmarks.push_back(
        make_entry("scalar_fma", "scalar", BenchmarkStatus::OK,
                   MetricKind::FLOATING_POINT_OPERATIONS, 244.2e9, 1000));
    auto j = floptic::report_to_json(report);
    CHECK_TRUE(j.contains("schema_version"));
    CHECK_EQ(j["schema_version"].get<int>(), 2);
}

// ---------------------------------------------------------------------
// Serialization for FLOP/s, OP/s, and B/s
// ---------------------------------------------------------------------

void test_json_serializes_floating_point_metric() {
    Report report;
    report.benchmarks.push_back(
        make_entry("scalar_fma", "scalar", BenchmarkStatus::OK,
                   MetricKind::FLOATING_POINT_OPERATIONS, 244.2e9, 1000));
    auto j = floptic::report_to_json(report);
    auto& m = j["benchmarks"][0]["results"]["metric"];
    CHECK_EQ(m["kind"].get<std::string>(), std::string("floating_point_operations"));
    CHECK_EQ(m["unit"].get<std::string>(), std::string("FLOP/s"));
    CHECK_TRUE(m.contains("operation_count"));
    CHECK_TRUE(!m.contains("byte_count"));
}

void test_json_serializes_integer_operations_metric() {
    Report report;
    report.benchmarks.push_back(
        make_entry("gemm_cublas_int8", "matrix", BenchmarkStatus::OK,
                   MetricKind::INTEGER_OPERATIONS, 5.0e11, 2000));
    auto j = floptic::report_to_json(report);
    auto& m = j["benchmarks"][0]["results"]["metric"];
    CHECK_EQ(m["kind"].get<std::string>(), std::string("integer_operations"));
    CHECK_EQ(m["unit"].get<std::string>(), std::string("OP/s"));
    CHECK_TRUE(m.contains("operation_count"));
    CHECK_TRUE(!m.contains("byte_count"));
}

void test_json_serializes_transferred_bytes_metric() {
    Report report;
    report.benchmarks.push_back(
        make_entry("stream_triad", "memory", BenchmarkStatus::OK,
                   MetricKind::TRANSFERRED_BYTES, 1.23e10, 3000));
    auto j = floptic::report_to_json(report);
    auto& m = j["benchmarks"][0]["results"]["metric"];
    CHECK_EQ(m["kind"].get<std::string>(), std::string("transferred_bytes"));
    CHECK_EQ(m["unit"].get<std::string>(), std::string("B/s"));
    CHECK_TRUE(m.contains("byte_count"));
    CHECK_TRUE(!m.contains("operation_count"));
}

// ---------------------------------------------------------------------
// Status serialization: failed/validation_failed retained, not omitted.
// ---------------------------------------------------------------------

void test_json_retains_failed_and_validation_failed_entries() {
    Report report;
    report.benchmarks.push_back(
        make_entry("scalar_fma", "scalar", BenchmarkStatus::OK,
                   MetricKind::FLOATING_POINT_OPERATIONS, 1e9, 100));
    report.benchmarks.push_back(
        make_entry("vector_axpy", "vector", BenchmarkStatus::FAILED,
                   MetricKind::FLOATING_POINT_OPERATIONS, 0.0, 0));
    report.benchmarks.push_back(
        make_entry("gemm_cublas", "matrix", BenchmarkStatus::VALIDATION_FAILED,
                   MetricKind::FLOATING_POINT_OPERATIONS, 0.0, 0));

    auto j = floptic::report_to_json(report);
    CHECK_EQ(j["benchmarks"].size(), static_cast<size_t>(3));
    CHECK_EQ(j["benchmarks"][0]["status"].get<std::string>(), std::string("ok"));
    CHECK_EQ(j["benchmarks"][1]["status"].get<std::string>(), std::string("failed"));
    CHECK_EQ(j["benchmarks"][2]["status"].get<std::string>(), std::string("validation_failed"));
}

// ---------------------------------------------------------------------
// Legacy normalization / compatibility field naming.
// ---------------------------------------------------------------------

void test_json_uses_legacy_gflops_not_ambiguous_gflops_key() {
    Report report;
    report.benchmarks.push_back(
        make_entry("scalar_fma", "scalar", BenchmarkStatus::OK,
                   MetricKind::FLOATING_POINT_OPERATIONS, 244.2e9, 1000));
    auto j = floptic::report_to_json(report);
    auto& results = j["benchmarks"][0]["results"];
    CHECK_TRUE(results.contains("legacy_gflops"));
    CHECK_TRUE(!results.contains("gflops"));
}

// ---------------------------------------------------------------------
// Markdown rendering: typed metric, never category-based unit guessing.
// Deliberately mismatches category ("memory") against a floating-point
// metric kind to prove the renderer does not infer units from category.
// ---------------------------------------------------------------------

std::string read_file_contents(const std::string& path) {
    std::ifstream in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void test_markdown_uses_typed_metric_not_category_guess() {
    Report report;
    report.devices.push_back(floptic::DeviceInfo{});
    report.devices[0].id = "cpu:0";
    report.devices[0].name = "Test CPU";
    report.devices[0].type = "cpu";

    // category is "memory" but the typed metric kind is floating-point —
    // a category-based guess would render this as GB/s; the typed-metric
    // renderer must render it as FLOP/s regardless of category.
    report.benchmarks.push_back(
        make_entry("mismatched_kernel", "memory", BenchmarkStatus::OK,
                   MetricKind::FLOATING_POINT_OPERATIONS, 244.2e9, 1000));

    std::string path = "test_markdown_typed_metric_output.md";
    floptic::write_markdown_report(report, path);
    std::string content = read_file_contents(path);
    std::remove(path.c_str());

    CHECK_TRUE(content.find("FLOP/s") != std::string::npos);
    CHECK_TRUE(content.find("GB/s") == std::string::npos);
}

void test_markdown_renders_non_ok_status_explicitly() {
    Report report;
    report.devices.push_back(floptic::DeviceInfo{});
    report.devices[0].id = "cpu:0";
    report.devices[0].name = "Test CPU";
    report.devices[0].type = "cpu";

    report.benchmarks.push_back(
        make_entry("broken_kernel", "scalar", BenchmarkStatus::FAILED,
                   MetricKind::FLOATING_POINT_OPERATIONS, 0.0, 0));

    std::string path = "test_markdown_status_output.md";
    floptic::write_markdown_report(report, path);
    std::string content = read_file_contents(path);
    std::remove(path.c_str());

    CHECK_TRUE(content.find("failed") != std::string::npos);
    // Must not silently render a failed kernel as zero throughput.
    CHECK_TRUE(content.find("0.0 FLOP/s") == std::string::npos);
}

} // namespace

int main() {
    test_json_report_declares_schema_version_2();
    test_json_serializes_floating_point_metric();
    test_json_serializes_integer_operations_metric();
    test_json_serializes_transferred_bytes_metric();
    test_json_retains_failed_and_validation_failed_entries();
    test_json_uses_legacy_gflops_not_ambiguous_gflops_key();
    test_markdown_uses_typed_metric_not_category_guess();
    test_markdown_renders_non_ok_status_explicitly();
    return floptic_test::finish();
}
