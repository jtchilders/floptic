#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "floptic/device_info.hpp"
#include "floptic/kernel_base.hpp"

namespace floptic {

struct BenchmarkEntry {
    std::string device_id;
    std::string kernel_name;
    std::string category;
    std::string precision;
    std::string mode;
    KernelConfig config;
    KernelResult result;
};

struct Report {
    std::string version = "0.1.0";
    std::string timestamp;
    std::string hostname;
    std::string os;
    std::string compiler;
    std::vector<std::string> build_backends;
    std::vector<DeviceInfo> devices;
    int iterations = 100;
    int warmup = 10;
    std::vector<BenchmarkEntry> benchmarks;
};

// Generate JSON from a report
nlohmann::json report_to_json(const Report& report);

// Write report to file or stdout
void write_json_report(const Report& report, const std::string& output_path);

} // namespace floptic
