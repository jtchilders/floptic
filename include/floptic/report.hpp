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

struct SystemInfo {
    std::string hostname;
    std::string os_name;          // e.g. "Linux"
    std::string os_release;       // e.g. "5.15.0-91-generic"
    std::string os_arch;          // e.g. "x86_64", "aarch64"
    std::string cpu_model;        // e.g. "Intel(R) Xeon(R) 6960P"

    // Compiler
    std::string compiler_name;    // e.g. "GCC", "Clang", "NVCC"
    std::string compiler_version; // e.g. "13.3.0"

    // CUDA-specific
    std::string cuda_runtime_version;  // e.g. "13.1"
    std::string cuda_driver_version;   // e.g. "570.86.16"
    std::string cublas_version;        // e.g. "12.9.1.4"
    std::string nvidia_smi_driver;     // full nvidia-smi driver string

    // Loaded modules / environment
    std::vector<std::string> loaded_modules;
};

// Collect system info at runtime
SystemInfo collect_system_info();

#ifdef FLOPTIC_HAS_CUDA
// Fill CUDA-specific fields (called after collect_system_info)
void fill_cuda_system_info(SystemInfo& info);
#endif

struct Report {
    std::string version = "0.1.0";
    std::string timestamp;
    SystemInfo system;
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

// Write markdown report to file
void write_markdown_report(const Report& report, const std::string& output_path);

} // namespace floptic
