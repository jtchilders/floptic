#include "floptic/report.hpp"
#include <iostream>
#include <fstream>
#include <ctime>
#include <unistd.h>

namespace floptic {

static std::string get_hostname() {
    char buf[256];
    if (gethostname(buf, sizeof(buf)) == 0)
        return std::string(buf);
    return "unknown";
}

static std::string get_timestamp() {
    time_t now = time(nullptr);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", localtime(&now));
    return std::string(buf);
}

nlohmann::json report_to_json(const Report& report) {
    nlohmann::json j;

    j["floptic_version"] = report.version;
    j["timestamp"] = report.timestamp.empty() ? get_timestamp() : report.timestamp;

    j["system"]["hostname"] = report.system.hostname.empty() ? get_hostname() : report.system.hostname;
    j["system"]["os_name"] = report.system.os_name;
    j["system"]["os_release"] = report.system.os_release;
    j["system"]["os_arch"] = report.system.os_arch;
    j["system"]["cpu_model"] = report.system.cpu_model;
    j["system"]["compiler_name"] = report.system.compiler_name;
    j["system"]["compiler_version"] = report.system.compiler_version;
    j["system"]["cuda_runtime_version"] = report.system.cuda_runtime_version;
    j["system"]["cuda_driver_version"] = report.system.cuda_driver_version;
    j["system"]["nvidia_smi_driver"] = report.system.nvidia_smi_driver;
    j["system"]["cublas_version"] = report.system.cublas_version;
    j["system"]["loaded_modules"] = report.system.loaded_modules;
    j["system"]["build_backends"] = report.build_backends;

    j["config"]["iterations"] = report.iterations;
    j["config"]["warmup"] = report.warmup;

    // Devices
    j["devices"] = nlohmann::json::array();
    for (auto& dev : report.devices) {
        nlohmann::json d;
        d["id"] = dev.id;
        d["name"] = dev.name;
        d["vendor"] = dev.vendor;
        d["arch"] = dev.arch;
        d["type"] = dev.type;
        d["memory_gb"] = dev.memory_bytes / (1024.0 * 1024.0 * 1024.0);
        d["compute_units"] = dev.compute_units;
        if (dev.physical_cores > 0) {
            d["physical_cores"] = dev.physical_cores;
            d["threads_per_core"] = dev.threads_per_core;
        }
        d["clock_mhz"] = dev.clock_mhz;
        d["boost_clock_mhz"] = dev.boost_clock_mhz;

        d["features"] = nlohmann::json::array();
        for (auto& f : dev.features)
            d["features"].push_back(feature_to_string(f));

        d["supported_precisions"] = nlohmann::json::array();
        for (auto& p : dev.supported_precisions)
            d["supported_precisions"].push_back(precision_to_string(p));

        d["theoretical_peak_gflops"] = dev.theoretical_peak_gflops;

        j["devices"].push_back(d);
    }

    // Benchmarks
    j["benchmarks"] = nlohmann::json::array();
    for (auto& entry : report.benchmarks) {
        nlohmann::json b;
        b["device_id"] = entry.device_id;
        b["kernel"] = entry.kernel_name;
        b["category"] = entry.category;
        b["precision"] = entry.precision;
        b["mode"] = entry.mode;

        b["results"]["gflops"] = entry.result.gflops;
        b["results"]["effective_gflops"] = entry.result.effective_gflops;
        b["results"]["peak_percent"] = entry.result.peak_percent;
        b["results"]["median_time_ms"] = entry.result.median_time_ms;
        b["results"]["min_time_ms"] = entry.result.min_time_ms;
        b["results"]["max_time_ms"] = entry.result.max_time_ms;
        b["results"]["total_flops"] = entry.result.total_flops;
        b["results"]["clock_mhz"] = entry.result.clock_mhz;

        if (entry.result.power_watts > 0) {
            b["results"]["power_watts"] = entry.result.power_watts;
            b["results"]["gflops_per_watt"] = entry.result.gflops_per_watt;
        }

        if (entry.result.accuracy_measured) {
            b["accuracy"]["max_ulp_error"] = entry.result.max_ulp_error;
            b["accuracy"]["sig_digits"] = entry.result.sig_digits;
        } else {
            b["accuracy"] = nullptr;
        }

        j["benchmarks"].push_back(b);
    }

    return j;
}

void write_json_report(const Report& report, const std::string& output_path) {
    if (output_path.empty()) {
        // No output file specified — summary table already printed to stderr
        return;
    }

    auto j = report_to_json(report);
    std::string pretty = j.dump(2);

    std::ofstream ofs(output_path);
    if (!ofs.is_open()) {
        std::cerr << "ERROR: Cannot open output file: " << output_path << std::endl;
        return;
    }
    ofs << pretty << std::endl;
}

} // namespace floptic
