#pragma once
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include "floptic/precision.hpp"
#include "floptic/device_info.hpp"

namespace floptic {

struct KernelConfig {
    Precision precision = Precision::FP64;
    std::string mode = "throughput";    // "throughput" or "latency"
    int64_t iterations = 100000;        // inner loop iterations
    int threads = 0;                    // 0 = auto-detect
    std::string device_id;              // target device
};

struct KernelResult {
    // Timing
    double median_time_ms = 0.0;
    double min_time_ms = 0.0;
    double max_time_ms = 0.0;

    // Performance
    double gflops = 0.0;
    double effective_gflops = 0.0;  // same as gflops for native precisions
    double peak_percent = 0.0;
    int64_t total_flops = 0;

    // Environment (best-effort)
    double clock_mhz = 0.0;
    double power_watts = 0.0;
    double gflops_per_watt = 0.0;

    // Accuracy (for emulated kernels)
    bool accuracy_measured = false;
    double max_ulp_error = 0.0;
    double sig_digits = 0.0;
};

class KernelBase {
public:
    virtual ~KernelBase() = default;

    virtual std::string name() const = 0;
    virtual std::string category() const = 0;
    virtual std::string backend() const = 0;    // "cpu", "cuda", "hip"
    virtual std::vector<Precision> supported_precisions() const = 0;
    virtual std::vector<std::string> supported_modes() const = 0;

    // Run the benchmark with given config, repeating for measurement_trials
    virtual KernelResult run(const KernelConfig& config,
                             const DeviceInfo& device,
                             int measurement_trials = 100) = 0;

    bool supports_precision(Precision p) const {
        for (auto& sp : supported_precisions())
            if (sp == p) return true;
        return false;
    }

    bool supports_device(const DeviceInfo& dev) const {
        return dev.type == "gpu" ? (backend() == "cuda" || backend() == "hip")
                                 : (backend() == "cpu");
    }
};

} // namespace floptic
