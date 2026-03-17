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
    int threads = 0;                    // 0 = auto-detect (CPU: all cores, GPU: all SMs)

    // GPU-specific launch config (0 = auto)
    int gpu_blocks = 0;                 // total thread blocks
    int gpu_threads_per_block = 256;    // threads per block
    int gpu_blocks_per_sm = 4;          // blocks per SM (used when gpu_blocks=0)

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

    // Check if this kernel can run on the given device.
    // Override to add architecture-specific guards (e.g. Blackwell-only).
    // Called before run() — if false, the kernel is silently skipped.
    virtual bool is_available(const DeviceInfo& device) const { return true; }

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
