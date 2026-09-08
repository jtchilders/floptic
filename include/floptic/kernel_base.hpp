#pragma once
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include "floptic/precision.hpp"
#include "floptic/device_info.hpp"
#include "floptic/benchmark_status.hpp"
#include "floptic/typed_metric.hpp"

namespace floptic {

struct KernelConfig {
    Precision precision = Precision::FP64;
    std::string mode = "throughput";    // "throughput" or "latency"
    int64_t iterations = 100000;        // inner loop iterations
    int threads = 0;                    // 0 = auto-detect (CPU: all cores, GPU: all SMs)
    int warmup_trials = 10;             // warmup iterations before/at each timed measurement;
                                         // 0 must perform no explicit warmup launches

    // GPU-specific launch config (0 = auto)
    int gpu_blocks = 0;                 // total thread blocks
    int gpu_threads_per_block = 256;    // threads per block
    int gpu_blocks_per_sm = 4;          // blocks per SM (used when gpu_blocks=0)

    std::string device_id;              // target device
};

struct KernelResult {
    // Explicit benchmark outcome. Everything below is only meaningful when
    // status == OK; other statuses render explicitly rather than as zero
    // performance (see benchmark_status.hpp).
    //
    // Defaults to a non-success state (FAILED) rather than OK. Status must
    // be authoritative and never inferred from a nonpositive legacy rate —
    // see floptic/dispatch_normalize.hpp, which is the single place that
    // promotes an unmigrated kernel's default-FAILED result to OK once it
    // has produced a valid nonzero legacy/typed measurement. A kernel that
    // sets its own explicit status (e.g. VALIDATION_FAILED) is left as-is.
    BenchmarkStatus status = BenchmarkStatus::FAILED;

    // Timing
    double median_time_ms = 0.0;
    double min_time_ms = 0.0;
    double max_time_ms = 0.0;

    // Typed primary metric (schema v2). This is the authoritative
    // performance representation; gflops/effective_gflops below remain
    // only as deprecated in-memory compatibility fields for callers that
    // have not migrated in this card.
    TypedMetric metric;

    // Performance (DEPRECATED: retained temporarily as in-memory
    // compatibility fields so existing kernels do not need per-kernel
    // migration in this card. Do not add new readers of these — read
    // `metric` instead. Serializers must emit these only as a clearly
    // named legacy field (`legacy_gflops`), never as an authoritative
    // `results.gflops`.)
    double gflops = 0.0;
    double effective_gflops = 0.0;  // same as gflops for native precisions
    double peak_percent = 0.0;
    int64_t total_flops = 0;
    // Explicit transferred-byte count for memory-bandwidth kernels
    // (category "memory"). Populated by the kernel itself alongside the
    // legacy `gflops` field, which such kernels repurpose to hold GB/s.
    // Read by floptic/dispatch_normalize.hpp when the inferred metric kind
    // is TRANSFERRED_BYTES — total_flops is never reused as a byte count.
    int64_t total_bytes = 0;

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
