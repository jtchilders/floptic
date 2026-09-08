// Regression test for warmup propagation: KernelConfig::warmup_trials must
// carry the CLI --warmup value through to whatever a kernel implementation
// uses for its warmup loop bound. This is a hardware-independent test double
// standing in for the real CPU/CUDA/HIP kernels (which require a live
// device to run() against) — it must not touch device discovery.
#include "floptic/kernel_base.hpp"
#include "check.hpp"

using floptic::DeviceInfo;
using floptic::KernelBase;
using floptic::KernelConfig;
using floptic::KernelResult;
using floptic::Precision;

namespace {

// A minimal test-double kernel whose run() records how many times its
// warmup body executed, driven entirely by config.warmup_trials — mirroring
// the "for (int w = 0; w < config.warmup_trials; w++)" pattern used by every
// real kernel implementation after this change.
class WarmupRecordingKernel : public KernelBase {
public:
    mutable int last_warmup_calls = -1;

    std::string name() const override { return "warmup_test_double"; }
    std::string category() const override { return "scalar"; }
    std::string backend() const override { return "cpu"; }
    std::vector<Precision> supported_precisions() const override { return {Precision::FP64}; }
    std::vector<std::string> supported_modes() const override { return {"throughput"}; }

    KernelResult run(const KernelConfig& config,
                     const DeviceInfo& /*device*/,
                     int /*measurement_trials*/) override {
        int warmup_calls = 0;
        for (int w = 0; w < config.warmup_trials; w++) {
            warmup_calls++;
        }
        last_warmup_calls = warmup_calls;

        KernelResult result;
        result.gflops = 1.0;  // nonzero so main.cpp would not treat this as skipped
        return result;
    }
};

void test_default_warmup_trials_is_ten() {
    KernelConfig config;
    CHECK_EQ(config.warmup_trials, 10);
}

void test_warmup_trials_propagates_to_kernel_run() {
    WarmupRecordingKernel kernel;
    DeviceInfo device;
    KernelConfig config;
    config.warmup_trials = 7;

    kernel.run(config, device, /*measurement_trials=*/1);
    CHECK_EQ(kernel.last_warmup_calls, 7);
}

void test_warmup_trials_zero_performs_no_warmup_launches() {
    WarmupRecordingKernel kernel;
    DeviceInfo device;
    KernelConfig config;
    config.warmup_trials = 0;

    kernel.run(config, device, /*measurement_trials=*/1);
    CHECK_EQ(kernel.last_warmup_calls, 0);
}

} // namespace

int main() {
    test_default_warmup_trials_is_ten();
    test_warmup_trials_propagates_to_kernel_run();
    test_warmup_trials_zero_performs_no_warmup_launches();
    return floptic_test::finish();
}
