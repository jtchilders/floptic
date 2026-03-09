#include "floptic/device_info.hpp"

#ifdef FLOPTIC_HAS_CUDA
#include <cuda_runtime.h>
#include <iostream>

namespace floptic {

// FP64 FLOPs per SM per clock for known architectures
static double fp64_ops_per_sm_per_clock(int major, int minor) {
    // Values are FMA ops (each = 2 FLOPs), so multiply by 2 for FLOPs
    switch (major) {
        case 7: // Volta / Turing
            if (minor == 0) return 32 * 2;  // V100: 32 FP64 FMA/SM/clk
            return 2 * 2;                    // Turing: 2 FP64 FMA/SM/clk
        case 8: // Ampere / Ada
            if (minor == 0) return 32 * 2;  // A100: 32 FP64 FMA/SM/clk
            if (minor == 6) return 2 * 2;   // RTX 3050 etc
            if (minor == 9) return 1 * 2;   // RTX 4090: 1 FP64 per 64 FP32 (approx)
            return 32 * 2;                   // default Ampere datacenter
        case 9: // Hopper
            return 32 * 2;                   // H100: 32 FP64 FMA/SM/clk
        case 10: // Blackwell
            return 32 * 2;                   // B200: estimated
        default:
            return 4 * 2;                    // conservative fallback
    }
}

static double fp32_ops_per_sm_per_clock(int major, int minor) {
    switch (major) {
        case 7:
            if (minor == 0) return 64 * 2;   // V100
            return 64 * 2;                    // Turing
        case 8:
            if (minor == 0) return 64 * 2;   // A100
            return 128 * 2;                   // Ada Lovelace
        case 9:
            return 128 * 2;                   // H100
        case 10:
            return 128 * 2;                   // B200 estimated
        default:
            return 64 * 2;
    }
}

std::vector<DeviceInfo> discover_cuda_devices() {
    std::vector<DeviceInfo> devices;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return devices;
    }

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);

        DeviceInfo dev;
        dev.id = "cuda:" + std::to_string(i);
        dev.name = props.name;
        dev.vendor = "nvidia";
        dev.arch = "sm_" + std::to_string(props.major * 10 + props.minor);
        dev.type = "gpu";
        dev.memory_bytes = props.totalGlobalMem;
        dev.compute_units = props.multiProcessorCount;
        dev.clock_mhz = props.clockRate / 1000;         // clockRate is in kHz
        dev.boost_clock_mhz = props.clockRate / 1000;   // same (boost by default)

        // Supported precisions
        dev.supported_precisions = {Precision::FP64, Precision::FP32};
        // FP16 supported on compute >= 5.3, but we start with FP64/FP32

        // Features based on compute capability
        if (props.major >= 7) {
            dev.features.push_back(Feature::TENSOR_CORES);
        }
        if (props.major >= 8 && props.minor == 0) {
            dev.features.push_back(Feature::FP64_TENSOR);
            dev.features.push_back(Feature::STRUCTURED_SPARSITY);
        }
        dev.features.push_back(Feature::FMA_HW);

        // Theoretical peak GFLOP/s
        double clock_ghz = dev.boost_clock_mhz / 1000.0;
        int sms = dev.compute_units;

        double fp64_gflops = sms * clock_ghz * fp64_ops_per_sm_per_clock(props.major, props.minor);
        double fp32_gflops = sms * clock_ghz * fp32_ops_per_sm_per_clock(props.major, props.minor);

        dev.theoretical_peak_gflops["FP64"] = fp64_gflops;
        dev.theoretical_peak_gflops["FP32"] = fp32_gflops;

        devices.push_back(dev);
    }

    return devices;
}

} // namespace floptic

#endif // FLOPTIC_HAS_CUDA
