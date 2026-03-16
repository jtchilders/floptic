#include "floptic/device_info.hpp"

#ifdef FLOPTIC_HAS_CUDA
#include <cuda_runtime.h>
#include <iostream>

namespace floptic {

// ============================================================================
// Per-architecture ops/SM/clock tables
// Values are FMA units per SM (each FMA = 2 FLOPs), so final FLOPs = units × 2
// ============================================================================

static double fp64_fma_per_sm_per_clock(int major, int minor) {
    switch (major) {
        case 7: // Volta / Turing
            if (minor == 0) return 32;  // V100: 32 FP64 FMA/SM/clk
            return 2;                    // Turing: 1/32 of FP32
        case 8: // Ampere / Ada
            if (minor == 0) return 32;  // A100: 32 FP64 FMA/SM/clk
            if (minor == 6) return 2;   // RTX 3050 etc
            if (minor == 9) return 1;   // RTX 4090: 1/64 of FP32
            return 32;                   // default Ampere datacenter
        case 9:  return 64;             // H100: 64 FP64 FMA/SM/clk (4th gen tensor arch)
        case 10: return 64;             // B200: 64 FP64 FMA/SM/clk (HGX 1000W: 37 TF/s)
        default: return 4;              // conservative
    }
}

static double fp32_fma_per_sm_per_clock(int major, int minor) {
    switch (major) {
        case 7:
            if (minor == 0) return 64;   // V100
            return 64;                    // Turing
        case 8:
            if (minor == 0) return 64;   // A100
            return 128;                   // Ada Lovelace
        case 9:  return 128;             // H100
        case 10: return 128;             // B200 estimated
        default: return 64;
    }
}

// FP16 on CUDA cores (not tensor cores)
// On most NVIDIA GPUs, FP16 CUDA core throughput = 2× FP32
static double fp16_fma_per_sm_per_clock(int major, int minor) {
    switch (major) {
        case 7:
            if (minor == 0) return 128;  // V100: 2× FP32 rate
            return 128;                   // Turing
        case 8:
            if (minor == 0) return 128;  // A100: 2× FP32 rate
            return 256;                   // Ada
        case 9:  return 256;             // Hopper
        case 10: return 256;             // Blackwell est.
        default: return 128;
    }
}

// BF16 on CUDA cores
// Ampere+: BF16 CUDA core rate = same as FP16 (2× FP32)
// Both FP16 and BF16 use the same half-precision FMA pipes
static double bf16_fma_per_sm_per_clock(int major, int minor) {
    switch (major) {
        case 7:  return 0;               // No BF16 on Volta/Turing CUDA cores
        case 8:
            if (minor == 0) return 128;  // A100: same as FP16 (2× FP32)
            return 256;                   // Ada
        case 9:  return 256;             // Hopper
        case 10: return 256;             // Blackwell est.
        default: return 0;
    }
}

// INT8 ops per SM per clock (CUDA cores, via DP4A-style)
static double int8_ops_per_sm_per_clock(int major, int minor) {
    switch (major) {
        case 7:
            if (minor == 0) return 256;  // V100
            return 256;                   // Turing
        case 8:
            if (minor == 0) return 256;  // A100
            return 512;                   // Ada
        case 9:  return 512;             // Hopper
        case 10: return 512;             // Blackwell est.
        default: return 256;
    }
}

// INT8 tensor core ops per SM per clock
// INT8 TC rate = 2× FP16 TC rate (each TC can do 2× as many INT8 ops)
static double tc_int8_ops_per_sm_per_clock(int major, int minor) {
    switch (major) {
        case 7:
            if (minor == 0) return 1024;  // V100: 2× FP16 TC
            return 1024;                   // Turing
        case 8:
            if (minor == 0) return 2048;  // A100: 2048 INT8 ops/SM/clk
            return 2048;                   // Ada
        case 9:  return 4096;             // H100: 4096 INT8 ops/SM/clk
        case 10: return 7737;             // B200: HGX 4500 TOPS dense @ 148 SM × 1965 MHz
        default: return 1024;
    }
}

// FP8 tensor core FMA per SM per clock (Hopper+)
// FP8 TC rate = 2× FP16 TC rate on Hopper; same as INT8 TC on Blackwell
static double tc_fp8_fma_per_sm_per_clock(int major, int minor) {
    switch (major) {
        case 9:  return 4096;             // H100: 2× FP16 TC
        case 10: return 7737;             // B200: HGX 4500 TF/s dense @ 148 SM × 1965 MHz
        default: return 0;                // No FP8 TC before Hopper
    }
}

// ============================================================================
// Tensor core theoretical peaks (GEMM-specific, per SM per clock)
// These are for reference in the device info; used by matrix kernels
// ============================================================================

static double tc_fp16_fma_per_sm_per_clock(int major, int minor) {
    switch (major) {
        case 7:
            if (minor == 0) return 512;   // V100
            return 512;                    // Turing
        case 8:
            if (minor == 0) return 1024;  // A100
            return 1024;                   // Ada (with sparsity: 2048)
        case 9:  return 2048;             // H100
        case 10: return 3868;             // B200: HGX 2250 TF/s dense @ 148 SM × 1965 MHz
        default: return 512;
    }
}

// TF32 tensor core: FP32 I/O, TF32 internal (19-bit mantissa)
// Rate is typically half of FP16 TC rate
static double tc_tf32_fma_per_sm_per_clock(int major, int minor) {
    switch (major) {
        case 7:  return 0;               // No TF32 on Volta/Turing
        case 8:
            if (minor == 0) return 512;  // A100: 512 TF32 FMA/SM/clk (half of FP16 TC)
            return 512;                   // Ada
        case 9:  return 1024;            // H100: 1024 TF32 FMA/SM/clk
        case 10: return 1934;            // B200: HGX 1125 TF/s dense @ 148 SM × 1965 MHz
        default: return 0;
    }
}

static double tc_fp64_fma_per_sm_per_clock(int major, int minor) {
    switch (major) {
        case 8:
            if (minor == 0) return 64;   // A100: 64 FP64 TC FMA/SM/clk
            return 0;
        case 9:  return 128;             // H100: 128 FP64 TC FMA/SM/clk (~67 TFLOP/s)
        case 10: return 0;              // B200: NO FP64 tensor cores (FP64 TC = CUDA core rate)
        default: return 0;               // No FP64 tensor on others
    }
}

// ============================================================================
// Device discovery
// ============================================================================

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
        // clockRate was removed from cudaDeviceProp in CUDA 13.
        // Use cudaDeviceGetAttribute instead (works on all CUDA versions).
        {
            int clockKHz = 0;
            cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, i);
            dev.clock_mhz = clockKHz / 1000;
            dev.boost_clock_mhz = clockKHz / 1000;
        }

        // --- Supported precisions ---
        dev.supported_precisions = {Precision::FP64, Precision::FP32};

        // FP16 on CUDA cores: compute >= 5.3 (practical usage from 7.0+)
        if (props.major >= 7) {
            dev.supported_precisions.push_back(Precision::FP16);
        }

        // BF16 on CUDA cores: compute >= 8.0
        if (props.major >= 8) {
            dev.supported_precisions.push_back(Precision::BF16);
        }

        // TF32: tensor cores via cuBLAS (Ampere+)
        if (props.major >= 8) {
            dev.supported_precisions.push_back(Precision::TF32);
        }

        // INT8: tensor cores from Volta+
        if (props.major >= 7) {
            dev.supported_precisions.push_back(Precision::INT8);
        }

        // FP8 E4M3/E5M2: Hopper+ tensor cores (sm_89/sm_90+)
        if (props.major >= 9 || (props.major == 8 && props.minor == 9)) {
            dev.supported_precisions.push_back(Precision::FP8_E4M3);
            dev.supported_precisions.push_back(Precision::FP8_E5M2);
        }

        // --- Features ---
        dev.features.push_back(Feature::FMA_HW);

        if (props.major >= 7) {
            dev.features.push_back(Feature::TENSOR_CORES);
        }
        if ((props.major == 8 && props.minor == 0) || props.major == 9) {
            dev.features.push_back(Feature::FP64_TENSOR);
        }
        // Note: Blackwell (sm_100) does NOT have FP64 tensor cores
        if (props.major >= 8) {
            dev.features.push_back(Feature::STRUCTURED_SPARSITY);
        }

        // --- Theoretical peak GFLOP/s ---
        double clock_ghz = dev.boost_clock_mhz / 1000.0;
        int sms = dev.compute_units;

        // CUDA core peaks (scalar/vector operations)
        dev.theoretical_peak_gflops["FP64"] = sms * clock_ghz * fp64_fma_per_sm_per_clock(props.major, props.minor) * 2.0;
        dev.theoretical_peak_gflops["FP32"] = sms * clock_ghz * fp32_fma_per_sm_per_clock(props.major, props.minor) * 2.0;

        double fp16_rate = fp16_fma_per_sm_per_clock(props.major, props.minor);
        if (fp16_rate > 0)
            dev.theoretical_peak_gflops["FP16"] = sms * clock_ghz * fp16_rate * 2.0;

        double bf16_rate = bf16_fma_per_sm_per_clock(props.major, props.minor);
        if (bf16_rate > 0)
            dev.theoretical_peak_gflops["BF16"] = sms * clock_ghz * bf16_rate * 2.0;

        // Tensor core peaks (for matrix operations)
        double tc_fp16 = tc_fp16_fma_per_sm_per_clock(props.major, props.minor);
        if (tc_fp16 > 0)
            dev.theoretical_peak_gflops["FP16_TC"] = sms * clock_ghz * tc_fp16 * 2.0;

        double tc_tf32 = tc_tf32_fma_per_sm_per_clock(props.major, props.minor);
        if (tc_tf32 > 0)
            dev.theoretical_peak_gflops["TF32_TC"] = sms * clock_ghz * tc_tf32 * 2.0;

        double tc_fp64 = tc_fp64_fma_per_sm_per_clock(props.major, props.minor);
        if (tc_fp64 > 0)
            dev.theoretical_peak_gflops["FP64_TC"] = sms * clock_ghz * tc_fp64 * 2.0;

        double tc_int8 = tc_int8_ops_per_sm_per_clock(props.major, props.minor);
        if (tc_int8 > 0)
            dev.theoretical_peak_gflops["INT8_TC"] = sms * clock_ghz * tc_int8 * 2.0;

        double tc_fp8 = tc_fp8_fma_per_sm_per_clock(props.major, props.minor);
        if (tc_fp8 > 0)
            dev.theoretical_peak_gflops["FP8_TC"] = sms * clock_ghz * tc_fp8 * 2.0;

        // Print summary
        std::cerr << "  CUDA " << dev.id << " (" << dev.name << "):" << std::endl;
        std::cerr << "    Arch: " << dev.arch << ", SMs: " << sms
                  << ", Clock: " << dev.boost_clock_mhz << " MHz" << std::endl;
        std::cerr << "    Theoretical peaks (CUDA cores):" << std::endl;
        for (auto& [key, val] : dev.theoretical_peak_gflops) {
            if (key.find("TC") == std::string::npos)
                std::cerr << "      " << key << ": " << val << " GFLOP/s" << std::endl;
        }
        std::cerr << "    Theoretical peaks (Tensor cores):" << std::endl;
        for (auto& [key, val] : dev.theoretical_peak_gflops) {
            if (key.find("TC") != std::string::npos)
                std::cerr << "      " << key << ": " << val << " GFLOP/s" << std::endl;
        }

        devices.push_back(dev);
    }

    return devices;
}

// ============================================================================
// Fill CUDA-specific system info
// ============================================================================

} // namespace floptic

#include "floptic/report.hpp"
#include <cublas_v2.h>

namespace floptic {

void fill_cuda_system_info(SystemInfo& info) {
    // CUDA runtime version
    int runtime_ver = 0;
    cudaRuntimeGetVersion(&runtime_ver);
    int rt_major = runtime_ver / 1000;
    int rt_minor = (runtime_ver % 1000) / 10;
    char vbuf[32];
    snprintf(vbuf, sizeof(vbuf), "%d.%d", rt_major, rt_minor);
    info.cuda_runtime_version = vbuf;

    // CUDA driver version
    int driver_ver = 0;
    cudaDriverGetVersion(&driver_ver);
    int drv_major = driver_ver / 1000;
    int drv_minor = (driver_ver % 1000) / 10;
    snprintf(vbuf, sizeof(vbuf), "%d.%d", drv_major, drv_minor);
    info.cuda_driver_version = vbuf;

    // cuBLAS version
    int major = 0, minor = 0, patch = 0;
    cublasGetProperty(MAJOR_VERSION, &major);
    cublasGetProperty(MINOR_VERSION, &minor);
    cublasGetProperty(PATCH_LEVEL, &patch);
    snprintf(vbuf, sizeof(vbuf), "%d.%d.%d", major, minor, patch);
    info.cublas_version = vbuf;
}

} // namespace floptic

#endif // FLOPTIC_HAS_CUDA
