#include "floptic/device_info.hpp"

#ifdef FLOPTIC_HAS_HIP
#include <hip/hip_runtime.h>
#include <iostream>

namespace floptic {

// ============================================================================
// Per-architecture ops/CU/clock tables for AMD CDNA GPUs
//
// AMD CDNA architecture uses Compute Units (CUs), each with:
//   - Vector ALUs: 64 lanes (SIMD32 × 2, or 4×SIMD16), process a 64-wide wavefront
//   - Matrix cores (MFMA): variable throughput per generation
//
// FP64 vector: 64 FMA/CU/clock on CDNA (full rate, unlike consumer GPUs)
// FP32 vector: 64 FMA/CU/clock on CDNA
// Matrix core rates vary by generation (CDNA1, CDNA2, CDNA3)
// ============================================================================

// GFX ID to CDNA generation mapping:
//   gfx908  = CDNA1 (MI100)
//   gfx90a  = CDNA2 (MI210, MI250, MI250X)
//   gfx940  = CDNA3 (MI300A)
//   gfx941  = CDNA3 (MI300X - different XCD config)
//   gfx942  = CDNA3 (MI300X/MI300A production)

static int gfx_to_cdna_gen(const std::string& gfx) {
    if (gfx == "gfx908") return 1;  // CDNA1
    if (gfx == "gfx90a") return 2;  // CDNA2
    if (gfx == "gfx940" || gfx == "gfx941" || gfx == "gfx942") return 3;  // CDNA3
    return 0;
}

// FP64 vector FMA/CU/clock (CDNA has full-rate FP64)
static double fp64_fma_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 1: return 32;   // MI100: 32 FP64 FMA/CU/clk (half-rate FP32)
        case 2: return 64;   // MI250X: 64 FP64 FMA/CU/clk (full-rate)
        case 3: return 64;   // MI300X: 64 FP64 FMA/CU/clk (full-rate)
        default: return 32;
    }
}

// FP32 vector FMA/CU/clock
static double fp32_fma_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 1: return 64;   // MI100: 64 FP32 FMA/CU/clk
        case 2: return 64;   // MI250X: 64 FP32 FMA/CU/clk
        case 3: return 64;   // MI300X: 64 FP32 FMA/CU/clk
        default: return 64;
    }
}

// FP16 vector FMA/CU/clock (packed, 2× FP32 rate)
static double fp16_fma_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 1: return 256;  // MI100: 4× FP32 rate via packed math
        case 2: return 256;  // MI250X: 4× FP32 (packed FP16 ops)
        case 3: return 256;  // MI300X: 4× FP32
        default: return 128;
    }
}

// BF16 vector FMA/CU/clock
static double bf16_fma_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 1: return 256;  // MI100: same as FP16
        case 2: return 256;  // MI250X: same as FP16
        case 3: return 256;  // MI300X: same as FP16
        default: return 0;
    }
}

// ============================================================================
// Matrix core (MFMA) rates per CU per clock
// These are the FMA operations issued by the matrix core unit per cycle.
// ============================================================================

// FP64 matrix FMA/CU/clock
static double mfma_fp64_fma_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 1: return 64;   // MI100: 2× vector rate
        case 2: return 128;  // MI250X: 2× vector rate (full MFMA FP64)
        case 3: return 128;  // MI300X: same as CDNA2 per CU
        default: return 0;
    }
}

// FP32 matrix FMA/CU/clock
static double mfma_fp32_fma_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 1: return 128;  // MI100: 2× vector rate
        case 2: return 128;  // MI250X: 2× vector
        case 3: return 128;  // MI300X: same per CU
        default: return 0;
    }
}

// FP16 matrix FMA/CU/clock
static double mfma_fp16_fma_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 1: return 512;  // MI100
        case 2: return 512;  // MI250X
        case 3: return 1024; // MI300X: 2× CDNA2 per CU
        default: return 0;
    }
}

// BF16 matrix FMA/CU/clock
static double mfma_bf16_fma_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 1: return 512;  // MI100
        case 2: return 512;  // MI250X
        case 3: return 1024; // MI300X: 2× CDNA2
        default: return 0;
    }
}

// TF32 matrix FMA/CU/clock (CDNA3 only)
static double mfma_tf32_fma_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 3: return 512;  // MI300X: new in CDNA3
        default: return 0;   // Not available on CDNA1/CDNA2
    }
}

// INT8 matrix ops/CU/clock
static double mfma_int8_ops_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 1: return 1024; // MI100
        case 2: return 1024; // MI250X
        case 3: return 2048; // MI300X: 2× CDNA2
        default: return 0;
    }
}

// FP8 matrix FMA/CU/clock (CDNA3 only)
static double mfma_fp8_fma_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 3: return 2048; // MI300X: same as INT8
        default: return 0;   // Not available before CDNA3
    }
}

// ============================================================================
// Device discovery
// ============================================================================

std::vector<DeviceInfo> discover_hip_devices() {
    std::vector<DeviceInfo> devices;

    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        return devices;
    }

    for (int i = 0; i < device_count; i++) {
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, i);

        DeviceInfo dev;
        dev.id = "hip:" + std::to_string(i);
        dev.name = props.name;
        dev.vendor = "amd";
        dev.type = "gpu";
        dev.memory_bytes = props.totalGlobalMem;

        // gcnArchName gives e.g. "gfx90a:sramecc+:xnack-"
        std::string gcn_arch = props.gcnArchName;
        // Extract just the gfx part
        auto colon = gcn_arch.find(':');
        if (colon != std::string::npos) {
            gcn_arch = gcn_arch.substr(0, colon);
        }
        dev.arch = gcn_arch;

        // MI250X has 2 GCDs, each reported as a separate HIP device
        // props.multiProcessorCount = CUs on this GCD
        dev.compute_units = props.multiProcessorCount;
        dev.clock_mhz = props.clockRate / 1000;
        dev.boost_clock_mhz = props.clockRate / 1000;

        int cdna_gen = gfx_to_cdna_gen(gcn_arch);

        // --- Supported precisions ---
        dev.supported_precisions = {Precision::FP64, Precision::FP32};

        if (cdna_gen >= 1) {
            dev.supported_precisions.push_back(Precision::FP16);
            dev.supported_precisions.push_back(Precision::BF16);
            dev.supported_precisions.push_back(Precision::INT8);
        }

        if (cdna_gen >= 3) {
            // TF32 (CDNA3), FP8 (CDNA3)
            dev.supported_precisions.push_back(Precision::TF32);
            dev.supported_precisions.push_back(Precision::FP8_E4M3);
            dev.supported_precisions.push_back(Precision::FP8_E5M2);
        }

        // --- Features ---
        dev.features.push_back(Feature::FMA_HW);
        if (cdna_gen >= 1) {
            dev.features.push_back(Feature::MATRIX_CORES);  // MFMA
            dev.features.push_back(Feature::FP64_MATRIX);
        }

        // --- Theoretical peaks ---
        double clock_ghz = dev.boost_clock_mhz / 1000.0;
        int cus = dev.compute_units;

        // Vector (CUDA-core equivalent) peaks
        dev.theoretical_peak_gflops["FP64"] = cus * clock_ghz * fp64_fma_per_cu_per_clock(cdna_gen) * 2.0;
        dev.theoretical_peak_gflops["FP32"] = cus * clock_ghz * fp32_fma_per_cu_per_clock(cdna_gen) * 2.0;

        double fp16_rate = fp16_fma_per_cu_per_clock(cdna_gen);
        if (fp16_rate > 0)
            dev.theoretical_peak_gflops["FP16"] = cus * clock_ghz * fp16_rate * 2.0;

        double bf16_rate = bf16_fma_per_cu_per_clock(cdna_gen);
        if (bf16_rate > 0)
            dev.theoretical_peak_gflops["BF16"] = cus * clock_ghz * bf16_rate * 2.0;

        // Matrix core (MFMA) peaks
        double mfma_fp64 = mfma_fp64_fma_per_cu_per_clock(cdna_gen);
        if (mfma_fp64 > 0)
            dev.theoretical_peak_gflops["FP64_MFMA"] = cus * clock_ghz * mfma_fp64 * 2.0;

        double mfma_fp32 = mfma_fp32_fma_per_cu_per_clock(cdna_gen);
        if (mfma_fp32 > 0)
            dev.theoretical_peak_gflops["FP32_MFMA"] = cus * clock_ghz * mfma_fp32 * 2.0;

        double mfma_fp16 = mfma_fp16_fma_per_cu_per_clock(cdna_gen);
        if (mfma_fp16 > 0)
            dev.theoretical_peak_gflops["FP16_MFMA"] = cus * clock_ghz * mfma_fp16 * 2.0;

        double mfma_bf16 = mfma_bf16_fma_per_cu_per_clock(cdna_gen);
        if (mfma_bf16 > 0)
            dev.theoretical_peak_gflops["BF16_MFMA"] = cus * clock_ghz * mfma_bf16 * 2.0;

        double mfma_tf32 = mfma_tf32_fma_per_cu_per_clock(cdna_gen);
        if (mfma_tf32 > 0)
            dev.theoretical_peak_gflops["TF32_MFMA"] = cus * clock_ghz * mfma_tf32 * 2.0;

        double mfma_int8 = mfma_int8_ops_per_cu_per_clock(cdna_gen);
        if (mfma_int8 > 0)
            dev.theoretical_peak_gflops["INT8_MFMA"] = cus * clock_ghz * mfma_int8 * 2.0;

        double mfma_fp8 = mfma_fp8_fma_per_cu_per_clock(cdna_gen);
        if (mfma_fp8 > 0)
            dev.theoretical_peak_gflops["FP8_MFMA"] = cus * clock_ghz * mfma_fp8 * 2.0;

        // Print summary
        std::cerr << "  HIP " << dev.id << " (" << dev.name << "):" << std::endl;
        std::cerr << "    Arch: " << dev.arch << " (CDNA" << cdna_gen << "), CUs: " << cus
                  << ", Clock: " << dev.boost_clock_mhz << " MHz" << std::endl;
        std::cerr << "    Theoretical peaks (vector):" << std::endl;
        for (auto& [key, val] : dev.theoretical_peak_gflops) {
            if (key.find("MFMA") == std::string::npos)
                std::cerr << "      " << key << ": " << val << " GFLOP/s" << std::endl;
        }
        std::cerr << "    Theoretical peaks (matrix cores):" << std::endl;
        for (auto& [key, val] : dev.theoretical_peak_gflops) {
            if (key.find("MFMA") != std::string::npos)
                std::cerr << "      " << key << ": " << val << " GFLOP/s" << std::endl;
        }

        devices.push_back(dev);
    }

    return devices;
}

} // namespace floptic

#endif // FLOPTIC_HAS_HIP
