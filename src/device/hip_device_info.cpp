#include "floptic/device_info.hpp"

#ifdef FLOPTIC_HAS_HIP
#include <hip/hip_runtime.h>
#include <iostream>

namespace floptic {

// ============================================================================
// Per-architecture FLOP/CU/clock tables for AMD CDNA GPUs
//
// All rates are in FLOP/CU/clock (already counting FMA as 2 FLOP).
// Source: AMD GPUOpen "AMD matrix cores" lab note, AMD datasheets.
//   https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
//
// CRITICAL: These values should NOT be multiplied by 2 again when computing
// peak GFLOP/s. Formula: peak = CUs × clock_GHz × flops_per_cu_per_clock.
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

// ============================================================================
// Vector (FMA) unit FLOP/CU/clock — already includes ×2 for FMA
// ============================================================================

// FP64 vector FLOP/CU/clock
static double fp64_flops_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 1: return 64;   // MI100: 32 FP64 FMA/CU/clk × 2 = 64 FLOP
        case 2: return 128;  // MI250X: full-rate FP64
        case 3: return 128;  // MI300X: full-rate FP64
        default: return 64;
    }
}

// FP32 vector FLOP/CU/clock
static double fp32_flops_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 1: return 128;  // MI100: 64 FP32 FMA/CU/clk × 2 = 128 FLOP
        case 2: return 256;  // MI250X: packed FP32 (v_pk_fma_f32) = 2× CDNA1
        case 3: return 256;  // MI300X: packed FP32
        default: return 128;
    }
}

// ============================================================================
// Matrix core (MFMA) FLOP/CU/clock — already includes ×2 for FMA
// Source: AMD GPUOpen table "Flops/Clock/CU"
// ============================================================================

// FP64 matrix FLOP/CU/clock
static double mfma_fp64_flops_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 1: return 0;    // MI100: NO FP64 matrix cores!
        case 2: return 256;  // MI250X: FP64 MFMA supported
        case 3: return 256;  // MI300X: same per CU as CDNA2
        default: return 0;
    }
}

// FP32 matrix FLOP/CU/clock
static double mfma_fp32_flops_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 1: return 256;  // MI100
        case 2: return 256;  // MI250X
        case 3: return 256;  // MI300X
        default: return 0;
    }
}

// FP16 matrix FLOP/CU/clock
static double mfma_fp16_flops_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 1: return 1024; // MI100
        case 2: return 1024; // MI250X
        case 3: return 2048; // MI300X: 2× CDNA2 per CU
        default: return 0;
    }
}

// BF16 matrix FLOP/CU/clock
static double mfma_bf16_flops_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 1: return 512;  // MI100: HALF the rate of FP16!
        case 2: return 1024; // MI250X: same as FP16
        case 3: return 2048; // MI300X: same as FP16
        default: return 0;
    }
}

// TF32 matrix FLOP/CU/clock (CDNA3 only)
static double mfma_tf32_flops_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 3: return 1024; // MI300X: new in CDNA3
        default: return 0;   // Not available on CDNA1/CDNA2
    }
}

// INT8 matrix OPS/CU/clock
static double mfma_int8_ops_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 1: return 1024; // MI100
        case 2: return 1024; // MI250X
        case 3: return 4096; // MI300X: 4× CDNA2 (same as FP8, per AMD datasheet: 2.6 POPs)
        default: return 0;
    }
}

// FP8 matrix FLOP/CU/clock (CDNA3 only)
static double mfma_fp8_flops_per_cu_per_clock(int cdna_gen) {
    switch (cdna_gen) {
        case 3: return 4096; // MI300X
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
        // All rates are already FLOP/CU/clock (FMA counted as 2 FLOP)
        // Peak GFLOP/s = CUs × clock_GHz × flops_per_cu_per_clock
        // NO extra ×2 — it's already in the rate tables!
        double clock_ghz = dev.boost_clock_mhz / 1000.0;
        int cus = dev.compute_units;

        // Vector (FMA unit) peaks
        dev.theoretical_peak_gflops["FP64"] = cus * clock_ghz * fp64_flops_per_cu_per_clock(cdna_gen);
        dev.theoretical_peak_gflops["FP32"] = cus * clock_ghz * fp32_flops_per_cu_per_clock(cdna_gen);
        // FP16/BF16 vector: packed operations, 2× FP32 rate on all CDNA gens
        dev.theoretical_peak_gflops["FP16"] = dev.theoretical_peak_gflops["FP32"] * 2.0;
        dev.theoretical_peak_gflops["BF16"] = dev.theoretical_peak_gflops["FP32"] * 2.0;

        // HBM bandwidth peak
        // hipDeviceProp reports memoryClockRate (kHz) and memoryBusWidth (bits)
        // but the effective multiplier varies by HBM generation:
        //   HBM2:  ×2 (DDR)
        //   HBM2e: ×2 (DDR)
        //   HBM3:  ×4 (QDR — quad data rate)
        // Use per-generation multipliers for correct bandwidth.
        double mem_clock_ghz = (props.memoryClockRate / 1e6);  // kHz to GHz
        double mem_bus_bytes = props.memoryBusWidth / 8.0;
        int hbm_multiplier = (cdna_gen >= 3) ? 4 : 2;  // CDNA3 (MI300) uses HBM3
        double hbm_bw_gbs = mem_clock_ghz * mem_bus_bytes * hbm_multiplier;
        if (hbm_bw_gbs > 0) {
            dev.theoretical_peak_gflops["HBM_BW"] = hbm_bw_gbs;  // GB/s stored in same map
        }

        // Matrix core (MFMA) peaks — only add entries that exist for this arch
        double mfma_fp64 = mfma_fp64_flops_per_cu_per_clock(cdna_gen);
        if (mfma_fp64 > 0)
            dev.theoretical_peak_gflops["FP64_MFMA"] = cus * clock_ghz * mfma_fp64;

        double mfma_fp32 = mfma_fp32_flops_per_cu_per_clock(cdna_gen);
        if (mfma_fp32 > 0)
            dev.theoretical_peak_gflops["FP32_MFMA"] = cus * clock_ghz * mfma_fp32;

        double mfma_fp16 = mfma_fp16_flops_per_cu_per_clock(cdna_gen);
        if (mfma_fp16 > 0)
            dev.theoretical_peak_gflops["FP16_MFMA"] = cus * clock_ghz * mfma_fp16;

        double mfma_bf16 = mfma_bf16_flops_per_cu_per_clock(cdna_gen);
        if (mfma_bf16 > 0)
            dev.theoretical_peak_gflops["BF16_MFMA"] = cus * clock_ghz * mfma_bf16;

        double mfma_tf32 = mfma_tf32_flops_per_cu_per_clock(cdna_gen);
        if (mfma_tf32 > 0)
            dev.theoretical_peak_gflops["TF32_MFMA"] = cus * clock_ghz * mfma_tf32;

        double mfma_int8 = mfma_int8_ops_per_cu_per_clock(cdna_gen);
        if (mfma_int8 > 0)
            dev.theoretical_peak_gflops["INT8_MFMA"] = cus * clock_ghz * mfma_int8;

        double mfma_fp8 = mfma_fp8_flops_per_cu_per_clock(cdna_gen);
        if (mfma_fp8 > 0)
            dev.theoretical_peak_gflops["FP8_MFMA"] = cus * clock_ghz * mfma_fp8;

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
