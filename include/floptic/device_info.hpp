#pragma once
#include <string>
#include <vector>
#include <map>
#include "floptic/precision.hpp"

namespace floptic {

enum class Feature {
    TENSOR_CORES,
    FP64_TENSOR,
    STRUCTURED_SPARSITY,
    MFMA,
    AMX_BF16,
    AMX_INT8,
    AVX2,
    AVX512,
    AVX512_FP16,
    FMA_HW
};

inline std::string feature_to_string(Feature f) {
    switch (f) {
        case Feature::TENSOR_CORES:       return "tensor_cores";
        case Feature::FP64_TENSOR:        return "fp64_tensor";
        case Feature::STRUCTURED_SPARSITY:return "structured_sparsity";
        case Feature::MFMA:               return "mfma";
        case Feature::AMX_BF16:           return "amx_bf16";
        case Feature::AMX_INT8:           return "amx_int8";
        case Feature::AVX2:               return "avx2";
        case Feature::AVX512:             return "avx512";
        case Feature::AVX512_FP16:        return "avx512_fp16";
        case Feature::FMA_HW:             return "fma_hw";
    }
    return "unknown";
}

struct DeviceInfo {
    std::string id;             // "cuda:0", "cpu:0", "hip:0"
    std::string name;           // "NVIDIA A100-SXM4-80GB"
    std::string vendor;         // "nvidia", "amd", "intel", "apple"
    std::string arch;           // "sm_80", "gfx942", "sapphire_rapids"
    std::string type;           // "gpu", "cpu"
    size_t memory_bytes = 0;
    int compute_units = 0;      // SMs, CUs, logical cores
    int physical_cores = 0;     // CPU: physical cores (0 if unknown)
    int threads_per_core = 1;   // CPU: SMT/HT threads per core
    int clock_mhz = 0;
    int boost_clock_mhz = 0;
    std::vector<Precision> supported_precisions;
    std::vector<Feature> features;
    std::map<std::string, double> theoretical_peak_gflops;

    bool has_feature(Feature f) const {
        for (auto& feat : features)
            if (feat == f) return true;
        return false;
    }

    bool supports_precision(Precision p) const {
        for (auto& sp : supported_precisions)
            if (sp == p) return true;
        return false;
    }
};

// Discover all available devices on this system
std::vector<DeviceInfo> discover_devices();

// Backend-specific discovery (defined in each backend)
std::vector<DeviceInfo> discover_cpu_devices();

#ifdef FLOPTIC_HAS_CUDA
std::vector<DeviceInfo> discover_cuda_devices();
#endif

#ifdef FLOPTIC_HAS_HIP
std::vector<DeviceInfo> discover_hip_devices();
#endif

} // namespace floptic
