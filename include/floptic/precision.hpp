#pragma once
#include <string>
#include <vector>
#include <cstddef>

namespace floptic {

enum class Precision {
    FP64,
    FP32,
    FP16,
    BF16,
    TF32,
    FP8_E4M3,
    FP8_E5M2,
    FP4,
    INT8,
    INT4
};

inline std::string precision_to_string(Precision p) {
    switch (p) {
        case Precision::FP64:     return "FP64";
        case Precision::FP32:     return "FP32";
        case Precision::FP16:     return "FP16";
        case Precision::BF16:     return "BF16";
        case Precision::TF32:     return "TF32";
        case Precision::FP8_E4M3: return "FP8_E4M3";
        case Precision::FP8_E5M2: return "FP8_E5M2";
        case Precision::FP4:      return "FP4";
        case Precision::INT8:     return "INT8";
        case Precision::INT4:     return "INT4";
    }
    return "UNKNOWN";
}

inline Precision string_to_precision(const std::string& s) {
    if (s == "fp64" || s == "FP64") return Precision::FP64;
    if (s == "fp32" || s == "FP32") return Precision::FP32;
    if (s == "fp16" || s == "FP16") return Precision::FP16;
    if (s == "bf16" || s == "BF16") return Precision::BF16;
    if (s == "tf32" || s == "TF32") return Precision::TF32;
    if (s == "fp8e4m3" || s == "FP8_E4M3") return Precision::FP8_E4M3;
    if (s == "fp8e5m2" || s == "FP8_E5M2") return Precision::FP8_E5M2;
    if (s == "fp4" || s == "FP4") return Precision::FP4;
    if (s == "int8" || s == "INT8") return Precision::INT8;
    if (s == "int4" || s == "INT4") return Precision::INT4;
    return Precision::FP64; // default
}

inline std::vector<Precision> all_standard_precisions() {
    return { Precision::FP64, Precision::FP32 };
    // Expand as backends support more types
}

// Type mapping for compile-time dispatch
template <Precision P> struct PrecisionType;
template <> struct PrecisionType<Precision::FP64> { using type = double; };
template <> struct PrecisionType<Precision::FP32> { using type = float; };

template <Precision P> struct PrecisionTraits {
    static constexpr size_t bytes = sizeof(typename PrecisionType<P>::type);
    static constexpr int fma_flops = 2;
    static constexpr const char* name =
        P == Precision::FP64 ? "FP64" :
        P == Precision::FP32 ? "FP32" : "UNKNOWN";
};

} // namespace floptic
