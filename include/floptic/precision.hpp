#pragma once
#include <string>
#include <vector>
#include <cstddef>
#include <optional>

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

// Strict parse: returns false (leaving `out` untouched) for any spelling that
// is not one of the accepted, exact aliases. Callers that must reject unknown
// input (e.g. CLI parsing) should use this instead of string_to_precision.
inline bool try_string_to_precision(const std::string& s, Precision& out) {
    if (s == "fp64" || s == "FP64") { out = Precision::FP64; return true; }
    if (s == "fp32" || s == "FP32") { out = Precision::FP32; return true; }
    if (s == "fp16" || s == "FP16") { out = Precision::FP16; return true; }
    if (s == "bf16" || s == "BF16") { out = Precision::BF16; return true; }
    if (s == "tf32" || s == "TF32") { out = Precision::TF32; return true; }
    if (s == "fp8e4m3" || s == "FP8_E4M3" || s == "fp8" || s == "FP8") { out = Precision::FP8_E4M3; return true; }
    if (s == "fp8e5m2" || s == "FP8_E5M2") { out = Precision::FP8_E5M2; return true; }
    if (s == "fp4" || s == "FP4") { out = Precision::FP4; return true; }
    if (s == "int8" || s == "INT8") { out = Precision::INT8; return true; }
    if (s == "int4" || s == "INT4") { out = Precision::INT4; return true; }
    return false;
}

// Strict parse returning std::optional<Precision>: std::nullopt for any
// spelling that is not one of the accepted, exact aliases. This is the
// public string-to-precision API; there is no silent fallback to FP64.
inline std::optional<Precision> string_to_precision(const std::string& s) {
    Precision p;
    if (try_string_to_precision(s, p)) return p;
    return std::nullopt;
}

inline std::vector<Precision> all_standard_precisions() {
    return { Precision::FP64, Precision::FP32, Precision::TF32, Precision::FP16, Precision::BF16,
             Precision::FP8_E4M3, Precision::FP8_E5M2, Precision::FP4, Precision::INT8 };
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
