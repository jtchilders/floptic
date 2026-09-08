#pragma once
// Deterministic operand generation for validation and throughput data.
//
// Every generator here is a pure function of (config, element index sequence
// via SplitMix64 draws) so identical (config, seed, type, count) reproduces
// bit-identical output on any conforming C++17 standard library on any
// platform -- see floptic/prng.hpp for exactly which reproducibility
// guarantee that is and why std::uniform_*_distribution is avoided.
//
// Generation is untimed, host-only, and must never appear inside a
// benchmark's measured region.
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "floptic/prng.hpp"

namespace floptic {

enum class OperandDistribution {
    UNIFORM_SIGNED,   // bounded values in [-scale, +scale]
    GAUSSIAN,         // zero-mean normal, configured stddev
    LOG_UNIFORM,      // random sign, exponents spanning a base-2 range
    SPARSE,           // deterministic target density; nonzeros from a base distribution
    CANCELLATION,     // paired/opposed values designed to exercise cancellation
    SPECIAL_VALUES,   // deterministic finite edge values + optional NaN/Inf
};

inline std::string operand_distribution_to_string(OperandDistribution d) {
    switch (d) {
        case OperandDistribution::UNIFORM_SIGNED: return "uniform_signed";
        case OperandDistribution::GAUSSIAN:       return "gaussian";
        case OperandDistribution::LOG_UNIFORM:    return "log_uniform";
        case OperandDistribution::SPARSE:         return "sparse";
        case OperandDistribution::CANCELLATION:   return "cancellation";
        case OperandDistribution::SPECIAL_VALUES: return "special_values";
    }
    return "unknown";
}

// Configuration for a single operand-generation call. Every field that
// affects the produced values is part of the reproducibility contract:
// (distribution, seed, count, and the fields relevant to that
// distribution) fully determine the output.
struct OperandConfig {
    OperandDistribution distribution = OperandDistribution::UNIFORM_SIGNED;
    uint64_t seed = 0;              // REQUIRED explicit seed (no implicit default entropy)
    int64_t count = 0;               // number of elements to generate

    // uniform_signed
    double scale = 1.0;              // values in [-scale, +scale]; must be > 0

    // gaussian
    double stddev = 1.0;             // must be > 0

    // log_uniform: sign is random; |value| = 2^e for e uniform in
    // [min_exponent, max_exponent] (inclusive), then jittered within the
    // octave by a uniform mantissa factor in [1, 2).
    int min_exponent = -10;
    int max_exponent = 10;           // must be >= min_exponent

    // sparse
    double density = 0.1;            // fraction nonzero, must be in [0, 1]
    OperandDistribution base_distribution = OperandDistribution::UNIFORM_SIGNED;

    // special_values
    bool inject_nan = false;         // validator-test-only
    bool inject_inf = false;         // validator-test-only

    // Validate parameters without generating data. Returns empty string on
    // success, otherwise a human-readable reason. Overflow-prone sizes are
    // rejected here rather than deep inside a loop.
    std::string validate() const {
        if (count < 0) return "count must be >= 0";
        // Guard against count * sizeof(T) overflowing size_t/ptrdiff_t
        // arithmetic downstream; 2^40 elements is already far beyond any
        // realistic validation or throughput buffer.
        if (count > (int64_t(1) << 40)) return "count is unreasonably large (overflow risk)";

        switch (distribution) {
            case OperandDistribution::UNIFORM_SIGNED:
                if (!(scale > 0.0)) return "scale must be > 0";
                if (!std::isfinite(scale)) return "scale must be finite";
                break;
            case OperandDistribution::GAUSSIAN:
                if (!(stddev > 0.0)) return "stddev must be > 0";
                if (!std::isfinite(stddev)) return "stddev must be finite";
                break;
            case OperandDistribution::LOG_UNIFORM:
                if (max_exponent < min_exponent) return "max_exponent must be >= min_exponent";
                break;
            case OperandDistribution::SPARSE:
                if (density < 0.0 || density > 1.0) return "density must be in [0, 1]";
                if (base_distribution == OperandDistribution::SPARSE)
                    return "sparse base_distribution must not be sparse (no recursive nesting)";
                break;
            case OperandDistribution::CANCELLATION:
                if (!(scale > 0.0)) return "scale must be > 0 for cancellation pairs";
                if (!std::isfinite(scale)) return "scale must be finite";
                break;
            case OperandDistribution::SPECIAL_VALUES:
                break;
        }
        return "";
    }
};

namespace detail {

// Draws one value from a "base" (non-sparse, non-special) distribution
// using the given RNG and config. Shared by top-level generation and by
// the nonzero-value path of `sparse`.
template <typename T>
inline T draw_base_value(SplitMix64& rng, const OperandConfig& cfg,
                          OperandDistribution dist) {
    switch (dist) {
        case OperandDistribution::UNIFORM_SIGNED: {
            return static_cast<T>(rng.next_signed_unit() * cfg.scale);
        }
        case OperandDistribution::GAUSSIAN: {
            return static_cast<T>(rng.next_standard_normal() * cfg.stddev);
        }
        case OperandDistribution::LOG_UNIFORM: {
            double sign = (rng.next_u64() & 1u) ? 1.0 : -1.0;
            int64_t e = rng.next_int_range(cfg.min_exponent, cfg.max_exponent);
            // Mantissa jitter in [1, 2) so values don't land exactly on
            // powers of two only.
            double mantissa = 1.0 + rng.next_double();
            double magnitude = mantissa * std::ldexp(1.0, static_cast<int>(e));
            return static_cast<T>(sign * magnitude);
        }
        case OperandDistribution::CANCELLATION: {
            // Pattern: index-paired opposed values. Caller (generate<T>)
            // handles actual pairing across the full array; here we just
            // produce one magnitude to be used with a sign chosen by the
            // caller.
            return static_cast<T>(rng.next_double() * cfg.scale);
        }
        case OperandDistribution::SPARSE:
        case OperandDistribution::SPECIAL_VALUES:
            // Not reachable: handled at the top level, never recursed into.
            return static_cast<T>(0);
    }
    return static_cast<T>(0);
}

}  // namespace detail

// Generates `cfg.count` values of type T (float or double) into `out`
// (resized as needed). Throws std::invalid_argument if cfg.validate() fails.
//
// Determinism contract: for a fixed cfg (all fields) and T, this function
// produces the exact same sequence of values every call, on every platform,
// because it only performs uint64_t arithmetic (via SplitMix64) and a fixed,
// explicit integer -> double mapping -- never a standard-library
// distribution object.
template <typename T>
inline void generate_operands(const OperandConfig& cfg, std::vector<T>& out) {
    std::string err = cfg.validate();
    if (!err.empty()) {
        throw std::invalid_argument("floptic::generate_operands: " + err);
    }

    out.assign(static_cast<size_t>(cfg.count), static_cast<T>(0));
    SplitMix64 rng(cfg.seed);

    switch (cfg.distribution) {
        case OperandDistribution::UNIFORM_SIGNED:
        case OperandDistribution::GAUSSIAN:
        case OperandDistribution::LOG_UNIFORM: {
            for (int64_t i = 0; i < cfg.count; i++) {
                out[static_cast<size_t>(i)] =
                    detail::draw_base_value<T>(rng, cfg, cfg.distribution);
            }
            break;
        }
        case OperandDistribution::CANCELLATION: {
            // Adjacent pairs (2k, 2k+1) hold +m, -m for a shared magnitude
            // m drawn per pair, so summing the pair (or the whole buffer,
            // for even count) is designed to cancel to (near) zero and
            // exercise catastrophic-cancellation error paths. A trailing
            // unpaired element (odd count) gets an unpaired positive value.
            for (int64_t i = 0; i + 1 < cfg.count; i += 2) {
                double magnitude = rng.next_double() * cfg.scale;
                // Ensure a visibly nonzero magnitude so the pair is a real
                // cancellation case, not a 0,0 pair.
                if (magnitude == 0.0) magnitude = cfg.scale;
                out[static_cast<size_t>(i)] = static_cast<T>(magnitude);
                out[static_cast<size_t>(i + 1)] = static_cast<T>(-magnitude);
            }
            if (cfg.count % 2 == 1) {
                double magnitude = rng.next_double() * cfg.scale;
                if (magnitude == 0.0) magnitude = cfg.scale;
                out[static_cast<size_t>(cfg.count - 1)] = static_cast<T>(magnitude);
            }
            break;
        }
        case OperandDistribution::SPARSE: {
            for (int64_t i = 0; i < cfg.count; i++) {
                double u = rng.next_double();
                if (u < cfg.density) {
                    T v = detail::draw_base_value<T>(rng, cfg, cfg.base_distribution);
                    // Nonzero guarantee for the "nonzero density" contract:
                    // resample once if the base distribution happened to
                    // produce exactly zero (astronomically unlikely for
                    // continuous distributions, but keep the guarantee
                    // explicit rather than probabilistic).
                    int guard = 0;
                    while (v == static_cast<T>(0) && guard < 8) {
                        v = detail::draw_base_value<T>(rng, cfg, cfg.base_distribution);
                        guard++;
                    }
                    out[static_cast<size_t>(i)] = v;
                } else {
                    out[static_cast<size_t>(i)] = static_cast<T>(0);
                }
            }
            break;
        }
        case OperandDistribution::SPECIAL_VALUES: {
            // Deterministic finite edge values, cycled across the buffer,
            // with NaN/Inf injected only when explicitly requested (for
            // validator negative-path tests -- never for throughput runs).
            const T kFiniteValues[] = {
                static_cast<T>(0.0),
                static_cast<T>(-0.0),
                static_cast<T>(1.0),
                static_cast<T>(-1.0),
                std::numeric_limits<T>::min(),           // smallest positive normal
                -std::numeric_limits<T>::min(),
                std::numeric_limits<T>::denorm_min(),    // smallest positive subnormal
                std::numeric_limits<T>::max(),
                std::numeric_limits<T>::lowest(),
                static_cast<T>(1e-30),
                static_cast<T>(-1e-30),
            };
            constexpr size_t kNumFinite = sizeof(kFiniteValues) / sizeof(kFiniteValues[0]);
            for (int64_t i = 0; i < cfg.count; i++) {
                out[static_cast<size_t>(i)] = kFiniteValues[static_cast<size_t>(i) % kNumFinite];
            }
            if (cfg.inject_nan && cfg.count > 0) {
                out[0] = std::numeric_limits<T>::quiet_NaN();
            }
            if (cfg.inject_inf && cfg.count > 0) {
                size_t idx = (cfg.count > 1) ? 1 : 0;
                out[idx] = std::numeric_limits<T>::infinity();
            }
            break;
        }
    }
}

}  // namespace floptic
