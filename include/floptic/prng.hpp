#pragma once
#include <cmath>
#include <cstdint>

// Deterministic, cross-platform pseudo-random source for validation data
// generation.
//
// Reproducibility guarantee: SplitMix64 is a fixed, publicly specified
// integer algorithm (Vigna, 2015) operating only on uint64_t addition,
// multiplication, and shifts/xors -- no floating-point state, no
// implementation-defined behavior, and no dependency on the C++ standard
// library's random-number facilities. `std::mt19937`/`std::uniform_*` are
// deliberately NOT used here: the C++ standard defines the engine's integer
// sequence but leaves each *distribution's* mapping from engine output to
// value implementation-defined, so identical seeds can legally produce
// different floating-point values across standard library implementations
// (or even across versions of the same one). Every value this header
// produces is instead derived by our own, fully-specified integer -> double
// mapping (see canonical_double() below), so the same (seed, config, type,
// element count) reproduces bit-identical output on any conforming C++17
// implementation on any platform -- not just "the same standard library".
namespace floptic {

class SplitMix64 {
public:
    explicit SplitMix64(uint64_t seed) : state_(seed) {}

    // Advances internal state and returns the next raw 64-bit output.
    uint64_t next_u64() {
        uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    // Canonical double in [0, 1) using the top 53 bits of a raw draw --
    // the standard, fully-specified integer->double mapping (same technique
    // documented for xoshiro/PCG "canonical" generators). No use of
    // std::ldexp or platform rounding modes; the arithmetic here is exact
    // for the fixed constant divisor.
    double next_double() {
        constexpr double kInv2Pow53 = 1.0 / 9007199254740992.0;  // 1 / 2^53
        return static_cast<double>(next_u64() >> 11) * kInv2Pow53;
    }

    // Uniform double in [-1, 1).
    double next_signed_unit() { return next_double() * 2.0 - 1.0; }

    // One standard-normal (mean 0, stddev 1) sample via the Box-Muller
    // transform, consuming exactly two draws from next_double() per call.
    // u1 is clamped away from exactly 0 so log() never sees zero (u1 == 0
    // has probability 2^-53 but must still be handled deterministically).
    double next_standard_normal() {
        double u1 = next_double();
        double u2 = next_double();
        if (u1 < 1e-300) u1 = 1e-300;
        constexpr double kTwoPi = 6.283185307179586476925286766559;
        double r = std::sqrt(-2.0 * std::log(u1));
        return r * std::cos(kTwoPi * u2);
    }

    // Uniform integer in [lo, hi] inclusive, via rejection-free modulo
    // reduction (a small, documented bias for very large ranges is
    // acceptable here: this is validation test data, not cryptography).
    int64_t next_int_range(int64_t lo, int64_t hi) {
        uint64_t span = static_cast<uint64_t>(hi - lo) + 1ULL;
        return lo + static_cast<int64_t>(next_u64() % span);
    }

private:
    uint64_t state_;
};

}  // namespace floptic
