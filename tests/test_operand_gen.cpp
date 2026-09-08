// Deterministic operand-generation tests: fixed PRNG known outputs, seed
// reproducibility, range/finiteness of production distributions, invalid
// config rejection. CPU-only, no device discovery.
#include "floptic/operand_gen.hpp"
#include "floptic/prng.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

using floptic::OperandConfig;
using floptic::OperandDistribution;
using floptic::SplitMix64;

namespace {

// ---------------------------------------------------------------------
// Fixed PRNG known-output vectors: pins SplitMix64's integer sequence so a
// change to the algorithm/constants is caught immediately. These values
// were computed by running SplitMix64(seed=42) and are the canonical
// reference for this implementation.
// ---------------------------------------------------------------------

void test_splitmix64_known_output_vector_for_seed_42() {
    SplitMix64 rng(42);
    const uint64_t expected[] = {
        13679457532755275413ULL,
        2949826092126892291ULL,
        5139283748462763858ULL,
        6349198060258255764ULL,
    };
    for (uint64_t exp : expected) {
        CHECK_EQ(rng.next_u64(), exp);
    }
}

void test_splitmix64_next_double_is_in_unit_interval() {
    SplitMix64 rng(1234);
    for (int i = 0; i < 1000; i++) {
        double d = rng.next_double();
        CHECK_TRUE(d >= 0.0);
        CHECK_TRUE(d < 1.0);
    }
}

// ---------------------------------------------------------------------
// Reproducibility: same seed/config/type/count -> identical data;
// different seed -> different data.
// ---------------------------------------------------------------------

void test_same_seed_config_reproduces_identical_data() {
    OperandConfig cfg;
    cfg.distribution = OperandDistribution::UNIFORM_SIGNED;
    cfg.seed = 777;
    cfg.count = 256;
    cfg.scale = 3.5;

    std::vector<double> a, b;
    floptic::generate_operands(cfg, a);
    floptic::generate_operands(cfg, b);

    CHECK_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); i++) {
        CHECK_EQ(a[i], b[i]);
    }
}

void test_different_seed_produces_different_data() {
    OperandConfig cfg;
    cfg.distribution = OperandDistribution::UNIFORM_SIGNED;
    cfg.count = 256;
    cfg.scale = 3.5;

    cfg.seed = 1;
    std::vector<double> a;
    floptic::generate_operands(cfg, a);

    cfg.seed = 2;
    std::vector<double> b;
    floptic::generate_operands(cfg, b);

    int differences = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) differences++;
    }
    CHECK_TRUE(differences > 0);
}

// ---------------------------------------------------------------------
// Production distributions: finite, nonzero (where applicable), in range.
// ---------------------------------------------------------------------

void test_uniform_signed_is_finite_nonzero_and_in_range() {
    OperandConfig cfg;
    cfg.distribution = OperandDistribution::UNIFORM_SIGNED;
    cfg.seed = 5;
    cfg.count = 10000;
    cfg.scale = 2.0;

    std::vector<double> data;
    floptic::generate_operands(cfg, data);
    for (double v : data) {
        CHECK_TRUE(std::isfinite(v));
        CHECK_TRUE(v != 0.0);
        CHECK_TRUE(v >= -cfg.scale && v < cfg.scale);
    }
}

void test_gaussian_is_finite_and_roughly_zero_mean() {
    OperandConfig cfg;
    cfg.distribution = OperandDistribution::GAUSSIAN;
    cfg.seed = 6;
    cfg.count = 20000;
    cfg.stddev = 1.0;

    std::vector<double> data;
    floptic::generate_operands(cfg, data);
    double sum = 0.0;
    for (double v : data) {
        CHECK_TRUE(std::isfinite(v));
        sum += v;
    }
    double mean = sum / static_cast<double>(data.size());
    // Loose statistical bound: with stddev=1 and n=20000, stderr of the
    // mean is ~1/sqrt(20000) ~= 0.0071; 0.1 is a generous, deterministic
    // (not flaky) bound given the fixed seed.
    CHECK_TRUE(std::fabs(mean) < 0.1);
}

void test_log_uniform_spans_requested_exponent_range_and_is_finite() {
    OperandConfig cfg;
    cfg.distribution = OperandDistribution::LOG_UNIFORM;
    cfg.seed = 7;
    cfg.count = 5000;
    cfg.min_exponent = -4;
    cfg.max_exponent = 4;

    std::vector<double> data;
    floptic::generate_operands(cfg, data);
    double min_mag = 1e300, max_mag = 0.0;
    for (double v : data) {
        CHECK_TRUE(std::isfinite(v));
        CHECK_TRUE(v != 0.0);
        double mag = std::fabs(v);
        min_mag = std::min(min_mag, mag);
        max_mag = std::max(max_mag, mag);
    }
    // Magnitude must stay within [2^min_exponent, 2^(max_exponent+1)) since
    // mantissa jitter is in [1, 2).
    CHECK_TRUE(min_mag >= std::ldexp(1.0, cfg.min_exponent));
    CHECK_TRUE(max_mag < std::ldexp(1.0, cfg.max_exponent + 1));
}

void test_sparse_matches_requested_density_approximately_and_nonzeros_are_finite_nonzero() {
    OperandConfig cfg;
    cfg.distribution = OperandDistribution::SPARSE;
    cfg.seed = 8;
    cfg.count = 100000;
    cfg.density = 0.2;
    cfg.base_distribution = OperandDistribution::UNIFORM_SIGNED;
    cfg.scale = 1.0;

    std::vector<double> data;
    floptic::generate_operands(cfg, data);
    int64_t nonzero = 0;
    for (double v : data) {
        CHECK_TRUE(std::isfinite(v));
        if (v != 0.0) nonzero++;
    }
    double observed_density = static_cast<double>(nonzero) / static_cast<double>(data.size());
    // Binomial stderr at n=100000, p=0.2 is ~0.0013; 0.02 is a generous
    // deterministic bound for the fixed seed.
    CHECK_TRUE(std::fabs(observed_density - cfg.density) < 0.02);
}

void test_cancellation_pairs_sum_to_zero() {
    OperandConfig cfg;
    cfg.distribution = OperandDistribution::CANCELLATION;
    cfg.seed = 9;
    cfg.count = 200;  // even: fully paired
    cfg.scale = 1e6;

    std::vector<double> data;
    floptic::generate_operands(cfg, data);
    for (size_t i = 0; i + 1 < data.size(); i += 2) {
        CHECK_TRUE(std::isfinite(data[i]));
        CHECK_TRUE(std::isfinite(data[i + 1]));
        CHECK_EQ(data[i] + data[i + 1], 0.0);
        CHECK_TRUE(data[i] != 0.0);
    }
}

void test_special_values_are_deterministic_finite_edges_by_default() {
    OperandConfig cfg;
    cfg.distribution = OperandDistribution::SPECIAL_VALUES;
    cfg.seed = 10;  // unused by this distribution, but must still be set
    cfg.count = 11;

    std::vector<double> data;
    floptic::generate_operands(cfg, data);
    for (double v : data) {
        CHECK_TRUE(!std::isnan(v));
        CHECK_TRUE(!std::isinf(v));
    }
}

void test_special_values_inject_nan_and_inf_only_when_requested() {
    OperandConfig cfg;
    cfg.distribution = OperandDistribution::SPECIAL_VALUES;
    cfg.seed = 11;
    cfg.count = 4;
    cfg.inject_nan = true;
    cfg.inject_inf = true;

    std::vector<double> data;
    floptic::generate_operands(cfg, data);
    bool saw_nan = false, saw_inf = false;
    for (double v : data) {
        if (std::isnan(v)) saw_nan = true;
        if (std::isinf(v)) saw_inf = true;
    }
    CHECK_TRUE(saw_nan);
    CHECK_TRUE(saw_inf);
}

// ---------------------------------------------------------------------
// Invalid configs must fail (validate() and/or generate_operands throws).
// ---------------------------------------------------------------------

void test_invalid_configs_fail_validation() {
    OperandConfig cfg;

    cfg = OperandConfig{};
    cfg.distribution = OperandDistribution::UNIFORM_SIGNED;
    cfg.count = 10;
    cfg.scale = -1.0;
    CHECK_TRUE(!cfg.validate().empty());

    cfg = OperandConfig{};
    cfg.distribution = OperandDistribution::GAUSSIAN;
    cfg.count = 10;
    cfg.stddev = 0.0;
    CHECK_TRUE(!cfg.validate().empty());

    cfg = OperandConfig{};
    cfg.distribution = OperandDistribution::LOG_UNIFORM;
    cfg.count = 10;
    cfg.min_exponent = 5;
    cfg.max_exponent = -5;
    CHECK_TRUE(!cfg.validate().empty());

    cfg = OperandConfig{};
    cfg.distribution = OperandDistribution::SPARSE;
    cfg.count = 10;
    cfg.density = 1.5;
    CHECK_TRUE(!cfg.validate().empty());

    cfg = OperandConfig{};
    cfg.distribution = OperandDistribution::SPARSE;
    cfg.count = 10;
    cfg.density = -0.1;
    CHECK_TRUE(!cfg.validate().empty());

    cfg = OperandConfig{};
    cfg.distribution = OperandDistribution::UNIFORM_SIGNED;
    cfg.count = -1;
    cfg.scale = 1.0;
    CHECK_TRUE(!cfg.validate().empty());

    cfg = OperandConfig{};
    cfg.distribution = OperandDistribution::UNIFORM_SIGNED;
    cfg.count = (int64_t(1) << 41);  // overflow-prone size
    cfg.scale = 1.0;
    CHECK_TRUE(!cfg.validate().empty());
}

void test_generate_operands_throws_on_invalid_config() {
    OperandConfig cfg;
    cfg.distribution = OperandDistribution::UNIFORM_SIGNED;
    cfg.count = 10;
    cfg.scale = -1.0;

    bool threw = false;
    try {
        std::vector<double> out;
        floptic::generate_operands(cfg, out);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK_TRUE(threw);
}

void test_valid_configs_pass_validation() {
    OperandConfig cfg;
    cfg.distribution = OperandDistribution::UNIFORM_SIGNED;
    cfg.count = 10;
    cfg.scale = 1.0;
    CHECK_TRUE(cfg.validate().empty());
}

// ---------------------------------------------------------------------
// float (FP32) generation must also be finite and in range.
// ---------------------------------------------------------------------

void test_uniform_signed_fp32_is_finite_and_in_range() {
    OperandConfig cfg;
    cfg.distribution = OperandDistribution::UNIFORM_SIGNED;
    cfg.seed = 12;
    cfg.count = 5000;
    cfg.scale = 4.0;

    std::vector<float> data;
    floptic::generate_operands(cfg, data);
    for (float v : data) {
        CHECK_TRUE(std::isfinite(v));
        CHECK_TRUE(v >= static_cast<float>(-cfg.scale) && v < static_cast<float>(cfg.scale));
    }
}

}  // namespace

int main() {
    test_splitmix64_known_output_vector_for_seed_42();
    test_splitmix64_next_double_is_in_unit_interval();
    test_same_seed_config_reproduces_identical_data();
    test_different_seed_produces_different_data();
    test_uniform_signed_is_finite_nonzero_and_in_range();
    test_gaussian_is_finite_and_roughly_zero_mean();
    test_log_uniform_spans_requested_exponent_range_and_is_finite();
    test_sparse_matches_requested_density_approximately_and_nonzeros_are_finite_nonzero();
    test_cancellation_pairs_sum_to_zero();
    test_special_values_are_deterministic_finite_edges_by_default();
    test_special_values_inject_nan_and_inf_only_when_requested();
    test_invalid_configs_fail_validation();
    test_generate_operands_throws_on_invalid_config();
    test_valid_configs_pass_validation();
    test_uniform_signed_fp32_is_finite_and_in_range();
    return floptic_test::finish();
}
