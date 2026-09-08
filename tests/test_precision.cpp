// Hardware-independent precision-mapping tests.
//
// These check pure enum <-> string logic in floptic/precision.hpp. They must
// not touch device discovery, CUDA/HIP, or kernel execution.
#include "floptic/precision.hpp"
#include "check.hpp"

#include <set>
#include <string>
#include <vector>

using floptic::Precision;
using floptic::all_standard_precisions;
using floptic::precision_to_string;
using floptic::string_to_precision;

namespace {

void test_every_accepted_spelling_maps_to_expected_enum() {
    const std::vector<std::pair<std::string, Precision>> cases = {
        {"fp64", Precision::FP64},       {"FP64", Precision::FP64},
        {"fp32", Precision::FP32},       {"FP32", Precision::FP32},
        {"fp16", Precision::FP16},       {"FP16", Precision::FP16},
        {"bf16", Precision::BF16},       {"BF16", Precision::BF16},
        {"tf32", Precision::TF32},       {"TF32", Precision::TF32},
        {"fp8e4m3", Precision::FP8_E4M3}, {"FP8_E4M3", Precision::FP8_E4M3},
        {"fp8", Precision::FP8_E4M3},     {"FP8", Precision::FP8_E4M3},
        {"fp8e5m2", Precision::FP8_E5M2}, {"FP8_E5M2", Precision::FP8_E5M2},
        {"fp4", Precision::FP4},          {"FP4", Precision::FP4},
        {"int8", Precision::INT8},        {"INT8", Precision::INT8},
        {"int4", Precision::INT4},        {"INT4", Precision::INT4},
    };

    for (const auto& [spelling, expected] : cases) {
        CHECK_EQ(static_cast<int>(string_to_precision(spelling)), static_cast<int>(expected));
    }
}

void test_precision_to_string_produces_canonical_spelling_for_every_enum() {
    CHECK_EQ(precision_to_string(Precision::FP64), std::string("FP64"));
    CHECK_EQ(precision_to_string(Precision::FP32), std::string("FP32"));
    CHECK_EQ(precision_to_string(Precision::FP16), std::string("FP16"));
    CHECK_EQ(precision_to_string(Precision::BF16), std::string("BF16"));
    CHECK_EQ(precision_to_string(Precision::TF32), std::string("TF32"));
    CHECK_EQ(precision_to_string(Precision::FP8_E4M3), std::string("FP8_E4M3"));
    CHECK_EQ(precision_to_string(Precision::FP8_E5M2), std::string("FP8_E5M2"));
    CHECK_EQ(precision_to_string(Precision::FP4), std::string("FP4"));
    CHECK_EQ(precision_to_string(Precision::INT8), std::string("INT8"));
    CHECK_EQ(precision_to_string(Precision::INT4), std::string("INT4"));
}

void test_standard_precision_enumeration_has_no_duplicates() {
    const auto precisions = all_standard_precisions();
    std::set<int> seen;
    for (const auto p : precisions) {
        seen.insert(static_cast<int>(p));
    }
    CHECK_EQ(seen.size(), precisions.size());
}

} // namespace

int main() {
    test_every_accepted_spelling_maps_to_expected_enum();
    test_precision_to_string_produces_canonical_spelling_for_every_enum();
    test_standard_precision_enumeration_has_no_duplicates();
    return floptic_test::finish();
}
