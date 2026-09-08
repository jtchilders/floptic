// Reference GEMM correctness tests: hand-computed rectangular alpha/beta
// cases, dimension/size-mismatch rejection. Untimed, CPU-only.
#include "floptic/reference_gemm.hpp"
#include "check.hpp"

#include <stdexcept>
#include <vector>

using floptic::GemmDims;
using floptic::reference_gemm;
using floptic::validate_gemm_dims;

namespace {

// ---------------------------------------------------------------------
// Hand-computed rectangular case.
//
// A (2x3, row-major):
//   [1 2 3]
//   [4 5 6]
// B (3x2, row-major):
//   [ 7  8]
//   [ 9 10]
//   [11 12]
// A*B (2x2):
//   row0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
//   row1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
// ---------------------------------------------------------------------

void test_rectangular_alpha1_beta0_matches_hand_computed_result() {
    GemmDims dims;
    dims.M = 2; dims.N = 2; dims.K = 3;
    dims.lda = 3; dims.ldb = 2; dims.ldc = 2;

    std::vector<double> a = {1, 2, 3, 4, 5, 6};
    std::vector<double> b = {7, 8, 9, 10, 11, 12};
    std::vector<double> c = {0, 0, 0, 0};

    reference_gemm<double>(dims, 1.0, a.data(), b.data(), 0.0, c.data());

    CHECK_EQ(c[0], 58.0);
    CHECK_EQ(c[1], 64.0);
    CHECK_EQ(c[2], 139.0);
    CHECK_EQ(c[3], 154.0);
}

void test_alpha_and_beta_scale_correctly() {
    GemmDims dims;
    dims.M = 2; dims.N = 2; dims.K = 3;
    dims.lda = 3; dims.ldb = 2; dims.ldc = 2;

    std::vector<double> a = {1, 2, 3, 4, 5, 6};
    std::vector<double> b = {7, 8, 9, 10, 11, 12};
    // Pre-existing C values that beta must scale and add.
    std::vector<double> c = {1, 1, 1, 1};

    reference_gemm<double>(dims, 2.0, a.data(), b.data(), 3.0, c.data());

    // C = 2*(A*B) + 3*prior
    CHECK_EQ(c[0], 2.0 * 58.0 + 3.0 * 1.0);
    CHECK_EQ(c[1], 2.0 * 64.0 + 3.0 * 1.0);
    CHECK_EQ(c[2], 2.0 * 139.0 + 3.0 * 1.0);
    CHECK_EQ(c[3], 2.0 * 154.0 + 3.0 * 1.0);
}

void test_beta_zero_ignores_prior_c_value_including_nonzero_garbage() {
    GemmDims dims;
    dims.M = 1; dims.N = 1; dims.K = 2;
    dims.lda = 2; dims.ldb = 1; dims.ldc = 1;

    std::vector<double> a = {2, 3};
    std::vector<double> b = {5, 7};
    std::vector<double> c = {999999.0};  // must be fully overwritten, not blended

    reference_gemm<double>(dims, 1.0, a.data(), b.data(), 0.0, c.data());
    CHECK_EQ(c[0], 2.0 * 5.0 + 3.0 * 7.0);  // 31
}

void test_transposed_a_matches_non_transposed_equivalent() {
    // A^T where A^T is 2x3 (same logical matrix as the hand-computed case
    // above), stored as its 3x2 transpose in memory: A_mem(k, r) = A(r, k).
    GemmDims dims;
    dims.M = 2; dims.N = 2; dims.K = 3;
    dims.trans_a = true;
    dims.lda = 2;   // A_mem is K x M = 3x2, row-major -> ld = M = 2
    dims.ldb = 2;
    dims.ldc = 2;

    // A_mem(k, r) = A(r, k): A row0=[1,2,3], row1=[4,5,6]
    // A_mem row0 (k=0): [A(0,0), A(1,0)] = [1, 4]
    // A_mem row1 (k=1): [A(0,1), A(1,1)] = [2, 5]
    // A_mem row2 (k=2): [A(0,2), A(1,2)] = [3, 6]
    std::vector<double> a_mem = {1, 4, 2, 5, 3, 6};
    std::vector<double> b = {7, 8, 9, 10, 11, 12};
    std::vector<double> c = {0, 0, 0, 0};

    reference_gemm<double>(dims, 1.0, a_mem.data(), b.data(), 0.0, c.data());

    CHECK_EQ(c[0], 58.0);
    CHECK_EQ(c[1], 64.0);
    CHECK_EQ(c[2], 139.0);
    CHECK_EQ(c[3], 154.0);
}

void test_fp32_input_still_computes_correct_result() {
    GemmDims dims;
    dims.M = 1; dims.N = 1; dims.K = 4;
    dims.lda = 4; dims.ldb = 1; dims.ldc = 1;

    std::vector<float> a = {1, 2, 3, 4};
    std::vector<float> b = {5, 6, 7, 8};
    std::vector<float> c = {0};

    reference_gemm<float>(dims, 1.0f, a.data(), b.data(), 0.0f, c.data());
    // 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
    CHECK_EQ(c[0], 70.0f);
}

// ---------------------------------------------------------------------
// Dimension/size-mismatch rejection.
// ---------------------------------------------------------------------

void test_zero_or_negative_dims_rejected() {
    GemmDims dims;
    dims.M = 0; dims.N = 2; dims.K = 2;
    dims.lda = 2; dims.ldb = 2; dims.ldc = 2;
    CHECK_TRUE(!validate_gemm_dims(dims).empty());

    dims.M = 2; dims.N = -1;
    CHECK_TRUE(!validate_gemm_dims(dims).empty());
}

void test_leading_dimension_too_small_rejected() {
    GemmDims dims;
    dims.M = 2; dims.N = 2; dims.K = 3;
    dims.lda = 2;  // must be >= K=3 for non-transposed A
    dims.ldb = 2; dims.ldc = 2;
    CHECK_TRUE(!validate_gemm_dims(dims).empty());
}

void test_valid_dims_pass_validation() {
    GemmDims dims;
    dims.M = 2; dims.N = 2; dims.K = 3;
    dims.lda = 3; dims.ldb = 2; dims.ldc = 2;
    CHECK_TRUE(validate_gemm_dims(dims).empty());
}

void test_reference_gemm_throws_on_invalid_dims_without_touching_memory() {
    GemmDims dims;
    dims.M = 0; dims.N = 2; dims.K = 2;
    dims.lda = 2; dims.ldb = 2; dims.ldc = 2;

    std::vector<double> a = {1, 2, 3, 4};
    std::vector<double> b = {1, 2, 3, 4};
    std::vector<double> c = {0, 0, 0, 0};

    bool threw = false;
    try {
        reference_gemm<double>(dims, 1.0, a.data(), b.data(), 0.0, c.data());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK_TRUE(threw);
}

}  // namespace

int main() {
    test_rectangular_alpha1_beta0_matches_hand_computed_result();
    test_alpha_and_beta_scale_correctly();
    test_beta_zero_ignores_prior_c_value_including_nonzero_garbage();
    test_transposed_a_matches_non_transposed_equivalent();
    test_fp32_input_still_computes_correct_result();
    test_zero_or_negative_dims_rejected();
    test_leading_dimension_too_small_rejected();
    test_valid_dims_pass_validation();
    test_reference_gemm_throws_on_invalid_dims_without_touching_memory();
    return floptic_test::finish();
}
