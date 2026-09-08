// Hardware-independent numerical correctness tests for the AXPY compute
// kernels in floptic/axpy_kernel.hpp: y = alpha*x + y for FP32 and FP64,
// across small/odd sizes that exercise every scalar/SIMD tail path
// (n < SIMD width, and n not a multiple of the width). No timing, no
// device discovery — this proves every element of y is actually updated
// and gets the numerically expected value, independent of which SIMD path
// (scalar/AVX2/AVX-512) this compile targets.
#include "floptic/axpy_kernel.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

// A width comfortably larger than any SIMD width compiled by this project
// (AVX-512 FP32 has the widest lanes at 16), so "width+1" sizes always
// straddle a real vector/scalar-tail boundary regardless of build target.
constexpr int64_t kProbeWidth = 16;

template <typename T>
void check_axpy_matches_reference(int64_t n, int num_threads) {
    std::vector<T> x(static_cast<size_t>(n)), y(static_cast<size_t>(n)),
        y_ref(static_cast<size_t>(n));
    T alpha = static_cast<T>(1.5);

    for (int64_t i = 0; i < n; i++) {
        x[static_cast<size_t>(i)] = static_cast<T>(1.0 + 0.25 * static_cast<double>(i));
        y[static_cast<size_t>(i)] = static_cast<T>(2.0 - 0.1 * static_cast<double>(i));
        y_ref[static_cast<size_t>(i)] = y[static_cast<size_t>(i)];
    }

    floptic::axpy_run(y.data(), x.data(), alpha, n, num_threads);

    for (int64_t i = 0; i < n; i++) {
        T expected = std::fma(alpha, x[static_cast<size_t>(i)], y_ref[static_cast<size_t>(i)]);
        // Every element must be updated (no skipped tail): compare against
        // the exact scalar fma reference, not the untouched initial value.
        CHECK_EQ(y[static_cast<size_t>(i)], expected);
    }
}

const int64_t kOddSizes[] = {1, 3, kProbeWidth + 1, 17};

void test_fp64_odd_sizes_single_threaded() {
    for (int64_t n : kOddSizes) {
        check_axpy_matches_reference<double>(n, /*num_threads=*/1);
    }
}

void test_fp32_odd_sizes_single_threaded() {
    for (int64_t n : kOddSizes) {
        check_axpy_matches_reference<float>(n, /*num_threads=*/1);
    }
}

// Same odd sizes but requesting multiple threads: with n this small the
// vectorized region may be empty for every OpenMP thread's static chunk,
// but the tail cleanup still must cover every element exactly once.
void test_fp64_odd_sizes_multi_threaded_request() {
    for (int64_t n : kOddSizes) {
        check_axpy_matches_reference<double>(n, /*num_threads=*/4);
    }
}

void test_fp32_odd_sizes_multi_threaded_request() {
    for (int64_t n : kOddSizes) {
        check_axpy_matches_reference<float>(n, /*num_threads=*/4);
    }
}

void test_zero_length_is_a_no_op() {
    // n=0: axpy_run must not read/write anything (guards against off-by-one
    // in the tail-boundary arithmetic at the smallest possible size).
    double* null_d = nullptr;
    float* null_f = nullptr;
    floptic::axpy_run(null_d, null_d, 1.0, 0, 1);
    floptic::axpy_run(null_f, null_f, 1.0f, 0, 1);
}

} // namespace

int main() {
    test_fp64_odd_sizes_single_threaded();
    test_fp32_odd_sizes_single_threaded();
    test_fp64_odd_sizes_multi_threaded_request();
    test_fp32_odd_sizes_multi_threaded_request();
    test_zero_length_is_a_no_op();
    return floptic_test::finish();
}
