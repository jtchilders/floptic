// Hardware-independent tests for effective CPU thread resolution
// (floptic/cpu_threads.hpp). Pure function, no device discovery, no OpenMP
// runtime dependency — exercises the exact logic used by scalar_fma_cpu and
// vector_axpy_cpu to turn a requested thread count into what actually runs.
#include "floptic/cpu_threads.hpp"
#include "check.hpp"

using floptic::resolve_effective_cpu_threads;

namespace {

void test_no_openmp_always_serial_regardless_of_request() {
    // The core regression this task exists to fix: requesting 8 threads in
    // a build without a parallel runtime must not multiply accounted work
    // by 8 — the effective count is always exactly 1.
    CHECK_EQ(resolve_effective_cpu_threads(/*requested=*/8, /*detected=*/16,
                                            /*openmp_available=*/false), 1);
}

void test_no_openmp_auto_request_is_still_serial() {
    CHECK_EQ(resolve_effective_cpu_threads(/*requested=*/0, /*detected=*/16,
                                            /*openmp_available=*/false), 1);
}

void test_no_openmp_serial_even_with_zero_detected_cores() {
    CHECK_EQ(resolve_effective_cpu_threads(/*requested=*/0, /*detected=*/0,
                                            /*openmp_available=*/false), 1);
}

void test_openmp_positive_request_is_honored() {
    CHECK_EQ(resolve_effective_cpu_threads(/*requested=*/8, /*detected=*/16,
                                            /*openmp_available=*/true), 8);
}

void test_openmp_auto_uses_detected_count() {
    CHECK_EQ(resolve_effective_cpu_threads(/*requested=*/0, /*detected=*/12,
                                            /*openmp_available=*/true), 12);
}

void test_openmp_auto_falls_back_to_one_when_detection_unavailable() {
    CHECK_EQ(resolve_effective_cpu_threads(/*requested=*/0, /*detected=*/0,
                                            /*openmp_available=*/true), 1);
    CHECK_EQ(resolve_effective_cpu_threads(/*requested=*/0, /*detected=*/-1,
                                            /*openmp_available=*/true), 1);
}

} // namespace

int main() {
    test_no_openmp_always_serial_regardless_of_request();
    test_no_openmp_auto_request_is_still_serial();
    test_no_openmp_serial_even_with_zero_detected_cores();
    test_openmp_positive_request_is_honored();
    test_openmp_auto_uses_detected_count();
    test_openmp_auto_falls_back_to_one_when_detection_unavailable();
    return floptic_test::finish();
}
