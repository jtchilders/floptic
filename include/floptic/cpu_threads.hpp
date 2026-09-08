#pragma once

// Effective CPU thread count resolution.
//
// A requested thread count only means something if the binary was actually
// built with a parallel runtime (OpenMP). Without one, every CPU kernel
// executes on exactly one thread no matter what the caller asked for, and
// work/FLOP accounting must reflect that single thread — not the requested
// count. This header isolates that resolution as a small, pure, testable
// function so it can be exercised without a live device or a real OpenMP
// build.

namespace floptic {

// True when this translation unit was compiled with OpenMP support
// (FLOPTIC_HAS_OPENMP defined by CMake when find_package(OpenMP) succeeds
// and FLOPTIC_ENABLE_OPENMP was not turned off).
constexpr bool openmp_compiled_in() {
#ifdef FLOPTIC_HAS_OPENMP
    return true;
#else
    return false;
#endif
}

// Resolves a requested CPU thread count (as parsed from --cpu-threads, where
// 0 means "auto") against a detected/default logical core count and
// compile-time OpenMP availability.
//
// Rules:
//   - openmp_available == false: the build can only ever execute serially,
//     so the effective count is always exactly 1 — including when the
//     caller requested a positive count (e.g. 8) or auto (0).
//   - openmp_available == true, requested_threads > 0: the request is
//     honored as-is; it becomes the effective thread count.
//   - openmp_available == true, requested_threads <= 0 (auto): use the
//     detected logical count when it is positive, otherwise fall back to 1.
//
// Callers (scalar FMA, CPU AXPY) must use the returned value for kernel
// execution (num_threads(...) clause), status/log text, and FLOP/work
// accounting — never the raw requested_threads or a raw device count.
constexpr int resolve_effective_cpu_threads(int requested_threads,
                                             int detected_logical_count,
                                             bool openmp_available) {
    if (!openmp_available) {
        return 1;
    }
    if (requested_threads > 0) {
        return requested_threads;
    }
    return detected_logical_count > 0 ? detected_logical_count : 1;
}

} // namespace floptic
