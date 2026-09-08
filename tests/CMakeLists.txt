# tests/CMakeLists.txt — CPU-only, dependency-free CTest unit tests.
#
# These test pure host-side logic only (precision mapping, CLI parsing).
# They must build and pass without any GPU present and without discovering
# devices or running kernels.

add_executable(test_precision test_precision.cpp)
target_link_libraries(test_precision PRIVATE floptic_core)
add_test(NAME precision COMMAND test_precision)

add_executable(test_cli_parser test_cli_parser.cpp)
target_link_libraries(test_cli_parser PRIVATE floptic_core)
add_test(NAME cli_parser COMMAND test_cli_parser)

add_executable(test_kernel_warmup test_kernel_warmup.cpp)
target_link_libraries(test_kernel_warmup PRIVATE floptic_core)
add_test(NAME kernel_warmup COMMAND test_kernel_warmup)

# Effective CPU thread resolution: pure function, no device/OpenMP runtime
# needed at test-binary build time — it tests the decision logic itself
# (see include/floptic/cpu_threads.hpp) against both openmp_available=true
# and openmp_available=false, independent of how *this* test binary was
# compiled.
add_executable(test_cpu_threads test_cpu_threads.cpp)
target_link_libraries(test_cpu_threads PRIVATE floptic_core)
add_test(NAME cpu_threads COMMAND test_cpu_threads)

# AXPY numerical tail correctness: exercises the exact compute kernel used
# by vector_axpy_cpu (floptic/axpy_kernel.hpp) for small/odd sizes across
# whatever SIMD path this build compiles (scalar/AVX2/AVX-512). Not timed.
add_executable(test_axpy_kernel test_axpy_kernel.cpp)
target_link_libraries(test_axpy_kernel PRIVATE floptic_core)
add_test(NAME axpy_kernel COMMAND test_axpy_kernel)

# Deterministic operand generation: fixed PRNG known outputs, seed
# reproducibility, production-distribution range/finiteness, invalid-config
# rejection. Header-only (floptic/prng.hpp, floptic/operand_gen.hpp), no
# device discovery.
add_executable(test_operand_gen test_operand_gen.cpp)
target_link_libraries(test_operand_gen PRIVATE floptic_core)
add_test(NAME operand_gen COMMAND test_operand_gen)

# Untimed host reference GEMM: hand-computed rectangular alpha/beta cases,
# transposed-operand equivalence, dimension/size-mismatch rejection.
# Header-only (floptic/reference_gemm.hpp).
add_executable(test_reference_gemm test_reference_gemm.cpp)
target_link_libraries(test_reference_gemm PRIVATE floptic_core)
add_test(NAME reference_gemm COMMAND test_reference_gemm)

# Validation metrics + precision/depth-aware policies: exact match,
# controlled perturbations, NaN/Inf visibility, near-zero relative-error
# floor, policy ordering across precisions, integer exactness and
# overflow-bound checks. Header-only (floptic/validation_metrics.hpp).
add_executable(test_validation_metrics test_validation_metrics.cpp)
target_link_libraries(test_validation_metrics PRIVATE floptic_core)
add_test(NAME validation_metrics COMMAND test_validation_metrics)

# Typed metric representation (schema v2): enum string values, legacy
# normalization, and the centralized kernel-semantics mapping used at the
# dispatch boundary. Header-only (floptic/typed_metric.hpp,
# floptic/benchmark_status.hpp).
add_executable(test_typed_metric test_typed_metric.cpp)
target_link_libraries(test_typed_metric PRIVATE floptic_core)
add_test(NAME typed_metric COMMAND test_typed_metric)

# Report schema v2: JSON schema_version, typed metric serialization
# (FLOP/s, OP/s, B/s), explicit status retention (failed/validation_failed
# not omitted), legacy compatibility field naming, and Markdown rendering
# driven by the typed metric rather than category-based unit guessing.
add_executable(test_report_schema test_report_schema.cpp)
target_link_libraries(test_report_schema PRIVATE floptic_core)
add_test(NAME report_schema COMMAND test_report_schema)

# Dispatch-boundary normalization (schema v2): centralized kernel-semantics
# mapping used at the dispatch boundary (main.cpp) — actual byte counts
# for memory kernels, authoritative status (never inferred from a
# nonpositive legacy rate), and arithmetic-convention population. Header-
# only (floptic/dispatch_normalize.hpp).
add_executable(test_dispatch_normalize test_dispatch_normalize.cpp)
target_link_libraries(test_dispatch_normalize PRIVATE floptic_core)
add_test(NAME dispatch_normalize COMMAND test_dispatch_normalize)

# Kernel invocation failure boundary (floptic/kernel_invoke.hpp): a thrown
# std::exception or unknown exception from a kernel/callable must become
# an explicit FAILED KernelResult with a diagnostic instead of propagating
# and aborting the whole run; a successful callable's result must pass
# through unmodified. Header-only, no device discovery.
add_executable(test_kernel_invoke test_kernel_invoke.cpp)
target_link_libraries(test_kernel_invoke PRIVATE floptic_core)
add_test(NAME kernel_invoke COMMAND test_kernel_invoke)

# Aggregate run-exit-code decision (floptic/run_outcome.hpp): all-OK is
# success, any FAILED/VALIDATION_FAILED is failure, unsupported-only/no-
# success is failure, and mixed OK+unsupported is success. Header-only,
# no device discovery.
add_executable(test_run_outcome test_run_outcome.cpp)
target_link_libraries(test_run_outcome PRIVATE floptic_core)
add_test(NAME run_outcome COMMAND test_run_outcome)

# Executable-level smoke test: drives the real `floptic` binary (not just
# the parser/test-double unit tests) to cover exit-code and stdout/stderr
# behavior end-to-end. CPU-only, kept fast via trials=1 / small inner-iters.
add_test(NAME executable_smoke
    COMMAND ${CMAKE_COMMAND} -DFLOPTIC_EXE=$<TARGET_FILE:floptic>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/test_executable_smoke.cmake)

# Real-executable regression: schema v2 JSON from an actual CPU scalar_fma
# run has status "ok", a typed FLOP/s metric (kind/unit/rate/operation_count),
# and a populated arithmetic convention — proving the dispatch_normalize.hpp
# -> json_writer.cpp path end-to-end, not just direct-construction tests.
add_test(NAME executable_schema_v2
    COMMAND ${CMAKE_COMMAND} -DFLOPTIC_EXE=$<TARGET_FILE:floptic>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/test_executable_schema_v2.cmake)

# Regression test for the no-OpenMP serial work-accounting defect: a build
# that was NOT linked against OpenMP must report/execute --cpu-threads=8 as
# exactly one effective thread, not silently multiply work by 8. Only
# meaningful (and only registered) when this configuration actually has no
# OpenMP, so it fails loudly if the intended no-OpenMP CI job accidentally
# picks up a parallel runtime instead of skipping silently either way.
if(NOT OpenMP_CXX_FOUND)
    add_test(NAME executable_no_openmp_thread_accounting
        COMMAND ${CMAKE_COMMAND} -DFLOPTIC_EXE=$<TARGET_FILE:floptic>
                -P ${CMAKE_CURRENT_SOURCE_DIR}/test_no_openmp_threads.cmake)
endif()
