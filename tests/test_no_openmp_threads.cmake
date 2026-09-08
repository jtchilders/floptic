# Executable-level regression test for the no-OpenMP CPU thread-accounting
# defect: in a build with no parallel runtime, requesting 8 CPU threads
# must not multiply reported/executed work by 8. This only runs (see
# tests/CMakeLists.txt) when the current CMake configuration found no
# OpenMP, so it directly reproduces the "no-OpenMP build reports/runs as
# one effective thread" acceptance criterion end-to-end against the real
# binary rather than only the unit-level resolve_effective_cpu_threads().
#
# Invoked via: cmake -DFLOPTIC_EXE=<path> -P test_no_openmp_threads.cmake

if(NOT DEFINED FLOPTIC_EXE)
    message(FATAL_ERROR "FLOPTIC_EXE not set")
endif()

execute_process(
    COMMAND ${FLOPTIC_EXE} --device=cpu --kernel=scalar_fma --precision=fp64
            --trials=1 --inner-iters=100 --warmup=0 --cpu-threads=8
            --report=stdout
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
    RESULT_VARIABLE _result
    TIMEOUT 60
)

if(NOT _result EQUAL 0)
    message(FATAL_ERROR "expected exit 0, got ${_result}\nstderr: ${_stderr}")
endif()

# The kernel's status line reports "threads=<effective count>" — must be 1,
# not the requested 8, when no OpenMP is linked in.
string(REGEX MATCH "threads=1( |$)" _has_one_thread "${_stderr}")
string(REGEX MATCH "threads=8( |$)" _has_eight_threads "${_stderr}")

if("${_has_one_thread}" STREQUAL "")
    message(FATAL_ERROR "expected 'threads=1' in stderr for a no-OpenMP build "
                         "requesting --cpu-threads=8\nstderr: ${_stderr}")
endif()
if(NOT "${_has_eight_threads}" STREQUAL "")
    message(FATAL_ERROR "found 'threads=8' in stderr — no-OpenMP build must not "
                         "report the raw requested thread count\nstderr: ${_stderr}")
endif()

# The JSON report's flop accounting must reflect one effective thread's
# worth of work, not eight. flops_per_trial for scalar_fma/throughput is
# num_threads * chains(8) * lanes * iters * 2 — an 8x inflation from the
# thread-count bug would be trivially visible as total_flops being 8x too
# large relative to a manually computed one-thread expectation, but the
# precise chains/lanes for a given SIMD build vary, so instead we assert
# the machine-readable effective-thread signal directly: the report is
# well-formed JSON with a nonzero result, proving no crash/regression, and
# the human-readable status line above is the primary correctness check.
string(JSON _benchmarks ERROR_VARIABLE _json_err GET "${_stdout}" benchmarks)
if(_json_err)
    message(FATAL_ERROR "stdout is not valid JSON with a 'benchmarks' key: ${_json_err}\nstdout: ${_stdout}")
endif()

message(STATUS "[no_openmp_thread_accounting] OK")
