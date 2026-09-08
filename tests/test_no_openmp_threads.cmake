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
# worth of work, not eight. On the AVX2/FP64 path exercised by this CI
# matrix (lanes=4), flops_per_trial for scalar_fma at inner-iters=100 and
# one effective thread is:
#   throughput: threads(1) * chains(8) * lanes(4) * iters(100) * 2 = 6400
#   latency:    threads(1) * lanes(4)             * iters(100) * 2 =  800
# An 8x inflation from the thread-count bug (using the raw requested
# --cpu-threads=8 instead of the resolved effective count) would instead
# report 51200 / 6400 respectively — asserting the exact expected values
# below fails loudly against either the old buggy total or any other
# unexpected drift, rather than only checking "nonzero".
string(JSON _num_benchmarks ERROR_VARIABLE _json_err LENGTH "${_stdout}" benchmarks)
if(_json_err)
    message(FATAL_ERROR "stdout is not valid JSON with a 'benchmarks' key: ${_json_err}\nstdout: ${_stdout}")
endif()

set(_throughput_flops)
set(_latency_flops)
if(_num_benchmarks GREATER 0)
    math(EXPR _last_index "${_num_benchmarks} - 1")
    foreach(_i RANGE 0 ${_last_index})
        string(JSON _kernel GET "${_stdout}" benchmarks ${_i} kernel)
        if(NOT _kernel STREQUAL "scalar_fma")
            continue()
        endif()
        string(JSON _mode GET "${_stdout}" benchmarks ${_i} mode)
        string(JSON _flops GET "${_stdout}" benchmarks ${_i} results total_flops)
        if(_mode STREQUAL "throughput")
            set(_throughput_flops "${_flops}")
        elseif(_mode STREQUAL "latency")
            set(_latency_flops "${_flops}")
        endif()
    endforeach()
endif()

if(NOT DEFINED _throughput_flops OR "${_throughput_flops}" STREQUAL "")
    message(FATAL_ERROR "no scalar_fma throughput benchmark entry found in JSON output\nstdout: ${_stdout}")
endif()
if(NOT DEFINED _latency_flops OR "${_latency_flops}" STREQUAL "")
    message(FATAL_ERROR "no scalar_fma latency benchmark entry found in JSON output\nstdout: ${_stdout}")
endif()

if(NOT _throughput_flops EQUAL 6400)
    message(FATAL_ERROR "scalar_fma throughput total_flops expected 6400 "
                         "(1 effective thread x 8 chains x 4 lanes x 100 iters x 2) "
                         "but got ${_throughput_flops} — the old thread-count bug would "
                         "report 51200 (8x inflation)\nstdout: ${_stdout}")
endif()
if(NOT _latency_flops EQUAL 800)
    message(FATAL_ERROR "scalar_fma latency total_flops expected 800 "
                         "(1 effective thread x 4 lanes x 100 iters x 2) "
                         "but got ${_latency_flops} — the old thread-count bug would "
                         "report 6400 (8x inflation)\nstdout: ${_stdout}")
endif()

message(STATUS "[no_openmp_thread_accounting] OK (throughput=${_throughput_flops}, latency=${_latency_flops})")
