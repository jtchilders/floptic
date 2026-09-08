# Executable-level smoke test for the built `floptic` binary.
#
# Unlike the parser/test-double unit tests, this exercises the real
# process: exit codes, stdout/stderr behavior, and end-to-end kernel
# selection against the live CPU backend. It is CPU-only and keeps
# runtime short (trials=1, tiny inner-iters) so it stays cheap in CI.
#
# Invoked via: cmake -DFLOPTIC_EXE=<path> -P test_executable_smoke.cmake

if(NOT DEFINED FLOPTIC_EXE)
    message(FATAL_ERROR "FLOPTIC_EXE not set")
endif()

set(_fail_count 0)

# Runs FLOPTIC_EXE with the given args, checks the resulting exit code
# against expected_code, and optionally checks stdout/stderr with regex
# (empty string skips that check). Reports failures instead of aborting
# immediately so the whole suite runs and all failures are visible.
function(run_case name expected_code stdout_regex stderr_regex)
    execute_process(
        COMMAND ${FLOPTIC_EXE} ${ARGN}
        OUTPUT_VARIABLE _stdout
        ERROR_VARIABLE _stderr
        RESULT_VARIABLE _result
        TIMEOUT 60
    )

    if(NOT _result EQUAL expected_code)
        message(SEND_ERROR "[${name}] expected exit ${expected_code}, got ${_result}\n"
                            "  args: ${ARGN}\n  stdout: ${_stdout}\n  stderr: ${_stderr}")
        math(EXPR _fail_count "${_fail_count}+1")
        set(_fail_count ${_fail_count} PARENT_SCOPE)
        return()
    endif()

    if(NOT "${stdout_regex}" STREQUAL "")
        string(REGEX MATCH "${stdout_regex}" _match "${_stdout}")
        if("${_match}" STREQUAL "")
            message(SEND_ERROR "[${name}] stdout did not match /${stdout_regex}/\n  stdout: ${_stdout}")
            math(EXPR _fail_count "${_fail_count}+1")
            set(_fail_count ${_fail_count} PARENT_SCOPE)
            return()
        endif()
    endif()

    if(NOT "${stderr_regex}" STREQUAL "")
        string(REGEX MATCH "${stderr_regex}" _match "${_stderr}")
        if("${_match}" STREQUAL "")
            message(SEND_ERROR "[${name}] stderr did not match /${stderr_regex}/\n  stderr: ${_stderr}")
            math(EXPR _fail_count "${_fail_count}+1")
            set(_fail_count ${_fail_count} PARENT_SCOPE)
            return()
        endif()
    endif()

    message(STATUS "[${name}] OK")
endfunction()

# 1. Malformed integer -> nonzero exit, no crash/throw.
run_case("malformed_integer" 1 "" "invalid integer value" --trials=abc)

# 2. Unknown precision -> nonzero exit, no silent FP64 fallback.
run_case("unknown_precision" 1 "" "unrecognized value" --precision=bogus)

# 3. Nonexistent kernel -> nonzero exit with clear diagnostic.
run_case("nonexistent_kernel" 1 "" "Unknown kernel" --kernel=does_not_exist_kernel_xyz)

# 4. --info produces parseable JSON on stdout with no --output needed.
execute_process(
    COMMAND ${FLOPTIC_EXE} --info
    OUTPUT_VARIABLE _info_stdout
    RESULT_VARIABLE _info_result
    TIMEOUT 60
)
if(NOT _info_result EQUAL 0)
    message(SEND_ERROR "[info_json] expected exit 0, got ${_info_result}")
    math(EXPR _fail_count "${_fail_count}+1")
else()
    string(JSON _devices ERROR_VARIABLE _json_err GET "${_info_stdout}" devices)
    if(_json_err)
        message(SEND_ERROR "[info_json] stdout is not valid JSON with a 'devices' key: ${_json_err}\n  stdout: ${_info_stdout}")
        math(EXPR _fail_count "${_fail_count}+1")
    else()
        message(STATUS "[info_json] OK")
    endif()
endif()

# 5. Exact --kernel filtering: only the requested kernel's rows appear.
execute_process(
    COMMAND ${FLOPTIC_EXE} --kernel=scalar_fma --device=cpu --trials=1 --inner-iters=100
    OUTPUT_VARIABLE _kf_stdout
    ERROR_VARIABLE _kf_stderr
    RESULT_VARIABLE _kf_result
    TIMEOUT 60
)
if(NOT _kf_result EQUAL 0)
    message(SEND_ERROR "[exact_kernel_filter] expected exit 0, got ${_kf_result}\n  stderr: ${_kf_stderr}")
    math(EXPR _fail_count "${_fail_count}+1")
else()
    string(REGEX MATCH "scalar_fma" _has_scalar "${_kf_stderr}")
    string(REGEX MATCH "vector_axpy" _has_vector "${_kf_stderr}")
    if("${_has_scalar}" STREQUAL "" OR NOT "${_has_vector}" STREQUAL "")
        message(SEND_ERROR "[exact_kernel_filter] expected only scalar_fma rows, found vector_axpy or missing scalar_fma\n  stderr: ${_kf_stderr}")
        math(EXPR _fail_count "${_fail_count}+1")
    else()
        message(STATUS "[exact_kernel_filter] OK")
    endif()
endif()

# 6. Duplicate/overlapping categories must not duplicate kernel execution:
#    --kernels=scalar,scalar should run scalar_fma exactly once per
#    precision/mode combination, not twice.
execute_process(
    COMMAND ${FLOPTIC_EXE} --kernels=scalar,scalar --device=cpu --precision=fp64 --trials=1 --inner-iters=100
    OUTPUT_VARIABLE _dup_stdout
    ERROR_VARIABLE _dup_stderr
    RESULT_VARIABLE _dup_result
    TIMEOUT 60
)
if(NOT _dup_result EQUAL 0)
    message(SEND_ERROR "[duplicate_category] expected exit 0, got ${_dup_result}\n  stderr: ${_dup_stderr}")
    math(EXPR _fail_count "${_fail_count}+1")
else()
    string(REGEX MATCHALL "--- scalar_fma \\| FP64 \\| throughput ---" _throughput_matches "${_dup_stderr}")
    list(LENGTH _throughput_matches _throughput_count)
    if(NOT _throughput_count EQUAL 1)
        message(SEND_ERROR "[duplicate_category] expected scalar_fma/FP64/throughput to run exactly once, ran ${_throughput_count} times\n  stderr: ${_dup_stderr}")
        math(EXPR _fail_count "${_fail_count}+1")
    else()
        message(STATUS "[duplicate_category] OK")
    endif()
endif()

# 7. Empty supported-combination (precision unsupported by every selected
#    kernel) must fail the process, not silently produce an empty report.
run_case("empty_combination_fails" 1 "" "no benchmarks were executed"
    --kernels=scalar --precision=int8 --device=cpu --trials=1 --inner-iters=100)

if(_fail_count GREATER 0)
    message(FATAL_ERROR "${_fail_count} executable smoke case(s) failed")
endif()

message(STATUS "All executable smoke cases passed.")
