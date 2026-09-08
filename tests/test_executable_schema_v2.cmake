# Real-executable regression test for JSON schema v2 (tests/CMakeLists.txt
# "regression that existing executable CPU JSON has typed metric and status
# ok"). Unlike tests/test_report_schema.cpp (which builds Report/
# BenchmarkEntry structs directly), this drives the actual `floptic`
# binary end-to-end against the real CPU scalar_fma kernel and parses its
# generated JSON, proving the full dispatch_normalize.hpp -> json_writer.cpp
# path produces a numeric schema_version 2, status "ok", and a typed
# FLOP/s metric with a populated arithmetic convention and nonzero
# operation_count for a real (not test-double) measurement.
#
# Invoked via: cmake -DFLOPTIC_EXE=<path> -P test_executable_schema_v2.cmake

if(NOT DEFINED FLOPTIC_EXE)
    message(FATAL_ERROR "FLOPTIC_EXE not set")
endif()

execute_process(
    COMMAND ${FLOPTIC_EXE} --device=cpu --kernel=scalar_fma --precision=fp64
            --trials=1 --inner-iters=100 --warmup=0
            --report=stdout
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
    RESULT_VARIABLE _result
    TIMEOUT 60
)

if(NOT _result EQUAL 0)
    message(FATAL_ERROR "expected exit 0, got ${_result}\nstderr: ${_stderr}")
endif()

# schema_version must be the numeric integer 2.
string(JSON _schema_version ERROR_VARIABLE _json_err GET "${_stdout}" schema_version)
if(_json_err)
    message(FATAL_ERROR "stdout is not valid JSON with a 'schema_version' key: ${_json_err}\nstdout: ${_stdout}")
endif()
if(NOT _schema_version EQUAL 2)
    message(FATAL_ERROR "expected schema_version 2, got ${_schema_version}\nstdout: ${_stdout}")
endif()

# Find the scalar_fma/throughput benchmark entry.
string(JSON _num_benchmarks ERROR_VARIABLE _json_err LENGTH "${_stdout}" benchmarks)
if(_json_err)
    message(FATAL_ERROR "stdout is not valid JSON with a 'benchmarks' key: ${_json_err}\nstdout: ${_stdout}")
endif()

set(_found_index -1)
if(_num_benchmarks GREATER 0)
    math(EXPR _last_index "${_num_benchmarks} - 1")
    foreach(_i RANGE 0 ${_last_index})
        string(JSON _kernel GET "${_stdout}" benchmarks ${_i} kernel)
        string(JSON _mode GET "${_stdout}" benchmarks ${_i} mode)
        if(_kernel STREQUAL "scalar_fma" AND _mode STREQUAL "throughput")
            set(_found_index ${_i})
            break()
        endif()
    endforeach()
endif()

if(_found_index EQUAL -1)
    message(FATAL_ERROR "no scalar_fma/throughput benchmark entry found in JSON output\nstdout: ${_stdout}")
endif()

# Status must be the explicit string "ok" for a real, successful CPU
# measurement — never silently defaulted or inferred from a rate.
string(JSON _status GET "${_stdout}" benchmarks ${_found_index} status)
if(NOT _status STREQUAL "ok")
    message(FATAL_ERROR "expected status 'ok' for a real CPU scalar_fma run, got '${_status}'\nstdout: ${_stdout}")
endif()

# Typed metric: kind, unit, rate, and operation_count must all be present
# and consistent — never inferred from category inside the serializer.
string(JSON _kind GET "${_stdout}" benchmarks ${_found_index} results metric kind)
if(NOT _kind STREQUAL "floating_point_operations")
    message(FATAL_ERROR "expected metric kind 'floating_point_operations', got '${_kind}'\nstdout: ${_stdout}")
endif()

string(JSON _unit GET "${_stdout}" benchmarks ${_found_index} results metric unit)
if(NOT _unit STREQUAL "FLOP/s")
    message(FATAL_ERROR "expected metric unit 'FLOP/s', got '${_unit}'\nstdout: ${_stdout}")
endif()

string(JSON _rate ERROR_VARIABLE _rate_err GET "${_stdout}" benchmarks ${_found_index} results metric rate_per_second)
if(_rate_err OR _rate LESS_EQUAL 0)
    message(FATAL_ERROR "expected a positive metric rate_per_second, got '${_rate}' (err=${_rate_err})\nstdout: ${_stdout}")
endif()

string(JSON _op_count ERROR_VARIABLE _count_err GET "${_stdout}" benchmarks ${_found_index} results metric operation_count)
if(_count_err OR _op_count LESS_EQUAL 0)
    message(FATAL_ERROR "expected a positive operation_count, got '${_op_count}' (err=${_count_err})\nstdout: ${_stdout}")
endif()

# Arithmetic convention must be populated for this real FMA-based kernel —
# the dispatch boundary must pass it through, not just define it unused.
string(JSON _convention ERROR_VARIABLE _conv_err GET "${_stdout}" benchmarks ${_found_index} results metric arithmetic_convention)
if(_conv_err OR "${_convention}" STREQUAL "")
    message(FATAL_ERROR "expected a nonempty arithmetic_convention for scalar_fma, got err=${_conv_err} value='${_convention}'\nstdout: ${_stdout}")
endif()

# Legacy compatibility field only, never the ambiguous authoritative key.
string(JSON _legacy ERROR_VARIABLE _legacy_err GET "${_stdout}" benchmarks ${_found_index} results legacy_gflops)
if(_legacy_err)
    message(FATAL_ERROR "expected a 'legacy_gflops' compatibility field: ${_legacy_err}\nstdout: ${_stdout}")
endif()

message(STATUS "[executable_schema_v2] OK (schema_version=${_schema_version}, status=${_status}, kind=${_kind}, unit=${_unit}, rate=${_rate}, op_count=${_op_count})")
