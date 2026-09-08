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

# Executable-level smoke test: drives the real `floptic` binary (not just
# the parser/test-double unit tests) to cover exit-code and stdout/stderr
# behavior end-to-end. CPU-only, kept fast via trials=1 / small inner-iters.
add_test(NAME executable_smoke
    COMMAND ${CMAKE_COMMAND} -DFLOPTIC_EXE=$<TARGET_FILE:floptic>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/test_executable_smoke.cmake)
