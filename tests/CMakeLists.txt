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
