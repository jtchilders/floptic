cmake_minimum_required(VERSION 3.22)
project(floptic VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Options ---
option(FLOPTIC_ENABLE_CUDA "Enable CUDA backend" ON)
option(FLOPTIC_ENABLE_HIP  "Enable HIP backend"  OFF)

# --- Backend Detection ---
include(cmake/DetectBackends.cmake)

# --- Compiler Flags ---
include(cmake/CompilerFlags.cmake)

# --- nlohmann/json (header-only) ---
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(json)

# --- Core library (always built) ---
add_library(floptic_core STATIC
    src/cli_parser.cpp
    src/kernel_registry.cpp
    src/device/cpu_device_info.cpp
    src/harness/timer_cpu.cpp
    src/harness/warmup.cpp
    src/harness/validator.cpp
    src/harness/power_monitor.cpp
    src/report/json_writer.cpp
    src/kernels/scalar/scalar_fma_cpu.cpp
    src/kernels/vector/vector_axpy_cpu.cpp
)
target_include_directories(floptic_core PUBLIC include)
target_link_libraries(floptic_core PUBLIC nlohmann_json::nlohmann_json)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(floptic_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(floptic_core PUBLIC FLOPTIC_HAS_OPENMP)
    message(STATUS "Floptic: OpenMP enabled")
endif()

# --- CUDA backend ---
if(FLOPTIC_HAS_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_library(floptic_cuda STATIC
        src/device/cuda_device_info.cu
        src/harness/timer_cuda.cu
        src/kernels/scalar/scalar_fma_cuda.cu
        src/kernels/vector/vector_axpy_cuda.cu
        src/kernels/matrix/gemm_cublas.cu
    )
    target_include_directories(floptic_cuda PUBLIC include)
    target_link_libraries(floptic_cuda PUBLIC nlohmann_json::nlohmann_json)
    target_compile_definitions(floptic_cuda PUBLIC FLOPTIC_HAS_CUDA)

    # Link cuBLAS for GEMM kernels
    find_library(CUBLAS_LIBRARY cublas HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64)
    if(CUBLAS_LIBRARY)
        target_link_libraries(floptic_cuda PUBLIC ${CUBLAS_LIBRARY})
        message(STATUS "Floptic: cuBLAS found for GEMM benchmarks")
    else()
        message(WARNING "Floptic: cuBLAS not found — GEMM benchmarks will fail at runtime")
    endif()

    # Link NVML for power monitoring (user-level)
    find_library(NVML_LIBRARY nvidia-ml HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/lib/x86_64-linux-gnu)
    if(NVML_LIBRARY)
        target_link_libraries(floptic_cuda PUBLIC ${NVML_LIBRARY})
        target_compile_definitions(floptic_cuda PUBLIC FLOPTIC_HAS_NVML)
        message(STATUS "Floptic: NVML found for power monitoring")
    endif()

    # Propagate CUDA flag to core
    target_compile_definitions(floptic_core PUBLIC FLOPTIC_HAS_CUDA)
endif()

# --- Main executable ---
add_executable(floptic src/main.cpp)
target_link_libraries(floptic PRIVATE floptic_core)
if(FLOPTIC_HAS_CUDA)
    target_link_libraries(floptic PRIVATE floptic_cuda)
endif()

# --- Summary ---
message(STATUS "")
message(STATUS "=== Floptic Build Configuration ===")
message(STATUS "  Version:    ${PROJECT_VERSION}")
message(STATUS "  CUDA:       ${FLOPTIC_HAS_CUDA}")
message(STATUS "  HIP:        ${FLOPTIC_HAS_HIP}")
message(STATUS "  OpenMP:     ${OpenMP_CXX_FOUND}")
message(STATUS "===================================")
message(STATUS "")
