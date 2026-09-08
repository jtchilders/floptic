# CompilerFlags.cmake — Per-backend optimization flags

# CPU: aggressive optimization, native arch
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-O3)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:-Wall>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wextra>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>"
    )
    # -march=native may not be available on all platforms (e.g., macOS ARM)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
    if(HAS_MARCH_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

# CUDA: default to native arch detection
if(FLOPTIC_HAS_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    message(STATUS "Floptic: CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
