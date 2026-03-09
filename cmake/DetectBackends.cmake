# DetectBackends.cmake — Auto-detect available GPU backends

set(FLOPTIC_HAS_CUDA OFF)
set(FLOPTIC_HAS_HIP OFF)
set(FLOPTIC_HAS_SYCL OFF)

# --- CUDA ---
if(FLOPTIC_ENABLE_CUDA)
    # First try finding the toolkit (works when CUDA_HOME or module is loaded)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(FLOPTIC_HAS_CUDA ON)
        message(STATUS "Floptic: CUDA detected via CUDAToolkit (${CUDAToolkit_VERSION})")
    else()
        # Fallback: check if nvcc is in PATH
        find_program(NVCC_EXECUTABLE nvcc)
        if(NVCC_EXECUTABLE)
            # Set CMAKE_CUDA_COMPILER so enable_language(CUDA) works
            set(CMAKE_CUDA_COMPILER ${NVCC_EXECUTABLE} CACHE FILEPATH "CUDA compiler")
            set(FLOPTIC_HAS_CUDA ON)
            message(STATUS "Floptic: CUDA detected via nvcc (${NVCC_EXECUTABLE})")
        else()
            include(CheckLanguage)
            check_language(CUDA)
            if(CMAKE_CUDA_COMPILER)
                set(FLOPTIC_HAS_CUDA ON)
                message(STATUS "Floptic: CUDA detected via check_language")
            else()
                message(STATUS "Floptic: CUDA not detected")
            endif()
        endif()
    endif()
endif()

# --- HIP ---
if(FLOPTIC_ENABLE_HIP)
    find_package(hip QUIET)
    if(hip_FOUND)
        set(FLOPTIC_HAS_HIP ON)
        message(STATUS "Floptic: HIP detected")
    else()
        message(STATUS "Floptic: HIP not detected")
    endif()
endif()
