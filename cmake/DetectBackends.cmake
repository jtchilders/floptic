# DetectBackends.cmake — Auto-detect available GPU backends

set(FLOPTIC_HAS_CUDA OFF)
set(FLOPTIC_HAS_HIP OFF)
set(FLOPTIC_HAS_SYCL OFF)

# --- CUDA ---
if(FLOPTIC_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(FLOPTIC_HAS_CUDA ON)
            message(STATUS "Floptic: CUDA detected (${CUDAToolkit_VERSION})")
        else()
            message(STATUS "Floptic: CUDA compiler found but toolkit not found")
        endif()
    else()
        message(STATUS "Floptic: CUDA not detected")
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
