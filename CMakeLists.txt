cmake_minimum_required(VERSION 3.18)

# Options (must be before project() for CUDA arch to take effect)
option(USE_MKL "Use Intel MKL for BLAS/LAPACK/ScaLAPACK" OFF)
option(USE_CUDA "Enable CUDA" OFF)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_PYTHON "Build Python bindings (pybind11)" OFF)

# Set CUDA architectures before project() so compiler identification uses valid arch.
# Use "native" to auto-detect the GPU on the build machine (CMake 3.24+).
# Force-set into CACHE so stale cache entries (e.g. UNINITIALIZED type) never
# override the intended value.  Users can still override via -DCMAKE_CUDA_ARCHITECTURES=...
if(USE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures" FORCE)
    endif()
    # Prefer /usr/local/cuda/bin/nvcc when no explicit compiler is set,
    # because /usr/bin/nvcc may be an older system-installed version.
    if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    endif()
    project(LYNX LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA enabled: architectures ${CMAKE_CUDA_ARCHITECTURES}")
else()
    project(LYNX LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP
find_package(OpenMP REQUIRED)

# MPI
find_package(MPI REQUIRED)

# --- Math libraries (BLAS, LAPACK, ScaLAPACK) ---
if(USE_MKL)
    set(MKL_THREADING openmp)
    set(MKL_INTERFACE lp64)
    find_package(MKL CONFIG QUIET)
    if(MKL_FOUND AND TARGET MKL::MKL)
        # oneAPI CMake config uses imported targets
        set(MATH_LIBRARIES MKL::MKL)
        get_target_property(_mkl_inc MKL::MKL INTERFACE_INCLUDE_DIRECTORIES)
        if(_mkl_inc)
            set(MATH_INCLUDE_DIRS ${_mkl_inc})
        else()
            set(MATH_INCLUDE_DIRS "")
        endif()
        set(MATH_DEFINES USE_MKL)
        message(STATUS "Using Intel MKL (imported target)")
    else()
        # Fallback: find libraries manually
        include(cmake/FindMKL.cmake)
        if(MKL_FOUND)
            set(MATH_LIBRARIES "-Wl,--start-group" ${MKL_LIBRARIES} "-Wl,--end-group")
            set(MATH_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
            set(MATH_DEFINES USE_MKL)
            message(STATUS "Using Intel MKL (manual)")
        else()
            message(WARNING "MKL requested but not found, falling back to OpenBLAS")
            set(USE_MKL OFF)
        endif()
    endif()
endif()

if(NOT USE_MKL)
    # Find OpenBLAS
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)

    # Find ScaLAPACK
    find_library(SCALAPACK_LIB scalapack-openmpi
        HINTS /usr/lib/x86_64-linux-gnu /usr/lib
    )
    if(NOT SCALAPACK_LIB)
        find_library(SCALAPACK_LIB scalapack
            HINTS /usr/lib/x86_64-linux-gnu /usr/lib
        )
    endif()

    if(SCALAPACK_LIB)
        message(STATUS "Found ScaLAPACK: ${SCALAPACK_LIB}")
    else()
        message(WARNING "ScaLAPACK not found — distributed eigensolvers will be unavailable")
    endif()

    set(MATH_LIBRARIES ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
    if(SCALAPACK_LIB)
        list(APPEND MATH_LIBRARIES ${SCALAPACK_LIB})
    endif()
    set(MATH_INCLUDE_DIRS "")
    set(MATH_DEFINES "")
    if(SCALAPACK_LIB)
        list(APPEND MATH_DEFINES USE_SCALAPACK)
        message(STATUS "Using OpenBLAS + LAPACK + ScaLAPACK")
    else()
        message(STATUS "Using OpenBLAS + LAPACK")
    endif()
endif()

# JSON library (header-only)
include(FetchContent)
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# Google Test
if(BUILD_TESTS)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# libxc (required — built from submodule)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ENABLE_FORTRAN OFF CACHE BOOL "" FORCE)
set(ENABLE_CUDA ${USE_CUDA} CACHE BOOL "" FORCE)  # libxc CUDA enabled when building with USE_CUDA — accepts device pointers directly
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "" FORCE)
add_subdirectory(external/libxc EXCLUDE_FROM_ALL)
if(USE_CUDA)
    # Resolve libxc's CUDA device symbols within the xc library itself,
    # so downstream executables (even pure C++) link cleanly.
    set_target_properties(xc PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
set(LIBXC_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/external/libxc/src
    ${CMAKE_CURRENT_BINARY_DIR}/external/libxc
    CACHE INTERNAL "")

add_subdirectory(src)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
if(BUILD_PYTHON)
    add_subdirectory(python)
endif()
