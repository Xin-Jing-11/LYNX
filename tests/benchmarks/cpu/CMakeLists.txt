# Standalone CPU kernel benchmarks for LYNX
# Build: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
# Or standalone: g++ -O3 -march=native -mavx2 -mfma -fopenmp -o bench_X bench_X.cpp

cmake_minimum_required(VERSION 3.18)
project(lynx_cpu_benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Performance flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mavx2 -mfma -DNDEBUG")

find_package(OpenMP REQUIRED)

# --- Pure standalone benchmarks (no external deps) ---
set(STANDALONE_BENCHMARKS
    bench_laplacian
    bench_hamiltonian
    bench_gradient
    bench_chebyshev
    bench_nonlocal
    bench_mixing
)

foreach(bench ${STANDALONE_BENCHMARKS})
    add_executable(${bench} ${bench}.cpp)
    target_link_libraries(${bench} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(${bench} PRIVATE -O3 -march=native -mavx2 -mfma)
endforeach()

# --- bench_xc needs libxc ---
# Try to find libxc from the LYNX build tree or system
find_library(LIBXC_LIBRARY xc
    HINTS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/external/libxc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/external/libxc/src
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
)
find_path(LIBXC_INCLUDE_DIR xc.h
    HINTS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/external/libxc/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../external/libxc/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/external/libxc
        /usr/include
        /usr/local/include
)
# Also need the build dir for generated headers (xc_version.h)
find_path(LIBXC_BUILD_INCLUDE_DIR xc_version.h
    HINTS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/external/libxc
)
# xc_funcs.h may be in the source tree
find_path(LIBXC_FUNCS_INCLUDE_DIR xc_funcs.h
    HINTS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../external/libxc/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/external/libxc/src
)

if(LIBXC_LIBRARY AND LIBXC_INCLUDE_DIR)
    add_executable(bench_xc bench_xc.cpp)
    target_link_libraries(bench_xc PRIVATE OpenMP::OpenMP_CXX ${LIBXC_LIBRARY})
    target_include_directories(bench_xc PRIVATE ${LIBXC_INCLUDE_DIR} ${LIBXC_BUILD_INCLUDE_DIR} ${LIBXC_FUNCS_INCLUDE_DIR})
    target_compile_options(bench_xc PRIVATE -O3 -march=native -mavx2 -mfma)
    message(STATUS "bench_xc: libxc found at ${LIBXC_LIBRARY}")
else()
    message(WARNING "bench_xc: libxc not found, skipping (set LIBXC_LIBRARY and LIBXC_INCLUDE_DIR)")
endif()
