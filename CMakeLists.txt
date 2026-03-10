cmake_minimum_required(VERSION 3.18)
project(SPARC LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(USE_MKL "Use Intel MKL" ON)
option(USE_CUDA "Enable CUDA" OFF)
option(BUILD_TESTS "Build tests" ON)

# MPI
find_package(MPI REQUIRED)

# MKL
if(USE_MKL)
    set(MKL_THREADING sequential)
    find_package(MKL CONFIG QUIET)
    if(NOT MKL_FOUND)
        include(cmake/FindMKL.cmake)
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

add_subdirectory(src)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
