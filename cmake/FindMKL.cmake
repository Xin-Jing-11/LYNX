# FindMKL.cmake — Fallback MKL finder using MKLROOT environment variable
#
# Sets:
#   MKL_FOUND
#   MKL_INCLUDE_DIRS
#   MKL_LIBRARIES

if(DEFINED ENV{MKLROOT})
    set(MKLROOT "$ENV{MKLROOT}")
else()
    set(MKLROOT "/opt/intel/mkl")
endif()

find_path(MKL_INCLUDE_DIR mkl.h
    PATHS ${MKLROOT}/include
    NO_DEFAULT_PATH
)

find_library(MKL_CORE_LIB mkl_core
    PATHS ${MKLROOT}/lib/intel64 ${MKLROOT}/lib
    NO_DEFAULT_PATH
)

find_library(MKL_SEQUENTIAL_LIB mkl_sequential
    PATHS ${MKLROOT}/lib/intel64 ${MKLROOT}/lib
    NO_DEFAULT_PATH
)

find_library(MKL_LP64_LIB mkl_intel_lp64
    PATHS ${MKLROOT}/lib/intel64 ${MKLROOT}/lib
    NO_DEFAULT_PATH
)

# Try GCC interface if Intel not found
if(NOT MKL_LP64_LIB)
    find_library(MKL_LP64_LIB mkl_gf_lp64
        PATHS ${MKLROOT}/lib/intel64 ${MKLROOT}/lib
        NO_DEFAULT_PATH
    )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG
    MKL_INCLUDE_DIR MKL_CORE_LIB MKL_SEQUENTIAL_LIB MKL_LP64_LIB
)

if(MKL_FOUND)
    set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
    set(MKL_LIBRARIES ${MKL_LP64_LIB} ${MKL_SEQUENTIAL_LIB} ${MKL_CORE_LIB} pthread m dl)
    message(STATUS "MKL found at ${MKLROOT}")
endif()
