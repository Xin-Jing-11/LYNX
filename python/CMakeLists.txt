include(FetchContent)

# Fetch pybind11
FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# Disable LTO for CUDA builds — it breaks fatbin symbol resolution
if(USE_CUDA)
    set(PYBIND11_LTO_CXX_FLAGS "")
    set(PYBIND11_LTO_LINKER_FLAGS "")
endif()

# The pybind11 module
pybind11_add_module(_core NO_EXTRAS
    src/bindings.cpp
    src/bind_types.cpp
    src/bind_core.cpp
    src/bind_operators.cpp
    src/bind_solvers.cpp
    src/bind_electronic.cpp
    src/bind_xc.cpp
    src/bind_physics.cpp
    src/bind_atoms.cpp
    src/bind_calculator.cpp
    src/Calculator.cpp
)

target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(_core PRIVATE lynx_core)

# When CUDA is enabled, enable separable compilation and device linking
# so that device code from lynx_core (static lib) is properly linked
# into the _core shared library.
if(USE_CUDA)
    set_target_properties(_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        INTERPROCEDURAL_OPTIMIZATION OFF
    )
endif()

# Put the .so in the lynx package directory for development use
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lynx
)

# GIL release for compute-heavy calls is done per-binding via py::call_guard

# Install the extension module into the lynx package directory
install(TARGETS _core DESTINATION lynx)
install(DIRECTORY lynx/ DESTINATION lynx FILES_MATCHING PATTERN "*.py")
