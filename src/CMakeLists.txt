add_library(lynx_core
    core/NDArray.cpp
    core/Lattice.cpp
    core/FDGrid.cpp
    core/Domain.cpp
    core/KPoints.cpp
    core/NumericalMethods.cpp
    core/ParameterDefaults.cpp
    parallel/MPIComm.cpp
    parallel/Parallelization.cpp
    parallel/HaloExchange.cpp
    atoms/AtomType.cpp
    atoms/Pseudopotential.cpp
    atoms/Crystal.cpp
    operators/FDStencil.cpp
    operators/Laplacian.cpp
    operators/Gradient.cpp
    operators/Hamiltonian.cpp
    operators/NonlocalProjector.cpp
    io/InputParser.cpp
    io/OutputWriter.cpp
    io/DensityIO.cpp
    electronic/Wavefunction.cpp
    electronic/ElectronDensity.cpp
    electronic/Occupation.cpp
    xc/XCFunctional.cpp
    xc/ExactExchange.cpp
    xc/ExchangePoissonSolver.cpp
    solvers/LinearSolver.cpp
    solvers/PoissonSolver.cpp
    solvers/Mixer.cpp
    solvers/KerkerPreconditioner.cpp
    solvers/EigenSolver.cpp
    physics/Energy.cpp
    physics/SCF.cpp
    physics/Forces.cpp
    physics/Stress.cpp
    physics/Electrostatics.cpp
)

# Need PIC when linking into Python shared library
if(BUILD_PYTHON)
    set_target_properties(lynx_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(lynx_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lynx_core PUBLIC MPI::MPI_CXX nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)

# Math libraries (MKL or OpenBLAS+LAPACK+ScaLAPACK)
if(MATH_LIBRARIES)
    target_link_libraries(lynx_core PUBLIC ${MATH_LIBRARIES})
endif()
if(MATH_INCLUDE_DIRS)
    target_include_directories(lynx_core PUBLIC ${MATH_INCLUDE_DIRS})
endif()
if(MATH_DEFINES)
    target_compile_definitions(lynx_core PUBLIC ${MATH_DEFINES})
endif()
# libxc
target_link_libraries(lynx_core PUBLIC xc)
target_include_directories(lynx_core SYSTEM PUBLIC ${LIBXC_INCLUDE_DIRS})

add_executable(lynx main.cpp)
target_link_libraries(lynx PRIVATE lynx_core)

if(USE_CUDA)
    target_sources(lynx_core PRIVATE
        core/gpu_common.cu
        parallel/HaloExchange.cu
        operators/Laplacian.cu
        operators/Gradient.cu
        operators/Hamiltonian.cu
        operators/NonlocalProjector.cu
        operators/ComplexOperators.cu
        operators/SOCOperators.cu
        solvers/EigenSolver.cu
        solvers/LinearSolver.cu
        xc/XCFunctional.cu
        xc/GPUExactExchange.cu
        xc/GPUExchangePoissonSolver.cu
        physics/GPUSCF.cu
        physics/GPUForceStress.cu
    )
    target_link_libraries(lynx_core PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::cusparse CUDA::curand CUDA::cufft CUDA::cufftw)
    target_compile_definitions(lynx_core PUBLIC USE_CUDA)
    set_target_properties(lynx_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(lynx_hpsi_compare ../tests/unit/test_HamiltonianCompare.cpp)
target_link_libraries(lynx_hpsi_compare PRIVATE lynx_core)
