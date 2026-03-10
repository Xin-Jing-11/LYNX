add_library(sparc_core
    core/NDArray.cpp
    core/Lattice.cpp
    core/FDGrid.cpp
    core/Domain.cpp
    parallel/MPIComm.cpp
    parallel/CartTopology.cpp
    parallel/Parallelization.cpp
    parallel/D2DTransfer.cpp
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
    electronic/Wavefunction.cpp
    electronic/ElectronDensity.cpp
    electronic/Occupation.cpp
    xc/XCFunctional.cpp
    solvers/LinearSolver.cpp
    solvers/PoissonSolver.cpp
    solvers/Mixer.cpp
    solvers/EigenSolver.cpp
    physics/Energy.cpp
    physics/SCF.cpp
)

target_include_directories(sparc_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sparc_core PUBLIC MPI::MPI_CXX nlohmann_json::nlohmann_json)

# Math libraries (MKL or OpenBLAS+LAPACK+ScaLAPACK)
if(MATH_LIBRARIES)
    target_link_libraries(sparc_core PUBLIC ${MATH_LIBRARIES})
endif()
if(MATH_INCLUDE_DIRS)
    target_include_directories(sparc_core PUBLIC ${MATH_INCLUDE_DIRS})
endif()
if(MATH_DEFINES)
    target_compile_definitions(sparc_core PUBLIC ${MATH_DEFINES})
endif()

add_executable(sparc main.cpp)
target_link_libraries(sparc PRIVATE sparc_core)
