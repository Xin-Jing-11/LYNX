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
)

target_include_directories(sparc_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sparc_core PUBLIC MPI::MPI_CXX nlohmann_json::nlohmann_json)

if(MKL_FOUND)
    target_include_directories(sparc_core PUBLIC ${MKL_INCLUDE_DIRS})
    target_link_libraries(sparc_core PUBLIC ${MKL_LIBRARIES})
    target_compile_definitions(sparc_core PUBLIC USE_MKL)
endif()

add_executable(sparc main.cpp)
target_link_libraries(sparc PRIVATE sparc_core)
