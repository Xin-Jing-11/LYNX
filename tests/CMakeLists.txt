add_executable(sparc_tests
    test_NDArray.cpp
    test_Lattice.cpp
    test_FDGrid.cpp
    test_Domain.cpp
    test_FDStencil.cpp
    test_InputParser.cpp
    test_Laplacian.cpp
    test_HaloExchange.cpp
    test_Hamiltonian.cpp
    test_Wavefunction.cpp
    test_Occupation.cpp
    test_XCFunctional.cpp
    test_LinearSolver.cpp
    test_ElectronDensity.cpp
    test_Energy.cpp
    test_Forces.cpp
    test_Stress.cpp
    test_EndToEnd.cpp
)
target_link_libraries(sparc_tests PRIVATE sparc_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(sparc_tests)

# MPI-based end-to-end SCF tests (separate executable with MPI init)
add_executable(sparc_e2e_tests
    test_main_mpi.cpp
    test_EndToEnd.cpp
)
target_link_libraries(sparc_e2e_tests PRIVATE sparc_core GTest::gtest)

# Register MPI tests to run with mpirun
# Single-process test
add_test(NAME E2E_BaTiO3_1proc
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:sparc_e2e_tests> --gtest_filter=EndToEnd.BaTiO3_SCF)
set_tests_properties(E2E_BaTiO3_1proc PROPERTIES TIMEOUT 600)

add_test(NAME E2E_Si8_1proc
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:sparc_e2e_tests> --gtest_filter=EndToEnd.Si8_SCF)
set_tests_properties(E2E_Si8_1proc PROPERTIES TIMEOUT 600)

# Reference comparison tests
add_executable(sparc_compare_tests
    test_CompareRef.cpp
)
target_link_libraries(sparc_compare_tests PRIVATE sparc_core GTest::gtest)
add_test(NAME CompareRef
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:sparc_compare_tests>)
set_tests_properties(CompareRef PROPERTIES TIMEOUT 120)
