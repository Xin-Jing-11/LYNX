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
)
target_link_libraries(sparc_tests PRIVATE sparc_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(sparc_tests)
