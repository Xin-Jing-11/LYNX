add_executable(lynx_tests
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
    test_KPoints.cpp
    test_DensityIO.cpp
    test_EndToEnd.cpp
)
target_link_libraries(lynx_tests PRIVATE lynx_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(lynx_tests)

# MPI-based end-to-end SCF tests (separate executable with MPI init)
add_executable(lynx_e2e_tests
    test_main_mpi.cpp
    test_EndToEnd.cpp
)
target_link_libraries(lynx_e2e_tests PRIVATE lynx_core GTest::gtest)

# Register MPI tests to run with mpirun
# Single-process test
add_test(NAME E2E_BaTiO3_1proc
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:lynx_e2e_tests> --gtest_filter=EndToEnd.BaTiO3_SCF)
set_tests_properties(E2E_BaTiO3_1proc PROPERTIES TIMEOUT 600)

add_test(NAME E2E_Si8_1proc
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:lynx_e2e_tests> --gtest_filter=EndToEnd.Si8_SCF)
set_tests_properties(E2E_Si8_1proc PROPERTIES TIMEOUT 600)

# Reference comparison tests
add_executable(lynx_compare_tests
    test_CompareRef.cpp
)
target_link_libraries(lynx_compare_tests PRIVATE lynx_core GTest::gtest)
add_test(NAME CompareRef
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:lynx_compare_tests>)
set_tests_properties(CompareRef PROPERTIES TIMEOUT 120)

# GPU kernel tests
if(USE_CUDA)
    add_executable(lynx_gpu_tests test_GPUKernels.cu)
    target_link_libraries(lynx_gpu_tests PRIVATE lynx_core GTest::gtest)
    set_target_properties(lynx_gpu_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME GPUKernels COMMAND $<TARGET_FILE:lynx_gpu_tests>)
    set_tests_properties(GPUKernels PROPERTIES TIMEOUT 120)

    add_executable(lynx_gpu_context_tests test_GPUContext.cu)
    target_link_libraries(lynx_gpu_context_tests PRIVATE lynx_core)
    set_target_properties(lynx_gpu_context_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME GPUContext COMMAND $<TARGET_FILE:lynx_gpu_context_tests>)
    set_tests_properties(GPUContext PROPERTIES TIMEOUT 60)

    add_executable(lynx_gpu_algo_tests test_GPUAlgorithms.cu)
    target_link_libraries(lynx_gpu_algo_tests PRIVATE lynx_core)
    set_target_properties(lynx_gpu_algo_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME GPUAlgorithms COMMAND $<TARGET_FILE:lynx_gpu_algo_tests>)
    set_tests_properties(GPUAlgorithms PROPERTIES TIMEOUT 120)

    add_executable(lynx_gpu_scf_tests test_GPUSCF.cu)
    target_link_libraries(lynx_gpu_scf_tests PRIVATE lynx_core GTest::gtest)
    set_target_properties(lynx_gpu_scf_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME GPUSCF COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:lynx_gpu_scf_tests>)
    set_tests_properties(GPUSCF PROPERTIES TIMEOUT 600)
endif()
