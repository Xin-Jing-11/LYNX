# =============================================================================
# Unit tests — fast component tests (~seconds each)
# =============================================================================
add_executable(lynx_unit_tests
    unit/test_DeviceArray.cpp
    unit/test_Lattice.cpp
    unit/test_FDGrid.cpp
    unit/test_Domain.cpp
    unit/test_FDStencil.cpp
    unit/test_InputParser.cpp
    unit/test_Laplacian.cpp
    unit/test_HaloExchange.cpp
    unit/test_Hamiltonian.cpp
    unit/test_Wavefunction.cpp
    unit/test_Occupation.cpp
    unit/test_XCFunctional.cpp
    unit/test_LinearSolver.cpp
    unit/test_ElectronDensity.cpp
    unit/test_Energy.cpp
    unit/test_Forces.cpp
    unit/test_Stress.cpp
    unit/test_KPoints.cpp
    unit/test_DensityIO.cpp
    unit/test_ResultWriter.cpp
)
target_link_libraries(lynx_unit_tests PRIVATE lynx_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(lynx_unit_tests PROPERTIES LABELS "unit")

# Reference comparison tests
add_executable(lynx_compare_tests
    unit/test_CompareRef.cpp
)
target_link_libraries(lynx_compare_tests PRIVATE lynx_core GTest::gtest)
add_test(NAME CompareRef
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:lynx_compare_tests>)
set_tests_properties(CompareRef PROPERTIES TIMEOUT 120 LABELS "unit"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# libxc comparison tests
add_executable(lynx_libxc_tests unit/test_LibxcCompare.cpp)
target_link_libraries(lynx_libxc_tests PRIVATE lynx_core GTest::gtest)
add_test(NAME LibxcCompare
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:lynx_libxc_tests>)
set_tests_properties(LibxcCompare PROPERTIES TIMEOUT 120 LABELS "unit")

# SCAN functional unit tests (hand-coded + libxc comparison)
add_executable(lynx_scan_tests unit/test_SCANFunctional.cpp)
target_link_libraries(lynx_scan_tests PRIVATE lynx_core GTest::gtest)
add_test(NAME SCANFunctional COMMAND $<TARGET_FILE:lynx_scan_tests>)
set_tests_properties(SCANFunctional PROPERTIES TIMEOUT 120 LABELS "unit")

# =============================================================================
# End-to-end tests — full SCF runs (~minutes each)
# =============================================================================

# MPI-based end-to-end SCF tests (separate executable with MPI init)
add_executable(lynx_e2e_tests
    e2e/test_main_mpi.cpp
    e2e/test_EndToEnd.cpp
)
target_link_libraries(lynx_e2e_tests PRIVATE lynx_core GTest::gtest)

# Register MPI tests to run with mpirun
add_test(NAME E2E_BaTiO3_1proc
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:lynx_e2e_tests> --gtest_filter=EndToEnd.BaTiO3_SCF)
set_tests_properties(E2E_BaTiO3_1proc PROPERTIES TIMEOUT 600 LABELS "e2e"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_test(NAME E2E_Si8_1proc
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:lynx_e2e_tests> --gtest_filter=EndToEnd.Si8_SCF)
set_tests_properties(E2E_Si8_1proc PROPERTIES TIMEOUT 600 LABELS "e2e"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_test(NAME E2E_PtAu_SOC
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:lynx_e2e_tests> --gtest_filter=EndToEnd.PtAu_SOC)
set_tests_properties(E2E_PtAu_SOC PROPERTIES TIMEOUT 1800 LABELS "e2e"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# =============================================================================
# GPU tests
# =============================================================================
if(USE_CUDA)
    # GPU unit tests
    add_executable(lynx_gpu_tests unit/gpu/test_GPUKernels.cu)
    target_link_libraries(lynx_gpu_tests PRIVATE lynx_core GTest::gtest)
    set_target_properties(lynx_gpu_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME GPUKernels COMMAND $<TARGET_FILE:lynx_gpu_tests>)
    set_tests_properties(GPUKernels PROPERTIES TIMEOUT 120 LABELS "unit")

    add_executable(lynx_gpu_context_tests unit/gpu/test_GPUContext.cu)
    target_link_libraries(lynx_gpu_context_tests PRIVATE lynx_core)
    set_target_properties(lynx_gpu_context_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME GPUContext COMMAND $<TARGET_FILE:lynx_gpu_context_tests>)
    set_tests_properties(GPUContext PROPERTIES TIMEOUT 60 LABELS "unit")

    add_executable(lynx_gpu_algo_tests unit/gpu/test_GPUAlgorithms.cu)
    target_link_libraries(lynx_gpu_algo_tests PRIVATE lynx_core)
    set_target_properties(lynx_gpu_algo_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME GPUAlgorithms COMMAND $<TARGET_FILE:lynx_gpu_algo_tests>)
    set_tests_properties(GPUAlgorithms PROPERTIES TIMEOUT 120 LABELS "unit")

    # GPU E2E tests
    add_executable(lynx_gpu_scf_tests e2e/test_GPUSCF.cu)
    target_link_libraries(lynx_gpu_scf_tests PRIVATE lynx_core GTest::gtest)
    set_target_properties(lynx_gpu_scf_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME GPUSCF COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:lynx_gpu_scf_tests>)
    set_tests_properties(GPUSCF PROPERTIES TIMEOUT 600 LABELS "e2e"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    add_executable(lynx_gpu_soc_tests e2e/test_GPUSOC.cu)
    target_link_libraries(lynx_gpu_soc_tests PRIVATE lynx_core GTest::gtest)
    set_target_properties(lynx_gpu_soc_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME GPUSOC COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:lynx_gpu_soc_tests>)
    set_tests_properties(GPUSOC PROPERTIES TIMEOUT 600 LABELS "e2e")
endif()
