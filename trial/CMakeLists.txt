cmake_minimum_required(VERSION 3.18)
project(trial LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(bench_ata bench_ata.cu)
target_link_libraries(bench_ata CUDA::cublas CUDA::cudart)
set_target_properties(bench_ata PROPERTIES CUDA_ARCHITECTURES "75;80;86;89;90;100;120")

add_executable(bench_ata2 bench_ata2.cu)
target_link_libraries(bench_ata2 CUDA::cublas CUDA::cudart)
set_target_properties(bench_ata2 PROPERTIES CUDA_ARCHITECTURES "75;80;86;89;90;100;120")

find_package(MPI REQUIRED)
add_executable(bench_ata_mpi bench_ata_mpi.cu)
target_link_libraries(bench_ata_mpi CUDA::cublas CUDA::cudart MPI::MPI_CXX)
set_target_properties(bench_ata_mpi PROPERTIES CUDA_ARCHITECTURES "75;80;86;89;90;100;120")

# NCCL benchmark — find NCCL from conda env
set(NCCL_INCLUDE_DIR "/home/xx/miniconda3/envs/llm/lib/python3.10/site-packages/nvidia/nccl/include")
set(NCCL_LIBRARY "/home/xx/miniconda3/envs/llm/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2")
add_executable(bench_ata_nccl bench_ata_nccl.cu)
target_include_directories(bench_ata_nccl PRIVATE ${NCCL_INCLUDE_DIR})
target_link_libraries(bench_ata_nccl CUDA::cublas CUDA::cudart ${NCCL_LIBRARY} MPI::MPI_CXX)
set_target_properties(bench_ata_nccl PROPERTIES CUDA_ARCHITECTURES "75;80;86;89;90;100;120")
