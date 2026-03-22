// Distributed A^T * A benchmark with MPI
// Each rank owns A_local (M_local x N), computes local C = A_local^T * A_local,
// then MPI_Allreduce to get global C = A_global^T * A_global.
//
// Tests:
// 1. Correctness: distributed result == single-process reference
// 2. Performance: compute vs communication breakdown
// 3. Scaling: 1, 2, 4, 8 ranks (all sharing 1 GPU)
//
// AI analogy: this is exactly data-parallel gradient AllReduce.
// Each GPU has local data → local matmul → AllReduce(sum) → global result.

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "[%d] CUDA %s:%d: %s\n", rank, __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "[%d] cuBLAS %s:%d: %d\n", rank, __FILE__, __LINE__, (int)s); MPI_Abort(MPI_COMM_WORLD, 1); } } while(0)

static int rank, nprocs;

// Custom dot kernel (from bench_ata)
__global__ void ata_dot_kernel(
    const double* __restrict__ A,
    double* __restrict__ C,
    int M, int N)
{
    int col_i = blockIdx.x;
    int col_j = blockIdx.y;
    if (col_j < col_i) return;

    extern __shared__ double sdata[];
    double sum = 0.0;
    for (int row = threadIdx.x; row < M; row += blockDim.x)
        sum += A[row + col_i * M] * A[row + col_j * M];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        C[col_i * N + col_j] = sdata[0];
        if (col_i != col_j)
            C[col_j * N + col_i] = sdata[0];
    }
}

void bench_distributed_ata(int M_global, int N, cublasHandle_t handle) {
    int M_local = M_global / nprocs;
    // Last rank gets remainder
    if (rank == nprocs - 1)
        M_local = M_global - M_local * (nprocs - 1);

    int M_local_base = M_global / nprocs;
    int my_offset = rank * M_local_base;

    size_t A_bytes = (size_t)M_local * N * sizeof(double);
    size_t C_bytes = (size_t)N * N * sizeof(double);

    // Initialize local A with deterministic data based on global row index
    std::vector<double> h_A(M_local * N);
    for (int col = 0; col < N; ++col)
        for (int row = 0; row < M_local; ++row) {
            int global_row = my_offset + row;
            h_A[row + col * M_local] = sin(0.001 * (global_row + col * M_global)) * 0.1;
        }

    double *d_A, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, A_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, C_bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_bytes, cudaMemcpyHostToDevice));

    // Host buffers for AllReduce
    std::vector<double> h_C_local(N * N);
    std::vector<double> h_C_global(N * N);

    // ============================================================
    // Method 1: cuBLAS DGEMM + MPI_Allreduce
    // ============================================================
    double alpha = 1.0, beta = 0.0;
    int iters = std::max(5, std::min(1000, (int)(5e8 / (2.0 * M_local * N * N))));

    // Warmup
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < 3; ++i) {
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 N, N, M_local, &alpha, d_A, M_local, d_A, M_local, &beta, d_C, N));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C_local.data(), d_C, C_bytes, cudaMemcpyDeviceToHost));
        MPI_Allreduce(h_C_local.data(), h_C_global.data(), N * N,
                       MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    double compute_total = 0, comm_total = 0, d2h_total = 0;
    for (int i = 0; i < iters; ++i) {
        double tc0 = MPI_Wtime();
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 N, N, M_local, &alpha, d_A, M_local, d_A, M_local, &beta, d_C, N));
        CUDA_CHECK(cudaDeviceSynchronize());
        double tc1 = MPI_Wtime();
        compute_total += tc1 - tc0;

        CUDA_CHECK(cudaMemcpy(h_C_local.data(), d_C, C_bytes, cudaMemcpyDeviceToHost));
        double tc2 = MPI_Wtime();
        d2h_total += tc2 - tc1;

        MPI_Allreduce(h_C_local.data(), h_C_global.data(), N * N,
                       MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double tc3 = MPI_Wtime();
        comm_total += tc3 - tc2;
    }
    double t1 = MPI_Wtime();
    double dgemm_total_ms = (t1 - t0) / iters * 1000;
    double dgemm_compute_ms = compute_total / iters * 1000;
    double dgemm_d2h_ms = d2h_total / iters * 1000;
    double dgemm_comm_ms = comm_total / iters * 1000;

    // Save reference for correctness check
    std::vector<double> C_ref_dgemm(h_C_global);

    // ============================================================
    // Method 2: Custom dot kernel + MPI_Allreduce
    // ============================================================
    int bs = 256;
    dim3 grid(N, N);

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < 3; ++i) {
        ata_dot_kernel<<<grid, bs, bs * sizeof(double)>>>(d_A, d_C, M_local, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C_local.data(), d_C, C_bytes, cudaMemcpyDeviceToHost));
        MPI_Allreduce(h_C_local.data(), h_C_global.data(), N * N,
                       MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    compute_total = 0; comm_total = 0; d2h_total = 0;
    for (int i = 0; i < iters; ++i) {
        double tc0 = MPI_Wtime();
        ata_dot_kernel<<<grid, bs, bs * sizeof(double)>>>(d_A, d_C, M_local, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        double tc1 = MPI_Wtime();
        compute_total += tc1 - tc0;

        CUDA_CHECK(cudaMemcpy(h_C_local.data(), d_C, C_bytes, cudaMemcpyDeviceToHost));
        double tc2 = MPI_Wtime();
        d2h_total += tc2 - tc1;

        MPI_Allreduce(h_C_local.data(), h_C_global.data(), N * N,
                       MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double tc3 = MPI_Wtime();
        comm_total += tc3 - tc2;
    }
    t1 = MPI_Wtime();
    double dot_total_ms = (t1 - t0) / iters * 1000;
    double dot_compute_ms = compute_total / iters * 1000;
    double dot_d2h_ms = d2h_total / iters * 1000;
    double dot_comm_ms = comm_total / iters * 1000;

    // Correctness check: dot vs dgemm reference
    double max_err = 0;
    for (int i = 0; i < N * N; ++i)
        max_err = std::max(max_err, std::abs(h_C_global[i] - C_ref_dgemm[i]));

    // ============================================================
    // Method 3: Custom dot kernel + GPU-side AllReduce via MPI
    // (Use MPI directly on device buffer if CUDA-aware MPI available,
    //  otherwise simulate: D2H -> Allreduce -> H2D)
    // ============================================================

    // Method 4: Custom dot + overlap compute/comm with double buffering
    // (pipeline: compute tile_k while allreduce tile_{k-1})
    // For N×N result this is overkill, but test it anyway

    if (rank == 0) {
        printf("  %7d  %3d  %4d  %5d  ", M_global, N, nprocs, M_local);
        printf("| %6.3f = %5.3f + %5.3f + %5.3f  ", dgemm_total_ms, dgemm_compute_ms, dgemm_d2h_ms, dgemm_comm_ms);
        printf("| %6.3f = %5.3f + %5.3f + %5.3f  ", dot_total_ms, dot_compute_ms, dot_d2h_ms, dot_comm_ms);
        printf("| %5.1fx  %.1e\n", dgemm_total_ms / dot_total_ms, max_err);
    }
}

// Single-process reference to verify distributed result
void verify_global(int M_global, int N, cublasHandle_t handle) {
    if (rank != 0) return;

    // Build full A on rank 0
    std::vector<double> h_A(M_global * N);
    for (int col = 0; col < N; ++col)
        for (int row = 0; row < M_global; ++row)
            h_A[row + col * M_global] = sin(0.001 * (row + col * M_global)) * 0.1;

    double *d_A, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)M_global * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)N * N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), (size_t)M_global * N * sizeof(double), cudaMemcpyHostToDevice));

    double alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             N, N, M_global, &alpha, d_A, M_global, d_A, M_global, &beta, d_C, N));

    std::vector<double> C_single(N * N);
    CUDA_CHECK(cudaMemcpy(C_single.data(), d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost));

    // Now compute distributed version
    cudaFree(d_A); cudaFree(d_C);

    // Distributed: split A across "virtual" ranks
    int M_local = M_global / nprocs;
    std::vector<double> C_sum(N * N, 0.0);

    for (int r = 0; r < nprocs; ++r) {
        int offset = r * M_local;
        int ml = (r == nprocs - 1) ? M_global - offset : M_local;

        std::vector<double> A_local(ml * N);
        for (int col = 0; col < N; ++col)
            for (int row = 0; row < ml; ++row)
                A_local[row + col * ml] = h_A[(offset + row) + col * M_global];

        CUDA_CHECK(cudaMalloc(&d_A, (size_t)ml * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_C, (size_t)N * N * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_A, A_local.data(), (size_t)ml * N * sizeof(double), cudaMemcpyHostToDevice));

        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 N, N, ml, &alpha, d_A, ml, d_A, ml, &beta, d_C, N));

        std::vector<double> C_part(N * N);
        CUDA_CHECK(cudaMemcpy(C_part.data(), d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N * N; ++i)
            C_sum[i] += C_part[i];

        cudaFree(d_A); cudaFree(d_C);
    }

    double max_err = 0;
    for (int i = 0; i < N * N; ++i)
        max_err = std::max(max_err, std::abs(C_single[i] - C_sum[i]));

    printf("  Correctness verify (M=%d, N=%d, %d ranks): single vs distributed max_err = %.2e %s\n",
           M_global, N, nprocs, max_err, max_err < 1e-8 ? "OK" : "FAIL");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // All ranks share GPU 0
    CUDA_CHECK(cudaSetDevice(0));

    if (rank == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("GPU: %s, %d MPI ranks sharing 1 GPU\n\n", prop.name, nprocs);
    }

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Verify correctness first
    if (rank == 0) printf("=== Correctness Verification ===\n");
    verify_global(10000, 12, handle);
    verify_global(10000, 30, handle);
    verify_global(100000, 30, handle);
    MPI_Barrier(MPI_COMM_WORLD);

    // Performance benchmark
    if (rank == 0) {
        printf("\n=== Distributed A^T*A Performance ===\n");
        printf("  M_global    N  nproc  M_loc  ");
        printf("| DGEMM total = comp + D2H + comm  ");
        printf("| DOT   total = comp + D2H + comm  ");
        printf("| Speedup  Err\n");
        printf("  -------  ---  -----  -----  ");
        printf("| --------------------------------  ");
        printf("| --------------------------------  ");
        printf("| -------  ---\n");
    }

    struct Config { int M; int N; };
    Config configs[] = {
        // Small (BaTiO3-like)
        {26*26*26, 12},
        {26*26*26, 30},
        // Medium
        {48*48*48, 12},
        {48*48*48, 30},
        // Large
        {100*100*100, 12},
        {100*100*100, 30},
        {100*100*100, 50},
    };

    for (auto& cfg : configs) {
        if (cfg.M % nprocs != 0 && rank == 0) {
            // Adjust M to be divisible
        }
        bench_distributed_ata(cfg.M, cfg.N, handle);
    }

    // Also test the "what if each rank has very few rows" scenario
    // (heavy band parallelism)
    if (rank == 0) printf("\n  --- Heavy band parallelism (small M_local) ---\n");
    for (int M_local_target : {256, 512, 1024, 4096}) {
        int M_global = M_local_target * nprocs;
        for (int N : {12, 30}) {
            bench_distributed_ata(M_global, N, handle);
        }
    }

    cublasDestroy(handle);
    MPI_Finalize();
    return 0;
}
