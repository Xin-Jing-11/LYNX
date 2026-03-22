// Benchmark: A^T * A for tall-thin matrices (band-parallel DFT use case)
// A is (M x N) where M >> N (e.g., M=100^3=1e6, N=12..50)
// Result C = A^T * A is (N x N), symmetric
//
// Approaches:
// 1. cuBLAS DGEMM:  C = A^T * A  via dgemm('T','N', N, N, M, 1, A, M, A, M, 0, C, N)
// 2. cuBLAS DSYRK:  C = A^T * A  via dsyrk('U','T', N, M, 1, A, M, 0, C, N) — symmetric
// 3. Custom kernel: Each thread-block computes a tile of C using shared memory reduction
// 4. Tiled DGEMM:   Split A into row-tiles, accumulate partial C = sum_t A_t^T * A_t
// 5. Multi-stream:  Overlap tiled compute with prior tile's reduction
//
// In distributed case: each GPU has A_local (M_local x N), computes local C, then AllReduce.
// The compute part is what we benchmark here.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t s = (x); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Approach 3: Custom dot-product kernel
// Each block computes one element C[col_i, col_j] = dot(A[:,col_i], A[:,col_j])
// using block-wide parallel reduction
// ============================================================
__global__ void ata_dot_kernel(
    const double* __restrict__ A,
    double* __restrict__ C,
    int M, int N)
{
    // Block (bx, by) computes C[bx][by]
    int col_i = blockIdx.x;
    int col_j = blockIdx.y;
    if (col_j < col_i) return;  // only upper triangle

    extern __shared__ double sdata[];

    double sum = 0.0;
    for (int row = threadIdx.x; row < M; row += blockDim.x) {
        sum += A[row + col_i * M] * A[row + col_j * M];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        C[col_i * N + col_j] = sdata[0];
        if (col_i != col_j)
            C[col_j * N + col_i] = sdata[0];  // fill lower triangle
    }
}

// ============================================================
// Approach 3b: Custom kernel — each block handles a TILE of C elements
// Reduces launch overhead for small N, amortizes A loads
// Grid: (num_tiles), Block: (256)
// Each block iterates over rows of A, accumulating multiple C entries
// ============================================================
__global__ void ata_tiled_dot_kernel(
    const double* __restrict__ A,
    double* __restrict__ C,
    int M, int N,
    const int* __restrict__ tile_i,  // column index i for each tile entry
    const int* __restrict__ tile_j,  // column index j for each tile entry
    int tile_size)                    // entries per tile
{
    extern __shared__ double sdata[];  // tile_size * blockDim.x

    int tid = threadIdx.x;
    int bs = blockDim.x;

    // Each thread accumulates tile_size partial sums
    double local_sums[64];  // max tile_size
    for (int t = 0; t < tile_size; ++t)
        local_sums[t] = 0.0;

    // Load tile column indices
    int tile_base = blockIdx.x * tile_size;

    // Iterate over rows of A
    for (int row = tid; row < M; row += bs) {
        // Load the row values for all relevant columns
        for (int t = 0; t < tile_size; ++t) {
            int ci = tile_i[tile_base + t];
            int cj = tile_j[tile_base + t];
            local_sums[t] += A[row + ci * M] * A[row + cj * M];
        }
    }

    // Reduce each partial sum across the block
    for (int t = 0; t < tile_size; ++t) {
        sdata[tid] = local_sums[t];
        __syncthreads();

        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }

        if (tid == 0) {
            int ci = tile_i[tile_base + t];
            int cj = tile_j[tile_base + t];
            C[ci * N + cj] = sdata[0];
            if (ci != cj)
                C[cj * N + ci] = sdata[0];
        }
        __syncthreads();
    }
}

// ============================================================
// Approach 5: Custom kernel — vectorized with warp shuffle reduction
// Each block computes one C[i,j], uses vectorized loads (double2)
// ============================================================
__global__ void ata_dot_warp_kernel(
    const double* __restrict__ A,
    double* __restrict__ C,
    int M, int N)
{
    int col_i = blockIdx.x;
    int col_j = blockIdx.y;
    if (col_j < col_i) return;

    double sum = 0.0;
    const double* ai = A + col_i * M;
    const double* aj = A + col_j * M;

    // Vectorized double2 loads where possible
    int M2 = M / 2;
    const double2* ai2 = reinterpret_cast<const double2*>(ai);
    const double2* aj2 = reinterpret_cast<const double2*>(aj);

    for (int row = threadIdx.x; row < M2; row += blockDim.x) {
        double2 va = ai2[row];
        double2 vb = aj2[row];
        sum += va.x * vb.x + va.y * vb.y;
    }
    // Handle remainder
    for (int row = M2 * 2 + threadIdx.x; row < M; row += blockDim.x) {
        sum += ai[row] * aj[row];
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Cross-warp reduction via shared memory
    __shared__ double warp_sums[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) {
            C[col_i * N + col_j] = sum;
            if (col_i != col_j)
                C[col_j * N + col_i] = sum;
        }
    }
}

// ============================================================
// Approach 6: Row-major A layout (for comparison)
// In DFT codes, A is often stored as (M x N) column-major,
// but A^T*A with row-major A means dot products are contiguous.
// Test if transposing A first then using dsyrk is faster.
// ============================================================

// ============================================================
// Benchmark harness
// ============================================================
int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices\n");
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (compute %d.%d, %.1f GB, L2=%d KB)\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e9, prop.l2CacheSize / 1024);
    printf("\n");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Test configurations: (M, N) pairs
    struct Config { int M; int N; const char* label; };
    Config configs[] = {
        {26*26*26,   12, "26³×12  (BaTiO3)"},
        {26*26*26,   30, "26³×30  (BaTiO3 large)"},
        {48*48*48,   12, "48³×12  (Si supercell)"},
        {48*48*48,   30, "48³×30  (Si supercell)"},
        {72*72*72,   12, "72³×12  (medium)"},
        {72*72*72,   30, "72³×30  (medium)"},
        {100*100*100, 12, "100³×12 (large)"},
        {100*100*100, 30, "100³×30 (large)"},
        {100*100*100, 50, "100³×50 (large, many bands)"},
        // Band-parallel: each GPU gets M/nproc rows
        {100*100*100/4, 30, "100³×30 / 4GPU (250K×30)"},
        {100*100*100/8, 30, "100³×30 / 8GPU (125K×30)"},
    };

    printf("%-28s  %10s  %10s  %10s  %10s  %10s\n",
           "Config", "DGEMM", "DSYRK", "Dot-naive", "Dot-warp", "DSYRK-BW");
    printf("%-28s  %10s  %10s  %10s  %10s  %10s\n",
           "", "(ms)", "(ms)", "(ms)", "(ms)", "(GB/s)");
    printf("-----------------------------------------------------------------------------------------------\n");

    for (auto& cfg : configs) {
        int M = cfg.M;
        int N = cfg.N;

        // Allocate
        double *d_A, *d_C;
        size_t A_bytes = (size_t)M * N * sizeof(double);
        size_t C_bytes = (size_t)N * N * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_A, A_bytes));
        CUDA_CHECK(cudaMalloc(&d_C, C_bytes));

        // Initialize A with random-ish data
        std::vector<double> h_A(M * N);
        for (size_t i = 0; i < h_A.size(); ++i)
            h_A[i] = sin(0.001 * i) * 0.1;
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_bytes, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float ms;

        // Determine iteration count (aim for ~200ms total)
        int iters = std::max(3, (int)(200000000.0 / (2.0 * M * N * N)));
        iters = std::min(iters, 500);

        // ---- 1. cuBLAS DGEMM: C = A^T * A ----
        double alpha = 1.0, beta = 0.0;
        // Warmup
        for (int i = 0; i < 3; ++i)
            CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     N, N, M, &alpha, d_A, M, d_A, M, &beta, d_C, N));
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i)
            CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     N, N, M, &alpha, d_A, M, d_A, M, &beta, d_C, N));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        double dgemm_ms = ms / iters;

        // Save reference result
        std::vector<double> C_ref(N * N);
        CUDA_CHECK(cudaMemcpy(C_ref.data(), d_C, C_bytes, cudaMemcpyDeviceToHost));

        // ---- 2. cuBLAS DSYRK: C = A^T * A (symmetric) ----
        for (int i = 0; i < 3; ++i)
            CUBLAS_CHECK(cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                     N, M, &alpha, d_A, M, &beta, d_C, N));
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i)
            CUBLAS_CHECK(cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                     N, M, &alpha, d_A, M, &beta, d_C, N));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        double dsyrk_ms = ms / iters;

        // Verify DSYRK
        {
            std::vector<double> C_test(N * N);
            // Fill lower triangle from upper for comparison
            CUBLAS_CHECK(cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                     N, M, &alpha, d_A, M, &beta, d_C, N));
            CUDA_CHECK(cudaMemcpy(C_test.data(), d_C, C_bytes, cudaMemcpyDeviceToHost));
            // Fill symmetric
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < i; ++j)
                    C_test[i * N + j] = C_test[j * N + i];
            double max_err = 0;
            for (int i = 0; i < N * N; ++i)
                max_err = std::max(max_err, std::abs(C_ref[i] - C_test[i]));
            if (max_err > 1e-10)
                printf("  WARNING: DSYRK error = %.2e\n", max_err);
        }

        // ---- 3. Custom dot kernel (naive, shared memory reduction) ----
        double dot_naive_ms = -1;
        if (N <= 50) {  // only for small N (otherwise N^2 blocks)
            int bs = 256;
            dim3 grid(N, N);
            for (int i = 0; i < 3; ++i)
                ata_dot_kernel<<<grid, bs, bs * sizeof(double)>>>(d_A, d_C, M, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaEventRecord(start);
            for (int i = 0; i < iters; ++i)
                ata_dot_kernel<<<grid, bs, bs * sizeof(double)>>>(d_A, d_C, M, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            dot_naive_ms = ms / iters;

            // Verify
            std::vector<double> C_test(N * N);
            ata_dot_kernel<<<grid, bs, bs * sizeof(double)>>>(d_A, d_C, M, N);
            CUDA_CHECK(cudaMemcpy(C_test.data(), d_C, C_bytes, cudaMemcpyDeviceToHost));
            double max_err = 0;
            for (int i = 0; i < N * N; ++i)
                max_err = std::max(max_err, std::abs(C_ref[i] - C_test[i]));
            if (max_err > 1e-8)
                printf("  WARNING: dot-naive error = %.2e\n", max_err);
        }

        // ---- 4. Custom dot kernel (warp shuffle, vectorized loads) ----
        double dot_warp_ms = -1;
        if (N <= 50) {
            int bs = 256;
            dim3 grid(N, N);
            for (int i = 0; i < 3; ++i)
                ata_dot_warp_kernel<<<grid, bs>>>(d_A, d_C, M, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaEventRecord(start);
            for (int i = 0; i < iters; ++i)
                ata_dot_warp_kernel<<<grid, bs>>>(d_A, d_C, M, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            dot_warp_ms = ms / iters;

            // Verify
            std::vector<double> C_test(N * N);
            ata_dot_warp_kernel<<<grid, bs>>>(d_A, d_C, M, N);
            CUDA_CHECK(cudaMemcpy(C_test.data(), d_C, C_bytes, cudaMemcpyDeviceToHost));
            double max_err = 0;
            for (int i = 0; i < N * N; ++i)
                max_err = std::max(max_err, std::abs(C_ref[i] - C_test[i]));
            if (max_err > 1e-8)
                printf("  WARNING: dot-warp error = %.2e\n", max_err);
        }

        // Compute effective bandwidth for DSYRK (reads A once, writes C)
        double dsyrk_bw = (A_bytes + C_bytes) / (dsyrk_ms * 1e-3) / 1e9;

        // Print results
        char buf1[16], buf2[16], buf3[16], buf4[16], buf5[16];
        snprintf(buf1, sizeof(buf1), "%.3f", dgemm_ms);
        snprintf(buf2, sizeof(buf2), "%.3f", dsyrk_ms);
        snprintf(buf3, sizeof(buf3), dot_naive_ms >= 0 ? "%.3f" : "n/a", dot_naive_ms);
        snprintf(buf4, sizeof(buf4), dot_warp_ms >= 0 ? "%.3f" : "n/a", dot_warp_ms);
        snprintf(buf5, sizeof(buf5), "%.0f", dsyrk_bw);

        printf("%-28s  %10s  %10s  %10s  %10s  %10s\n",
               cfg.label, buf1, buf2, buf3, buf4, buf5);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_C);
    }

    printf("\n");

    // ============================================================
    // Part 2: Measure communication overhead estimate
    // In band-parallel, after local A^T*A, we need AllReduce of N×N matrix.
    // Simulate with cudaMemcpy D2H + H2D (proxy for NCCL/MPI)
    // ============================================================
    printf("AllReduce cost estimate (N×N matrix, D2H + H2D round-trip):\n");
    printf("%-10s  %10s  %10s\n", "N", "D2H+H2D", "vs DSYRK");
    printf("--------------------------------------\n");
    for (int N : {12, 30, 50}) {
        size_t C_bytes = N * N * sizeof(double);
        double *d_C;
        std::vector<double> h_C(N * N);
        CUDA_CHECK(cudaMalloc(&d_C, C_bytes));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int iters = 1000;
        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i) {
            CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, C_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), C_bytes, cudaMemcpyHostToDevice));
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        double rt_ms = ms / iters;

        printf("N=%-8d  %8.3f ms  (%.1f KB)\n", N, rt_ms, C_bytes / 1024.0);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_C);
    }

    printf("\n");

    // ============================================================
    // Part 3: Tiled DSYRK — split M into chunks to stay in L2 cache
    // C = sum_t A_t^T * A_t where each A_t is (tile_rows x N)
    // First call uses beta=0, subsequent use beta=1 to accumulate
    // ============================================================
    printf("Tiled DSYRK (L2-friendly) vs single DSYRK:\n");
    printf("%-28s  %10s", "Config", "Single");

    int tile_sizes[] = {4096, 8192, 16384, 32768, 65536};
    int n_tiles = sizeof(tile_sizes) / sizeof(tile_sizes[0]);
    for (int t = 0; t < n_tiles; ++t)
        printf("  tile=%-5d", tile_sizes[t]);
    printf("\n");

    for (auto& cfg : configs) {
        int M = cfg.M;
        int N = cfg.N;

        double *d_A, *d_C;
        size_t A_bytes = (size_t)M * N * sizeof(double);
        size_t C_bytes = (size_t)N * N * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_A, A_bytes));
        CUDA_CHECK(cudaMalloc(&d_C, C_bytes));

        std::vector<double> h_A(M * N);
        for (size_t i = 0; i < h_A.size(); ++i)
            h_A[i] = sin(0.001 * i) * 0.1;
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_bytes, cudaMemcpyHostToDevice));

        int iters = std::max(3, (int)(200000000.0 / (2.0 * M * N * N)));
        iters = std::min(iters, 500);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float ms;
        double alpha = 1.0, beta_zero = 0.0, beta_one = 1.0;

        // Single DSYRK
        for (int i = 0; i < 3; ++i)
            CUBLAS_CHECK(cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                     N, M, &alpha, d_A, M, &beta_zero, d_C, N));
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i)
            CUBLAS_CHECK(cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                     N, M, &alpha, d_A, M, &beta_zero, d_C, N));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        double single_ms = ms / iters;

        // Save reference
        std::vector<double> C_ref(N * N);
        CUBLAS_CHECK(cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                 N, M, &alpha, d_A, M, &beta_zero, d_C, N));
        CUDA_CHECK(cudaMemcpy(C_ref.data(), d_C, C_bytes, cudaMemcpyDeviceToHost));

        printf("%-28s  %8.3f ms", cfg.label, single_ms);

        // Tiled DSYRK
        for (int ti = 0; ti < n_tiles; ++ti) {
            int tile_rows = tile_sizes[ti];
            if (tile_rows >= M) {
                printf("  %8s ms", "=single");
                continue;
            }

            // Warmup
            for (int w = 0; w < 3; ++w) {
                for (int row = 0; row < M; row += tile_rows) {
                    int rows_this = std::min(tile_rows, M - row);
                    // A is column-major: tile starts at A + row, with leading dim M
                    double* b = (row == 0) ? &beta_zero : &beta_one;
                    CUBLAS_CHECK(cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                             N, rows_this, &alpha, d_A + row, M, b, d_C, N));
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaEventRecord(start);
            for (int i = 0; i < iters; ++i) {
                for (int row = 0; row < M; row += tile_rows) {
                    int rows_this = std::min(tile_rows, M - row);
                    double* b = (row == 0) ? &beta_zero : &beta_one;
                    CUBLAS_CHECK(cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                             N, rows_this, &alpha, d_A + row, M, b, d_C, N));
                }
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            double tiled_ms = ms / iters;

            // Verify
            {
                for (int row = 0; row < M; row += tile_rows) {
                    int rows_this = std::min(tile_rows, M - row);
                    double* b = (row == 0) ? &beta_zero : &beta_one;
                    CUBLAS_CHECK(cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                             N, rows_this, &alpha, d_A + row, M, b, d_C, N));
                }
                std::vector<double> C_test(N * N);
                CUDA_CHECK(cudaMemcpy(C_test.data(), d_C, C_bytes, cudaMemcpyDeviceToHost));
                double max_err = 0;
                for (int i = 0; i < N * N; ++i)
                    max_err = std::max(max_err, std::abs(C_ref[i] - C_test[i]));
                if (max_err > 1e-8)
                    printf("\n  WARNING: tiled dsyrk tile=%d error = %.2e\n", tile_rows, max_err);
            }

            printf("  %8.3f ms", tiled_ms);
        }
        printf("\n");

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_C);
    }

    cublasDestroy(handle);
    printf("\nDone.\n");
    return 0;
}
