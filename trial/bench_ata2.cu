// Benchmark: A^T * A — sweep M/N ratio to find crossover point
// between custom dot kernel and cuBLAS DGEMM

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS %s:%d: %d\n", __FILE__, __LINE__, (int)s); exit(1); } } while(0)

// Custom dot kernel: one block per C[i,j], shared-memory reduction
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

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    struct Config { int M; int N; };
    Config configs[] = {
        // N=12: sweep M
        {48,    12},
        {96,    12},
        {192,   12},
        {384,   12},
        {768,   12},
        {1536,  12},
        {3072,  12},
        {6144,  12},
        {12288, 12},
        {26*26*26, 12},  // 17576
        {50000, 12},
        {100000, 12},
        // N=30: sweep M
        {60,    30},
        {120,   30},
        {240,   30},
        {480,   30},
        {960,   30},
        {1920,  30},
        {3840,  30},
        {7680,  30},
        {15360, 30},
        {30720, 30},
        {50000, 30},
        {100000, 30},
        // N=50: sweep M
        {100,   50},
        {200,   50},
        {500,   50},
        {1000,  50},
        {2000,  50},
        {5000,  50},
        {10000, 50},
        {50000, 50},
        {100000, 50},
        // N=100: sweep M (band-parallel with many bands)
        {200,   100},
        {500,   100},
        {1000,  100},
        {2000,  100},
        {5000,  100},
        {10000, 100},
        {50000, 100},
        // N=200: edge case
        {400,   200},
        {1000,  200},
        {2000,  200},
        {5000,  200},
        {10000, 200},
    };

    printf("%-8s  %-6s  %6s  %10s  %10s  %10s  %s\n",
           "M", "N", "M/N", "DGEMM(ms)", "Dot(ms)", "Ratio", "Winner");
    printf("--------------------------------------------------------------------------\n");

    int last_N = -1;
    for (auto& cfg : configs) {
        int M = cfg.M;
        int N = cfg.N;

        if (N != last_N) {
            if (last_N != -1) printf("\n");
            last_N = N;
        }

        double *d_A, *d_C;
        size_t A_bytes = (size_t)M * N * sizeof(double);
        size_t C_bytes = (size_t)N * N * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_A, A_bytes));
        CUDA_CHECK(cudaMalloc(&d_C, C_bytes));

        // Init
        std::vector<double> h_A(M * N);
        for (size_t i = 0; i < h_A.size(); ++i)
            h_A[i] = sin(0.001 * i) * 0.1;
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_bytes, cudaMemcpyHostToDevice));

        // Auto-tune iterations
        double flops = 2.0 * M * N * N;
        int iters = std::max(5, std::min(5000, (int)(1e9 / flops)));

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float ms;
        double alpha = 1.0, beta = 0.0;

        // cuBLAS DGEMM
        for (int i = 0; i < 3; ++i)
            CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     N, N, M, &alpha, d_A, M, d_A, M, &beta, d_C, N));
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i)
            CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     N, N, M, &alpha, d_A, M, d_A, M, &beta, d_C, N));
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        double dgemm_ms = ms / iters;

        // Custom dot kernel
        int bs = 256;
        dim3 grid(N, N);
        for (int i = 0; i < 3; ++i)
            ata_dot_kernel<<<grid, bs, bs * sizeof(double)>>>(d_A, d_C, M, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i)
            ata_dot_kernel<<<grid, bs, bs * sizeof(double)>>>(d_A, d_C, M, N);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        double dot_ms = ms / iters;

        double ratio = dgemm_ms / dot_ms;
        const char* winner = (ratio > 1.0) ? "DOT" : "DGEMM";

        printf("%-8d  %-6d  %6.1f  %10.3f  %10.3f  %10.2f  %s\n",
               M, N, (double)M/N, dgemm_ms, dot_ms, ratio, winner);

        cudaEventDestroy(start); cudaEventDestroy(stop);
        cudaFree(d_A); cudaFree(d_C);
    }

    cublasDestroy(handle);
    printf("\nDone.\n");
    return 0;
}
