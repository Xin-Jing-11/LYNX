// Standalone benchmark for gradient kernels (V1 per-column launch vs V2 batched)
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define MAX_FD_COEFF 7
__host__ __device__ int ceildiv(int a, int b) { return (a + b - 1) / b; }

__constant__ double d_D1x[MAX_FD_COEFF];
__constant__ double d_D1y[MAX_FD_COEFF];
__constant__ double d_D1z[MAX_FD_COEFF];

// V1: one launch per column
__global__ void gradient_x_kernel(
    const double* __restrict__ x_ex,
    double* __restrict__ y,
    int nx, int ny, int nz,
    int FDn, int nx_ex, int ny_ex, int nxny_ex)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int idx = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
    int loc = i + j * nx + k * nx * ny;

    double val = 0.0;
    for (int p = 1; p <= FDn; ++p)
        val += d_D1x[p] * (x_ex[idx + p] - x_ex[idx - p]);
    y[loc] = val;
}

// V2: template FDn, batched columns, grid.z = nz
template<int FDN>
__global__ void gradient_x_kernel_v2(
    const double* __restrict__ x_ex,
    double* __restrict__ y,
    int nx, int ny, int nz,
    int nx_ex, int ny_ex, int nxny_ex,
    int nd, int nd_ex, int ncol)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int idx_base = (i + FDN) + (j + FDN) * nx_ex + (k + FDN) * nxny_ex;
    int loc = i + j * nx + k * nx * ny;

    for (int n = 0; n < ncol; ++n) {
        int idx = idx_base + n * nd_ex;
        double val = 0.0;
        #pragma unroll
        for (int p = 1; p <= FDN; ++p)
            val += d_D1x[p] * (x_ex[idx + p] - x_ex[idx - p]);
        y[loc + n * nd] = val;
    }
}

int main() {
    const int nx = 25, ny = 26, nz = 27;
    const int FDn = 6;
    const int ncol = 20;
    const int NREPS = 500;

    const int nx_ex = nx + 2 * FDn;
    const int ny_ex = ny + 2 * FDn;
    const int nxny_ex = nx_ex * ny_ex;
    const int nz_ex = nz + 2 * FDn;
    const int nd = nx * ny * nz;
    const int nd_ex = nxny_ex * nz_ex;

    // Upload coefficients
    double h_D1[MAX_FD_COEFF] = {0.0, 0.8, -0.2, 0.038, -0.0079, 0.0018, -0.0004};
    CUDA_CHECK(cudaMemcpyToSymbol(d_D1x, h_D1, MAX_FD_COEFF * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_D1y, h_D1, MAX_FD_COEFF * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_D1z, h_D1, MAX_FD_COEFF * sizeof(double)));

    double *d_x_ex, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x_ex, (size_t)nd_ex * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)nd * ncol * sizeof(double)));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniformDouble(gen, d_x_ex, nd_ex * ncol);
    curandDestroyGenerator(gen);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("=== Gradient Benchmark: nx=%d ny=%d nz=%d FDn=%d ncol=%d (NREPS=%d) ===\n",
           nx, ny, nz, FDn, ncol, NREPS);

    // V1: per-column launch
    {
        dim3 block(32, 4, 4);
        dim3 grid(ceildiv(nx, 32), ceildiv(ny, 4), ceildiv(nz, 4));

        for (int i = 0; i < 5; i++)
            for (int n = 0; n < ncol; n++)
                gradient_x_kernel<<<grid, block>>>(d_x_ex + n * nd_ex, d_y + n * nd,
                    nx, ny, nz, FDn, nx_ex, ny_ex, nxny_ex);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++)
            for (int n = 0; n < ncol; n++)
                gradient_x_kernel<<<grid, block>>>(d_x_ex + n * nd_ex, d_y + n * nd,
                    nx, ny, nz, FDn, nx_ex, ny_ex, nxny_ex);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        ms /= NREPS;
        double bytes = (double)nd * ncol * (2 * FDn + 1.0) * sizeof(double);
        printf("  V1 (per-col): %8.4f ms  | %6.1f GB/s\n", ms, bytes / (ms * 1e6));
    }

    // V2: batched
    {
        dim3 block(32, 8);
        dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz);

        for (int i = 0; i < 5; i++)
            gradient_x_kernel_v2<6><<<grid, block>>>(d_x_ex, d_y,
                nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++)
            gradient_x_kernel_v2<6><<<grid, block>>>(d_x_ex, d_y,
                nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        ms /= NREPS;
        double bytes = (double)nd * ncol * (2 * FDn + 1.0) * sizeof(double);
        printf("  V2 (batched): %8.4f ms  | %6.1f GB/s\n", ms, bytes / (ms * 1e6));
    }

    printf("\n");
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_x_ex); cudaFree(d_y);
    return 0;
}
