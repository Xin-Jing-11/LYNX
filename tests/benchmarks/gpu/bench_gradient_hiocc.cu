// High-occupancy gradient kernel benchmark
// Problem: gradient_v2 loops over columns internally → only nz grid blocks in z
// Fix: V3 uses grid.z = nz*ncol (same pattern as laplacian_v7), eliminating inner loop
// Also adds __launch_bounds__(256, 6) to guarantee ≤40 regs
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

// ============================================================
// BASELINE: V2 (batched, inner column loop)
// ============================================================
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

// ============================================================
// OPTIMIZED: V3 — one block per (i-tile, j-tile, k*col)
// Same pattern as laplacian_v7: grid.z = nz * ncol
// Eliminates inner loop → 20x more blocks → much higher occupancy
// ============================================================
template<int FDN>
__global__ __launch_bounds__(256, 6)
void gradient_x_kernel_v3(
    const double* __restrict__ x_ex,
    double* __restrict__ y,
    int nx, int ny, int nz,
    int nx_ex, int ny_ex, int nxny_ex,
    int nd, int nd_ex, int ncol)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int flat_z = blockIdx.z;
    int k = flat_z % nz;
    int n = flat_z / nz;
    if (i >= nx || j >= ny) return;

    int idx = (i + FDN) + (j + FDN) * nx_ex + (k + FDN) * nxny_ex + n * nd_ex;
    int loc = i + j * nx + k * nx * ny + n * nd;

    double val = 0.0;
    #pragma unroll
    for (int p = 1; p <= FDN; ++p)
        val += d_D1x[p] * (x_ex[idx + p] - x_ex[idx - p]);
    y[loc] = val;
}

// ============================================================
// V3b: Same as V3 but with precomputed a*coefficients in constant memory
// (matching laplacian_v7 pattern for potential FMA optimization)
// ============================================================
__constant__ double d_aD1x[MAX_FD_COEFF];  // precomputed a * D1x

template<int FDN>
__global__ __launch_bounds__(256, 6)
void gradient_x_kernel_v3b(
    const double* __restrict__ x_ex,
    double* __restrict__ y,
    int nx, int ny, int nz,
    int nx_ex, int ny_ex, int nxny_ex,
    int nd, int nd_ex, int ncol,
    double a)  // scaling factor
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int flat_z = blockIdx.z;
    int k = flat_z % nz;
    int n = flat_z / nz;
    if (i >= nx || j >= ny) return;

    int idx = (i + FDN) + (j + FDN) * nx_ex + (k + FDN) * nxny_ex + n * nd_ex;
    int loc = i + j * nx + k * nx * ny + n * nd;

    double val = 0.0;
    #pragma unroll
    for (int p = 1; p <= FDN; ++p)
        val = fma(d_D1x[p], x_ex[idx + p] - x_ex[idx - p], val);
    y[loc] = val;
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

    double h_D1[MAX_FD_COEFF] = {0.0, 0.8, -0.2, 0.038, -0.0079, 0.0018, -0.0004};
    CUDA_CHECK(cudaMemcpyToSymbol(d_D1x, h_D1, MAX_FD_COEFF * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_aD1x, h_D1, MAX_FD_COEFF * sizeof(double)));

    double *d_x_ex, *d_y, *d_y2;
    CUDA_CHECK(cudaMalloc(&d_x_ex, (size_t)nd_ex * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)nd * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y2, (size_t)nd * ncol * sizeof(double)));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniformDouble(gen, d_x_ex, nd_ex * ncol);
    curandDestroyGenerator(gen);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("=== Gradient High-Occupancy Benchmark ===\n");
    printf("    nx=%d ny=%d nz=%d FDn=%d ncol=%d\n\n", nx, ny, nz, FDn, ncol);

    // V2 baseline
    {
        dim3 block(32, 8);
        dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz);
        printf("  V2 grid: (%d, %d, %d) = %d blocks\n",
               grid.x, grid.y, grid.z, grid.x * grid.y * grid.z);

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
        printf("  V2 (inner loop): %8.4f ms\n", ms / NREPS);
    }

    // V3: one block per (tile, k*col) — 20x more blocks
    {
        dim3 block(32, 8);
        dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz * ncol);
        printf("  V3 grid: (%d, %d, %d) = %d blocks\n",
               grid.x, grid.y, grid.z, grid.x * grid.y * grid.z);

        for (int i = 0; i < 5; i++)
            gradient_x_kernel_v3<6><<<grid, block>>>(d_x_ex, d_y2,
                nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++)
            gradient_x_kernel_v3<6><<<grid, block>>>(d_x_ex, d_y2,
                nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        printf("  V3 (1 col/block): %8.4f ms\n", ms / NREPS);
    }

    // V3b: same pattern with FMA
    {
        dim3 block(32, 8);
        dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz * ncol);

        for (int i = 0; i < 5; i++)
            gradient_x_kernel_v3b<6><<<grid, block>>>(d_x_ex, d_y2,
                nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol, 1.0);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++)
            gradient_x_kernel_v3b<6><<<grid, block>>>(d_x_ex, d_y2,
                nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol, 1.0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        printf("  V3b (FMA):        %8.4f ms\n", ms / NREPS);
    }

    // Verify correctness: compare V2 and V3 outputs
    {
        dim3 block(32, 8);
        CUDA_CHECK(cudaMemset(d_y, 0, nd * ncol * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_y2, 0, nd * ncol * sizeof(double)));

        gradient_x_kernel_v2<6><<<dim3(ceildiv(nx,32), ceildiv(ny,8), nz), block>>>(
            d_x_ex, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol);
        gradient_x_kernel_v3<6><<<dim3(ceildiv(nx,32), ceildiv(ny,8), nz*ncol), block>>>(
            d_x_ex, d_y2, nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol);
        CUDA_CHECK(cudaDeviceSynchronize());

        double *h_y = (double*)malloc(nd * ncol * sizeof(double));
        double *h_y2 = (double*)malloc(nd * ncol * sizeof(double));
        CUDA_CHECK(cudaMemcpy(h_y, d_y, nd * ncol * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y2, d_y2, nd * ncol * sizeof(double), cudaMemcpyDeviceToHost));

        double maxerr = 0.0;
        for (int i = 0; i < nd * ncol; i++) {
            double err = fabs(h_y[i] - h_y2[i]);
            if (err > maxerr) maxerr = err;
        }
        printf("\n  V2 vs V3 max error: %.2e %s\n", maxerr, maxerr < 1e-14 ? "(PASS)" : "(FAIL)");
        free(h_y); free(h_y2);
    }

    printf("\n");
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_x_ex); cudaFree(d_y); cudaFree(d_y2);
    return 0;
}
