// Standalone benchmark for orthogonal Laplacian kernels (V2, V6, V7, V8, V3)
// Tests the FD stencil that dominates CheFSI iteration time.
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define MAX_FDN 6
#define MAX_FD_COEFF 7

__host__ __device__ int ceildiv(int a, int b) { return (a + b - 1) / b; }

// Constant memory for FD stencil coefficients
__constant__ double d_D2x[MAX_FD_COEFF];
__constant__ double d_D2y[MAX_FD_COEFF];
__constant__ double d_D2z[MAX_FD_COEFF];
__constant__ double d_aD2x[7];
__constant__ double d_aD2y[7];
__constant__ double d_aD2z[7];

// ============================================================
// V2: Template FDn + multi-column batching (blockIdx.z = k)
// ============================================================
template<int FDN>
__global__ void laplacian_orth_kernel_v2(
    const double* __restrict__ x_ex,
    const double* __restrict__ V,
    double* __restrict__ y,
    int nx, int ny, int nz,
    int nx_ex, int ny_ex, int nxny_ex,
    int nd, int nd_ex,
    double a, double b, double diag_coeff,
    int ncol)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int idx_base = (i + FDN) + (j + FDN) * nx_ex + (k + FDN) * nxny_ex;
    int loc = i + j * nx + k * nx * ny;

    double v_val = 0.0;
    if (V) v_val = b * V[loc];

    for (int n = 0; n < ncol; ++n) {
        int idx = idx_base + n * nd_ex;
        double center = x_ex[idx];
        double val = diag_coeff * center;

        #pragma unroll
        for (int p = 1; p <= FDN; ++p) {
            val += a * d_D2x[p] * (x_ex[idx + p] + x_ex[idx - p]);
            val += a * d_D2y[p] * (x_ex[idx + p * nx_ex] + x_ex[idx - p * nx_ex]);
            val += a * d_D2z[p] * (x_ex[idx + p * nxny_ex] + x_ex[idx - p * nxny_ex]);
        }

        val += v_val * center;
        y[loc + n * nd] = val;
    }
}

// ============================================================
// V6: One column per block (blockIdx.z = k*ncol + col)
// ============================================================
template<int FDN>
__global__ __launch_bounds__(256, 6)
void laplacian_orth_kernel_v6(
    const double* __restrict__ x_ex,
    const double* __restrict__ V,
    double* __restrict__ y,
    int nx, int ny, int nz,
    int nx_ex, int ny_ex, int nxny_ex,
    int nd, int nd_ex,
    double a, double b, double diag_coeff,
    int ncol)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int flat_z = blockIdx.z;
    int k = flat_z % nz;
    int n = flat_z / nz;
    if (i >= nx || j >= ny) return;

    int idx = (i + FDN) + (j + FDN) * nx_ex + (k + FDN) * nxny_ex + n * nd_ex;
    int loc = i + j * nx + k * nx * ny;

    double center = x_ex[idx];
    double val = diag_coeff * center;

    #pragma unroll
    for (int p = 1; p <= FDN; ++p) {
        val += a * d_D2x[p] * (x_ex[idx + p] + x_ex[idx - p]);
        val += a * d_D2y[p] * (x_ex[idx + p * nx_ex] + x_ex[idx - p * nx_ex]);
        val += a * d_D2z[p] * (x_ex[idx + p * nxny_ex] + x_ex[idx - p * nxny_ex]);
    }

    if (V) val += b * V[loc] * center;
    y[loc + n * nd] = val;
}

// ============================================================
// V7: Precomputed a*coeff (FMA-friendly)
// ============================================================
template<int FDN>
__global__ __launch_bounds__(256, 6)
void laplacian_orth_kernel_v7(
    const double* __restrict__ x_ex,
    const double* __restrict__ V,
    double* __restrict__ y,
    int nx, int ny, int nz,
    int nx_ex, int ny_ex, int nxny_ex,
    int nd, int nd_ex,
    double b, double diag_coeff,
    int ncol)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int flat_z = blockIdx.z;
    int k = flat_z % nz;
    int n = flat_z / nz;
    if (i >= nx || j >= ny) return;

    int idx = (i + FDN) + (j + FDN) * nx_ex + (k + FDN) * nxny_ex + n * nd_ex;
    int loc = i + j * nx + k * nx * ny;

    double center = x_ex[idx];
    double val = diag_coeff * center;

    #pragma unroll
    for (int p = 1; p <= FDN; ++p) {
        val += d_aD2x[p] * (x_ex[idx + p] + x_ex[idx - p]);
        val += d_aD2y[p] * (x_ex[idx + p * nx_ex] + x_ex[idx - p * nx_ex]);
        val += d_aD2z[p] * (x_ex[idx + p * nxny_ex] + x_ex[idx - p * nxny_ex]);
    }

    if (V) val += b * V[loc] * center;
    y[loc + n * nd] = val;
}

// ============================================================
// V8: Multi-column loop + precomputed coefficients
// ============================================================
template<int FDN>
__global__ void laplacian_orth_kernel_v8(
    const double* __restrict__ x_ex,
    const double* __restrict__ V,
    double* __restrict__ y,
    int nx, int ny, int nz,
    int nx_ex, int ny_ex, int nxny_ex,
    int nd, int nd_ex,
    double b, double diag_coeff,
    int ncol)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int idx_base = (i + FDN) + (j + FDN) * nx_ex + (k + FDN) * nxny_ex;
    int loc = i + j * nx + k * nx * ny;

    double v_val = 0.0;
    if (V) v_val = b * V[loc];

    for (int n = 0; n < ncol; ++n) {
        int idx = idx_base + n * nd_ex;
        double center = x_ex[idx];
        double val = diag_coeff * center;

        #pragma unroll
        for (int p = 1; p <= FDN; ++p) {
            val += d_aD2x[p] * (x_ex[idx + p] + x_ex[idx - p]);
            val += d_aD2y[p] * (x_ex[idx + p * nx_ex] + x_ex[idx - p * nx_ex]);
            val += d_aD2z[p] * (x_ex[idx + p * nxny_ex] + x_ex[idx - p * nxny_ex]);
        }

        val += v_val * center;
        y[loc + n * nd] = val;
    }
}

// ============================================================
// V3: Shared memory 2D tiling
// ============================================================
template<int FDN, int BX, int BY>
__global__ void laplacian_orth_kernel_v3(
    const double* __restrict__ x_ex,
    const double* __restrict__ V,
    double* __restrict__ y,
    int nx, int ny, int nz,
    int nx_ex, int ny_ex, int nxny_ex,
    int nd, int nd_ex,
    double a, double b, double diag_coeff,
    int ncol)
{
    constexpr int TILE_X_L = BX + 2 * FDN;
    constexpr int TILE_Y_L = BY + 2 * FDN;
    constexpr int TILE_SIZE = TILE_X_L * TILE_Y_L;
    constexpr int BLOCK_SIZE = BX * BY;

    int bx = blockIdx.x * BX;
    int by = blockIdx.y * BY;
    int k  = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = bx + tx;
    int j = by + ty;
    bool active = (i < nx && j < ny);

    extern __shared__ double smem[];
    int tid = ty * BX + tx;

    double v_val = 0.0;
    int loc = 0;
    if (active) {
        loc = i + j * nx + k * nx * ny;
        if (V) v_val = b * V[loc];
    }

    for (int n = 0; n < ncol; ++n) {
        const double* x_col = x_ex + n * nd_ex;
        int z_off = (k + FDN) * nxny_ex;

        for (int idx = tid; idx < TILE_SIZE; idx += BLOCK_SIZE) {
            int ti = idx % TILE_X_L;
            int tj = idx / TILE_X_L;
            int gi = bx + ti;
            int gj = by + tj;
            smem[tj * TILE_X_L + ti] = (gi < nx_ex && gj < ny_ex)
                ? x_col[gi + gj * nx_ex + z_off] : 0.0;
        }
        __syncthreads();

        if (active) {
            int si = tx + FDN;
            int sj = ty + FDN;
            double center = smem[sj * TILE_X_L + si];
            double val = diag_coeff * center;

            #pragma unroll
            for (int p = 1; p <= FDN; ++p) {
                val += a * d_D2x[p] * (smem[sj * TILE_X_L + si + p] +
                                        smem[sj * TILE_X_L + si - p]);
                val += a * d_D2y[p] * (smem[(sj + p) * TILE_X_L + si] +
                                        smem[(sj - p) * TILE_X_L + si]);
            }

            int gidx = (i + FDN) + (j + FDN) * nx_ex + z_off;
            #pragma unroll
            for (int p = 1; p <= FDN; ++p) {
                val += a * d_D2z[p] * (x_col[gidx + p * nxny_ex] +
                                        x_col[gidx - p * nxny_ex]);
            }

            val += v_val * center;
            y[loc + n * nd] = val;
        }
        __syncthreads();
    }
}

// ============================================================
// Benchmark harness
// ============================================================
int main() {
    // DFT-representative grid: ~25x26x27 (Si4 benchmark)
    const int nx = 25, ny = 26, nz = 27;
    const int FDn = 6;
    const int ncol = 20;  // typical band count
    const int NREPS = 200;

    const int nx_ex = nx + 2 * FDn;
    const int ny_ex = ny + 2 * FDn;
    const int nxny_ex = nx_ex * ny_ex;
    const int nz_ex = nz + 2 * FDn;
    const int nd = nx * ny * nz;
    const int nd_ex = nxny_ex * nz_ex;

    // Upload fake stencil coefficients
    double h_D2[MAX_FD_COEFF] = {-4.0, 1.333, -0.0833, 0.0159, -0.00397, 0.00119, -0.000397};
    CUDA_CHECK(cudaMemcpyToSymbol(d_D2x, h_D2, MAX_FD_COEFF * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_D2y, h_D2, MAX_FD_COEFF * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_D2z, h_D2, MAX_FD_COEFF * sizeof(double)));

    double a = 1.0;
    double h_aD2[7];
    for (int i = 0; i < 7; i++) h_aD2[i] = a * h_D2[i];
    CUDA_CHECK(cudaMemcpyToSymbol(d_aD2x, h_aD2, 7 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_aD2y, h_aD2, 7 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_aD2z, h_aD2, 7 * sizeof(double)));

    // Allocate device memory
    double *d_x_ex, *d_V, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x_ex, (size_t)nd_ex * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, (size_t)nd * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)nd * ncol * sizeof(double)));

    // Fill with random data
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniformDouble(gen, d_x_ex, nd_ex * ncol);
    curandGenerateUniformDouble(gen, d_V, nd);
    curandDestroyGenerator(gen);

    double b = 1.0, diag_coeff = -4.0;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto bench = [&](const char* name, auto launcher) {
        // Warmup
        for (int i = 0; i < 5; i++) launcher();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++) launcher();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= NREPS;

        // Bandwidth estimate: each grid point reads center + 2*FDn*3 neighbors + V, writes 1
        // Per column: reads ~(1 + 36) doubles, writes 1 double
        double bytes_per_col = (double)nd * (38.0 + 1.0) * sizeof(double);
        double total_bytes = bytes_per_col * ncol;
        double gbps = total_bytes / (ms * 1e6);

        printf("  %-12s: %8.4f ms  |  %6.1f GB/s  |  %6.1f GFLOP/s\n",
               name, ms, gbps,
               (double)nd * ncol * (1 + 6 * 6) / (ms * 1e6));  // ~37 FLOPs/point
    };

    printf("=== Laplacian Benchmark: nx=%d ny=%d nz=%d FDn=%d ncol=%d (NREPS=%d) ===\n",
           nx, ny, nz, FDn, ncol, NREPS);

    // V2: multi-column loop, grid.z = nz
    bench("V2", [&]() {
        dim3 block(32, 8);
        dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz);
        laplacian_orth_kernel_v2<6><<<grid, block>>>(
            d_x_ex, d_V, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex,
            nd, nd_ex, a, b, diag_coeff, ncol);
    });

    // V6: 1 col/block, grid.z = nz*ncol
    bench("V6", [&]() {
        dim3 block(32, 8);
        dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz * ncol);
        laplacian_orth_kernel_v6<6><<<grid, block>>>(
            d_x_ex, d_V, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex,
            nd, nd_ex, a, b, diag_coeff, ncol);
    });

    // V7: precomputed a*coeff, 1 col/block
    bench("V7", [&]() {
        dim3 block(32, 8);
        dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz * ncol);
        laplacian_orth_kernel_v7<6><<<grid, block>>>(
            d_x_ex, d_V, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex,
            nd, nd_ex, b, diag_coeff, ncol);
    });

    // V8: multi-column + precomputed
    bench("V8", [&]() {
        dim3 block(32, 8);
        dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz);
        laplacian_orth_kernel_v8<6><<<grid, block>>>(
            d_x_ex, d_V, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex,
            nd, nd_ex, b, diag_coeff, ncol);
    });

    // V3: shared memory tiling
    bench("V3-smem", [&]() {
        constexpr int BX = 32, BY = 8, FDN = 6;
        constexpr int smem_bytes = (BX + 2*FDN) * (BY + 2*FDN) * sizeof(double);
        dim3 block(BX, BY);
        dim3 grid(ceildiv(nx, BX), ceildiv(ny, BY), nz);
        laplacian_orth_kernel_v3<FDN, BX, BY><<<grid, block, smem_bytes>>>(
            d_x_ex, d_V, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex,
            nd, nd_ex, a, b, diag_coeff, ncol);
    });

    printf("\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_x_ex); cudaFree(d_V); cudaFree(d_y);
    return 0;
}
