#ifdef USE_CUDA
#include "core/gpu_common.cuh"

namespace sparc {
namespace gpu {

// Gradient kernel for direction=0 (x), stride=1, coefficients from d_D1x
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
    for (int p = 1; p <= FDn; ++p) {
        val += d_D1x[p] * (x_ex[idx + p] - x_ex[idx - p]);
    }
    y[loc] = val;
}

// Gradient kernel for direction=1 (y), stride=nx_ex
__global__ void gradient_y_kernel(
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
    for (int p = 1; p <= FDn; ++p) {
        val += d_D1y[p] * (x_ex[idx + p * nx_ex] - x_ex[idx - p * nx_ex]);
    }
    y[loc] = val;
}

// Gradient kernel for direction=2 (z), stride=nxny_ex
__global__ void gradient_z_kernel(
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
    for (int p = 1; p <= FDn; ++p) {
        val += d_D1z[p] * (x_ex[idx + p * nxny_ex] - x_ex[idx - p * nxny_ex]);
    }
    y[loc] = val;
}

// Host wrapper: applies gradient in direction dir (0=x, 1=y, 2=z) for ncol columns
void gradient_gpu(
    const double* d_x_ex, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    int direction, int ncol)
{
    int nxny_ex = nx_ex * ny_ex;
    int nd = nx * ny * nz;
    int nd_ex = nxny_ex * (nz + 2 * FDn);

    dim3 block(32, 4, 4);
    dim3 grid(ceildiv(nx, 32), ceildiv(ny, 4), ceildiv(nz, 4));

    for (int n = 0; n < ncol; ++n) {
        const double* xn = d_x_ex + n * nd_ex;
        double* yn = d_y + n * nd;

        switch (direction) {
            case 0:
                gradient_x_kernel<<<grid, block>>>(xn, yn, nx, ny, nz, FDn, nx_ex, ny_ex, nxny_ex);
                break;
            case 1:
                gradient_y_kernel<<<grid, block>>>(xn, yn, nx, ny, nz, FDn, nx_ex, ny_ex, nxny_ex);
                break;
            case 2:
                gradient_z_kernel<<<grid, block>>>(xn, yn, nx, ny, nz, FDn, nx_ex, ny_ex, nxny_ex);
                break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// V2: Template FDn + multi-column batching (same pattern as Laplacian V2)
// Single launch, blockIdx.z = k, inner loop over columns
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

template<int FDN>
__global__ void gradient_y_kernel_v2(
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
            val += d_D1y[p] * (x_ex[idx + p * nx_ex] - x_ex[idx - p * nx_ex]);
        y[loc + n * nd] = val;
    }
}

template<int FDN>
__global__ void gradient_z_kernel_v2(
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
            val += d_D1z[p] * (x_ex[idx + p * nxny_ex] - x_ex[idx - p * nxny_ex]);
        y[loc + n * nd] = val;
    }
}

void gradient_v2_gpu(
    const double* d_x_ex, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    int direction, int ncol)
{
    int nxny_ex = nx_ex * ny_ex;
    int nd = nx * ny * nz;
    int nz_ex = nz + 2 * FDn;
    int nd_ex = nxny_ex * nz_ex;

    dim3 block(32, 8);
    dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz);

    if (FDn == 6) {
        switch (direction) {
            case 0: gradient_x_kernel_v2<6><<<grid, block>>>(d_x_ex, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol); break;
            case 1: gradient_y_kernel_v2<6><<<grid, block>>>(d_x_ex, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol); break;
            case 2: gradient_z_kernel_v2<6><<<grid, block>>>(d_x_ex, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol); break;
        }
    } else {
        switch (direction) {
            case 0: gradient_x_kernel_v2<3><<<grid, block>>>(d_x_ex, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol); break;
            case 1: gradient_y_kernel_v2<3><<<grid, block>>>(d_x_ex, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol); break;
            case 2: gradient_z_kernel_v2<3><<<grid, block>>>(d_x_ex, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol); break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
} // namespace sparc
#endif // USE_CUDA
