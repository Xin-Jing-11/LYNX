#ifdef USE_CUDA
#include "core/gpu_common.cuh"
#include "parallel/HaloExchange.cuh"

namespace lynx {
namespace gpu {

// ============================================================
// V0: Original per-column halo exchange (kept for compatibility)
// ============================================================

// Copy local data into interior of extended array (one column).
__global__ void halo_copy_interior_kernel(
    const double* __restrict__ x,
    double* __restrict__ x_ex,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex, int nxny_ex)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int loc = i + j * nx + k * nx * ny;
    int idx_ex = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
    x_ex[idx_ex] = x[loc];
}

// Periodic BC for one face direction.
// face_id: 0=z_lo, 1=z_hi, 2=y_lo, 3=y_hi, 4=x_lo, 5=x_hi
__global__ void halo_periodic_bc_kernel(
    double* __restrict__ x_ex,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex, int nz_ex,
    int nxny_ex, int face_id)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (face_id == 0) {
        int total = nxny_ex * FDn;
        if (tid >= total) return;
        int p = tid / nxny_ex;
        int rem = tid % nxny_ex;
        x_ex[p * nxny_ex + rem] = x_ex[(nz + p) * nxny_ex + rem];
    }
    else if (face_id == 1) {
        int total = nxny_ex * FDn;
        if (tid >= total) return;
        int p = tid / nxny_ex;
        int rem = tid % nxny_ex;
        x_ex[(nz + FDn + p) * nxny_ex + rem] = x_ex[(FDn + p) * nxny_ex + rem];
    }
    else if (face_id == 2) {
        int total = nz_ex * FDn * nx_ex;
        if (tid >= total) return;
        int i = tid % nx_ex;
        int tmp = tid / nx_ex;
        int j = tmp % FDn;
        int k = tmp / FDn;
        x_ex[j * nx_ex + k * nxny_ex + i] =
            x_ex[(ny + j) * nx_ex + k * nxny_ex + i];
    }
    else if (face_id == 3) {
        int total = nz_ex * FDn * nx_ex;
        if (tid >= total) return;
        int i = tid % nx_ex;
        int tmp = tid / nx_ex;
        int j = tmp % FDn;
        int k = tmp / FDn;
        x_ex[(ny + FDn + j) * nx_ex + k * nxny_ex + i] =
            x_ex[(FDn + j) * nx_ex + k * nxny_ex + i];
    }
    else if (face_id == 4) {
        int total = nz_ex * ny_ex * FDn;
        if (tid >= total) return;
        int i = tid % FDn;
        int tmp = tid / FDn;
        int j = tmp % ny_ex;
        int k = tmp / ny_ex;
        int base = j * nx_ex + k * nxny_ex;
        x_ex[base + i] = x_ex[base + nx + i];
    }
    else if (face_id == 5) {
        int total = nz_ex * ny_ex * FDn;
        if (tid >= total) return;
        int i = tid % FDn;
        int tmp = tid / FDn;
        int j = tmp % ny_ex;
        int k = tmp / ny_ex;
        int base = j * nx_ex + k * nxny_ex;
        x_ex[base + nx + FDn + i] = x_ex[base + FDn + i];
    }
}

// V0: Per-column halo exchange (original)
void halo_exchange_gpu(
    const double* d_x,
    double* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol,
    bool periodic_x, bool periodic_y, bool periodic_z,
    cudaStream_t stream)
{
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nz_ex = nz + 2 * FDn;
    int nxny_ex = nx_ex * ny_ex;
    int nd = nx * ny * nz;
    int nd_ex = nxny_ex * nz_ex;

    dim3 block(32, 4, 4);
    dim3 grid(ceildiv(nx, 32), ceildiv(ny, 4), ceildiv(nz, 4));

    for (int n = 0; n < ncol; ++n) {
        CUDA_CHECK(cudaMemsetAsync(d_x_ex + n * nd_ex, 0, nd_ex * sizeof(double), stream));

        halo_copy_interior_kernel<<<grid, block, 0, stream>>>(
            d_x + n * nd, d_x_ex + n * nd_ex,
            nx, ny, nz, FDn, nx_ex, ny_ex, nxny_ex);

        int bs = 256;
        if (periodic_z) {
            int total_z = nxny_ex * FDn;
            int gz = ceildiv(total_z, bs);
            halo_periodic_bc_kernel<<<gz, bs, 0, stream>>>(d_x_ex + n * nd_ex,
                nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, 0);
            halo_periodic_bc_kernel<<<gz, bs, 0, stream>>>(d_x_ex + n * nd_ex,
                nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, 1);
        }
        if (periodic_y) {
            int total_y = nz_ex * FDn * nx_ex;
            int gy = ceildiv(total_y, bs);
            halo_periodic_bc_kernel<<<gy, bs, 0, stream>>>(d_x_ex + n * nd_ex,
                nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, 2);
            halo_periodic_bc_kernel<<<gy, bs, 0, stream>>>(d_x_ex + n * nd_ex,
                nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, 3);
        }
        if (periodic_x) {
            int total_x = nz_ex * ny_ex * FDn;
            int gx = ceildiv(total_x, bs);
            halo_periodic_bc_kernel<<<gx, bs, 0, stream>>>(d_x_ex + n * nd_ex,
                nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, 4);
            halo_periodic_bc_kernel<<<gx, bs, 0, stream>>>(d_x_ex + n * nd_ex,
                nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, 5);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// V2: Batched halo exchange — all columns in single kernel launches
// Reduces 8*ncol launches to just 8 launches total
// ============================================================

__global__ void halo_copy_interior_batched_kernel(
    const double* __restrict__ x,
    double* __restrict__ x_ex,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex, int nxny_ex,
    int nd, int nd_ex)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int flat_z = blockIdx.z;
    int k = flat_z % nz;
    int n = flat_z / nz;
    if (i >= nx || j >= ny) return;

    int loc = i + j * nx + k * nx * ny + n * nd;
    int idx_ex = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex + n * nd_ex;
    x_ex[idx_ex] = x[loc];
}

__global__ void halo_periodic_bc_batched_kernel(
    double* __restrict__ x_ex,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex, int nz_ex,
    int nxny_ex, int nd_ex,
    int face_id, int ncol)
{
    int flat_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine per-face element count
    int face_size;
    if (face_id <= 1) face_size = nxny_ex * FDn;
    else if (face_id <= 3) face_size = nz_ex * FDn * nx_ex;
    else face_size = nz_ex * ny_ex * FDn;

    int n = flat_tid / face_size;
    int tid = flat_tid % face_size;
    if (n >= ncol) return;

    double* col_ex = x_ex + n * nd_ex;

    if (face_id == 0) {
        int p = tid / nxny_ex;
        int rem = tid % nxny_ex;
        col_ex[p * nxny_ex + rem] = col_ex[(nz + p) * nxny_ex + rem];
    }
    else if (face_id == 1) {
        int p = tid / nxny_ex;
        int rem = tid % nxny_ex;
        col_ex[(nz + FDn + p) * nxny_ex + rem] = col_ex[(FDn + p) * nxny_ex + rem];
    }
    else if (face_id == 2) {
        int i = tid % nx_ex;
        int tmp = tid / nx_ex;
        int j = tmp % FDn;
        int k = tmp / FDn;
        col_ex[j * nx_ex + k * nxny_ex + i] =
            col_ex[(ny + j) * nx_ex + k * nxny_ex + i];
    }
    else if (face_id == 3) {
        int i = tid % nx_ex;
        int tmp = tid / nx_ex;
        int j = tmp % FDn;
        int k = tmp / FDn;
        col_ex[(ny + FDn + j) * nx_ex + k * nxny_ex + i] =
            col_ex[(FDn + j) * nx_ex + k * nxny_ex + i];
    }
    else if (face_id == 4) {
        int i = tid % FDn;
        int tmp = tid / FDn;
        int j = tmp % ny_ex;
        int k = tmp / ny_ex;
        int base = j * nx_ex + k * nxny_ex;
        col_ex[base + i] = col_ex[base + nx + i];
    }
    else if (face_id == 5) {
        int i = tid % FDn;
        int tmp = tid / FDn;
        int j = tmp % ny_ex;
        int k = tmp / ny_ex;
        int base = j * nx_ex + k * nxny_ex;
        col_ex[base + nx + FDn + i] = col_ex[base + FDn + i];
    }
}

void halo_exchange_batched_gpu(
    const double* d_x,
    double* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol,
    bool periodic_x, bool periodic_y, bool periodic_z,
    cudaStream_t stream)
{
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nz_ex = nz + 2 * FDn;
    int nxny_ex = nx_ex * ny_ex;
    int nd = nx * ny * nz;
    int nd_ex = nxny_ex * nz_ex;

    // Zero all x_ex columns at once
    CUDA_CHECK(cudaMemsetAsync(d_x_ex, 0, (size_t)nd_ex * ncol * sizeof(double), stream));

    // Copy interior: one block per (tile_x, tile_y, k * ncol)
    {
        dim3 block(32, 8);
        dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz * ncol);
        halo_copy_interior_batched_kernel<<<grid, block, 0, stream>>>(
            d_x, d_x_ex, nx, ny, nz, FDn,
            nx_ex, ny_ex, nxny_ex, nd, nd_ex);
    }

    // Periodic BC: one launch per face, all columns batched
    int bs = 256;
    if (periodic_z) {
        int total = nxny_ex * FDn * ncol;
        halo_periodic_bc_batched_kernel<<<ceildiv(total, bs), bs, 0, stream>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 0, ncol);
        halo_periodic_bc_batched_kernel<<<ceildiv(total, bs), bs, 0, stream>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 1, ncol);
    }
    if (periodic_y) {
        int total = nz_ex * FDn * nx_ex * ncol;
        halo_periodic_bc_batched_kernel<<<ceildiv(total, bs), bs, 0, stream>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 2, ncol);
        halo_periodic_bc_batched_kernel<<<ceildiv(total, bs), bs, 0, stream>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 3, ncol);
    }
    if (periodic_x) {
        int total = nz_ex * ny_ex * FDn * ncol;
        halo_periodic_bc_batched_kernel<<<ceildiv(total, bs), bs, 0, stream>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 4, ncol);
        halo_periodic_bc_batched_kernel<<<ceildiv(total, bs), bs, 0, stream>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 5, ncol);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// V3: No-memset batched halo exchange
// For fully-periodic systems, every x_ex element is written by either
// interior copy or periodic BC, so cudaMemset is unnecessary.
// ============================================================

void halo_exchange_batched_nomemset_gpu(
    const double* d_x,
    double* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol,
    bool periodic_x, bool periodic_y, bool periodic_z,
    cudaStream_t stream)
{
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nz_ex = nz + 2 * FDn;
    int nxny_ex = nx_ex * ny_ex;
    int nd = nx * ny * nz;
    int nd_ex = nxny_ex * nz_ex;

    // Copy interior
    {
        dim3 block(32, 8);
        dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz * ncol);
        halo_copy_interior_batched_kernel<<<grid, block, 0, stream>>>(
            d_x, d_x_ex, nx, ny, nz, FDn,
            nx_ex, ny_ex, nxny_ex, nd, nd_ex);
    }

    // Periodic BC: z first, then y, then x (dependency order)
    int bs = 256;
    if (periodic_z) {
        int total = nxny_ex * FDn * ncol;
        halo_periodic_bc_batched_kernel<<<ceildiv(total, bs), bs, 0, stream>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 0, ncol);
        halo_periodic_bc_batched_kernel<<<ceildiv(total, bs), bs, 0, stream>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 1, ncol);
    }
    if (periodic_y) {
        int total = nz_ex * FDn * nx_ex * ncol;
        halo_periodic_bc_batched_kernel<<<ceildiv(total, bs), bs, 0, stream>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 2, ncol);
        halo_periodic_bc_batched_kernel<<<ceildiv(total, bs), bs, 0, stream>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 3, ncol);
    }
    if (periodic_x) {
        int total = nz_ex * ny_ex * FDn * ncol;
        halo_periodic_bc_batched_kernel<<<ceildiv(total, bs), bs, 0, stream>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 4, ncol);
        halo_periodic_bc_batched_kernel<<<ceildiv(total, bs), bs, 0, stream>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 5, ncol);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
} // namespace lynx
#endif // USE_CUDA
