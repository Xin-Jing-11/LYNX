#ifdef USE_CUDA
#include "core/gpu_common.cuh"
#include <cuComplex.h>

namespace lynx {
namespace gpu {

// ============================================================
// Helper: real * complex scaling
// ============================================================
__device__ __forceinline__ cuDoubleComplex rcmul(double s, cuDoubleComplex z) {
    return make_cuDoubleComplex(s * z.x, s * z.y);
}

// Atomic add for cuDoubleComplex (two double atomicAdds)
__device__ __forceinline__ void atomicAddZ(cuDoubleComplex* addr, cuDoubleComplex val) {
    atomicAdd(&(reinterpret_cast<double*>(addr)[0]), val.x);
    atomicAdd(&(reinterpret_cast<double*>(addr)[1]), val.y);
}

// ============================================================
//  COMPLEX HALO EXCHANGE
// ============================================================

// Copy interior data into extended array (complex, batched)
__global__ void halo_copy_interior_z_batched_kernel(
    const cuDoubleComplex* __restrict__ x,
    cuDoubleComplex* __restrict__ x_ex,
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

// Periodic BC for complex data (batched, same structure as real)
__global__ void halo_periodic_bc_z_batched_kernel(
    cuDoubleComplex* __restrict__ x_ex,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex, int nz_ex,
    int nxny_ex, int nd_ex,
    int face_id, int ncol)
{
    int flat_tid = blockIdx.x * blockDim.x + threadIdx.x;

    int face_size;
    if (face_id <= 1) face_size = nxny_ex * FDn;
    else if (face_id <= 3) face_size = nz_ex * FDn * nx_ex;
    else face_size = nz_ex * ny_ex * FDn;

    int n = flat_tid / face_size;
    int tid = flat_tid % face_size;
    if (n >= ncol) return;

    cuDoubleComplex* col_ex = x_ex + n * nd_ex;

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

// Apply Bloch phase factors to ghost zones after periodic wrap.
// face_id: 0=z_lo (phase_l), 1=z_hi (phase_r), 2=y_lo, 3=y_hi, 4=x_lo, 5=x_hi
// phase = make_cuDoubleComplex(cos(theta), sin(theta)) for the appropriate direction
__global__ void halo_bloch_phase_z_kernel(
    cuDoubleComplex* __restrict__ x_ex,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex, int nz_ex,
    int nxny_ex, int nd_ex,
    int face_id, int ncol,
    double phase_re, double phase_im)
{
    int flat_tid = blockIdx.x * blockDim.x + threadIdx.x;

    int face_size;
    if (face_id <= 1) face_size = nxny_ex * FDn;
    else if (face_id <= 3) face_size = nz_ex * FDn * nx_ex;
    else face_size = nz_ex * ny_ex * FDn;

    int n = flat_tid / face_size;
    int tid = flat_tid % face_size;
    if (n >= ncol) return;

    cuDoubleComplex phase = make_cuDoubleComplex(phase_re, phase_im);
    cuDoubleComplex* col_ex = x_ex + n * nd_ex;
    int idx = -1;

    if (face_id == 0) {
        // z lo ghost: rows 0..FDn-1
        int p = tid / nxny_ex;
        int rem = tid % nxny_ex;
        idx = p * nxny_ex + rem;
    }
    else if (face_id == 1) {
        // z hi ghost: rows nz+FDn..nz+2*FDn-1
        int p = tid / nxny_ex;
        int rem = tid % nxny_ex;
        idx = (nz + FDn + p) * nxny_ex + rem;
    }
    else if (face_id == 2) {
        // y lo ghost
        int i = tid % nx_ex;
        int tmp = tid / nx_ex;
        int j = tmp % FDn;
        int k = tmp / FDn;
        idx = j * nx_ex + k * nxny_ex + i;
    }
    else if (face_id == 3) {
        // y hi ghost
        int i = tid % nx_ex;
        int tmp = tid / nx_ex;
        int j = tmp % FDn;
        int k = tmp / FDn;
        idx = (ny + FDn + j) * nx_ex + k * nxny_ex + i;
    }
    else if (face_id == 4) {
        // x lo ghost
        int i = tid % FDn;
        int tmp = tid / FDn;
        int j = tmp % ny_ex;
        int k = tmp / ny_ex;
        idx = j * nx_ex + k * nxny_ex + i;
    }
    else if (face_id == 5) {
        // x hi ghost
        int i = tid % FDn;
        int tmp = tid / FDn;
        int j = tmp % ny_ex;
        int k = tmp / ny_ex;
        idx = j * nx_ex + k * nxny_ex + (nx + FDn + i);
    }

    if (idx >= 0) {
        col_ex[idx] = cuCmul(col_ex[idx], phase);
    }
}

// Host wrapper: complex halo exchange with Bloch phases
void halo_exchange_z_gpu(
    const cuDoubleComplex* d_x,
    cuDoubleComplex* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol,
    bool periodic_x, bool periodic_y, bool periodic_z,
    double kx_Lx, double ky_Ly, double kz_Lz)
{
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nz_ex = nz + 2 * FDn;
    int nxny_ex = nx_ex * ny_ex;
    int nd = nx * ny * nz;
    int nd_ex = nxny_ex * nz_ex;

    // Zero all x_ex columns
    CUDA_CHECK(cudaMemset(d_x_ex, 0, (size_t)nd_ex * ncol * sizeof(cuDoubleComplex)));

    // Copy interior
    {
        dim3 block(32, 8);
        dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz * ncol);
        halo_copy_interior_z_batched_kernel<<<grid, block>>>(
            d_x, d_x_ex, nx, ny, nz, FDn,
            nx_ex, ny_ex, nxny_ex, nd, nd_ex);
    }

    // Periodic BC: copy ghost zones (same data as interior, no phase yet)
    int bs = 256;
    if (periodic_z) {
        int total = nxny_ex * FDn * ncol;
        halo_periodic_bc_z_batched_kernel<<<ceildiv(total, bs), bs>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 0, ncol);
        halo_periodic_bc_z_batched_kernel<<<ceildiv(total, bs), bs>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 1, ncol);
    }
    if (periodic_y) {
        int total = nz_ex * FDn * nx_ex * ncol;
        halo_periodic_bc_z_batched_kernel<<<ceildiv(total, bs), bs>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 2, ncol);
        halo_periodic_bc_z_batched_kernel<<<ceildiv(total, bs), bs>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 3, ncol);
    }
    if (periodic_x) {
        int total = nz_ex * ny_ex * FDn * ncol;
        halo_periodic_bc_z_batched_kernel<<<ceildiv(total, bs), bs>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 4, ncol);
        halo_periodic_bc_z_batched_kernel<<<ceildiv(total, bs), bs>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex, 5, ncol);
    }

    // Apply Bloch phase factors to ghost zones
    // phase_l = e^{-ikL} = cos(kL) - i*sin(kL), phase_r = e^{+ikL} = cos(kL) + i*sin(kL)
    if (periodic_z && (kz_Lz != 0.0)) {
        double cos_z = cos(kz_Lz), sin_z = sin(kz_Lz);
        int total = nxny_ex * FDn * ncol;
        // z_lo ghost: phase_l = e^{-ikL_z}
        halo_bloch_phase_z_kernel<<<ceildiv(total, bs), bs>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex,
            0, ncol, cos_z, -sin_z);
        // z_hi ghost: phase_r = e^{+ikL_z}
        halo_bloch_phase_z_kernel<<<ceildiv(total, bs), bs>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex,
            1, ncol, cos_z, sin_z);
    }
    if (periodic_y && (ky_Ly != 0.0)) {
        double cos_y = cos(ky_Ly), sin_y = sin(ky_Ly);
        int total = nz_ex * FDn * nx_ex * ncol;
        halo_bloch_phase_z_kernel<<<ceildiv(total, bs), bs>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex,
            2, ncol, cos_y, -sin_y);
        halo_bloch_phase_z_kernel<<<ceildiv(total, bs), bs>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex,
            3, ncol, cos_y, sin_y);
    }
    if (periodic_x && (kx_Lx != 0.0)) {
        double cos_x = cos(kx_Lx), sin_x = sin(kx_Lx);
        int total = nz_ex * ny_ex * FDn * ncol;
        halo_bloch_phase_z_kernel<<<ceildiv(total, bs), bs>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex,
            4, ncol, cos_x, -sin_x);
        halo_bloch_phase_z_kernel<<<ceildiv(total, bs), bs>>>(
            d_x_ex, nx, ny, nz, FDn, nx_ex, ny_ex, nz_ex, nxny_ex, nd_ex,
            5, ncol, cos_x, sin_x);
    }

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
//  COMPLEX LAPLACIAN (orthogonal)
//  Real stencil coefficients applied to complex data
// ============================================================

template<int FDN>
__global__ void laplacian_orth_z_kernel(
    const cuDoubleComplex* __restrict__ x_ex,
    const double* __restrict__ V,
    cuDoubleComplex* __restrict__ y,
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

    cuDoubleComplex center = x_ex[idx];
    cuDoubleComplex val = rcmul(diag_coeff, center);

    #pragma unroll
    for (int p = 1; p <= FDN; ++p) {
        cuDoubleComplex sum_x = cuCadd(x_ex[idx + p], x_ex[idx - p]);
        cuDoubleComplex sum_y = cuCadd(x_ex[idx + p * nx_ex], x_ex[idx - p * nx_ex]);
        cuDoubleComplex sum_z = cuCadd(x_ex[idx + p * nxny_ex], x_ex[idx - p * nxny_ex]);
        val = cuCadd(val, rcmul(a * d_D2x[p], sum_x));
        val = cuCadd(val, rcmul(a * d_D2y[p], sum_y));
        val = cuCadd(val, rcmul(a * d_D2z[p], sum_z));
    }

    if (V) val = cuCadd(val, rcmul(b * V[loc], center));
    y[loc + n * nd] = val;
}

void laplacian_orth_z_gpu(
    const cuDoubleComplex* d_x_ex, const double* d_V, cuDoubleComplex* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c,
    double diag_coeff, int ncol)
{
    int nxny_ex = nx_ex * ny_ex;
    int nd = nx * ny * nz;
    int nd_ex = nxny_ex * (nz + 2 * FDn);

    dim3 block(32, 8);
    dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz * ncol);

    if (FDn == 6) {
        laplacian_orth_z_kernel<6><<<grid, block>>>(
            d_x_ex, d_V, d_y,
            nx, ny, nz, nx_ex, ny_ex, nxny_ex,
            nd, nd_ex, a, b, diag_coeff, ncol);
    } else {
        laplacian_orth_z_kernel<3><<<grid, block>>>(
            d_x_ex, d_V, d_y,
            nx, ny, nz, nx_ex, ny_ex, nxny_ex,
            nd, nd_ex, a, b, diag_coeff, ncol);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
//  COMPLEX LAPLACIAN (non-orthogonal, with mixed derivatives)
// ============================================================

__global__ void laplacian_nonorth_z_kernel(
    const cuDoubleComplex* __restrict__ x_ex,
    const double* __restrict__ V,
    cuDoubleComplex* __restrict__ y,
    int nx, int ny, int nz,
    int FDn, int nx_ex, int ny_ex, int nxny_ex,
    double a, double b, double diag_coeff,
    bool has_xy, bool has_xz, bool has_yz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int idx = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
    int loc = i + j * nx + k * nx * ny;

    cuDoubleComplex center = x_ex[idx];
    cuDoubleComplex val = rcmul(diag_coeff, center);

    for (int p = 1; p <= FDn; ++p) {
        cuDoubleComplex sum_x = cuCadd(x_ex[idx + p], x_ex[idx - p]);
        cuDoubleComplex sum_y = cuCadd(x_ex[idx + p * nx_ex], x_ex[idx - p * nx_ex]);
        cuDoubleComplex sum_z = cuCadd(x_ex[idx + p * nxny_ex], x_ex[idx - p * nxny_ex]);
        val = cuCadd(val, rcmul(a * d_D2x[p], sum_x));
        val = cuCadd(val, rcmul(a * d_D2y[p], sum_y));
        val = cuCadd(val, rcmul(a * d_D2z[p], sum_z));
    }

    if (has_xy) {
        for (int p = 1; p <= FDn; ++p) {
            for (int q = 1; q <= FDn; ++q) {
                cuDoubleComplex mixed = cuCsub(
                    cuCsub(cuCadd(x_ex[idx + p + q * nx_ex],
                                  x_ex[idx - p - q * nx_ex]),
                           x_ex[idx + p - q * nx_ex]),
                    x_ex[idx - p + q * nx_ex]);
                // mixed = (+p,+q) - (+p,-q) - (-p,+q) + (-p,-q)
                // Rewrite correctly:
                cuDoubleComplex pp = x_ex[idx + p + q * nx_ex];
                cuDoubleComplex pm = x_ex[idx + p - q * nx_ex];
                cuDoubleComplex mp = x_ex[idx - p + q * nx_ex];
                cuDoubleComplex mm = x_ex[idx - p - q * nx_ex];
                mixed = cuCadd(cuCsub(cuCsub(pp, pm), mp), mm);
                val = cuCadd(val, rcmul(a * d_D2xy[p] * d_D1y[q], mixed));
            }
        }
    }

    if (has_xz) {
        for (int p = 1; p <= FDn; ++p) {
            for (int q = 1; q <= FDn; ++q) {
                cuDoubleComplex pp = x_ex[idx + p + q * nxny_ex];
                cuDoubleComplex pm = x_ex[idx + p - q * nxny_ex];
                cuDoubleComplex mp = x_ex[idx - p + q * nxny_ex];
                cuDoubleComplex mm = x_ex[idx - p - q * nxny_ex];
                cuDoubleComplex mixed = cuCadd(cuCsub(cuCsub(pp, pm), mp), mm);
                val = cuCadd(val, rcmul(a * d_D2xz[p] * d_D1z[q], mixed));
            }
        }
    }

    if (has_yz) {
        for (int p = 1; p <= FDn; ++p) {
            for (int q = 1; q <= FDn; ++q) {
                cuDoubleComplex pp = x_ex[idx + p * nx_ex + q * nxny_ex];
                cuDoubleComplex pm = x_ex[idx + p * nx_ex - q * nxny_ex];
                cuDoubleComplex mp = x_ex[idx - p * nx_ex + q * nxny_ex];
                cuDoubleComplex mm = x_ex[idx - p * nx_ex - q * nxny_ex];
                cuDoubleComplex mixed = cuCadd(cuCsub(cuCsub(pp, pm), mp), mm);
                val = cuCadd(val, rcmul(a * d_D2yz[p] * d_D1z[q], mixed));
            }
        }
    }

    if (V) val = cuCadd(val, rcmul(b * V[loc], center));
    y[loc] = val;
}

void laplacian_nonorth_z_gpu(
    const cuDoubleComplex* d_x_ex, const double* d_V, cuDoubleComplex* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c,
    double diag_coeff,
    bool has_xy, bool has_xz, bool has_yz,
    int ncol)
{
    int nxny_ex = nx_ex * ny_ex;
    int nd = nx * ny * nz;
    int nd_ex = nxny_ex * (nz + 2 * FDn);

    dim3 block(32, 4, 4);
    dim3 grid(ceildiv(nx, 32), ceildiv(ny, 4), ceildiv(nz, 4));

    for (int n = 0; n < ncol; ++n) {
        laplacian_nonorth_z_kernel<<<grid, block>>>(
            d_x_ex + n * nd_ex, d_V, d_y + n * nd,
            nx, ny, nz, FDn, nx_ex, ny_ex, nxny_ex,
            a, b, diag_coeff,
            has_xy, has_xz, has_yz);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
//  COMPLEX GRADIENT
//  Real FD stencil coefficients applied to complex data
// ============================================================

template<int FDN>
__global__ void gradient_x_z_kernel(
    const cuDoubleComplex* __restrict__ x_ex,
    cuDoubleComplex* __restrict__ y,
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
        cuDoubleComplex val = make_cuDoubleComplex(0.0, 0.0);
        #pragma unroll
        for (int p = 1; p <= FDN; ++p)
            val = cuCadd(val, rcmul(d_D1x[p], cuCsub(x_ex[idx + p], x_ex[idx - p])));
        y[loc + n * nd] = val;
    }
}

template<int FDN>
__global__ void gradient_y_z_kernel(
    const cuDoubleComplex* __restrict__ x_ex,
    cuDoubleComplex* __restrict__ y,
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
        cuDoubleComplex val = make_cuDoubleComplex(0.0, 0.0);
        #pragma unroll
        for (int p = 1; p <= FDN; ++p)
            val = cuCadd(val, rcmul(d_D1y[p], cuCsub(x_ex[idx + p * nx_ex], x_ex[idx - p * nx_ex])));
        y[loc + n * nd] = val;
    }
}

template<int FDN>
__global__ void gradient_z_z_kernel(
    const cuDoubleComplex* __restrict__ x_ex,
    cuDoubleComplex* __restrict__ y,
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
        cuDoubleComplex val = make_cuDoubleComplex(0.0, 0.0);
        #pragma unroll
        for (int p = 1; p <= FDN; ++p)
            val = cuCadd(val, rcmul(d_D1z[p], cuCsub(x_ex[idx + p * nxny_ex], x_ex[idx - p * nxny_ex])));
        y[loc + n * nd] = val;
    }
}

void gradient_z_gpu(
    const cuDoubleComplex* d_x_ex, cuDoubleComplex* d_y,
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
            case 0: gradient_x_z_kernel<6><<<grid, block>>>(d_x_ex, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol); break;
            case 1: gradient_y_z_kernel<6><<<grid, block>>>(d_x_ex, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol); break;
            case 2: gradient_z_z_kernel<6><<<grid, block>>>(d_x_ex, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol); break;
        }
    } else {
        switch (direction) {
            case 0: gradient_x_z_kernel<3><<<grid, block>>>(d_x_ex, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol); break;
            case 1: gradient_y_z_kernel<3><<<grid, block>>>(d_x_ex, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol); break;
            case 2: gradient_z_z_kernel<3><<<grid, block>>>(d_x_ex, d_y, nx, ny, nz, nx_ex, ny_ex, nxny_ex, nd, nd_ex, ncol); break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
//  COMPLEX HAMILTONIAN (local part)
//  H_local*psi = -0.5*Lap*psi + Veff*psi + c*psi
//  Uses complex halo exchange (with Bloch phases) + complex Laplacian
// ============================================================

void hamiltonian_apply_local_z_gpu(
    const cuDoubleComplex* d_psi, const double* d_Veff, cuDoubleComplex* d_Hpsi,
    cuDoubleComplex* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol, double c,
    bool is_orthogonal,
    bool periodic_x, bool periodic_y, bool periodic_z,
    double diag_coeff,
    bool has_xy, bool has_xz, bool has_yz,
    double kx_Lx, double ky_Ly, double kz_Lz)
{
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;

    // Step 1: Complex halo exchange with Bloch phases
    halo_exchange_z_gpu(d_psi, d_x_ex, nx, ny, nz, FDn, ncol,
                        periodic_x, periodic_y, periodic_z,
                        kx_Lx, ky_Ly, kz_Lz);

    // Step 2: Apply (-0.5*Lap + Veff + c*I)
    if (is_orthogonal) {
        laplacian_orth_z_gpu(d_x_ex, d_Veff, d_Hpsi,
                             nx, ny, nz, FDn, nx_ex, ny_ex,
                             -0.5, 1.0, c, diag_coeff, ncol);
    } else {
        laplacian_nonorth_z_gpu(d_x_ex, d_Veff, d_Hpsi,
                                nx, ny, nz, FDn, nx_ex, ny_ex,
                                -0.5, 1.0, c, diag_coeff,
                                has_xy, has_xz, has_yz, ncol);
    }
}

// ============================================================
//  COMPLEX NONLOCAL PROJECTOR
//  Chi is REAL, psi is COMPLEX. Bloch phases are per-atom scalars.
//  bloch_fac[2*iat+0] = cos(theta), bloch_fac[2*iat+1] = sin(theta)
//  where theta = -k . R_image for each influence atom
// ============================================================

// Fused gather + Chi^T * psi for complex wavefunctions
// alpha_z = dV * bloch_fac * Chi^T * psi_z
__global__ void fused_gather_chitpsi_z_kernel(
    const cuDoubleComplex* __restrict__ psi,
    const double* __restrict__ Chi_flat,
    const int* __restrict__ gpos_flat,
    const int* __restrict__ gpos_offsets,
    const int* __restrict__ chi_offsets,
    const int* __restrict__ ndc_arr,
    const int* __restrict__ nproj_arr,
    const int* __restrict__ IP_displ,
    const double* __restrict__ bloch_fac,  // [n_atoms * 2]: (cos, sin) per atom
    cuDoubleComplex* __restrict__ alpha,
    int Nd, int ncol_this,
    int ncol_stride,
    int col_start,
    double dV, int n_atoms,
    int smem_available)  // 0 = no shared memory, >0 = bytes of shared memory allocated
{
    int iat = blockIdx.x;
    if (iat >= n_atoms) return;

    int ndc = ndc_arr[iat];
    int np = nproj_arr[iat];
    if (ndc == 0 || np == 0) return;

    int goff = gpos_offsets[iat];
    int coff = chi_offsets[iat];
    int abase = IP_displ[iat];

    // Bloch phase for this atom: e^{-ik.R_image}
    double bf_re = bloch_fac[2 * iat];
    double bf_im = bloch_fac[2 * iat + 1];
    cuDoubleComplex bloch = make_cuDoubleComplex(bf_re, bf_im);

    extern __shared__ char smem_raw[];
    cuDoubleComplex* psi_sh = reinterpret_cast<cuDoubleComplex*>(smem_raw);

    // Gather psi into shared memory if it fits; otherwise use global memory
    int total_gather = ndc * ncol_this;
    bool use_smem = (smem_available > 0 && total_gather * (int)sizeof(cuDoubleComplex) <= smem_available);
    if (use_smem) {
        for (int idx = threadIdx.x; idx < total_gather; idx += blockDim.x) {
            int ig = idx % ndc;
            int n = idx / ndc;
            psi_sh[idx] = psi[gpos_flat[goff + ig] + (col_start + n) * Nd];
        }
        __syncthreads();
    }

    // Compute alpha = dV * bloch * Chi^T * psi_gathered
    int total_out = np * ncol_this;
    for (int idx = threadIdx.x; idx < total_out; idx += blockDim.x) {
        int jp = idx / ncol_this;
        int n = idx % ncol_this;

        // Chi is REAL, psi is COMPLEX => dot is complex
        cuDoubleComplex dot = make_cuDoubleComplex(0.0, 0.0);
        const double* chi_col = Chi_flat + coff + jp * ndc;
        if (use_smem) {
            const cuDoubleComplex* psi_col = psi_sh + n * ndc;
            for (int ig = 0; ig < ndc; ++ig)
                dot = cuCadd(dot, rcmul(chi_col[ig], psi_col[ig]));
        } else {
            // Global memory fallback for large ndc
            for (int ig = 0; ig < ndc; ++ig)
                dot = cuCadd(dot, rcmul(chi_col[ig], psi[gpos_flat[goff + ig] + (col_start + n) * Nd]));
        }

        // Apply dV and Bloch phase: alpha += bloch * dV * dot
        cuDoubleComplex contrib = cuCmul(bloch, rcmul(dV, dot));
        atomicAddZ(&alpha[(abase + jp) * ncol_stride + (col_start + n)], contrib);
    }
}

// Gamma scaling for complex alpha: alpha[ip*ncol+n] *= Gamma[ip]
// Gamma is REAL
__global__ void gamma_scale_z_kernel(
    cuDoubleComplex* __restrict__ alpha,
    const double* __restrict__ Gamma,
    int total_nproj, int ncol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_nproj * ncol) return;
    int ip = idx / ncol;
    alpha[idx] = rcmul(Gamma[ip], alpha[idx]);
}

// Fused Chi * alpha + scatter-add for complex
// Hpsi += conj(bloch) * Chi * alpha_z
__global__ void fused_chialpha_scatter_z_kernel(
    cuDoubleComplex* __restrict__ Hpsi,
    const double* __restrict__ Chi_flat,
    const int* __restrict__ gpos_flat,
    const int* __restrict__ gpos_offsets,
    const int* __restrict__ chi_offsets,
    const int* __restrict__ ndc_arr,
    const int* __restrict__ nproj_arr,
    const int* __restrict__ IP_displ,
    const double* __restrict__ bloch_fac,  // same (cos,sin) as gather — will conjugate
    const cuDoubleComplex* __restrict__ alpha,
    int Nd, int ncol_this,
    int ncol_stride,
    int col_start,
    int n_atoms)
{
    int iat = blockIdx.x;
    if (iat >= n_atoms) return;

    int ndc = ndc_arr[iat];
    int np = nproj_arr[iat];
    if (ndc == 0 || np == 0) return;

    int goff = gpos_offsets[iat];
    int coff = chi_offsets[iat];
    int abase = IP_displ[iat];

    // Conjugate Bloch phase: e^{+ik.R_image} = conj(e^{-ik.R_image})
    double bf_re = bloch_fac[2 * iat];
    double bf_im = bloch_fac[2 * iat + 1];
    cuDoubleComplex bloch_conj = make_cuDoubleComplex(bf_re, -bf_im);

    // Load alpha tile into shared memory
    extern __shared__ char smem_raw[];
    cuDoubleComplex* alpha_sh = reinterpret_cast<cuDoubleComplex*>(smem_raw);
    int alpha_size = np * ncol_this;
    for (int idx = threadIdx.x; idx < alpha_size; idx += blockDim.x) {
        int jp = idx / ncol_this;
        int n = idx % ncol_this;
        alpha_sh[idx] = alpha[(abase + jp) * ncol_stride + (col_start + n)];
    }
    __syncthreads();

    // Compute Chi * alpha and scatter-add with conjugate Bloch phase
    int total_out = ndc * ncol_this;
    for (int idx = threadIdx.x; idx < total_out; idx += blockDim.x) {
        int ig = idx % ndc;
        int n = idx / ndc;

        // Chi is REAL, alpha is COMPLEX => val is complex
        cuDoubleComplex val = make_cuDoubleComplex(0.0, 0.0);
        const double* chi_row_base = Chi_flat + coff + ig;
        const cuDoubleComplex* alpha_col = alpha_sh + n;
        for (int jp = 0; jp < np; ++jp)
            val = cuCadd(val, rcmul(chi_row_base[jp * ndc], alpha_col[jp * ncol_this]));

        // Apply conjugate Bloch phase
        val = cuCmul(bloch_conj, val);
        atomicAddZ(&Hpsi[gpos_flat[goff + ig] + (col_start + n) * Nd], val);
    }
}

// Host wrapper for complex nonlocal projector
void nonlocal_projector_apply_z_gpu(
    const cuDoubleComplex* d_psi,
    cuDoubleComplex* d_Hpsi,
    const double* d_Chi_flat,
    const int* d_gpos_flat,
    const int* d_gpos_offsets,
    const int* d_chi_offsets,
    const int* d_ndc_arr,
    const int* d_nproj_arr,
    const int* d_IP_displ,
    const double* d_Gamma,
    cuDoubleComplex* d_alpha,
    const double* d_bloch_fac,   // [n_atoms * 2] on device
    int Nd, int ncol, double dV,
    int n_atoms, int total_nproj,
    int max_ndc, int max_nproj)
{
    if (n_atoms == 0 || total_nproj == 0) return;

    // Zero alpha
    CUDA_CHECK(cudaMemset(d_alpha, 0, total_nproj * ncol * sizeof(cuDoubleComplex)));

    int block_size = 256;

    // Query device shared memory limit
    int device;
    cudaGetDevice(&device);
    int max_smem;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    // Step 1: Fused gather + Chi^T * psi -> alpha (with Bloch phases)
    size_t smem_per_col = (size_t)max_ndc * sizeof(cuDoubleComplex);
    int ncol_batch1 = ncol;
    if (smem_per_col * ncol > (size_t)max_smem)
        ncol_batch1 = std::max(1, (int)(max_smem / smem_per_col));

    // Check if max_ndc fits in shared memory for at least 1 column
    bool smem_fits = (smem_per_col <= (size_t)max_smem);

    for (int col_start = 0; col_start < ncol; col_start += ncol_batch1) {
        int cols_this = std::min(ncol_batch1, ncol - col_start);

        if (smem_fits) {
            size_t smem1 = (size_t)max_ndc * cols_this * sizeof(cuDoubleComplex);
            if (smem1 > 48 * 1024)
                cudaFuncSetAttribute(fused_gather_chitpsi_z_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem1);

            fused_gather_chitpsi_z_kernel<<<n_atoms, block_size, smem1>>>(
                d_psi, d_Chi_flat, d_gpos_flat,
                d_gpos_offsets, d_chi_offsets, d_ndc_arr, d_nproj_arr, d_IP_displ,
                d_bloch_fac, d_alpha, Nd, cols_this, ncol, col_start, dV, n_atoms,
                (int)smem1);
        } else {
            // Large ndc: use no-smem path. Pass smem_available=16 (1 element) to avoid
            // NVCC zero-smem issues, but kernel won't use smem (ndc > 1)
            fused_gather_chitpsi_z_kernel<<<n_atoms, block_size, 16>>>(
                d_psi, d_Chi_flat, d_gpos_flat,
                d_gpos_offsets, d_chi_offsets, d_ndc_arr, d_nproj_arr, d_IP_displ,
                d_bloch_fac, d_alpha, Nd, 1, ncol, col_start, dV, n_atoms,
                16);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 2: Gamma scaling (Gamma is real)
    {
        int total = total_nproj * ncol;
        gamma_scale_z_kernel<<<ceildiv(total, block_size), block_size>>>(
            d_alpha, d_Gamma, total_nproj, ncol);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 3: Fused Chi * alpha + scatter -> Hpsi (with conjugate Bloch phase)
    size_t smem_per_col3 = (size_t)max_nproj * sizeof(cuDoubleComplex);
    int ncol_batch3 = ncol;
    if (smem_per_col3 * ncol > (size_t)max_smem)
        ncol_batch3 = std::max(1, (int)(max_smem / smem_per_col3));

    for (int col_start = 0; col_start < ncol; col_start += ncol_batch3) {
        int cols_this = std::min(ncol_batch3, ncol - col_start);
        size_t smem3 = (size_t)max_nproj * cols_this * sizeof(cuDoubleComplex);

        if (smem3 > 48 * 1024) {
            cudaFuncSetAttribute(fused_chialpha_scatter_z_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem3);
        }

        fused_chialpha_scatter_z_kernel<<<n_atoms, block_size, smem3>>>(
            d_Hpsi, d_Chi_flat, d_gpos_flat,
            d_gpos_offsets, d_chi_offsets, d_ndc_arr, d_nproj_arr, d_IP_displ,
            d_bloch_fac, d_alpha, Nd, cols_this, ncol, col_start, n_atoms);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Convenience wrapper: takes host-side metadata, uploads to device
void nonlocal_projector_apply_z_gpu(
    const cuDoubleComplex* d_psi,
    cuDoubleComplex* d_Hpsi,
    const double* d_Chi_flat,
    const int* d_gpos_flat,
    const double* d_Gamma,
    cuDoubleComplex* d_alpha,
    const double* d_bloch_fac,
    int Nd, int ncol, double dV,
    int n_atoms, int total_nproj,
    const int* h_gpos_offsets,
    const int* h_chi_offsets,
    const int* h_ndc_arr,
    const int* h_nproj_arr,
    const int* h_IP_displ,
    int max_ndc, int max_nproj)
{
    if (n_atoms == 0 || total_nproj == 0) return;

    int *d_gpos_off, *d_chi_off, *d_ndc, *d_nproj, *d_ip;
    CUDA_CHECK(cudaMallocAsync(&d_gpos_off, (n_atoms + 1) * sizeof(int), 0));
    CUDA_CHECK(cudaMallocAsync(&d_chi_off, (n_atoms + 1) * sizeof(int), 0));
    CUDA_CHECK(cudaMallocAsync(&d_ndc, n_atoms * sizeof(int), 0));
    CUDA_CHECK(cudaMallocAsync(&d_nproj, n_atoms * sizeof(int), 0));
    CUDA_CHECK(cudaMallocAsync(&d_ip, n_atoms * sizeof(int), 0));

    CUDA_CHECK(cudaMemcpy(d_gpos_off, h_gpos_offsets, (n_atoms + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chi_off, h_chi_offsets, (n_atoms + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ndc, h_ndc_arr, n_atoms * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nproj, h_nproj_arr, n_atoms * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ip, h_IP_displ, n_atoms * sizeof(int), cudaMemcpyHostToDevice));

    nonlocal_projector_apply_z_gpu(
        d_psi, d_Hpsi, d_Chi_flat, d_gpos_flat,
        d_gpos_off, d_chi_off, d_ndc, d_nproj, d_ip,
        d_Gamma, d_alpha, d_bloch_fac,
        Nd, ncol, dV, n_atoms, total_nproj,
        max_ndc, max_nproj);

    cudaFreeAsync(d_gpos_off, 0); cudaFreeAsync(d_chi_off, 0);
    cudaFreeAsync(d_ndc, 0); cudaFreeAsync(d_nproj, 0); cudaFreeAsync(d_ip, 0);
}

} // namespace gpu
} // namespace lynx
#endif // USE_CUDA
