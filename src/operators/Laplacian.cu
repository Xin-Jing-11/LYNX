#ifdef USE_CUDA
#include "core/gpu_common.cuh"

namespace lynx {
namespace gpu {

// ============================================================
// V0 (baseline): Orthogonal Laplacian kernel
// One kernel launch per column, runtime FDn
// ============================================================
__global__ void laplacian_orth_kernel(
    const double* __restrict__ x_ex,
    const double* __restrict__ V,
    double* __restrict__ y,
    int nx, int ny, int nz,
    int FDn, int nx_ex, int ny_ex, int nxny_ex,
    double a, double b, double diag_coeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int idx = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
    int loc = i + j * nx + k * nx * ny;

    double val = diag_coeff * x_ex[idx];

    for (int p = 1; p <= FDn; ++p) {
        val += a * d_D2x[p] * (x_ex[idx + p] + x_ex[idx - p]);
        val += a * d_D2y[p] * (x_ex[idx + p * nx_ex] + x_ex[idx - p * nx_ex]);
        val += a * d_D2z[p] * (x_ex[idx + p * nxny_ex] + x_ex[idx - p * nxny_ex]);
    }

    if (V) val += b * V[loc] * x_ex[idx];
    y[loc] = val;
}

// Host wrapper for baseline orthogonal Laplacian
void laplacian_orth_gpu(
    const double* d_x_ex, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c,
    double diag_coeff,
    int ncol)
{
    int nxny_ex = nx_ex * ny_ex;
    int nd = nx * ny * nz;
    int nd_ex = nxny_ex * (nz + 2 * FDn);

    dim3 block(32, 4, 4);  // 512 threads
    dim3 grid(ceildiv(nx, 32), ceildiv(ny, 4), ceildiv(nz, 4));

    for (int n = 0; n < ncol; ++n) {
        laplacian_orth_kernel<<<grid, block>>>(
            d_x_ex + n * nd_ex, d_V, d_y + n * nd,
            nx, ny, nz, FDn, nx_ex, ny_ex, nxny_ex,
            a, b, diag_coeff);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// V2: Template FDn + multi-column batching (blockIdx.z = column)
// Single kernel launch for all columns
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

void laplacian_orth_v2_gpu(
    const double* d_x_ex, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c,
    double diag_coeff,
    int ncol)
{
    int nxny_ex = nx_ex * ny_ex;
    int nd = nx * ny * nz;
    int nd_ex = nxny_ex * (nz + 2 * FDn);

    // 2D block over (x,y), z mapped to grid
    dim3 block(32, 8);  // 256 threads
    dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz);

    if (FDn == 6) {
        laplacian_orth_kernel_v2<6><<<grid, block>>>(
            d_x_ex, d_V, d_y,
            nx, ny, nz, nx_ex, ny_ex, nxny_ex,
            nd, nd_ex, a, b, diag_coeff, ncol);
    } else {
        laplacian_orth_kernel_v2<3><<<grid, block>>>(
            d_x_ex, d_V, d_y,
            nx, ny, nz, nx_ex, ny_ex, nxny_ex,
            nd, nd_ex, a, b, diag_coeff, ncol);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// V6: One column per block — blockIdx.z = k * ncol + col
// Template FDn for unrolling, LOW register pressure (~40 regs)
// Maximum occupancy by avoiding ncol loop in registers
// ============================================================
template<int FDN>
__global__ __launch_bounds__(256, 6)  // target 6 blocks/SM
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

void laplacian_orth_v6_gpu(
    const double* d_x_ex, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c,
    double diag_coeff,
    int ncol)
{
    int nxny_ex = nx_ex * ny_ex;
    int nd = nx * ny * nz;
    int nd_ex = nxny_ex * (nz + 2 * FDn);

    dim3 block(32, 8);
    dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz * ncol);

    if (FDn == 6) {
        laplacian_orth_kernel_v6<6><<<grid, block>>>(
            d_x_ex, d_V, d_y,
            nx, ny, nz, nx_ex, ny_ex, nxny_ex,
            nd, nd_ex, a, b, diag_coeff, ncol);
    } else {
        laplacian_orth_kernel_v6<3><<<grid, block>>>(
            d_x_ex, d_V, d_y,
            nx, ny, nz, nx_ex, ny_ex, nxny_ex,
            nd, nd_ex, a, b, diag_coeff, ncol);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// V7: Like V6 (1 col/block) + precomputed a*coeff to enable FMA
// Loads a*D2x etc from constant memory (uploaded separately)
// ============================================================

// Additional constant memory for precomputed a*coefficients
__constant__ double d_aD2x[7];
__constant__ double d_aD2y[7];
__constant__ double d_aD2z[7];

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

    // Use precomputed a*coeff — enables val += aD2x[p] * sum → single FMA
    #pragma unroll
    for (int p = 1; p <= FDN; ++p) {
        val += d_aD2x[p] * (x_ex[idx + p] + x_ex[idx - p]);
        val += d_aD2y[p] * (x_ex[idx + p * nx_ex] + x_ex[idx - p * nx_ex]);
        val += d_aD2z[p] * (x_ex[idx + p * nxny_ex] + x_ex[idx - p * nxny_ex]);
    }

    if (V) val += b * V[loc] * center;
    y[loc + n * nd] = val;
}

void upload_precomputed_coefficients(const double* D2x, const double* D2y, const double* D2z,
                                      double a, int FDn)
{
    double aD2x[7], aD2y[7], aD2z[7];
    for (int p = 0; p <= FDn; ++p) {
        aD2x[p] = a * D2x[p];
        aD2y[p] = a * D2y[p];
        aD2z[p] = a * D2z[p];
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_aD2x, aD2x, (FDn + 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_aD2y, aD2y, (FDn + 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_aD2z, aD2z, (FDn + 1) * sizeof(double)));
}

// Cached value of 'a' used for precomputed coefficients.
// Re-uploads only when a changes (avoids redundant cudaMemcpyToSymbol).
static double s_cached_a = 0.0;
static bool s_coeffs_initialized = false;

void laplacian_orth_v7_gpu(
    const double* d_x_ex, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c,
    double diag_coeff,
    int ncol)
{
    // Re-upload precomputed a*D2 coefficients if 'a' changed
    if (!s_coeffs_initialized || a != s_cached_a) {
        // Read current D2 from constant memory and re-upload scaled
        double h_D2x[MAX_FD_COEFF], h_D2y[MAX_FD_COEFF], h_D2z[MAX_FD_COEFF];
        CUDA_CHECK(cudaMemcpyFromSymbol(h_D2x, d_D2x, (FDn + 1) * sizeof(double)));
        CUDA_CHECK(cudaMemcpyFromSymbol(h_D2y, d_D2y, (FDn + 1) * sizeof(double)));
        CUDA_CHECK(cudaMemcpyFromSymbol(h_D2z, d_D2z, (FDn + 1) * sizeof(double)));
        upload_precomputed_coefficients(h_D2x, h_D2y, h_D2z, a, FDn);
        s_cached_a = a;
        s_coeffs_initialized = true;
    }

    int nxny_ex = nx_ex * ny_ex;
    int nd = nx * ny * nz;
    int nd_ex = nxny_ex * (nz + 2 * FDn);

    dim3 block(32, 8);
    dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz * ncol);

    if (FDn == 6) {
        laplacian_orth_kernel_v7<6><<<grid, block>>>(
            d_x_ex, d_V, d_y,
            nx, ny, nz, nx_ex, ny_ex, nxny_ex,
            nd, nd_ex, b, diag_coeff, ncol);
    } else {
        laplacian_orth_kernel_v7<3><<<grid, block>>>(
            d_x_ex, d_V, d_y,
            nx, ny, nz, nx_ex, ny_ex, nxny_ex,
            nd, nd_ex, b, diag_coeff, ncol);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// V8: Multi-column loop (like V2) + precomputed a*coeff (like V7)
// Best of both: FMA fusion + register reuse across columns
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

void laplacian_orth_v8_gpu(
    const double* d_x_ex, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c,
    double diag_coeff,
    int ncol)
{
    int nxny_ex = nx_ex * ny_ex;
    int nd = nx * ny * nz;
    int nd_ex = nxny_ex * (nz + 2 * FDn);

    dim3 block(32, 8);
    dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz);

    if (FDn == 6) {
        laplacian_orth_kernel_v8<6><<<grid, block>>>(
            d_x_ex, d_V, d_y,
            nx, ny, nz, nx_ex, ny_ex, nxny_ex,
            nd, nd_ex, b, diag_coeff, ncol);
    } else {
        laplacian_orth_kernel_v8<3><<<grid, block>>>(
            d_x_ex, d_V, d_y,
            nx, ny, nz, nx_ex, ny_ex, nxny_ex,
            nd, nd_ex, b, diag_coeff, ncol);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// V3: Shared memory 2D tiling + blockIdx.z per z-plane
// Block (BX, BY) loads (BX+2*FDN) x (BY+2*FDN) tile into smem
// x,y stencil from shared memory, z stencil from global memory
// One z-plane per block (full parallelism, no z-sweep bottleneck)
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
    int k  = blockIdx.z;  // one z-plane per block
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

        // Cooperative tile load: (TILE_X_L x TILE_Y_L) from x_ex
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

            // x,y stencil from shared memory
            #pragma unroll
            for (int p = 1; p <= FDN; ++p) {
                val += a * d_D2x[p] * (smem[sj * TILE_X_L + si + p] +
                                        smem[sj * TILE_X_L + si - p]);
                val += a * d_D2y[p] * (smem[(sj + p) * TILE_X_L + si] +
                                        smem[(sj - p) * TILE_X_L + si]);
            }

            // z stencil from global memory
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

void laplacian_orth_v3_gpu(
    const double* d_x_ex, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c,
    double diag_coeff,
    int ncol)
{
    constexpr int BX = 32, BY = 8;
    constexpr int FDN = 6;
    constexpr int smem_bytes = (BX + 2 * FDN) * (BY + 2 * FDN) * sizeof(double);

    int nxny_ex = nx_ex * ny_ex;
    int nd = nx * ny * nz;
    int nd_ex = nxny_ex * (nz + 2 * FDn);

    dim3 block(BX, BY);
    dim3 grid(ceildiv(nx, BX), ceildiv(ny, BY), nz);

    laplacian_orth_kernel_v3<FDN, BX, BY><<<grid, block, smem_bytes>>>(
        d_x_ex, d_V, d_y,
        nx, ny, nz, nx_ex, ny_ex, nxny_ex,
        nd, nd_ex, a, b, diag_coeff, ncol);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// V4: Fused periodic Laplacian — no x_ex, no halo exchange
// Reads directly from compact psi array with periodic wrapping
// Eliminates x_ex allocation and all halo exchange kernels
// ============================================================
__device__ __forceinline__ int wrap(int i, int n) {
    return i >= n ? i - n : (i < 0 ? i + n : i);
}

template<int FDN>
__global__ void laplacian_orth_fused_kernel(
    const double* __restrict__ psi,   // compact (nd) per column
    const double* __restrict__ V,
    double* __restrict__ y,
    int nx, int ny, int nz,
    int nxny,                         // nx * ny
    int nd,                           // nx * ny * nz
    double a, double b, double diag_coeff,
    int ncol)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int loc = i + j * nx + k * nxny;

    double v_val = 0.0;
    if (V) v_val = b * V[loc];

    // Precompute base indices for y and z stencil access
    int base_jk = j * nx + k * nxny;  // for x-stencil: psi[wrap_i + base_jk]
    int base_ik = i + k * nxny;       // for y-stencil: psi[base_ik + wrap_j * nx]
    int base_ij = i + j * nx;         // for z-stencil: psi[base_ij + wrap_k * nxny]

    for (int n = 0; n < ncol; ++n) {
        const double* col = psi + n * nd;
        double center = col[loc];
        double val = diag_coeff * center;

        #pragma unroll
        for (int p = 1; p <= FDN; ++p) {
            // x-direction with periodic wrapping
            int ip = wrap(i + p, nx);
            int im = wrap(i - p, nx);
            val += a * d_D2x[p] * (col[ip + base_jk] + col[im + base_jk]);

            // y-direction with periodic wrapping
            int jp = wrap(j + p, ny);
            int jm = wrap(j - p, ny);
            val += a * d_D2y[p] * (col[base_ik + jp * nx] + col[base_ik + jm * nx]);

            // z-direction with periodic wrapping
            int kp = wrap(k + p, nz);
            int km = wrap(k - p, nz);
            val += a * d_D2z[p] * (col[base_ij + kp * nxny] + col[base_ij + km * nxny]);
        }

        val += v_val * center;
        y[loc + n * nd] = val;
    }
}

void laplacian_orth_fused_gpu(
    const double* d_psi, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    double a, double b, double c,
    double diag_coeff,
    int ncol)
{
    int nxny = nx * ny;
    int nd = nxny * nz;

    dim3 block(32, 8);  // 256 threads
    dim3 grid(ceildiv(nx, 32), ceildiv(ny, 8), nz);

    if (FDn == 6) {
        laplacian_orth_fused_kernel<6><<<grid, block>>>(
            d_psi, d_V, d_y,
            nx, ny, nz, nxny, nd,
            a, b, diag_coeff, ncol);
    } else {
        laplacian_orth_fused_kernel<3><<<grid, block>>>(
            d_psi, d_V, d_y,
            nx, ny, nz, nxny, nd,
            a, b, diag_coeff, ncol);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// V5: Fused periodic + shared memory 2D tiling
// Best of both: no x_ex + shared memory for x,y stencil reuse
// blockIdx.z = k (one z-plane per block, full parallelism)
// ============================================================
template<int FDN, int BX, int BY>
__global__ void laplacian_orth_fused_smem_kernel(
    const double* __restrict__ psi,
    const double* __restrict__ V,
    double* __restrict__ y,
    int nx, int ny, int nz,
    int nxny,
    int nd,
    double a, double b, double diag_coeff,
    int ncol)
{
    constexpr int TILE_X_L = BX + 2 * FDN;
    constexpr int TILE_Y_L = BY + 2 * FDN;
    constexpr int TILE_SIZE = TILE_X_L * TILE_Y_L;
    constexpr int BLOCK_SIZE = BX * BY;

    int bx = blockIdx.x * BX;
    int by = blockIdx.y * BY;
    int k  = blockIdx.z;  // one z-plane per block
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
        loc = i + j * nx + k * nxny;
        if (V) v_val = b * V[loc];
    }

    for (int n = 0; n < ncol; ++n) {
        const double* col = psi + n * nd;

        // Cooperative tile load with periodic wrapping
        for (int idx = tid; idx < TILE_SIZE; idx += BLOCK_SIZE) {
            int ti = idx % TILE_X_L;
            int tj = idx / TILE_X_L;
            int gi = wrap(bx + ti - FDN, nx);
            int gj = wrap(by + tj - FDN, ny);
            smem[tj * TILE_X_L + ti] = col[gi + gj * nx + k * nxny];
        }
        __syncthreads();

        if (active) {
            int si = tx + FDN;
            int sj = ty + FDN;
            double center = smem[sj * TILE_X_L + si];
            double val = diag_coeff * center;

            // x,y stencil from shared memory
            #pragma unroll
            for (int p = 1; p <= FDN; ++p) {
                val += a * d_D2x[p] * (smem[sj * TILE_X_L + si + p] +
                                        smem[sj * TILE_X_L + si - p]);
                val += a * d_D2y[p] * (smem[(sj + p) * TILE_X_L + si] +
                                        smem[(sj - p) * TILE_X_L + si]);
            }

            // z-stencil from global memory with periodic wrapping
            int base_ij = i + j * nx;
            #pragma unroll
            for (int p = 1; p <= FDN; ++p) {
                int kp = wrap(k + p, nz);
                int km = wrap(k - p, nz);
                val += a * d_D2z[p] * (col[base_ij + kp * nxny] +
                                        col[base_ij + km * nxny]);
            }

            val += v_val * center;
            y[loc + n * nd] = val;
        }
        __syncthreads();
    }
}

void laplacian_orth_v5_gpu(
    const double* d_psi, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    double a, double b, double c,
    double diag_coeff,
    int ncol)
{
    constexpr int BX = 32, BY = 8;
    constexpr int FDN = 6;
    constexpr int smem_size = (BX + 2 * FDN) * (BY + 2 * FDN) * sizeof(double);

    dim3 block(BX, BY);  // 256 threads
    dim3 grid(ceildiv(nx, BX), ceildiv(ny, BY), nz);

    laplacian_orth_fused_smem_kernel<FDN, BX, BY><<<grid, block, smem_size>>>(
        d_psi, d_V, d_y,
        nx, ny, nz, nx * ny, nx * ny * nz,
        a, b, diag_coeff, ncol);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Non-orthogonal Laplacian kernel (with mixed derivatives)
// ============================================================
__global__ void laplacian_nonorth_kernel(
    const double* __restrict__ x_ex,
    const double* __restrict__ V,
    double* __restrict__ y,
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

    double val = diag_coeff * x_ex[idx];
    for (int p = 1; p <= FDn; ++p) {
        val += a * d_D2x[p] * (x_ex[idx + p] + x_ex[idx - p]);
        val += a * d_D2y[p] * (x_ex[idx + p * nx_ex] + x_ex[idx - p * nx_ex]);
        val += a * d_D2z[p] * (x_ex[idx + p * nxny_ex] + x_ex[idx - p * nxny_ex]);
    }

    if (has_xy) {
        for (int p = 1; p <= FDn; ++p) {
            for (int q = 1; q <= FDn; ++q) {
                double mixed = x_ex[idx + p + q * nx_ex]
                             - x_ex[idx + p - q * nx_ex]
                             - x_ex[idx - p + q * nx_ex]
                             + x_ex[idx - p - q * nx_ex];
                val += a * d_D2xy[p] * d_D1y[q] * mixed;
            }
        }
    }

    if (has_xz) {
        for (int p = 1; p <= FDn; ++p) {
            for (int q = 1; q <= FDn; ++q) {
                double mixed = x_ex[idx + p + q * nxny_ex]
                             - x_ex[idx + p - q * nxny_ex]
                             - x_ex[idx - p + q * nxny_ex]
                             + x_ex[idx - p - q * nxny_ex];
                val += a * d_D2xz[p] * d_D1z[q] * mixed;
            }
        }
    }

    if (has_yz) {
        for (int p = 1; p <= FDn; ++p) {
            for (int q = 1; q <= FDn; ++q) {
                double mixed = x_ex[idx + p * nx_ex + q * nxny_ex]
                             - x_ex[idx + p * nx_ex - q * nxny_ex]
                             - x_ex[idx - p * nx_ex + q * nxny_ex]
                             + x_ex[idx - p * nx_ex - q * nxny_ex];
                val += a * d_D2yz[p] * d_D1z[q] * mixed;
            }
        }
    }

    if (V) val += b * V[loc] * x_ex[idx];
    y[loc] = val;
}

void laplacian_nonorth_gpu(
    const double* d_x_ex, const double* d_V, double* d_y,
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
        laplacian_nonorth_kernel<<<grid, block>>>(
            d_x_ex + n * nd_ex, d_V, d_y + n * nd,
            nx, ny, nz, FDn, nx_ex, ny_ex, nxny_ex,
            a, b, diag_coeff,
            has_xy, has_xz, has_yz);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
} // namespace lynx
#endif // USE_CUDA
