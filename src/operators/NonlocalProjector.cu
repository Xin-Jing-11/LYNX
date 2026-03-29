#ifdef USE_CUDA
#include "core/gpu_common.cuh"

namespace lynx {
namespace gpu {

// ============================================================
// Device helpers: warp and block reduction
// ============================================================
__device__ __forceinline__ double warpReduceSum_d(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-level sum reduction. Returns correct sum in thread 0 only.
// smem must have at least (blockDim.x / 32) doubles.
__device__ double blockReduceSum_d(double val, double* smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;

    val = warpReduceSum_d(val);
    if (lane == 0) smem[warp] = val;
    __syncthreads();

    if (warp == 0) {
        val = (lane < nwarps) ? smem[lane] : 0.0;
        val = warpReduceSum_d(val);
    }
    __syncthreads();
    return val;
}

// ============================================================
// Tiled gather + Chi^T × psi kernel
//
// One block per atom. Processes ndc in fixed-size tiles.
// Shared memory: (NL_TILE + nwarps) doubles ≈ 2 KB, O(1) w.r.t. system size.
//
// Algorithm per atom:
//   For each wavefunction column n:
//     For each tile of ndc grid points:
//       1. Cooperatively gather scattered psi values into shared tile
//       2. Each thread accumulates chi[jp,ig]*psi[ig] for its ig stride
//     Block-reduce partial sums per projector, thread 0 writes to alpha
// ============================================================
static constexpr int NL_TILE = 256;
static constexpr int NL_MAX_NP = 32;

__global__ void fused_gather_chitpsi_kernel(
    const double* __restrict__ psi,        // (Nd, ncol_stride)
    const double* __restrict__ Chi_flat,
    const int* __restrict__ gpos_flat,
    const int* __restrict__ gpos_offsets,
    const int* __restrict__ chi_offsets,
    const int* __restrict__ ndc_arr,
    const int* __restrict__ nproj_arr,
    const int* __restrict__ IP_displ,
    double* __restrict__ alpha,            // (total_nproj, ncol_stride)
    int Nd, int ncol_this,
    int ncol_stride,
    int col_start,
    double dV, int n_atoms)
{
    int iat = blockIdx.x;
    if (iat >= n_atoms) return;

    int ndc = ndc_arr[iat];
    int np = nproj_arr[iat];
    if (ndc == 0 || np == 0) return;

    int goff = gpos_offsets[iat];
    int coff = chi_offsets[iat];
    int abase = IP_displ[iat];

    extern __shared__ double smem[];
    double* psi_tile = smem;
    double* reduce_buf = smem + NL_TILE;

    for (int n = 0; n < ncol_this; n++) {
        // Per-thread accumulators for each projector
        double dots[NL_MAX_NP];
        #pragma unroll 4
        for (int jp = 0; jp < NL_MAX_NP; jp++)
            dots[jp] = 0.0;

        // Tile over ndc
        for (int tile = 0; tile < ndc; tile += NL_TILE) {
            int tile_len = min(NL_TILE, ndc - tile);

            // Cooperative gather: scattered psi → shared tile
            for (int i = threadIdx.x; i < tile_len; i += blockDim.x)
                psi_tile[i] = psi[gpos_flat[goff + tile + i] + (col_start + n) * Nd];
            __syncthreads();

            // Each thread accumulates for its stride of ig values
            for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
                double pv = psi_tile[i];
                const double* chi_base = Chi_flat + coff + tile + i;
                for (int jp = 0; jp < np; jp++)
                    dots[jp] += chi_base[jp * ndc] * pv;
            }
            __syncthreads();
        }

        // Block-reduce each projector and write to alpha
        for (int jp = 0; jp < np; jp++) {
            double val = blockReduceSum_d(dots[jp], reduce_buf);
            if (threadIdx.x == 0)
                atomicAdd(&alpha[(abase + jp) * ncol_stride + (col_start + n)], val * dV);
        }
    }
}

// ============================================================
// Fused Chi × alpha + scatter-add kernel (unchanged)
//
// One block per atom. Shared memory: nproj * ncol_this doubles (tiny, ~2 KB)
// ============================================================
__global__ void fused_chialpha_scatter_kernel(
    double* __restrict__ Hpsi,
    const double* __restrict__ Chi_flat,
    const int* __restrict__ gpos_flat,
    const int* __restrict__ gpos_offsets,
    const int* __restrict__ chi_offsets,
    const int* __restrict__ ndc_arr,
    const int* __restrict__ nproj_arr,
    const int* __restrict__ IP_displ,
    const double* __restrict__ alpha,
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

    extern __shared__ double alpha_sh[];
    int alpha_size = np * ncol_this;
    for (int idx = threadIdx.x; idx < alpha_size; idx += blockDim.x) {
        int jp = idx / ncol_this;
        int n = idx % ncol_this;
        alpha_sh[idx] = alpha[(abase + jp) * ncol_stride + (col_start + n)];
    }
    __syncthreads();

    int total_out = ndc * ncol_this;
    for (int idx = threadIdx.x; idx < total_out; idx += blockDim.x) {
        int ig = idx % ndc;
        int n = idx / ndc;

        double val = 0.0;
        const double* chi_row_base = Chi_flat + coff + ig;
        const double* alpha_col = alpha_sh + n;
        for (int jp = 0; jp < np; ++jp)
            val += chi_row_base[jp * ndc] * alpha_col[jp * ncol_this];

        atomicAdd(&Hpsi[gpos_flat[goff + ig] + (col_start + n) * Nd], val);
    }
}

// ============================================================
// Gamma scaling: alpha[ip * ncol + n] *= Gamma[ip]
// ============================================================
__global__ void gamma_scale_kernel(
    double* __restrict__ alpha,
    const double* __restrict__ Gamma,
    int total_nproj, int ncol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_nproj * ncol) return;
    int ip = idx / ncol;
    alpha[idx] *= Gamma[ip];
}

// ============================================================
// Host wrapper: 3 kernel launches (gather, gamma, scatter)
//
// Shared memory is fixed at ~2 KB regardless of system size.
// No device smem query, no branching, no fallback paths.
// ============================================================
void nonlocal_projector_apply_gpu(
    const double* d_psi,
    double* d_Hpsi,
    const double* d_Chi_flat,
    const int* d_gpos_flat,
    const int* d_gpos_offsets,
    const int* d_chi_offsets,
    const int* d_ndc_arr,
    const int* d_nproj_arr,
    const int* d_IP_displ,
    const double* d_Gamma,
    double* d_alpha,
    int Nd, int ncol, double dV,
    int n_atoms, int total_nproj,
    int max_ndc, int max_nproj)
{
    if (n_atoms == 0 || total_nproj == 0) return;

    CUDA_CHECK(cudaMemset(d_alpha, 0, total_nproj * ncol * sizeof(double)));

    int block_size = 256;

    // Step 1: Tiled gather + Chi^T * psi → alpha
    // Fixed shared memory: tile buffer + warp reduction buffer
    size_t smem1 = (NL_TILE + block_size / 32) * sizeof(double);
    fused_gather_chitpsi_kernel<<<n_atoms, block_size, smem1>>>(
        d_psi, d_Chi_flat, d_gpos_flat,
        d_gpos_offsets, d_chi_offsets, d_ndc_arr, d_nproj_arr, d_IP_displ,
        d_alpha, Nd, ncol, ncol, 0, dV, n_atoms);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Gamma scaling
    {
        int total = total_nproj * ncol;
        gamma_scale_kernel<<<ceildiv(total, block_size), block_size>>>(
            d_alpha, d_Gamma, total_nproj, ncol);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 3: Chi * alpha + scatter → Hpsi
    // Shared memory: max_nproj * ncol_batch doubles (typically tiny, ~2 KB)
    int device;
    cudaGetDevice(&device);
    int max_smem;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    size_t smem_per_col3 = (size_t)max_nproj * sizeof(double);
    int ncol_batch3 = ncol;
    if (smem_per_col3 * ncol > (size_t)max_smem)
        ncol_batch3 = std::max(1, (int)(max_smem / smem_per_col3));

    for (int col_start = 0; col_start < ncol; col_start += ncol_batch3) {
        int cols_this = std::min(ncol_batch3, ncol - col_start);
        size_t smem3 = (size_t)max_nproj * cols_this * sizeof(double);

        if (smem3 > 48 * 1024) {
            cudaFuncSetAttribute(fused_chialpha_scatter_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem3);
        }

        fused_chialpha_scatter_kernel<<<n_atoms, block_size, smem3>>>(
            d_Hpsi, d_Chi_flat, d_gpos_flat,
            d_gpos_offsets, d_chi_offsets, d_ndc_arr, d_nproj_arr, d_IP_displ,
            d_alpha, Nd, cols_this, ncol, col_start, n_atoms);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================
// Convenience wrapper: takes host-side metadata, uploads to device
// ============================================================
void nonlocal_projector_apply_gpu(
    const double* d_psi,
    double* d_Hpsi,
    const double* d_Chi_flat,
    const int* d_gpos_flat,
    const double* d_Gamma,
    double* d_alpha,
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

    nonlocal_projector_apply_gpu(
        d_psi, d_Hpsi, d_Chi_flat, d_gpos_flat,
        d_gpos_off, d_chi_off, d_ndc, d_nproj, d_ip,
        d_Gamma, d_alpha,
        Nd, ncol, dV, n_atoms, total_nproj,
        max_ndc, max_nproj);

    cudaFreeAsync(d_gpos_off, 0); cudaFreeAsync(d_chi_off, 0);
    cudaFreeAsync(d_ndc, 0); cudaFreeAsync(d_nproj, 0); cudaFreeAsync(d_ip, 0);
}

} // namespace gpu
} // namespace lynx
#endif // USE_CUDA
