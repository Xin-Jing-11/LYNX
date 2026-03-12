#ifdef USE_CUDA
#include "core/gpu_common.cuh"

namespace sparc {
namespace gpu {

// ============================================================
// Fused gather + Chi^T × psi kernel
//
// One block per atom. Threads cooperate to:
//   1. Gather psi values at scattered grid positions into shared memory
//   2. Compute alpha_atom(nproj, ncol) = dV * Chi^T(nproj, ndc) × psi_gathered(ndc, ncol)
//
// Alpha layout: ROW-MAJOR alpha[(IP_displ[iat] + jp) * ncol + n]
//
// Block size: 256 threads
// Shared memory: ndc * ncol doubles (for psi_gathered tile)
//   Max: ~500 * 30 * 8 = 120 KB — fits in L1/smem on modern GPUs
// ============================================================
__global__ void fused_gather_chitpsi_kernel(
    const double* __restrict__ psi,        // offset to col_start already applied
    const double* __restrict__ Chi_flat,
    const int* __restrict__ gpos_flat,
    const int* __restrict__ gpos_offsets,
    const int* __restrict__ chi_offsets,
    const int* __restrict__ ndc_arr,
    const int* __restrict__ nproj_arr,
    const int* __restrict__ IP_displ,
    double* __restrict__ alpha,            // full alpha array
    int Nd, int ncol_this,                 // columns in this batch
    int ncol_stride,                       // full alpha row stride
    int col_start,                         // first column index
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

    extern __shared__ double psi_sh[];

    // Gather psi into shared memory: psi_sh[ig + n * ndc]
    int total_gather = ndc * ncol_this;
    for (int idx = threadIdx.x; idx < total_gather; idx += blockDim.x) {
        int ig = idx % ndc;
        int n = idx / ndc;
        psi_sh[idx] = psi[gpos_flat[goff + ig] + (col_start + n) * Nd];
    }
    __syncthreads();

    // Compute alpha = dV * Chi^T * psi_gathered
    int total_out = np * ncol_this;
    for (int idx = threadIdx.x; idx < total_out; idx += blockDim.x) {
        int jp = idx / ncol_this;
        int n = idx % ncol_this;

        double dot = 0.0;
        const double* chi_col = Chi_flat + coff + jp * ndc;
        const double* psi_col = psi_sh + n * ndc;
        for (int ig = 0; ig < ndc; ++ig)
            dot += chi_col[ig] * psi_col[ig];

        atomicAdd(&alpha[(abase + jp) * ncol_stride + (col_start + n)], dot * dV);
    }
}

// ============================================================
// Fused Chi × alpha + scatter-add kernel
//
// One block per atom. Threads cooperate to:
//   1. Load alpha_atom into shared memory
//   2. Compute buf(ndc, ncol) = Chi(ndc, nproj) × alpha_atom(nproj, ncol)
//   3. Scatter-add buf to Hpsi at grid positions
//
// Shared memory: nproj * ncol doubles (for alpha tile)
//   Max: ~9 * 30 * 8 = 2.2 KB — tiny
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

    // Load alpha_atom tile into shared memory: alpha_sh[jp * ncol_this + n]
    extern __shared__ double alpha_sh[];
    int alpha_size = np * ncol_this;
    for (int idx = threadIdx.x; idx < alpha_size; idx += blockDim.x) {
        int jp = idx / ncol_this;
        int n = idx % ncol_this;
        alpha_sh[idx] = alpha[(abase + jp) * ncol_stride + (col_start + n)];
    }
    __syncthreads();

    // Compute Chi * alpha and scatter-add
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
// Host wrapper: 3 kernel launches total (was 200+ cuBLAS calls)
//
// All metadata arrays (offsets, ndc, nproj, IP_displ) are on DEVICE.
// Host only needs n_atoms, total_nproj, max_ndc for launch config.
// ============================================================
void nonlocal_projector_apply_gpu(
    const double* d_psi,
    double* d_Hpsi,
    const double* d_Chi_flat,
    const int* d_gpos_flat,
    const int* d_gpos_offsets,   // [n_atoms+1] on device
    const int* d_chi_offsets,    // [n_atoms+1] on device
    const int* d_ndc_arr,        // [n_atoms] on device
    const int* d_nproj_arr,      // [n_atoms] on device
    const int* d_IP_displ,       // [n_atoms] on device
    const double* d_Gamma,       // [total_nproj] on device
    double* d_alpha,             // workspace [total_nproj * ncol] on device
    int Nd, int ncol, double dV,
    int n_atoms, int total_nproj,
    int max_ndc, int max_nproj)
{
    if (n_atoms == 0 || total_nproj == 0) return;

    // Zero alpha
    CUDA_CHECK(cudaMemset(d_alpha, 0, total_nproj * ncol * sizeof(double)));

    int block_size = 256;

    // Query device shared memory limit
    int device;
    cudaGetDevice(&device);
    int max_smem;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    // Step 1: Fused gather + Chi^T * psi → alpha
    // Shared memory: max_ndc * ncol_batch doubles
    // Tile over columns if shared memory would exceed limit
    size_t smem_per_col = (size_t)max_ndc * sizeof(double);
    int ncol_batch1 = ncol;
    if (smem_per_col * ncol > (size_t)max_smem)
        ncol_batch1 = std::max(1, (int)(max_smem / smem_per_col));

    for (int col_start = 0; col_start < ncol; col_start += ncol_batch1) {
        int cols_this = std::min(ncol_batch1, ncol - col_start);
        size_t smem1 = (size_t)max_ndc * cols_this * sizeof(double);

        if (smem1 > 48 * 1024) {
            cudaFuncSetAttribute(fused_gather_chitpsi_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem1);
        }

        fused_gather_chitpsi_kernel<<<n_atoms, block_size, smem1>>>(
            d_psi, d_Chi_flat, d_gpos_flat,
            d_gpos_offsets, d_chi_offsets, d_ndc_arr, d_nproj_arr, d_IP_displ,
            d_alpha, Nd, cols_this, ncol, col_start, dV, n_atoms);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 2: Gamma scaling
    {
        int total = total_nproj * ncol;
        gamma_scale_kernel<<<ceildiv(total, block_size), block_size>>>(
            d_alpha, d_Gamma, total_nproj, ncol);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 3: Fused Chi * alpha + scatter → Hpsi
    // Shared memory: max_nproj * ncol_batch doubles (typically tiny, ~9*30*8=2KB)
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

    // Upload small metadata to device
    int *d_gpos_off, *d_chi_off, *d_ndc, *d_nproj, *d_ip;
    CUDA_CHECK(cudaMalloc(&d_gpos_off, (n_atoms + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_chi_off, (n_atoms + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ndc, n_atoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nproj, n_atoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ip, n_atoms * sizeof(int)));

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

    cudaFree(d_gpos_off); cudaFree(d_chi_off);
    cudaFree(d_ndc); cudaFree(d_nproj); cudaFree(d_ip);
}

} // namespace gpu
} // namespace sparc
#endif // USE_CUDA
