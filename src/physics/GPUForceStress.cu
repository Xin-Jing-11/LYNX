#ifdef USE_CUDA

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "core/GPUContext.cuh"
#include "core/gpu_common.cuh"

namespace lynx {
namespace gpu {

// ============================================================
// Forward declarations for GPU functions defined elsewhere
// ============================================================

void halo_exchange_gpu(
    const double* d_x, double* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol,
    bool periodic_x, bool periodic_y, bool periodic_z);

void gradient_gpu(
    const double* d_x_ex, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    int direction, int ncol);

// fused_gather_chitpsi_kernel from NonlocalProjector.cu
// With CUDA_SEPARABLE_COMPILATION ON, __global__ functions have external linkage.
__global__ void fused_gather_chitpsi_kernel(
    const double* __restrict__ psi,
    const double* __restrict__ Chi_flat,
    const int* __restrict__ gpos_flat,
    const int* __restrict__ gpos_offsets,
    const int* __restrict__ chi_offsets,
    const int* __restrict__ ndc_arr,
    const int* __restrict__ nproj_arr,
    const int* __restrict__ IP_displ,
    double* __restrict__ alpha,
    int Nd, int ncol_this,
    int ncol_stride,
    int col_start,
    double dV, int n_atoms);

// ============================================================
// GPU Kernels for Force & Stress
// ============================================================

// ------------------------------------------------------------
// Nonlocal force reduction kernel
// One thread per (atom, dim). Loops over projectors and bands.
// f_nloc[atom*3+dim] = -spn_fac_wk * sum_n(g_n * sum_jp(Gamma[off+jp] * alpha[..] * beta[..]))
//
// alpha layout: alpha[(abase+jp) * Nband + n]
// beta  layout: beta[(abase+jp) * Nband + n]
// IP_displ_phys: [n_phys_atoms+1] CSR prefix sum of projectors per physical atom
// ------------------------------------------------------------
__global__ void nonlocal_force_reduce_kernel(
    const double* __restrict__ d_alpha,
    const double* __restrict__ d_beta,
    const double* __restrict__ d_occ,
    const double* __restrict__ d_Gamma,
    const int* __restrict__ d_IP_displ_phys,
    int n_phys_atoms,
    int Nband,
    double spn_fac_wk,
    double* __restrict__ d_force,   // [n_phys_atoms * 3], only writes [atom*3+dim]
    int dim)
{
    int ia = blockIdx.x * blockDim.x + threadIdx.x;
    if (ia >= n_phys_atoms) return;

    int off = d_IP_displ_phys[ia];
    int nproj = d_IP_displ_phys[ia + 1] - off;
    if (nproj == 0) return;

    double fJ = 0.0;
    for (int n = 0; n < Nband; ++n) {
        double g_n = d_occ[n];
        if (fabs(g_n) < 1e-15) continue;
        double band_sum = 0.0;
        for (int jp = 0; jp < nproj; ++jp) {
            int idx = (off + jp) * Nband + n;
            band_sum += d_Gamma[off + jp] * d_alpha[idx] * d_beta[idx];
        }
        fJ += g_n * band_sum;
    }

    d_force[ia * 3 + dim] -= spn_fac_wk * fJ;
}

// ------------------------------------------------------------
// Kinetic stress kernel
// One block per Voigt pair (6 total). Reduces over all bands and grid points.
// sk[voigt] = -occfac * dV * sum_n(g_n * dot(dpsi_a(:,n), dpsi_b(:,n)))
//
// Uses a two-level reduction: threads reduce over grid points for one band,
// then accumulate across bands.
// ------------------------------------------------------------
__global__ void kinetic_stress_kernel(
    const double* __restrict__ d_Dpsi_a,   // (Nd, Nband) gradient in direction a
    const double* __restrict__ d_Dpsi_b,   // (Nd, Nband) gradient in direction b
    const double* __restrict__ d_occ,      // (Nband) occupations
    double* __restrict__ d_sk_out,         // [1] partial result for this Voigt pair
    int Nd, int Nband,
    double neg_occfac_dV)                  // = -occfac * dV
{
    // blockDim.x threads cooperate per band, blockIdx.x selects band
    int band = blockIdx.x;
    if (band >= Nband) return;

    double g_n = d_occ[band];

    extern __shared__ double sdata[];

    // Each thread sums over a subset of grid points
    double local_dot = 0.0;
    const double* pa = d_Dpsi_a + band * Nd;
    const double* pb = d_Dpsi_b + band * Nd;
    for (int i = threadIdx.x; i < Nd; i += blockDim.x) {
        local_dot += pa[i] * pb[i];
    }

    sdata[threadIdx.x] = local_dot;
    __syncthreads();

    // Shared memory reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    // Thread 0 accumulates into output
    if (threadIdx.x == 0) {
        atomicAdd(d_sk_out, neg_occfac_dV * g_n * sdata[0]);
    }
}

// ------------------------------------------------------------
// Nonlocal energy kernel (single block, serial reduction)
// energy_nl_raw = sum_n(g_n * sum_jp(Gamma[jp] * alpha[jp,n]^2))
// (caller multiplies by wk; final result scaled by occfac on host)
// ------------------------------------------------------------
__global__ void nonlocal_energy_kernel(
    const double* __restrict__ d_alpha,
    const double* __restrict__ d_occ,
    const double* __restrict__ d_Gamma,
    const int* __restrict__ d_IP_displ_phys,
    int n_phys_atoms,
    int Nband,
    int total_nproj,
    double* __restrict__ d_energy_out)   // [1]
{
    double enl = 0.0;
    for (int n = 0; n < Nband; ++n) {
        double g_n = d_occ[n];
        if (fabs(g_n) < 1e-15) continue;
        for (int ia = 0; ia < n_phys_atoms; ++ia) {
            int off = d_IP_displ_phys[ia];
            int nproj = d_IP_displ_phys[ia + 1] - off;
            for (int jp = 0; jp < nproj; ++jp) {
                int idx = (off + jp) * Nband + n;
                double a = d_alpha[idx];
                enl += g_n * d_Gamma[off + jp] * a * a;
            }
        }
    }
    *d_energy_out = enl;
}

// ------------------------------------------------------------
// Weighted gather kernel for nonlocal stress:
// beta_stress[(abase+jp)*Nband + n] += dV * sum_ig(chi(ig,jp) * xR[ig] * Dpsi[gpos[ig] + n*Nd])
//
// This is like fused_gather_chitpsi_kernel but with position weighting.
// One block per influence atom. xR depends on (dim2, grid coordinates, atom position).
// For orthogonal cells: xR = (grid_coord - atom_coord) in Cartesian.
// ------------------------------------------------------------
__global__ void weighted_gather_chitpsi_kernel(
    const double* __restrict__ Dpsi,       // (Nd, Nband) gradient in direction 'dim'
    const double* __restrict__ Chi_flat,
    const int* __restrict__ gpos_flat,
    const int* __restrict__ gpos_offsets,
    const int* __restrict__ chi_offsets,
    const int* __restrict__ ndc_arr,
    const int* __restrict__ nproj_arr,
    const int* __restrict__ IP_displ,
    const double* __restrict__ atom_pos,   // [n_influence * 3] atom position (x,y,z)
    double* __restrict__ beta,             // [total_nproj * Nband]
    int Nd, int ncol_this, int ncol_stride, int col_start,
    double dV, int n_atoms,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    int xs, int ys, int zs,
    int dim2)                              // which coordinate (0=x,1=y,2=z)
{
    int iat = blockIdx.x;
    if (iat >= n_atoms) return;

    int ndc = ndc_arr[iat];
    int np = nproj_arr[iat];
    if (ndc == 0 || np == 0) return;

    int goff = gpos_offsets[iat];
    int coff = chi_offsets[iat];
    int abase = IP_displ[iat];

    double ap_x = atom_pos[iat * 3 + 0];
    double ap_y = atom_pos[iat * 3 + 1];
    double ap_z = atom_pos[iat * 3 + 2];

    // Shared memory: psi_sh[ig + n * ndc] AND xR_sh[ig]
    extern __shared__ double shared_mem[];
    double* psi_sh = shared_mem;
    double* xR_sh = shared_mem + ndc * ncol_this;

    // Gather Dpsi and compute xR
    int total_gather = ndc * ncol_this;
    for (int idx = threadIdx.x; idx < total_gather; idx += blockDim.x) {
        int ig = idx % ndc;
        int n = idx / ndc;
        psi_sh[idx] = Dpsi[gpos_flat[goff + ig] + (col_start + n) * Nd];
    }
    // Compute xR per grid point
    for (int ig = threadIdx.x; ig < ndc; ig += blockDim.x) {
        int flat = gpos_flat[goff + ig];
        int li = flat % nx;
        int lj = (flat / nx) % ny;
        // int lk = flat / (nx * ny);  // unused unless dim2==2
        double r;
        if (dim2 == 0) {
            r = (li + xs) * dx - ap_x;
        } else if (dim2 == 1) {
            r = (lj + ys) * dy - ap_y;
        } else {
            int lk = flat / (nx * ny);
            r = (lk + zs) * dz - ap_z;
        }
        xR_sh[ig] = r;
    }
    __syncthreads();

    // Compute weighted inner product: beta[abase+jp, n] += dV * sum_ig(chi * xR * Dpsi)
    int total_out = np * ncol_this;
    for (int idx = threadIdx.x; idx < total_out; idx += blockDim.x) {
        int jp = idx / ncol_this;
        int n = idx % ncol_this;

        double dot = 0.0;
        const double* chi_col = Chi_flat + coff + jp * ndc;
        const double* psi_col = psi_sh + n * ndc;
        for (int ig = 0; ig < ndc; ++ig)
            dot += chi_col[ig] * xR_sh[ig] * psi_col[ig];

        atomicAdd(&beta[(abase + jp) * ncol_stride + (col_start + n)], dot * dV);
    }
}

// ------------------------------------------------------------
// Nonlocal stress reduction kernel (same structure as force reduce)
// snl[voigt] -= spn_fac * sum_n(wk * g_n * sum_jp(Gamma * alpha * beta_stress))
// Run once per Voigt pair with appropriate beta_stress.
// ------------------------------------------------------------
__global__ void nonlocal_stress_reduce_kernel(
    const double* __restrict__ d_alpha,
    const double* __restrict__ d_beta_stress,
    const double* __restrict__ d_occ,
    const double* __restrict__ d_Gamma,
    const int* __restrict__ d_IP_displ_phys,
    int n_phys_atoms,
    int Nband,
    double wk,
    double* __restrict__ d_snl_out)   // [1] atomicAdd into this
{
    // Single block, thread 0 does all work (small data)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    double sum = 0.0;
    for (int n = 0; n < Nband; ++n) {
        double g_n = d_occ[n];
        if (fabs(g_n) < 1e-15) continue;
        for (int ia = 0; ia < n_phys_atoms; ++ia) {
            int off = d_IP_displ_phys[ia];
            int nproj = d_IP_displ_phys[ia + 1] - off;
            for (int jp = 0; jp < nproj; ++jp) {
                int idx = (off + jp) * Nband + n;
                sum += g_n * d_Gamma[off + jp] * d_alpha[idx] * d_beta_stress[idx];
            }
        }
    }
    atomicAdd(d_snl_out, -wk * sum);  // negative: snl -= Gamma * alpha * beta * wk * g_n
}

// ============================================================
// Host function: compute_force_stress_gpu
// ============================================================
void compute_force_stress_gpu(
    const double* d_psi,       // (Nd, Nband) wavefunctions on device
    const double* d_occ,       // (Nband) occupations on device
    // Nonlocal data (all on device, from GPUNonlocalData)
    const double* d_Chi_flat,
    const int*    d_gpos_flat,
    const int*    d_gpos_offsets,
    const int*    d_chi_offsets,
    const int*    d_ndc_arr,
    const int*    d_nproj_arr,
    const int*    d_IP_displ,     // per influence atom
    const double* d_Gamma,
    int n_influence,
    int total_nproj,
    int max_ndc,
    int max_nproj,
    int n_phys_atoms,
    // Physical atom IP_displ (host, will upload)
    const int* h_IP_displ_phys,   // [n_phys_atoms+1]
    // Atom position data for nonlocal stress (host)
    const double* h_atom_pos,     // [n_influence * 3]
    // Grid parameters
    int nx, int ny, int nz, int FDn, int Nd, int Nband,
    double dV, double dx, double dy, double dz,
    int xs, int ys, int zs,
    double occfac,
    // Output (host)
    double* h_f_nloc,       // [3 * n_phys_atoms]
    double* h_stress_k,     // [6] kinetic stress (Voigt: xx,xy,xz,yy,yz,zz)
    double* h_stress_nl,    // [6] nonlocal stress (Voigt)
    double* h_energy_nl)    // scalar
{
    auto& ctx = GPUContext::instance();

    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;

    double spn_fac = occfac * 2.0;  // matches CPU: spn_fac = occfac * 2.0
    double wk = 1.0;                // gamma-point

    // ----------------------------------------------------------------
    // Allocate memory for large gradient arrays (cudaMalloc — scratch pool too small)
    // ----------------------------------------------------------------
    size_t grad_size = (size_t)Nd * Nband;
    size_t grad_bytes = grad_size * sizeof(double);

    double* d_Dpsi_x = nullptr;
    double* d_Dpsi_y = nullptr;
    double* d_Dpsi_z = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_Dpsi_x, grad_bytes, 0));
    CUDA_CHECK(cudaMallocAsync(&d_Dpsi_y, grad_bytes, 0));
    CUDA_CHECK(cudaMallocAsync(&d_Dpsi_z, grad_bytes, 0));

    // Small buffers from scratch pool
    size_t scratch_cp = ctx.scratch_pool.checkpoint();

    double* d_alpha = ctx.scratch_pool.alloc<double>((size_t)total_nproj * Nband);
    double* d_beta  = ctx.scratch_pool.alloc<double>((size_t)total_nproj * Nband);

    // Force output on device
    double* d_force = ctx.scratch_pool.alloc<double>(3 * n_phys_atoms);
    CUDA_CHECK(cudaMemset(d_force, 0, 3 * n_phys_atoms * sizeof(double)));

    // Kinetic stress: 6 Voigt components on device
    double* d_sk = ctx.scratch_pool.alloc<double>(6);
    CUDA_CHECK(cudaMemset(d_sk, 0, 6 * sizeof(double)));

    // Nonlocal stress: 6 Voigt components on device
    double* d_snl = ctx.scratch_pool.alloc<double>(6);
    CUDA_CHECK(cudaMemset(d_snl, 0, 6 * sizeof(double)));

    // Nonlocal energy: single scalar
    double* d_enl = ctx.scratch_pool.alloc<double>(1);
    CUDA_CHECK(cudaMemset(d_enl, 0, sizeof(double)));

    // Upload physical atom IP_displ
    double* d_atom_pos = nullptr;
    int* d_IP_displ_phys = nullptr;
    if (total_nproj > 0) {
        d_IP_displ_phys = ctx.scratch_pool.alloc<int>(n_phys_atoms + 1);
        CUDA_CHECK(cudaMemcpy(d_IP_displ_phys, h_IP_displ_phys,
                              (n_phys_atoms + 1) * sizeof(int), cudaMemcpyHostToDevice));

        // Upload atom positions for stress weighted gather
        if (n_influence > 0 && h_atom_pos) {
            d_atom_pos = ctx.scratch_pool.alloc<double>(n_influence * 3);
            CUDA_CHECK(cudaMemcpy(d_atom_pos, h_atom_pos,
                                  n_influence * 3 * sizeof(double), cudaMemcpyHostToDevice));
        }
    }

    // Use ctx.buf.x_ex for halo exchange workspace (already sized for Nband columns)
    double* d_x_ex = ctx.buf.x_ex;

    // ----------------------------------------------------------------
    // Step 1: Halo exchange all bands (once)
    // ----------------------------------------------------------------
    halo_exchange_gpu(d_psi, d_x_ex, nx, ny, nz, FDn, Nband, true, true, true);

    // ----------------------------------------------------------------
    // Step 2: Compute 3 gradient directions
    // ----------------------------------------------------------------
    gradient_gpu(d_x_ex, d_Dpsi_x, nx, ny, nz, FDn, nx_ex, ny_ex, 0, Nband);
    gradient_gpu(d_x_ex, d_Dpsi_y, nx, ny, nz, FDn, nx_ex, ny_ex, 1, Nband);
    gradient_gpu(d_x_ex, d_Dpsi_z, nx, ny, nz, FDn, nx_ex, ny_ex, 2, Nband);

    // ----------------------------------------------------------------
    // Step 3: Kinetic stress (6 Voigt pairs)
    // ----------------------------------------------------------------
    {
        // Voigt ordering: xx(0), xy(1), xz(2), yy(3), yz(4), zz(5)
        const double* d_Dpsi[3] = { d_Dpsi_x, d_Dpsi_y, d_Dpsi_z };
        int voigt_a[6] = {0, 0, 0, 1, 1, 2};
        int voigt_b[6] = {0, 1, 2, 1, 2, 2};

        int bs = 256;
        size_t smem = bs * sizeof(double);
        double neg_occfac_dV = -occfac * dV;

        for (int v = 0; v < 6; ++v) {
            kinetic_stress_kernel<<<Nband, bs, smem>>>(
                d_Dpsi[voigt_a[v]], d_Dpsi[voigt_b[v]],
                d_occ, d_sk + v, Nd, Nband, neg_occfac_dV);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // ----------------------------------------------------------------
    // Step 4: Compute alpha = dV * Chi^T * psi (all bands)
    // ----------------------------------------------------------------
    if (total_nproj > 0 && n_influence > 0) {
        CUDA_CHECK(cudaMemset(d_alpha, 0, (size_t)total_nproj * Nband * sizeof(double)));

        int block_size = 256;

        // Query device shared memory limit
        int device;
        cudaGetDevice(&device);
        int max_smem;
        cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

        size_t smem_per_col = (size_t)max_ndc * sizeof(double);
        int ncol_batch = Nband;
        if (smem_per_col * Nband > (size_t)max_smem)
            ncol_batch = std::max(1, (int)(max_smem / smem_per_col));

        for (int col_start = 0; col_start < Nband; col_start += ncol_batch) {
            int cols_this = std::min(ncol_batch, Nband - col_start);
            size_t smem1 = (size_t)max_ndc * cols_this * sizeof(double);
            if (smem1 > 48 * 1024) {
                cudaFuncSetAttribute(fused_gather_chitpsi_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem1);
            }
            fused_gather_chitpsi_kernel<<<n_influence, block_size, smem1>>>(
                d_psi, d_Chi_flat, d_gpos_flat,
                d_gpos_offsets, d_chi_offsets, d_ndc_arr, d_nproj_arr, d_IP_displ,
                d_alpha, Nd, cols_this, Nband, col_start, dV, n_influence);
            CUDA_CHECK(cudaGetLastError());
        }

        // ----------------------------------------------------------------
        // Step 5: Nonlocal energy (uses alpha before Gamma scaling)
        // energy_nl = wk * sum_n(g_n * sum(Gamma * alpha^2))
        // Note: alpha here has NOT been Gamma-scaled yet (raw dV*Chi^T*psi)
        // ----------------------------------------------------------------
        nonlocal_energy_kernel<<<1, 1>>>(
            d_alpha, d_occ, d_Gamma, d_IP_displ_phys,
            n_phys_atoms, Nband, total_nproj, d_enl);
        CUDA_CHECK(cudaGetLastError());

        // ----------------------------------------------------------------
        // Step 6: Nonlocal force (3 dims)
        // For each dim: beta = dV * Chi^T * Dpsi_dim, then reduce
        // f_nloc[atom*3+dim] -= spn_fac * wk * sum_n(g_n * sum_jp(Gamma * alpha * beta))
        // ----------------------------------------------------------------
        {
            const double* d_Dpsi_arr[3] = { d_Dpsi_x, d_Dpsi_y, d_Dpsi_z };
            int bs_force = 256;

            for (int dim = 0; dim < 3; ++dim) {
                CUDA_CHECK(cudaMemset(d_beta, 0, (size_t)total_nproj * Nband * sizeof(double)));

                for (int col_start = 0; col_start < Nband; col_start += ncol_batch) {
                    int cols_this = std::min(ncol_batch, Nband - col_start);
                    size_t smem1 = (size_t)max_ndc * cols_this * sizeof(double);
                    if (smem1 > 48 * 1024) {
                        cudaFuncSetAttribute(fused_gather_chitpsi_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem1);
                    }
                    fused_gather_chitpsi_kernel<<<n_influence, block_size, smem1>>>(
                        d_Dpsi_arr[dim], d_Chi_flat, d_gpos_flat,
                        d_gpos_offsets, d_chi_offsets, d_ndc_arr, d_nproj_arr, d_IP_displ,
                        d_beta, Nd, cols_this, Nband, col_start, dV, n_influence);
                    CUDA_CHECK(cudaGetLastError());
                }

                // Reduce force for this dim
                nonlocal_force_reduce_kernel<<<ceildiv(n_phys_atoms, bs_force), bs_force>>>(
                    d_alpha, d_beta, d_occ, d_Gamma, d_IP_displ_phys,
                    n_phys_atoms, Nband, spn_fac * wk, d_force, dim);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        // ----------------------------------------------------------------
        // Step 7: Nonlocal stress (6 Voigt pairs)
        // For each (dim, dim2):
        //   beta_stress = dV * sum(chi * xR_dim2 * Dpsi_dim)
        //   snl[voigt] -= spn_fac * wk * sum_n(g_n * sum_jp(Gamma * alpha * beta_stress))
        // ----------------------------------------------------------------
        if (d_atom_pos) {
            const double* d_Dpsi_arr[3] = { d_Dpsi_x, d_Dpsi_y, d_Dpsi_z };

            int voigt_a[6] = {0, 0, 0, 1, 1, 2};
            int voigt_b[6] = {0, 1, 2, 1, 2, 2};

            for (int v = 0; v < 6; ++v) {
                int dim  = voigt_a[v];
                int dim2 = voigt_b[v];

                CUDA_CHECK(cudaMemset(d_beta, 0, (size_t)total_nproj * Nband * sizeof(double)));

                // Compute weighted gather: beta_stress
                // Extra shared memory for xR array, so use smaller batch
                size_t smem_xR = (size_t)max_ndc * sizeof(double);
                int ncol_batch_w = Nband;
                if ((smem_per_col * Nband + smem_xR) > (size_t)max_smem)
                    ncol_batch_w = std::max(1, (int)((max_smem - smem_xR) / smem_per_col));

                for (int col_start = 0; col_start < Nband; col_start += ncol_batch_w) {
                    int cols_this = std::min(ncol_batch_w, Nband - col_start);
                    size_t smem_w = (size_t)max_ndc * cols_this * sizeof(double) + smem_xR;
                    if (smem_w > 48 * 1024) {
                        cudaFuncSetAttribute(weighted_gather_chitpsi_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_w);
                    }
                    weighted_gather_chitpsi_kernel<<<n_influence, block_size, smem_w>>>(
                        d_Dpsi_arr[dim], d_Chi_flat, d_gpos_flat,
                        d_gpos_offsets, d_chi_offsets, d_ndc_arr, d_nproj_arr, d_IP_displ,
                        d_atom_pos, d_beta,
                        Nd, cols_this, Nband, col_start,
                        dV, n_influence,
                        nx, ny, nz, dx, dy, dz, xs, ys, zs, dim2);
                    CUDA_CHECK(cudaGetLastError());
                }

                // Reduce nonlocal stress for this Voigt pair
                nonlocal_stress_reduce_kernel<<<1, 1>>>(
                    d_alpha, d_beta, d_occ, d_Gamma, d_IP_displ_phys,
                    n_phys_atoms, Nband, wk, d_snl + v);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    // ----------------------------------------------------------------
    // Download results to host
    // ----------------------------------------------------------------
    CUDA_CHECK(cudaDeviceSynchronize());

    // Force
    if (total_nproj > 0) {
        CUDA_CHECK(cudaMemcpy(h_f_nloc, d_force,
                              3 * n_phys_atoms * sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        std::memset(h_f_nloc, 0, 3 * n_phys_atoms * sizeof(double));
    }

    // Kinetic stress
    CUDA_CHECK(cudaMemcpy(h_stress_k, d_sk, 6 * sizeof(double), cudaMemcpyDeviceToHost));

    // Nonlocal stress & energy
    if (total_nproj > 0) {
        double h_snl_raw[6];
        CUDA_CHECK(cudaMemcpy(h_snl_raw, d_snl, 6 * sizeof(double), cudaMemcpyDeviceToHost));

        double h_enl_raw;
        CUDA_CHECK(cudaMemcpy(&h_enl_raw, d_enl, sizeof(double), cudaMemcpyDeviceToHost));

        // Scale: snl *= spn_fac, energy_nl *= occfac * wk
        double energy_nl = h_enl_raw * occfac * wk;

        for (int i = 0; i < 6; ++i) h_snl_raw[i] *= spn_fac;

        // Subtract energy_nl from diagonal: xx(0), yy(3), zz(5)
        h_stress_nl[0] = h_snl_raw[0] - energy_nl;
        h_stress_nl[1] = h_snl_raw[1];
        h_stress_nl[2] = h_snl_raw[2];
        h_stress_nl[3] = h_snl_raw[3] - energy_nl;
        h_stress_nl[4] = h_snl_raw[4];
        h_stress_nl[5] = h_snl_raw[5] - energy_nl;

        *h_energy_nl = energy_nl;
    } else {
        std::memset(h_stress_nl, 0, 6 * sizeof(double));
        *h_energy_nl = 0.0;
    }

    // Free gradient arrays
    cudaFreeAsync(d_Dpsi_x, 0);
    cudaFreeAsync(d_Dpsi_y, 0);
    cudaFreeAsync(d_Dpsi_z, 0);

    // Restore scratch pool
    ctx.scratch_pool.restore(scratch_cp);
}

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
