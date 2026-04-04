#ifdef USE_CUDA

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>
#include <complex>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>

#include "core/GPUContext.cuh"
#include "core/gpu_common.cuh"
#include "physics/GPUForce.cuh"
#include "physics/GPUKernelUtils.cuh"
#include "parallel/HaloExchange.cuh"
#include "operators/Gradient.cuh"
#include "operators/NonlocalProjector.cuh"
#include "operators/ComplexOperators.cuh"

namespace lynx {
namespace gpu {

// ============================================================
// GPU Kernels for Force
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
// Tiled weighted gather + Chi^T × (xR * Dpsi) kernel for stress.
// Same tiling strategy as fused_gather_chitpsi_kernel.
// Shared memory: (NL_TILE + nwarps) doubles ≈ 2 KB, O(1) w.r.t. system size.
// xR (position weight) is computed on the fly from the grid index.
// ------------------------------------------------------------
__global__ void weighted_gather_chitpsi_kernel(
    const double* __restrict__ Dpsi,
    const double* __restrict__ Chi_flat,
    const int* __restrict__ gpos_flat,
    const int* __restrict__ gpos_offsets,
    const int* __restrict__ chi_offsets,
    const int* __restrict__ ndc_arr,
    const int* __restrict__ nproj_arr,
    const int* __restrict__ IP_displ,
    const double* __restrict__ atom_pos,
    double* __restrict__ beta,
    int Nd, int ncol_this, int ncol_stride, int col_start,
    double dV, int n_atoms,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    int xs, int ys, int zs,
    int dim2)
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

    extern __shared__ double smem_w[];
    double* psi_tile = smem_w;
    double* reduce_buf = smem_w + NL_TILE_FS;

    for (int n = 0; n < ncol_this; n++) {
        double dots[NL_MAX_NP_FS];
        #pragma unroll 4
        for (int jp = 0; jp < NL_MAX_NP_FS; jp++) dots[jp] = 0.0;

        for (int tile = 0; tile < ndc; tile += NL_TILE_FS) {
            int tile_len = min(NL_TILE_FS, ndc - tile);

            // Gather Dpsi into tile
            for (int i = threadIdx.x; i < tile_len; i += blockDim.x)
                psi_tile[i] = Dpsi[gpos_flat[goff + tile + i] + (col_start + n) * Nd];
            __syncthreads();

            // Accumulate chi * xR * Dpsi for each projector
            for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
                int flat = gpos_flat[goff + tile + i];
                double r;
                if (dim2 == 0) r = (flat % nx + xs) * dx - ap_x;
                else if (dim2 == 1) r = ((flat / nx) % ny + ys) * dy - ap_y;
                else r = (flat / (nx * ny) + zs) * dz - ap_z;

                double pv = psi_tile[i] * r;
                const double* chi_base = Chi_flat + coff + tile + i;
                for (int jp = 0; jp < np; jp++)
                    dots[jp] += chi_base[jp * ndc] * pv;
            }
            __syncthreads();
        }

        for (int jp = 0; jp < np; jp++) {
            double val = blockReduceSum_fs(dots[jp], reduce_buf);
            if (threadIdx.x == 0)
                atomicAdd(&beta[(abase + jp) * ncol_stride + (col_start + n)], val * dV);
        }
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
    double* h_energy_nl,
    cudaStream_t stream)    // scalar
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
    CUDA_CHECK(cudaMallocAsync(&d_Dpsi_x, grad_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_Dpsi_y, grad_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_Dpsi_z, grad_bytes, stream));

    // Small buffers from scratch pool
    size_t scratch_cp = ctx.scratch_pool.checkpoint();

    double* d_alpha = ctx.scratch_pool.alloc<double>((size_t)total_nproj * Nband);
    double* d_beta  = ctx.scratch_pool.alloc<double>((size_t)total_nproj * Nband);

    // Force output on device
    double* d_force = ctx.scratch_pool.alloc<double>(3 * n_phys_atoms);
    CUDA_CHECK(cudaMemsetAsync(d_force, 0, 3 * n_phys_atoms * sizeof(double), stream));

    // Kinetic stress: 6 Voigt components on device
    double* d_sk = ctx.scratch_pool.alloc<double>(6);
    CUDA_CHECK(cudaMemsetAsync(d_sk, 0, 6 * sizeof(double), stream));

    // Nonlocal stress: 6 Voigt components on device
    double* d_snl = ctx.scratch_pool.alloc<double>(6);
    CUDA_CHECK(cudaMemsetAsync(d_snl, 0, 6 * sizeof(double), stream));

    // Nonlocal energy: single scalar
    double* d_enl = ctx.scratch_pool.alloc<double>(1);
    CUDA_CHECK(cudaMemsetAsync(d_enl, 0, sizeof(double), stream));

    // Upload physical atom IP_displ
    double* d_atom_pos = nullptr;
    int* d_IP_displ_phys = nullptr;
    if (total_nproj > 0) {
        d_IP_displ_phys = ctx.scratch_pool.alloc<int>(n_phys_atoms + 1);
        CUDA_CHECK(cudaMemcpyAsync(d_IP_displ_phys, h_IP_displ_phys,
                              (n_phys_atoms + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));

        // Upload atom positions for stress weighted gather
        if (n_influence > 0 && h_atom_pos) {
            d_atom_pos = ctx.scratch_pool.alloc<double>(n_influence * 3);
            CUDA_CHECK(cudaMemcpyAsync(d_atom_pos, h_atom_pos,
                                  n_influence * 3 * sizeof(double), cudaMemcpyHostToDevice, stream));
        }
    }

    // Use ctx.buf.x_ex for halo exchange workspace (already sized for Nband columns)
    double* d_x_ex = ctx.buf.x_ex;

    // ----------------------------------------------------------------
    // Step 1: Halo exchange all bands (once)
    // ----------------------------------------------------------------
    halo_exchange_batched_nomemset_gpu(d_psi, d_x_ex, nx, ny, nz, FDn, Nband, true, true, true, stream);

    // ----------------------------------------------------------------
    // Step 2: Compute 3 gradient directions (batched V2 — single launch per direction)
    // ----------------------------------------------------------------
    gradient_v3_gpu(d_x_ex, d_Dpsi_x, nx, ny, nz, FDn, nx_ex, ny_ex, 0, Nband, stream);
    gradient_v3_gpu(d_x_ex, d_Dpsi_y, nx, ny, nz, FDn, nx_ex, ny_ex, 1, Nband, stream);
    gradient_v3_gpu(d_x_ex, d_Dpsi_z, nx, ny, nz, FDn, nx_ex, ny_ex, 2, Nband, stream);

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
            kinetic_stress_kernel<<<Nband, bs, smem, stream>>>(
                d_Dpsi[voigt_a[v]], d_Dpsi[voigt_b[v]],
                d_occ, d_sk + v, Nd, Nband, neg_occfac_dV);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // ----------------------------------------------------------------
    // Step 4: Compute alpha = dV * Chi^T * psi (all bands)
    // Tiled kernel: fixed shared memory ≈ 2 KB, independent of system size
    // ----------------------------------------------------------------
    if (total_nproj > 0 && n_influence > 0) {
        CUDA_CHECK(cudaMemsetAsync(d_alpha, 0, (size_t)total_nproj * Nband * sizeof(double), stream));

        int block_size = 256;
        size_t smem_tiled = (NL_TILE_FS + block_size / 32) * sizeof(double);

        nonlocal_gather_chitpsi_gpu(
            d_psi, d_Chi_flat, d_gpos_flat,
            d_gpos_offsets, d_chi_offsets, d_ndc_arr, d_nproj_arr, d_IP_displ,
            d_alpha, Nd, Nband, Nband, 0, dV, n_influence,
            max_ndc, max_nproj, stream);

        // ----------------------------------------------------------------
        // Step 5: Nonlocal energy (uses alpha before Gamma scaling)
        // energy_nl = wk * sum_n(g_n * sum(Gamma * alpha^2))
        // Note: alpha here has NOT been Gamma-scaled yet (raw dV*Chi^T*psi)
        // ----------------------------------------------------------------
        nonlocal_energy_kernel<<<1, 1, 0, stream>>>(
            d_alpha, d_occ, d_Gamma, d_IP_displ_phys,
            n_phys_atoms, Nband, total_nproj, d_enl);
        CUDA_CHECK(cudaGetLastError());

        // ----------------------------------------------------------------
        // Step 6: Nonlocal force (3 dims)
        // For each dim: beta = dV * Chi^T * Dpsi_dim, then reduce
        // ----------------------------------------------------------------
        {
            const double* d_Dpsi_arr[3] = { d_Dpsi_x, d_Dpsi_y, d_Dpsi_z };
            int bs_force = 256;

            for (int dim = 0; dim < 3; ++dim) {
                CUDA_CHECK(cudaMemsetAsync(d_beta, 0, (size_t)total_nproj * Nband * sizeof(double), stream));

                nonlocal_gather_chitpsi_gpu(
                    d_Dpsi_arr[dim], d_Chi_flat, d_gpos_flat,
                    d_gpos_offsets, d_chi_offsets, d_ndc_arr, d_nproj_arr, d_IP_displ,
                    d_beta, Nd, Nband, Nband, 0, dV, n_influence,
                    max_ndc, max_nproj, stream);

                nonlocal_force_reduce_kernel<<<ceildiv(n_phys_atoms, bs_force), bs_force, 0, stream>>>(
                    d_alpha, d_beta, d_occ, d_Gamma, d_IP_displ_phys,
                    n_phys_atoms, Nband, spn_fac * wk, d_force, dim);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        // ----------------------------------------------------------------
        // Step 7: Nonlocal stress (6 Voigt pairs)
        // Tiled weighted kernel: fixed shared memory ≈ 2 KB
        // ----------------------------------------------------------------
        if (d_atom_pos) {
            const double* d_Dpsi_arr[3] = { d_Dpsi_x, d_Dpsi_y, d_Dpsi_z };

            int voigt_a[6] = {0, 0, 0, 1, 1, 2};
            int voigt_b[6] = {0, 1, 2, 1, 2, 2};

            for (int v = 0; v < 6; ++v) {
                int dim  = voigt_a[v];
                int dim2 = voigt_b[v];

                CUDA_CHECK(cudaMemsetAsync(d_beta, 0, (size_t)total_nproj * Nband * sizeof(double), stream));

                weighted_gather_chitpsi_kernel<<<n_influence, block_size, smem_tiled, stream>>>(
                    d_Dpsi_arr[dim], d_Chi_flat, d_gpos_flat,
                    d_gpos_offsets, d_chi_offsets, d_ndc_arr, d_nproj_arr, d_IP_displ,
                    d_atom_pos, d_beta,
                    Nd, Nband, Nband, 0,
                    dV, n_influence,
                    nx, ny, nz, dx, dy, dz, xs, ys, zs, dim2);
                CUDA_CHECK(cudaGetLastError());

                nonlocal_stress_reduce_kernel<<<1, 1, 0, stream>>>(
                    d_alpha, d_beta, d_occ, d_Gamma, d_IP_displ_phys,
                    n_phys_atoms, Nband, wk, d_snl + v);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    // ----------------------------------------------------------------
    // Download results to host
    // ----------------------------------------------------------------
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Force
    if (total_nproj > 0) {
        CUDA_CHECK(cudaMemcpyAsync(h_f_nloc, d_force,
                              3 * n_phys_atoms * sizeof(double), cudaMemcpyDeviceToHost, stream));
    } else {
        std::memset(h_f_nloc, 0, 3 * n_phys_atoms * sizeof(double));
    }

    // Kinetic stress
    CUDA_CHECK(cudaMemcpyAsync(h_stress_k, d_sk, 6 * sizeof(double), cudaMemcpyDeviceToHost, stream));

    // Nonlocal stress & energy
    if (total_nproj > 0) {
        double h_snl_raw[6];
        CUDA_CHECK(cudaMemcpyAsync(h_snl_raw, d_snl, 6 * sizeof(double), cudaMemcpyDeviceToHost, stream));

        double h_enl_raw;
        CUDA_CHECK(cudaMemcpyAsync(&h_enl_raw, d_enl, sizeof(double), cudaMemcpyDeviceToHost, stream));

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
    cudaFreeAsync(d_Dpsi_x, stream);
    cudaFreeAsync(d_Dpsi_y, stream);
    cudaFreeAsync(d_Dpsi_z, stream);

    // Restore scratch pool
    ctx.scratch_pool.restore(scratch_cp);
}

// SOC gather kernel (defined in SOCOperators.cu, external linkage via separable compilation)
__global__
void soc_gather_alpha_z_kernel(
    const cuDoubleComplex* __restrict__ psi,
    const cuDoubleComplex* __restrict__ Chi_soc_flat,
    const int*             __restrict__ gpos_flat,
    const int*             __restrict__ gpos_offsets,
    const int*             __restrict__ chi_soc_offsets,
    const int*             __restrict__ ndc_arr,
    const int*             __restrict__ nproj_soc_arr,
    const int*             __restrict__ IP_displ_soc,
    const double*          __restrict__ bloch_fac,
    cuDoubleComplex*       __restrict__ alpha_up,
    cuDoubleComplex*       __restrict__ alpha_dn,
    int Nd_d, int ncol_this, int ncol_stride, int col_start,
    double dV, int n_atoms);

// Local spinor extract kernel for SOC force
// Extracts component (0=up, 1=dn) from interleaved spinor layout
__global__ void soc_force_spinor_extract_kernel(
    const cuDoubleComplex* __restrict__ spinor,  // [up0|dn0|up1|dn1|...] stride 2*Nd per band
    cuDoubleComplex* __restrict__ out,            // [col0|col1|...] stride Nd per band
    int Nd, int ncol, int component)              // component: 0=up, 1=dn
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nd * ncol) return;
    int ig = i % Nd;
    int n = i / Nd;
    out[n * Nd + ig] = spinor[n * 2 * Nd + component * Nd + ig];
}

// ============================================================
// Host function: compute_soc_force_gpu
//
// SOC nonlocal force using spinor wavefunctions.
// Algorithm:
//   1. Extract psi_up/dn from spinor layout
//   2. Compute alpha_up/dn = bloch * dV * Chi_soc^T * psi_up/dn (gather)
//   3. For each dim:
//      a. Complex halo exchange + gradient → Dpsi_up/dn
//      b. Compute beta_up/dn = bloch * dV * Chi_soc^T * Dpsi_up/dn
//      c. Download alpha, beta, occ to host
//      d. CPU reduction: Term 1 (Lz·Sz) + Term 2 (L±S∓)
// ============================================================
void compute_soc_force_gpu(
    const cuDoubleComplex* d_psi_spinor,  // (2*Nd_d, Nband) spinor psi on device
    const double* d_occ,                   // (Nband) occupations on device
    // SOC projector data (all on device, from GPUSOCData)
    const cuDoubleComplex* d_Chi_soc_flat,
    const int* d_gpos_flat,
    const int* d_gpos_offsets_soc,
    const int* d_chi_soc_offsets,
    const int* d_ndc_arr_soc,
    const int* d_nproj_soc_arr,
    const int* d_IP_displ_soc,
    const double* d_Gamma_soc,
    const int* d_proj_l,
    const int* d_proj_m,
    const double* d_bloch_fac,
    int n_influence_soc,
    int total_soc_nproj,
    int max_ndc_soc,
    int max_nproj_soc,
    int n_phys_atoms,
    const int* h_IP_displ_phys_soc,  // [n_phys+1] physical atom SOC projector offsets (host)
    const int* h_proj_l,             // [total_soc_nproj] projector l values (host)
    const int* h_proj_m,             // [total_soc_nproj] projector m values (host)
    const double* h_Gamma_soc,       // [total_soc_nproj] SOC gamma values (host)
    // Grid parameters
    int nx, int ny, int nz, int FDn, int Nd_d, int Nband,
    double dV,
    double kx_Lx, double ky_Ly, double kz_Lz,  // Bloch phase products
    double spn_fac, double wk,
    // Output (host)
    double* h_f_soc,
    cudaStream_t stream)  // [3 * n_phys_atoms]
{
    if (n_influence_soc == 0 || total_soc_nproj == 0) {
        std::memset(h_f_soc, 0, 3 * n_phys_atoms * sizeof(double));
        return;
    }

    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nz_ex = nz + 2 * FDn;
    int Nd_ex = nx_ex * ny_ex * nz_ex;
    int Nd_d_spinor = 2 * Nd_d;

    // ----------------------------------------------------------------
    // Allocate scratch buffers
    // ----------------------------------------------------------------
    size_t alpha_elems = (size_t)total_soc_nproj * Nband;
    size_t alpha_bytes = alpha_elems * sizeof(cuDoubleComplex);

    cuDoubleComplex* d_alpha_up = nullptr;
    cuDoubleComplex* d_alpha_dn = nullptr;
    cuDoubleComplex* d_beta_up  = nullptr;
    cuDoubleComplex* d_beta_dn  = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_alpha_up, alpha_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_alpha_dn, alpha_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_beta_up,  alpha_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_beta_dn,  alpha_bytes, stream));

    // Extracted psi_up/dn (Nd_d * Nband complex each)
    size_t psi_comp_bytes = (size_t)Nd_d * Nband * sizeof(cuDoubleComplex);
    cuDoubleComplex* d_psi_up = nullptr;
    cuDoubleComplex* d_psi_dn = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_psi_up, psi_comp_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_psi_dn, psi_comp_bytes, stream));

    // Gradient output buffers (Nd_d * Nband complex)
    cuDoubleComplex* d_Dpsi_up = nullptr;
    cuDoubleComplex* d_Dpsi_dn = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_Dpsi_up, psi_comp_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_Dpsi_dn, psi_comp_bytes, stream));

    // Halo exchange buffer (Nd_ex * Nband complex)
    size_t ex_bytes = (size_t)Nd_ex * Nband * sizeof(cuDoubleComplex);
    cuDoubleComplex* d_x_ex = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_x_ex, ex_bytes, stream));

    // ----------------------------------------------------------------
    // Step 1: Extract up/dn from spinor layout
    // ----------------------------------------------------------------
    {
        int total_elems = Nd_d * Nband;
        int bs = 256;
        int gs = (total_elems + bs - 1) / bs;
        soc_force_spinor_extract_kernel<<<gs, bs, 0, stream>>>(d_psi_spinor, d_psi_up, Nd_d, Nband, 0);
        CUDA_CHECK(cudaGetLastError());
        soc_force_spinor_extract_kernel<<<gs, bs, 0, stream>>>(d_psi_spinor, d_psi_dn, Nd_d, Nband, 1);
        CUDA_CHECK(cudaGetLastError());
    }

    // ----------------------------------------------------------------
    // Step 2: Compute alpha_up/dn = bloch * dV * Chi_soc^T * psi_up/dn
    // We use the gather kernel with the full spinor input but passing
    // extracted components as if they were spinor (need to fake spinor layout).
    // Actually, the gather kernel expects spinor layout [up|dn] per band.
    // So we just call it with d_psi_spinor directly.
    // ----------------------------------------------------------------
    CUDA_CHECK(cudaMemsetAsync(d_alpha_up, 0, alpha_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(d_alpha_dn, 0, alpha_bytes, stream));

    {
        int threads = 256;
        if (max_nproj_soc * Nband < threads)
            threads = ((max_nproj_soc * Nband + 31) / 32) * 32;
        if (threads < 32) threads = 32;

        soc_gather_alpha_z_kernel<<<n_influence_soc, threads, 0, stream>>>(
            d_psi_spinor, d_Chi_soc_flat, d_gpos_flat,
            d_gpos_offsets_soc, d_chi_soc_offsets,
            d_ndc_arr_soc, d_nproj_soc_arr, d_IP_displ_soc,
            d_bloch_fac,
            d_alpha_up, d_alpha_dn,
            Nd_d, Nband, Nband, 0, dV, n_influence_soc);
        CUDA_CHECK(cudaGetLastError());
    }

    // Download alpha to host (small: ~total_soc_nproj * Nband * 16 bytes)
    std::vector<cuDoubleComplex> h_alpha_up(alpha_elems);
    std::vector<cuDoubleComplex> h_alpha_dn(alpha_elems);
    std::vector<double> h_occ(Nband);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(h_alpha_up.data(), d_alpha_up, alpha_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_alpha_dn.data(), d_alpha_dn, alpha_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_occ.data(), d_occ, Nband * sizeof(double), cudaMemcpyDeviceToHost, stream));

    // Initialize output
    std::memset(h_f_soc, 0, 3 * n_phys_atoms * sizeof(double));

    // ----------------------------------------------------------------
    // Step 3: For each direction, compute gradient → beta → reduce
    // ----------------------------------------------------------------
    for (int dim = 0; dim < 3; ++dim) {
        // 3a. Complex halo exchange + gradient for psi_up → Dpsi_up
        halo_exchange_z_gpu(d_psi_up, d_x_ex, nx, ny, nz, FDn, Nband,
                            true, true, true, kx_Lx, ky_Ly, kz_Lz, stream);
        gradient_z_gpu(d_x_ex, d_Dpsi_up, nx, ny, nz, FDn, nx_ex, ny_ex, dim, Nband, stream);

        // 3b. Complex halo exchange + gradient for psi_dn → Dpsi_dn
        halo_exchange_z_gpu(d_psi_dn, d_x_ex, nx, ny, nz, FDn, Nband,
                            true, true, true, kx_Lx, ky_Ly, kz_Lz, stream);
        gradient_z_gpu(d_x_ex, d_Dpsi_dn, nx, ny, nz, FDn, nx_ex, ny_ex, dim, Nband, stream);

        // 3c. Compute beta_up/dn using soc_gather on Dpsi
        // We need to pack Dpsi_up/dn into spinor layout for the gather kernel
        // Allocate temporary spinor-layout gradient buffer
        cuDoubleComplex* d_Dpsi_spinor = nullptr;
        size_t spinor_grad_bytes = (size_t)Nd_d_spinor * Nband * sizeof(cuDoubleComplex);
        CUDA_CHECK(cudaMallocAsync(&d_Dpsi_spinor, spinor_grad_bytes, stream));

        // Scatter Dpsi_up/dn into spinor layout: [up0|dn0|up1|dn1|...]
        {
            for (int n = 0; n < Nband; ++n) {
                CUDA_CHECK(cudaMemcpyAsync(
                    d_Dpsi_spinor + n * Nd_d_spinor,
                    d_Dpsi_up + n * Nd_d,
                    Nd_d * sizeof(cuDoubleComplex),
                    cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    d_Dpsi_spinor + n * Nd_d_spinor + Nd_d,
                    d_Dpsi_dn + n * Nd_d,
                    Nd_d * sizeof(cuDoubleComplex),
                    cudaMemcpyDeviceToDevice, stream));
            }
        }

        CUDA_CHECK(cudaMemsetAsync(d_beta_up, 0, alpha_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(d_beta_dn, 0, alpha_bytes, stream));

        {
            int threads = 256;
            if (max_nproj_soc * Nband < threads)
                threads = ((max_nproj_soc * Nband + 31) / 32) * 32;
            if (threads < 32) threads = 32;

            soc_gather_alpha_z_kernel<<<n_influence_soc, threads, 0, stream>>>(
                d_Dpsi_spinor, d_Chi_soc_flat, d_gpos_flat,
                d_gpos_offsets_soc, d_chi_soc_offsets,
                d_ndc_arr_soc, d_nproj_soc_arr, d_IP_displ_soc,
                d_bloch_fac,
                d_beta_up, d_beta_dn,
                Nd_d, Nband, Nband, 0, dV, n_influence_soc);
            CUDA_CHECK(cudaGetLastError());
        }

        cudaFreeAsync(d_Dpsi_spinor, stream);

        // 3d. Download beta to host
        std::vector<cuDoubleComplex> h_beta_up(alpha_elems);
        std::vector<cuDoubleComplex> h_beta_dn(alpha_elems);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpyAsync(h_beta_up.data(), d_beta_up, alpha_bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_beta_dn.data(), d_beta_dn, alpha_bytes, cudaMemcpyDeviceToHost, stream));

        // 3e. CPU reduction: accumulate SOC force for this dim
        for (int ia = 0; ia < n_phys_atoms; ++ia) {
            int off = h_IP_displ_phys_soc[ia];
            int nproj = h_IP_displ_phys_soc[ia + 1] - off;
            if (nproj == 0) continue;

            double f_dim = 0.0;
            for (int n = 0; n < Nband; ++n) {
                double g_n = h_occ[n];
                if (fabs(g_n) < 1e-15) continue;

                double fJ = 0.0;
                for (int jp = 0; jp < nproj; ++jp) {
                    int glob_jp = off + jp;
                    int l = h_proj_l[glob_jp];
                    int m = h_proj_m[glob_jp];
                    double gamma_soc = h_Gamma_soc[glob_jp];

                    int idx = glob_jp * Nband + n;

                    // Term 1: on-diagonal (Lz·Sz)
                    if (m != 0) {
                        // Re(conj(alpha)*beta) = alpha.x*beta.x + alpha.y*beta.y
                        double re_au_bu = h_alpha_up[idx].x * h_beta_up[idx].x
                                        + h_alpha_up[idx].y * h_beta_up[idx].y;
                        double re_ad_bd = h_alpha_dn[idx].x * h_beta_dn[idx].x
                                        + h_alpha_dn[idx].y * h_beta_dn[idx].y;
                        fJ += 0.5 * (double)m * gamma_soc * (re_au_bu - re_ad_bd);
                    }

                    // Term 2: L+S- (m -> m+1)
                    if (m + 1 <= l) {
                        double ladder = sqrt((double)(l * (l + 1) - m * (m + 1)));
                        int jp_shifted = jp + 1;
                        int idx_s = (off + jp_shifted) * Nband + n;
                        // Re(conj(alpha_dn[m+1])*beta_up[m] + conj(alpha_up[m])*beta_dn[m+1])
                        double re1 = h_alpha_dn[idx_s].x * h_beta_up[idx].x
                                   + h_alpha_dn[idx_s].y * h_beta_up[idx].y;
                        double re2 = h_alpha_up[idx].x * h_beta_dn[idx_s].x
                                   + h_alpha_up[idx].y * h_beta_dn[idx_s].y;
                        fJ += 0.5 * ladder * gamma_soc * (re1 + re2);
                    }

                    // Term 2: L-S+ (m -> m-1)
                    if (m - 1 >= -l) {
                        double ladder = sqrt((double)(l * (l + 1) - m * (m - 1)));
                        int jp_shifted = jp - 1;
                        int idx_s = (off + jp_shifted) * Nband + n;
                        // Re(conj(alpha_up[m-1])*beta_dn[m] + conj(alpha_dn[m])*beta_up[m-1])
                        double re1 = h_alpha_up[idx_s].x * h_beta_dn[idx].x
                                   + h_alpha_up[idx_s].y * h_beta_dn[idx].y;
                        double re2 = h_alpha_dn[idx].x * h_beta_up[idx_s].x
                                   + h_alpha_dn[idx].y * h_beta_up[idx_s].y;
                        fJ += 0.5 * ladder * gamma_soc * (re1 + re2);
                    }
                }

                f_dim -= spn_fac * wk * g_n * fJ;
            }

            h_f_soc[ia * 3 + dim] += f_dim;
        }
    }

    // Free scratch
    cudaFreeAsync(d_alpha_up, stream);
    cudaFreeAsync(d_alpha_dn, stream);
    cudaFreeAsync(d_beta_up, stream);
    cudaFreeAsync(d_beta_dn, stream);
    cudaFreeAsync(d_psi_up, stream);
    cudaFreeAsync(d_psi_dn, stream);
    cudaFreeAsync(d_Dpsi_up, stream);
    cudaFreeAsync(d_Dpsi_dn, stream);
    cudaFreeAsync(d_x_ex, stream);
}

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
