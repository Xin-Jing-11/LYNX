/**
 * @file    SOCOperators.cu
 * @brief   GPU kernels for spin-orbit coupling (SOC) in spinor DFT calculations.
 *
 *          Implements the GPU counterpart of the CPU SOC routines in
 *          LYNX/src/operators/NonlocalProjector.cpp (apply_soc_kpt).
 *
 *          Kernel 1: soc_gather_alpha_z_kernel  -- inner product Chi_soc^T * psi
 *          Kernel 2: soc_scatter_z_kernel       -- scatter Term 1 (Lz*Sz) + Term 2 (ladder)
 *          Kernel 3: spinor_offdiag_veff_kernel -- off-diagonal Veff coupling
 *          Kernel 4: spinor_density_kernel      -- density / magnetization from spinors
 *
 * @authors GPU SOC implementation
 */

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "operators/SOCOperators.cuh"
#include <math.h>
#include <stdio.h>

namespace lynx {
namespace gpu {

/* =========================================================================
 * Helper device functions
 * ========================================================================= */

static __device__ __forceinline__
cuDoubleComplex cmul_rd(double a, cuDoubleComplex z) {
    // real * complex
    return make_cuDoubleComplex(a * z.x, a * z.y);
}

static __device__ __forceinline__
cuDoubleComplex cmul_rc(cuDoubleComplex a, double chi) {
    // complex * real
    return make_cuDoubleComplex(a.x * chi, a.y * chi);
}

static __device__ __forceinline__
void atomicAddComplex(cuDoubleComplex* addr, cuDoubleComplex val) {
    atomicAdd(&(reinterpret_cast<double*>(addr)[0]), val.x);
    atomicAdd(&(reinterpret_cast<double*>(addr)[1]), val.y);
}


/* =========================================================================
 * Kernel 1: soc_gather_alpha_z_kernel
 *
 * Computes the inner products alpha_up and alpha_dn for SOC projectors:
 *   alpha_sigma[IP_displ + jp, col] += bloch_fac * dV * sum_ig Chi_soc[ig,jp] * psi_sigma[gpos[ig]]
 *
 * Launch: one block per influence atom, threads loop over (grid_point, column) pairs.
 * Shared memory is used to cache grid positions for the current atom.
 * ========================================================================= */

__global__
void soc_gather_alpha_z_kernel(
    const cuDoubleComplex* __restrict__ psi,        // (Nd_d_spinor, ncol_stride)
    const cuDoubleComplex* __restrict__ Chi_soc_flat,// flattened SOC projectors [ndc x nproj per atom] (complex)
    const int*             __restrict__ gpos_flat,   // flattened grid positions
    const int*             __restrict__ gpos_offsets, // per-influence-atom offsets into gpos_flat
    const int*             __restrict__ chi_soc_offsets, // per-influence-atom offsets into Chi_soc_flat
    const int*             __restrict__ ndc_arr,     // ndc per influence atom
    const int*             __restrict__ nproj_soc_arr, // nproj_soc per influence atom
    const int*             __restrict__ IP_displ_soc,// per-influence-atom SOC proj displacement (global atom)
    const double*          __restrict__ bloch_fac,   // [n_atoms * 2]: cos, sin per atom
    cuDoubleComplex*       __restrict__ alpha_up,    // [total_soc_nproj * ncol_stride]
    cuDoubleComplex*       __restrict__ alpha_dn,    // [total_soc_nproj * ncol_stride]
    int Nd_d, int ncol_this, int ncol_stride, int col_start,
    double dV, int n_atoms)
{
    int iat = blockIdx.x;
    if (iat >= n_atoms) return;

    int ndc     = ndc_arr[iat];
    int nproj   = nproj_soc_arr[iat];
    if (ndc == 0 || nproj == 0) return;

    int gpos_off = gpos_offsets[iat];
    int chi_off  = chi_soc_offsets[iat];
    int ip_off   = IP_displ_soc[iat]; // projector offset for this atom

    double bcos = bloch_fac[iat * 2 + 0];
    double bsin = bloch_fac[iat * 2 + 1];
    // alpha_scale = bloch_fac * dV
    double scale_re = bcos * dV;
    double scale_im = bsin * dV;

    int Nd_d_spinor = 2 * Nd_d;

    // Each thread processes a subset of (jp, col) pairs
    int total_work = nproj * ncol_this;
    for (int tid = threadIdx.x; tid < total_work; tid += blockDim.x) {
        int jp  = tid % nproj;
        int col = tid / nproj;
        int n   = col_start + col;

        const cuDoubleComplex* psi_up = psi + n * Nd_d_spinor;
        const cuDoubleComplex* psi_dn = psi_up + Nd_d;

        double dot_up_re = 0.0, dot_up_im = 0.0;
        double dot_dn_re = 0.0, dot_dn_im = 0.0;

        for (int ig = 0; ig < ndc; ++ig) {
            int grid_idx = gpos_flat[gpos_off + ig];
            cuDoubleComplex chi_val = Chi_soc_flat[chi_off + jp * ndc + ig];
            // conj(chi) * psi (gather uses conjugate of Chi_soc)
            cuDoubleComplex chi_conj = make_cuDoubleComplex(chi_val.x, -chi_val.y);

            cuDoubleComplex pu = psi_up[grid_idx];
            dot_up_re += chi_conj.x * pu.x - chi_conj.y * pu.y;
            dot_up_im += chi_conj.x * pu.y + chi_conj.y * pu.x;

            cuDoubleComplex pd = psi_dn[grid_idx];
            dot_dn_re += chi_conj.x * pd.x - chi_conj.y * pd.y;
            dot_dn_im += chi_conj.x * pd.y + chi_conj.y * pd.x;
        }

        // Multiply by bloch_fac * dV: (scale_re + i*scale_im) * (dot_re + i*dot_im)
        double au_re = scale_re * dot_up_re - scale_im * dot_up_im;
        double au_im = scale_re * dot_up_im + scale_im * dot_up_re;

        double ad_re = scale_re * dot_dn_re - scale_im * dot_dn_im;
        double ad_im = scale_re * dot_dn_im + scale_im * dot_dn_re;

        // Accumulate into alpha arrays: layout alpha[(ip_off + jp) * ncol_stride + col]
        int alpha_idx = (ip_off + jp) * ncol_stride + col;
        atomicAddComplex(&alpha_up[alpha_idx], make_cuDoubleComplex(au_re, au_im));
        atomicAddComplex(&alpha_dn[alpha_idx], make_cuDoubleComplex(ad_re, ad_im));
    }
}


/* =========================================================================
 * Kernel 2: soc_scatter_z_kernel
 *
 * Scatter SOC contributions back to Hpsi. Implements:
 *   Term 1 (on-diagonal, Lz*Sz): m != 0 terms
 *   Term 2 (off-diagonal, ladder L+S- and L-S+)
 *
 * Launch: one block per influence atom.
 * ========================================================================= */

__global__
void soc_scatter_z_kernel(
    cuDoubleComplex*       __restrict__ Hpsi,        // (Nd_d_spinor, ncol_stride)
    const cuDoubleComplex* __restrict__ Chi_soc_flat, // complex Chi_soc
    const int*             __restrict__ gpos_flat,
    const int*             __restrict__ gpos_offsets,
    const int*             __restrict__ chi_soc_offsets,
    const int*             __restrict__ ndc_arr,
    const int*             __restrict__ nproj_soc_arr,
    const int*             __restrict__ IP_displ_soc,
    const double*          __restrict__ bloch_fac,    // [n_atoms * 2]: cos, sin (kernel negates sin for conj)
    const cuDoubleComplex* __restrict__ alpha_up,
    const cuDoubleComplex* __restrict__ alpha_dn,
    const double*          __restrict__ Gamma_soc,    // per-projector gamma
    const int*             __restrict__ proj_l,       // l per projector (global index)
    const int*             __restrict__ proj_m,       // m per projector (global index)
    int Nd_d, int ncol_this, int ncol_stride, int col_start,
    int n_atoms)
{
    int iat = blockIdx.x;
    if (iat >= n_atoms) return;

    int ndc     = ndc_arr[iat];
    int nproj   = nproj_soc_arr[iat];
    if (ndc == 0 || nproj == 0) return;

    int gpos_off = gpos_offsets[iat];
    int chi_off  = chi_soc_offsets[iat];
    int ip_off   = IP_displ_soc[iat];

    // Conjugate bloch factor for scatter: negate sin to get conj
    double bcos = bloch_fac[iat * 2 + 0];
    double bsin = -bloch_fac[iat * 2 + 1]; // negate sin for conjugate

    int Nd_d_spinor = 2 * Nd_d;

    // Each thread processes a subset of (ig, col, jp) work
    // Strategy: iterate over columns and projectors, scatter to grid
    for (int col = 0; col < ncol_this; ++col) {
        int n = col_start + col;
        cuDoubleComplex* Hpsi_up = Hpsi + n * Nd_d_spinor;
        cuDoubleComplex* Hpsi_dn = Hpsi_up + Nd_d;

        for (int jp = 0; jp < nproj; ++jp) {
            int glob_jp = ip_off + jp;
            int l = proj_l[glob_jp];
            int m = proj_m[glob_jp];
            double gamma = Gamma_soc[glob_jp];

            int alpha_idx = glob_jp * ncol_stride + col;

            // ---- Term 1: on-diagonal (Lz * Sz) ----
            // Hpsi_up += +0.5 * m * gamma * alpha_up * bloch_conj * chi
            // Hpsi_dn += -0.5 * m * gamma * alpha_dn * bloch_conj * chi
            if (m != 0) {
                double coeff1_fac = 0.5 * (double)m * gamma;
                cuDoubleComplex au = alpha_up[alpha_idx];
                cuDoubleComplex ad = alpha_dn[alpha_idx];

                // coeff_up = coeff1_fac * alpha_up
                cuDoubleComplex coeff_up = cmul_rd(coeff1_fac, au);
                // coeff_dn = -coeff1_fac * alpha_dn
                cuDoubleComplex coeff_dn = cmul_rd(-coeff1_fac, ad);

                // bloch_conj * coeff: (bcos + i*bsin) * coeff
                cuDoubleComplex bc_cu = make_cuDoubleComplex(
                    bcos * coeff_up.x - bsin * coeff_up.y,
                    bcos * coeff_up.y + bsin * coeff_up.x);
                cuDoubleComplex bc_cd = make_cuDoubleComplex(
                    bcos * coeff_dn.x - bsin * coeff_dn.y,
                    bcos * coeff_dn.y + bsin * coeff_dn.x);

                for (int ig = threadIdx.x; ig < ndc; ig += blockDim.x) {
                    int grid_idx = gpos_flat[gpos_off + ig];
                    cuDoubleComplex chi_val = Chi_soc_flat[chi_off + jp * ndc + ig];
                    atomicAddComplex(&Hpsi_up[grid_idx], cuCmul(bc_cu, chi_val));
                    atomicAddComplex(&Hpsi_dn[grid_idx], cuCmul(bc_cd, chi_val));
                }
            }

            // ---- Term 2: L+ S- ----
            if (m + 1 <= l) {
                double ladder = sqrt((double)(l * (l + 1) - m * (m + 1)));
                double fac = 0.5 * ladder * gamma;
                int jp_shifted = jp + 1; // m+1 column is next (sequential m ordering)
                int alpha_idx_shifted = (ip_off + jp_shifted) * ncol_stride + col;
                cuDoubleComplex a_dn_shifted = alpha_dn[alpha_idx_shifted];

                cuDoubleComplex coeff = cmul_rd(fac, a_dn_shifted);
                cuDoubleComplex bc_c = make_cuDoubleComplex(
                    bcos * coeff.x - bsin * coeff.y,
                    bcos * coeff.y + bsin * coeff.x);

                for (int ig = threadIdx.x; ig < ndc; ig += blockDim.x) {
                    int grid_idx = gpos_flat[gpos_off + ig];
                    cuDoubleComplex chi_val = Chi_soc_flat[chi_off + jp * ndc + ig];
                    atomicAddComplex(&Hpsi_up[grid_idx], cuCmul(bc_c, chi_val));
                }
            }

            // ---- Term 2: L- S+ ----
            if (m - 1 >= -l) {
                double ladder = sqrt((double)(l * (l + 1) - m * (m - 1)));
                double fac = 0.5 * ladder * gamma;
                int jp_shifted = jp - 1; // m-1 column is previous
                int alpha_idx_shifted = (ip_off + jp_shifted) * ncol_stride + col;
                cuDoubleComplex a_up_shifted = alpha_up[alpha_idx_shifted];

                cuDoubleComplex coeff = cmul_rd(fac, a_up_shifted);
                cuDoubleComplex bc_c = make_cuDoubleComplex(
                    bcos * coeff.x - bsin * coeff.y,
                    bcos * coeff.y + bsin * coeff.x);

                for (int ig = threadIdx.x; ig < ndc; ig += blockDim.x) {
                    int grid_idx = gpos_flat[gpos_off + ig];
                    cuDoubleComplex chi_val = Chi_soc_flat[chi_off + jp * ndc + ig];
                    atomicAddComplex(&Hpsi_dn[grid_idx], cuCmul(bc_c, chi_val));
                }
            }
        } // jp
    } // col
}


/* =========================================================================
 * Kernel 3: spinor_offdiag_veff_kernel
 *
 * Apply off-diagonal Veff coupling for spinor Hamiltonian:
 *   Hpsi_up[i] += V_ud[i] * psi_dn[i]
 *   Hpsi_dn[i] += conj(V_ud[i]) * psi_up[i]
 *
 * where V_ud = V_ud_re + i * V_ud_im.
 * ========================================================================= */

__global__
void spinor_offdiag_veff_kernel(
    cuDoubleComplex*       __restrict__ Hpsi,
    const cuDoubleComplex* __restrict__ psi,
    const double*          __restrict__ V_ud_re,
    const double*          __restrict__ V_ud_im,
    int Nd_d, int ncol)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nd_d) return;

    int Nd_d_spinor = 2 * Nd_d;

    double vre = V_ud_re[i];
    double vim = V_ud_im[i];

    for (int n = blockIdx.y; n < ncol; n += gridDim.y) {
        cuDoubleComplex* Hpsi_up = Hpsi + n * Nd_d_spinor;
        cuDoubleComplex* Hpsi_dn = Hpsi_up + Nd_d;

        const cuDoubleComplex* psi_up = psi + n * Nd_d_spinor;
        const cuDoubleComplex* psi_dn = psi_up + Nd_d;

        // V_ud * psi_dn
        cuDoubleComplex pd = psi_dn[i];
        double h_up_re = vre * pd.x - vim * pd.y;
        double h_up_im = vre * pd.y + vim * pd.x;
        Hpsi_up[i] = cuCadd(Hpsi_up[i], make_cuDoubleComplex(h_up_re, h_up_im));

        // conj(V_ud) * psi_up = (vre - i*vim) * psi_up
        cuDoubleComplex pu = psi_up[i];
        double h_dn_re = vre * pu.x + vim * pu.y;
        double h_dn_im = vre * pu.y - vim * pu.x;
        Hpsi_dn[i] = cuCadd(Hpsi_dn[i], make_cuDoubleComplex(h_dn_re, h_dn_im));
    }
}


/* =========================================================================
 * Kernel 4: spinor_density_kernel
 *
 * Compute charge density and magnetization from spinor wavefunctions:
 *   rho[i]   += weight * sum_n occ[n] * (|psi_up|^2 + |psi_dn|^2)
 *   mag_x[i] += weight * sum_n occ[n] * 2 * Re(conj(psi_up) * psi_dn)
 *   mag_y[i] -= weight * sum_n occ[n] * 2 * Im(conj(psi_up) * psi_dn)
 *   mag_z[i] += weight * sum_n occ[n] * (|psi_up|^2 - |psi_dn|^2)
 * ========================================================================= */

__global__
void spinor_density_kernel(
    const cuDoubleComplex* __restrict__ psi,
    const double*          __restrict__ occ,
    double*                __restrict__ rho,
    double*                __restrict__ mag_x,
    double*                __restrict__ mag_y,
    double*                __restrict__ mag_z,
    int Nd_d, int Nband, double weight)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nd_d) return;

    int Nd_d_spinor = 2 * Nd_d;

    double rho_acc = 0.0;
    double mx_acc  = 0.0;
    double my_acc  = 0.0;
    double mz_acc  = 0.0;

    for (int n = 0; n < Nband; ++n) {
        double fn = occ[n];
        if (fn < 1e-16) continue;

        double w = weight * fn;

        const cuDoubleComplex* psi_up = psi + n * Nd_d_spinor;
        const cuDoubleComplex* psi_dn = psi_up + Nd_d;

        cuDoubleComplex pu = psi_up[i];
        cuDoubleComplex pd = psi_dn[i];

        double up2 = pu.x * pu.x + pu.y * pu.y; // |psi_up|^2
        double dn2 = pd.x * pd.x + pd.y * pd.y; // |psi_dn|^2

        // conj(psi_up) * psi_dn = (pu.x - i*pu.y) * (pd.x + i*pd.y)
        double cross_re = pu.x * pd.x + pu.y * pd.y;
        double cross_im = pu.x * pd.y - pu.y * pd.x;

        rho_acc += w * (up2 + dn2);
        mx_acc  += w * 2.0 * cross_re;
        my_acc  -= w * 2.0 * cross_im;
        mz_acc  += w * (up2 - dn2);
    }

    atomicAdd(&rho[i],   rho_acc);
    atomicAdd(&mag_x[i], mx_acc);
    atomicAdd(&mag_y[i], my_acc);
    atomicAdd(&mag_z[i], mz_acc);
}


/* =========================================================================
 * Host function: soc_apply_z_gpu
 *
 * Orchestrates the full SOC application:
 *   1. Zero alpha_up, alpha_dn
 *   2. Launch gather kernel (inner products)
 *   3. Launch scatter kernel (Term 1 + Term 2)
 * ========================================================================= */

void spinor_offdiag_veff_gpu(
    cuDoubleComplex* d_Hpsi, const cuDoubleComplex* d_psi,
    const double* d_V_ud_re, const double* d_V_ud_im,
    int Nd_d, int ncol,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (Nd_d + threads - 1) / threads;
    for (int n = 0; n < ncol; ++n) {
        spinor_offdiag_veff_kernel<<<blocks, threads, 0, stream>>>(
            d_Hpsi + n * 2 * Nd_d, d_psi + n * 2 * Nd_d,
            d_V_ud_re, d_V_ud_im, Nd_d, 1);
    }
}

void soc_apply_z_gpu(
    const cuDoubleComplex* d_psi, cuDoubleComplex* d_Hpsi,
    // SOC projector data (all on device)
    const cuDoubleComplex* d_Chi_soc_flat, const int* d_gpos_flat,
    const int* d_gpos_offsets, const int* d_chi_soc_offsets,
    const int* d_ndc_arr, const int* d_nproj_soc_arr,
    const int* d_IP_displ_soc,
    const double* d_Gamma_soc,
    const int* d_proj_l, const int* d_proj_m,
    const double* d_bloch_fac,
    cuDoubleComplex* d_alpha_up, cuDoubleComplex* d_alpha_dn,
    int Nd_d, int ncol, double dV,
    int n_influence, int total_soc_nproj,
    int max_ndc_soc, int max_nproj_soc,
    cudaStream_t stream)
{
    if (n_influence == 0 || total_soc_nproj == 0) return;

    // 1. Zero alpha arrays
    cudaMemsetAsync(d_alpha_up, 0, total_soc_nproj * ncol * sizeof(cuDoubleComplex), stream);
    cudaMemsetAsync(d_alpha_dn, 0, total_soc_nproj * ncol * sizeof(cuDoubleComplex), stream);

    // 2. Launch gather kernel: one block per influence atom
    {
        int threads = 256;
        if (max_nproj_soc * ncol < threads)
            threads = ((max_nproj_soc * ncol + 31) / 32) * 32;
        if (threads < 32) threads = 32;

        dim3 grid(n_influence);
        dim3 block(threads);

        soc_gather_alpha_z_kernel<<<grid, block, 0, stream>>>(
            d_psi, d_Chi_soc_flat, d_gpos_flat,
            d_gpos_offsets, d_chi_soc_offsets,
            d_ndc_arr, d_nproj_soc_arr, d_IP_displ_soc,
            d_bloch_fac,
            d_alpha_up, d_alpha_dn,
            Nd_d, ncol, ncol, 0, dV, n_influence);
    }

    cudaStreamSynchronize(stream);

    // 3. Launch scatter kernel: one block per influence atom
    //    The scatter kernel uses the same bloch_fac array as the gather kernel
    //    but internally negates the sin component to form the conjugate phase.
    {
        int threads = 256;
        if (max_ndc_soc < threads)
            threads = ((max_ndc_soc + 31) / 32) * 32;
        if (threads < 32) threads = 32;

        dim3 grid(n_influence);
        dim3 block(threads);

        soc_scatter_z_kernel<<<grid, block, 0, stream>>>(
            d_Hpsi, d_Chi_soc_flat, d_gpos_flat,
            d_gpos_offsets, d_chi_soc_offsets,
            d_ndc_arr, d_nproj_soc_arr, d_IP_displ_soc,
            d_bloch_fac,   // scatter kernel negates sin internally for conjugate
            d_alpha_up, d_alpha_dn,
            d_Gamma_soc, d_proj_l, d_proj_m,
            Nd_d, ncol, ncol, 0, n_influence);
    }

    cudaStreamSynchronize(stream);
}


/* =========================================================================
 * Host function: spinor_density_gpu
 *
 * Launch the spinor density kernel.
 * ========================================================================= */

void spinor_density_gpu(
    const cuDoubleComplex* d_psi, const double* d_occ,
    double* d_rho, double* d_mag_x, double* d_mag_y, double* d_mag_z,
    int Nd_d, int Nband, double weight,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (Nd_d + threads - 1) / threads;

    spinor_density_kernel<<<blocks, threads, 0, stream>>>(
        d_psi, d_occ, d_rho, d_mag_x, d_mag_y, d_mag_z,
        Nd_d, Nband, weight);

    cudaStreamSynchronize(stream);
}


} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
