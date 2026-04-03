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

#include "core/GPUContext.cuh"
#include "core/gpu_common.cuh"
#include "physics/GPUStress.cuh"
#include "physics/GPUKernelUtils.cuh"
#include "parallel/HaloExchange.cuh"
#include "operators/Gradient.cuh"
#include "operators/ComplexOperators.cuh"

namespace lynx {
namespace gpu {

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

// Spinor extract kernel (defined in GPUForce.cu, external linkage via separable compilation)
__global__ void soc_force_spinor_extract_kernel(
    const cuDoubleComplex* __restrict__ spinor,
    cuDoubleComplex* __restrict__ out,
    int Nd, int ncol, int component);

// ============================================================
// Host function: compute_soc_stress_gpu
//
// SOC nonlocal stress using spinor wavefunctions.
// Algorithm:
//   1. Extract psi_up/dn from spinor layout
//   2. Compute alpha_up/dn = bloch * dV * Chi_soc^T * psi_up/dn
//   3. Download alpha to host, compute SOC energy_nl on CPU
//   4. For each Voigt pair (dim, dim2):
//      a. Complex halo exchange + gradient for psi_up/dn in direction dim
//      b. Download gradient to host
//      c. Compute position-weighted beta on CPU
//      d. Accumulate stress reduction on CPU
//   5. Apply spn_fac, subtract energy_soc from diagonal
// ============================================================
void compute_soc_stress_gpu(
    const cuDoubleComplex* d_psi_spinor,  // (2*Nd_d, Nband)
    const double* d_occ,
    // SOC data (all on device)
    const cuDoubleComplex* d_Chi_soc_flat, const int* d_gpos_flat,
    const int* d_gpos_offsets_soc, const int* d_chi_soc_offsets,
    const int* d_ndc_arr_soc, const int* d_nproj_soc_arr,
    const int* d_IP_displ_soc, const double* d_Gamma_soc,
    const int* d_proj_l, const int* d_proj_m,
    const double* d_bloch_fac,
    int n_influence_soc, int total_soc_nproj,
    int max_ndc_soc, int max_nproj_soc,
    int n_phys_atoms,
    const int* h_IP_displ_phys_soc,
    // Atom positions for position-weighted gather (host)
    const double* h_atom_pos_soc,  // [n_influence_soc * 3]
    // Grid parameters
    int nx, int ny, int nz, int FDn, int Nd_d, int Nband,
    double dV, double dx, double dy, double dz,
    int xs, int ys, int zs,
    bool is_orth,
    const double* uvec,      // [9] lattice vectors (row-major) for nonCart2Cart_coord, may be null if orth
    const double* uvec_inv,  // [9] inverse lat vectors for nonCart2Cart_grad, may be null if orth
    double kx_Lx, double ky_Ly, double kz_Lz,
    // Host-side proj info for reduction
    const int* h_proj_l, const int* h_proj_m,
    const double* h_Gamma_soc,
    // Host-side SOC projector data for position-weighted beta (complex)
    const std::complex<double>* h_Chi_soc_flat,
    const int* h_gpos_flat,
    const int* h_gpos_offsets_soc,
    const int* h_chi_soc_offsets,
    const int* h_ndc_arr_soc,
    const int* h_nproj_soc_arr,
    const int* h_IP_displ_soc_inf,  // per-influence-atom IP displ
    const double* h_bloch_fac,      // [n_influence_soc * 2] (cos, sin)
    double spn_fac, double wk,
    // Output (host)
    double* h_stress_soc,   // [6] Voigt stress
    double* h_energy_soc,
    cudaStream_t stream)   // scalar SOC energy for diagonal
{
    if (n_influence_soc == 0 || total_soc_nproj == 0) {
        std::memset(h_stress_soc, 0, 6 * sizeof(double));
        *h_energy_soc = 0.0;
        return;
    }

    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nz_ex = nz + 2 * FDn;
    int Nd_ex = nx_ex * ny_ex * nz_ex;
    // ----------------------------------------------------------------
    // Allocate GPU scratch buffers
    // ----------------------------------------------------------------
    size_t alpha_elems = (size_t)total_soc_nproj * Nband;
    size_t alpha_bytes = alpha_elems * sizeof(cuDoubleComplex);

    cuDoubleComplex* d_alpha_up = nullptr;
    cuDoubleComplex* d_alpha_dn = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_alpha_up, alpha_bytes, 0));
    CUDA_CHECK(cudaMallocAsync(&d_alpha_dn, alpha_bytes, 0));

    // Extracted psi_up/dn (Nd_d * Nband complex each)
    size_t psi_comp_bytes = (size_t)Nd_d * Nband * sizeof(cuDoubleComplex);
    cuDoubleComplex* d_psi_up = nullptr;
    cuDoubleComplex* d_psi_dn = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_psi_up, psi_comp_bytes, 0));
    CUDA_CHECK(cudaMallocAsync(&d_psi_dn, psi_comp_bytes, 0));

    // Gradient output buffers
    cuDoubleComplex* d_Dpsi_up = nullptr;
    cuDoubleComplex* d_Dpsi_dn = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_Dpsi_up, psi_comp_bytes, 0));
    CUDA_CHECK(cudaMallocAsync(&d_Dpsi_dn, psi_comp_bytes, 0));

    // Halo exchange buffer
    size_t ex_bytes = (size_t)Nd_ex * Nband * sizeof(cuDoubleComplex);
    cuDoubleComplex* d_x_ex = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_x_ex, ex_bytes, 0));

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
    // ----------------------------------------------------------------
    CUDA_CHECK(cudaMemset(d_alpha_up, 0, alpha_bytes));
    CUDA_CHECK(cudaMemset(d_alpha_dn, 0, alpha_bytes));

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

    // Download alpha and occupations to host
    std::vector<cuDoubleComplex> h_alpha_up(alpha_elems);
    std::vector<cuDoubleComplex> h_alpha_dn(alpha_elems);
    std::vector<double> h_occ(Nband);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(h_alpha_up.data(), d_alpha_up, alpha_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_alpha_dn.data(), d_alpha_dn, alpha_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_occ.data(), d_occ, Nband * sizeof(double), cudaMemcpyDeviceToHost, stream));

    // ----------------------------------------------------------------
    // Step 3: Compute SOC energy_nl on CPU
    // ----------------------------------------------------------------
    double energy_soc = 0.0;
    for (int ia = 0; ia < n_phys_atoms; ++ia) {
        int off = h_IP_displ_phys_soc[ia];
        int nproj = h_IP_displ_phys_soc[ia + 1] - off;
        if (nproj == 0) continue;

        for (int n = 0; n < Nband; ++n) {
            double g_n = h_occ[n];
            if (fabs(g_n) < 1e-15) continue;

            for (int jp = 0; jp < nproj; ++jp) {
                int glob_jp = off + jp;
                int l = h_proj_l[glob_jp];
                int m = h_proj_m[glob_jp];
                double gamma_soc = h_Gamma_soc[glob_jp];
                int idx = glob_jp * Nband + n;

                // Term 1: 0.5 * m * gamma_soc * (|alpha_up|^2 - |alpha_dn|^2)
                if (m != 0) {
                    double norm_up = h_alpha_up[idx].x * h_alpha_up[idx].x
                                   + h_alpha_up[idx].y * h_alpha_up[idx].y;
                    double norm_dn = h_alpha_dn[idx].x * h_alpha_dn[idx].x
                                   + h_alpha_dn[idx].y * h_alpha_dn[idx].y;
                    energy_soc += wk * g_n * 0.5 * (double)m * gamma_soc * (norm_up - norm_dn);
                }

                // Term 2: L+S- (m -> m+1)
                if (m + 1 <= l) {
                    double ladder = sqrt((double)(l * (l + 1) - m * (m + 1)));
                    int idx_s = (off + jp + 1) * Nband + n;
                    // Re(conj(alpha_up[m]) * alpha_dn[m+1])
                    double re_val = h_alpha_up[idx].x * h_alpha_dn[idx_s].x
                                  + h_alpha_up[idx].y * h_alpha_dn[idx_s].y;
                    energy_soc += wk * g_n * 0.5 * ladder * gamma_soc * re_val;
                }

                // Term 2: L-S+ (m -> m-1)
                if (m - 1 >= -l) {
                    double ladder = sqrt((double)(l * (l + 1) - m * (m - 1)));
                    int idx_s = (off + jp - 1) * Nband + n;
                    // Re(conj(alpha_dn[m]) * alpha_up[m-1])
                    double re_val = h_alpha_dn[idx].x * h_alpha_up[idx_s].x
                                  + h_alpha_dn[idx].y * h_alpha_up[idx_s].y;
                    energy_soc += wk * g_n * 0.5 * ladder * gamma_soc * re_val;
                }
            }
        }
    }

    // ----------------------------------------------------------------
    // Step 4: For each Voigt pair, compute gradient on GPU, then
    //         position-weighted beta and stress reduction on CPU
    // ----------------------------------------------------------------
    double snl[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // We need the gradient for each direction (0,1,2). To avoid recomputing,
    // cache gradients: compute all 3 directions once, download, then do
    // all 6 Voigt pairs on CPU.
    // Each gradient: Nd_d * Nband * sizeof(cuDoubleComplex) = Nd_d * Nband * 16 bytes
    std::vector<cuDoubleComplex> h_Dpsi_up_all[3];
    std::vector<cuDoubleComplex> h_Dpsi_dn_all[3];

    for (int dim = 0; dim < 3; ++dim) {
        // Complex halo exchange + gradient for psi_up
        halo_exchange_z_gpu(d_psi_up, d_x_ex, nx, ny, nz, FDn, Nband,
                            true, true, true, kx_Lx, ky_Ly, kz_Lz, stream);
        gradient_z_gpu(d_x_ex, d_Dpsi_up, nx, ny, nz, FDn, nx_ex, ny_ex, dim, Nband, stream);

        // Complex halo exchange + gradient for psi_dn
        halo_exchange_z_gpu(d_psi_dn, d_x_ex, nx, ny, nz, FDn, Nband,
                            true, true, true, kx_Lx, ky_Ly, kz_Lz, stream);
        gradient_z_gpu(d_x_ex, d_Dpsi_dn, nx, ny, nz, FDn, nx_ex, ny_ex, dim, Nband, stream);

        // Download gradients to host
        h_Dpsi_up_all[dim].resize((size_t)Nd_d * Nband);
        h_Dpsi_dn_all[dim].resize((size_t)Nd_d * Nband);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpyAsync(h_Dpsi_up_all[dim].data(), d_Dpsi_up, psi_comp_bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_Dpsi_dn_all[dim].data(), d_Dpsi_dn, psi_comp_bytes, cudaMemcpyDeviceToHost, stream));
    }

    // For non-orthogonal: transform gradients to Cartesian
    // dpsi_cart[dim] = sum_d uvec_inv[dim][d] * dpsi_nc[d]
    // Do this transform once for all 3 Cartesian directions
    std::vector<cuDoubleComplex> h_Dpsi_up_cart[3], h_Dpsi_dn_cart[3];
    if (!is_orth && uvec_inv) {
        for (int dim = 0; dim < 3; dim++) {
            h_Dpsi_up_cart[dim].resize((size_t)Nd_d * Nband);
            h_Dpsi_dn_cart[dim].resize((size_t)Nd_d * Nband);
            for (int idx = 0; idx < Nd_d * Nband; idx++) {
                cuDoubleComplex d0_u = h_Dpsi_up_all[0][idx], d1_u = h_Dpsi_up_all[1][idx], d2_u = h_Dpsi_up_all[2][idx];
                cuDoubleComplex d0_d = h_Dpsi_dn_all[0][idx], d1_d = h_Dpsi_dn_all[1][idx], d2_d = h_Dpsi_dn_all[2][idx];
                h_Dpsi_up_cart[dim][idx] = make_cuDoubleComplex(
                    uvec_inv[dim*3+0]*d0_u.x + uvec_inv[dim*3+1]*d1_u.x + uvec_inv[dim*3+2]*d2_u.x,
                    uvec_inv[dim*3+0]*d0_u.y + uvec_inv[dim*3+1]*d1_u.y + uvec_inv[dim*3+2]*d2_u.y);
                h_Dpsi_dn_cart[dim][idx] = make_cuDoubleComplex(
                    uvec_inv[dim*3+0]*d0_d.x + uvec_inv[dim*3+1]*d1_d.x + uvec_inv[dim*3+2]*d2_d.x,
                    uvec_inv[dim*3+0]*d0_d.y + uvec_inv[dim*3+1]*d1_d.y + uvec_inv[dim*3+2]*d2_d.y);
            }
        }
    }

    // CPU: for each Voigt pair (dim, dim2), compute position-weighted beta and reduce
    int cnt = 0;
    for (int dim = 0; dim < 3; ++dim) {
        for (int dim2 = dim; dim2 < 3; ++dim2) {
            const cuDoubleComplex* h_dpsi_up = is_orth ? h_Dpsi_up_all[dim].data() : h_Dpsi_up_cart[dim].data();
            const cuDoubleComplex* h_dpsi_dn = is_orth ? h_Dpsi_dn_all[dim].data() : h_Dpsi_dn_cart[dim].data();

            // Compute beta_soc_up/dn: bloch * dV * Chi_soc^T * (xR_dim2 * dpsi_dim)
            std::vector<cuDoubleComplex> beta_soc_up(alpha_elems, make_cuDoubleComplex(0.0, 0.0));
            std::vector<cuDoubleComplex> beta_soc_dn(alpha_elems, make_cuDoubleComplex(0.0, 0.0));

            for (int iat = 0; iat < n_influence_soc; ++iat) {
                int ndc = h_ndc_arr_soc[iat];
                int np = h_nproj_soc_arr[iat];
                if (ndc == 0 || np == 0) continue;

                int goff = h_gpos_offsets_soc[iat];
                int coff = h_chi_soc_offsets[iat];
                int abase = h_IP_displ_soc_inf[iat];

                double ap_x = h_atom_pos_soc[iat * 3 + 0];
                double ap_y = h_atom_pos_soc[iat * 3 + 1];
                double ap_z = h_atom_pos_soc[iat * 3 + 2];

                // Bloch factor for this influence atom
                double bloch_cos = h_bloch_fac[iat * 2 + 0];
                double bloch_sin = h_bloch_fac[iat * 2 + 1];
                // beta_scale = bloch * dV
                double bs_re = bloch_cos * dV;
                double bs_im = bloch_sin * dV;

                for (int jp = 0; jp < np; ++jp) {
                    for (int n = 0; n < Nband; ++n) {
                        double dot_up_re = 0.0, dot_up_im = 0.0;
                        double dot_dn_re = 0.0, dot_dn_im = 0.0;

                        for (int ig = 0; ig < ndc; ++ig) {
                            int flat = h_gpos_flat[goff + ig];
                            int li = flat % nx;
                            int lj = (flat / nx) % ny;

                            int lk = flat / (nx * ny);
                            double r1 = (li + xs) * dx - ap_x;
                            double r2 = (lj + ys) * dy - ap_y;
                            double r3 = (lk + zs) * dz - ap_z;
                            // For non-orth: transform to Cartesian
                            if (!is_orth && uvec) {
                                double a = r1, b = r2, c = r3;
                                r1 = uvec[0]*a + uvec[3]*b + uvec[6]*c;
                                r2 = uvec[1]*a + uvec[4]*b + uvec[7]*c;
                                r3 = uvec[2]*a + uvec[5]*b + uvec[8]*c;
                            }
                            double xR;
                            if (dim2 == 0) xR = r1;
                            else if (dim2 == 1) xR = r2;
                            else xR = r3;

                            std::complex<double> chi_val = std::conj(h_Chi_soc_flat[coff + jp * ndc + ig]);
                            // conj(chi) * xR: real-scale the complex conjugate
                            double w_re = chi_val.real() * xR;
                            double w_im = chi_val.imag() * xR;

                            // dpsi_up[gpos[ig] + n * Nd_d]
                            int psi_idx = n * Nd_d + flat;
                            cuDoubleComplex dp_up = h_dpsi_up[psi_idx];
                            cuDoubleComplex dp_dn = h_dpsi_dn[psi_idx];

                            // (w_re + i*w_im) * (dp.x + i*dp.y)
                            dot_up_re += w_re * dp_up.x - w_im * dp_up.y;
                            dot_up_im += w_re * dp_up.y + w_im * dp_up.x;
                            dot_dn_re += w_re * dp_dn.x - w_im * dp_dn.y;
                            dot_dn_im += w_re * dp_dn.y + w_im * dp_dn.x;
                        }

                        // Multiply by beta_scale = bloch * dV (complex)
                        int out_idx = (abase + jp) * Nband + n;
                        beta_soc_up[out_idx].x += bs_re * dot_up_re - bs_im * dot_up_im;
                        beta_soc_up[out_idx].y += bs_re * dot_up_im + bs_im * dot_up_re;
                        beta_soc_dn[out_idx].x += bs_re * dot_dn_re - bs_im * dot_dn_im;
                        beta_soc_dn[out_idx].y += bs_re * dot_dn_im + bs_im * dot_dn_re;
                    }
                }
            }

            // Accumulate SOC nonlocal stress for this Voigt pair
            for (int ia = 0; ia < n_phys_atoms; ++ia) {
                int off = h_IP_displ_phys_soc[ia];
                int nproj = h_IP_displ_phys_soc[ia + 1] - off;
                if (nproj == 0) continue;

                for (int n = 0; n < Nband; ++n) {
                    double g_n = h_occ[n];
                    if (fabs(g_n) < 1e-15) continue;

                    for (int jp = 0; jp < nproj; ++jp) {
                        int glob_jp = off + jp;
                        int l = h_proj_l[glob_jp];
                        int m = h_proj_m[glob_jp];
                        double gamma_soc = h_Gamma_soc[glob_jp];
                        int idx = glob_jp * Nband + n;

                        // Term 1: on-diagonal (Lz*Sz)
                        if (m != 0) {
                            double re_au_bu = h_alpha_up[idx].x * beta_soc_up[idx].x
                                            + h_alpha_up[idx].y * beta_soc_up[idx].y;
                            double re_ad_bd = h_alpha_dn[idx].x * beta_soc_dn[idx].x
                                            + h_alpha_dn[idx].y * beta_soc_dn[idx].y;
                            snl[cnt] -= 0.5 * (double)m * gamma_soc * (re_au_bu - re_ad_bd) * wk * g_n;
                        }

                        // Term 2: L+S- (m -> m+1)
                        if (m + 1 <= l) {
                            double ladder = sqrt((double)(l * (l + 1) - m * (m + 1)));
                            int idx_s = (off + jp + 1) * Nband + n;
                            double re_val = h_alpha_up[idx].x * beta_soc_dn[idx_s].x
                                          + h_alpha_up[idx].y * beta_soc_dn[idx_s].y;
                            snl[cnt] -= 0.5 * ladder * gamma_soc * re_val * wk * g_n;
                        }

                        // Term 2: L-S+ (m -> m-1)
                        if (m - 1 >= -l) {
                            double ladder = sqrt((double)(l * (l + 1) - m * (m - 1)));
                            int idx_s = (off + jp - 1) * Nband + n;
                            double re_val = h_alpha_dn[idx].x * beta_soc_up[idx_s].x
                                          + h_alpha_dn[idx].y * beta_soc_up[idx_s].y;
                            snl[cnt] -= 0.5 * ladder * gamma_soc * re_val * wk * g_n;
                        }
                    }
                }
            }
            cnt++;
        }
    }

    // ----------------------------------------------------------------
    // Step 5: Apply spn_fac, output results
    // ----------------------------------------------------------------
    for (int i = 0; i < 6; ++i)
        snl[i] *= spn_fac;

    // energy_soc scaled by spn_fac (=2.0 for SOC spinor)
    energy_soc *= spn_fac;

    // Subtract energy_soc from diagonal: xx(0), yy(3), zz(5)
    h_stress_soc[0] = snl[0] - energy_soc;
    h_stress_soc[1] = snl[1];
    h_stress_soc[2] = snl[2];
    h_stress_soc[3] = snl[3] - energy_soc;
    h_stress_soc[4] = snl[4];
    h_stress_soc[5] = snl[5] - energy_soc;

    *h_energy_soc = energy_soc;

    // Free GPU scratch
    cudaFreeAsync(d_alpha_up, 0);
    cudaFreeAsync(d_alpha_dn, 0);
    cudaFreeAsync(d_psi_up, 0);
    cudaFreeAsync(d_psi_dn, 0);
    cudaFreeAsync(d_Dpsi_up, 0);
    cudaFreeAsync(d_Dpsi_dn, 0);
    cudaFreeAsync(d_x_ex, 0);
}

// ============================================================
// mGGA psi stress kernel
// Like kinetic_stress_kernel but weighted by vtau:
// sk[voigt] = -occfac * dV * sum_n(g_n * sum_i(vtau[i] * dpsi_a[i,n] * dpsi_b[i,n]))
// ============================================================
// mGGA psi stress kernel for non-orthogonal cells.
// Computes all 6 Voigt components in one launch, transforming lattice-frame
// gradients to Cartesian via uvec_inv.
// One block per band; threads reduce over grid points.
__global__ void mgga_psi_stress_kernel_nonorth(
    const double* __restrict__ d_Dpsi_0,   // (Nd, Nband) gradient along lattice dir 0
    const double* __restrict__ d_Dpsi_1,   // (Nd, Nband) gradient along lattice dir 1
    const double* __restrict__ d_Dpsi_2,   // (Nd, Nband) gradient along lattice dir 2
    const double* __restrict__ d_vtau,     // (Nd) vtau for this spin
    const double* __restrict__ d_occ,      // (Nband) occupations
    double* __restrict__ d_stress,         // [6] Voigt stress output (xx,xy,xz,yy,yz,zz)
    int Nd, int Nband,
    double neg_occfac_dV,
    // uvec_inv stored row-major: [a*3+j] = uvec_inv(a,j)
    double uinv00, double uinv01, double uinv02,
    double uinv10, double uinv11, double uinv12,
    double uinv20, double uinv21, double uinv22)
{
    int band = blockIdx.x;
    if (band >= Nband) return;

    double g_n = d_occ[band];

    extern __shared__ double sdata[];

    // Each thread accumulates 6 Voigt sums
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0;

    const double* p0 = d_Dpsi_0 + band * Nd;
    const double* p1 = d_Dpsi_1 + band * Nd;
    const double* p2 = d_Dpsi_2 + band * Nd;

    for (int i = threadIdx.x; i < Nd; i += blockDim.x) {
        double v = d_vtau[i];
        double d0 = p0[i], d1 = p1[i], d2 = p2[i];

        // Transform to Cartesian: ga = uvec_inv(a,:) · (d0, d1, d2)
        double gx = uinv00*d0 + uinv01*d1 + uinv02*d2;
        double gy = uinv10*d0 + uinv11*d1 + uinv12*d2;
        double gz = uinv20*d0 + uinv21*d1 + uinv22*d2;

        s0 += v * gx * gx;  // xx
        s1 += v * gx * gy;  // xy
        s2 += v * gx * gz;  // xz
        s3 += v * gy * gy;  // yy
        s4 += v * gy * gz;  // yz
        s5 += v * gz * gz;  // zz
    }

    // Reduce each Voigt component sequentially using shared memory
    double sums[6] = {s0, s1, s2, s3, s4, s5};
    for (int v = 0; v < 6; v++) {
        sdata[threadIdx.x] = sums[v];
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            atomicAdd(&d_stress[v], neg_occfac_dV * g_n * sdata[0]);
        }
    }
}

// mGGA psi stress kernel for orthogonal cells (single Voigt pair per launch).
__global__ void mgga_psi_stress_kernel(
    const double* __restrict__ d_Dpsi_a,   // (Nd, Nband) gradient in direction a
    const double* __restrict__ d_Dpsi_b,   // (Nd, Nband) gradient in direction b
    const double* __restrict__ d_vtau,     // (Nd) vtau for this spin
    const double* __restrict__ d_occ,      // (Nband) occupations
    double* __restrict__ d_sk_out,         // [1] partial result for this Voigt pair
    int Nd, int Nband,
    double neg_occfac_dV)                  // = -occfac * dV
{
    int band = blockIdx.x;
    if (band >= Nband) return;

    double g_n = d_occ[band];

    extern __shared__ double sdata[];

    double local_dot = 0.0;
    const double* pa = d_Dpsi_a + band * Nd;
    const double* pb = d_Dpsi_b + band * Nd;
    for (int i = threadIdx.x; i < Nd; i += blockDim.x) {
        local_dot += d_vtau[i] * pa[i] * pb[i];
    }

    sdata[threadIdx.x] = local_dot;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(d_sk_out, neg_occfac_dV * g_n * sdata[0]);
    }
}

// ============================================================
// GPU dot product reduction kernel: result = sum(a[i] * b[i])
// ============================================================
__global__ void dot_product_reduce_kernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ result,
    int N)
{
    extern __shared__ double sdata[];

    double local_sum = 0.0;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += blockDim.x * gridDim.x) {
        local_sum += a[i] * b[i];
    }

    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// ============================================================
// Host function: compute_mgga_stress_gpu
// Computes mGGA psi stress on GPU (avoids downloading psi/vtau)
// Also computes tau_vtau_dot = ∫ τ·vtau dV
// ============================================================
void compute_mgga_stress_gpu(
    const double* d_psi,       // (Nd, Nband) wavefunctions on GPU
    const double* d_occ,       // (Nband) occupations on GPU
    const double* d_vtau,      // (Nd) vtau for this spin channel
    const double* d_tau,       // (Nd * tau_len) tau array on GPU
    const double* d_vtau_full, // (Nd * vtau_len) vtau array on GPU
    int nx, int ny, int nz, int FDn, int Nd, int Nband,
    double dV, double occfac,
    int tau_dot_len,           // number of elements for tau·vtau dot (Nd for nospin, 2*Nd for spin)
    bool is_orth,              // true if orthogonal cell
    const double* uvec_inv,    // [9] row-major uvec_inv matrix (only used if !is_orth)
    // Output (host)
    double* h_stress_mgga,     // [6] mGGA psi stress (Voigt)
    double* h_tau_vtau_dot,
    cudaStream_t stream)    // scalar: ∫ τ·vtau dV
{
    auto& ctx = GPUContext::instance();

    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;

    // Allocate gradient arrays
    size_t grad_size = (size_t)Nd * Nband;
    size_t grad_bytes = grad_size * sizeof(double);

    double* d_Dpsi_x = nullptr;
    double* d_Dpsi_y = nullptr;
    double* d_Dpsi_z = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_Dpsi_x, grad_bytes, 0));
    CUDA_CHECK(cudaMallocAsync(&d_Dpsi_y, grad_bytes, 0));
    CUDA_CHECK(cudaMallocAsync(&d_Dpsi_z, grad_bytes, 0));

    // Scratch pool for small buffers
    size_t scratch_cp = ctx.scratch_pool.checkpoint();

    // mGGA psi stress: 6 Voigt components
    double* d_smgga = ctx.scratch_pool.alloc<double>(6);
    CUDA_CHECK(cudaMemset(d_smgga, 0, 6 * sizeof(double)));

    // tau·vtau dot product result
    double* d_dot = ctx.scratch_pool.alloc<double>(1);
    CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(double)));

    // Halo exchange + gradients (batched V2 — single launch per direction)
    double* d_x_ex = ctx.buf.x_ex;
    halo_exchange_batched_nomemset_gpu(d_psi, d_x_ex, nx, ny, nz, FDn, Nband, true, true, true, stream);
    gradient_v3_gpu(d_x_ex, d_Dpsi_x, nx, ny, nz, FDn, nx_ex, ny_ex, 0, Nband, stream);
    gradient_v3_gpu(d_x_ex, d_Dpsi_y, nx, ny, nz, FDn, nx_ex, ny_ex, 1, Nband, stream);
    gradient_v3_gpu(d_x_ex, d_Dpsi_z, nx, ny, nz, FDn, nx_ex, ny_ex, 2, Nband, stream);

    // Compute mGGA psi stress: 6 Voigt pairs
    {
        int bs = 256;
        size_t smem = bs * sizeof(double);
        double neg_occfac_dV = -occfac * dV;

        if (is_orth) {
            const double* d_Dpsi[3] = { d_Dpsi_x, d_Dpsi_y, d_Dpsi_z };
            int voigt_a[6] = {0, 0, 0, 1, 1, 2};
            int voigt_b[6] = {0, 1, 2, 1, 2, 2};

            for (int v = 0; v < 6; ++v) {
                mgga_psi_stress_kernel<<<Nband, bs, smem, stream>>>(
                    d_Dpsi[voigt_a[v]], d_Dpsi[voigt_b[v]],
                    d_vtau, d_occ, d_smgga + v, Nd, Nband, neg_occfac_dV);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            // Non-orthogonal: single kernel computes all 6 Voigt with uvec_inv transform
            mgga_psi_stress_kernel_nonorth<<<Nband, bs, smem, stream>>>(
                d_Dpsi_x, d_Dpsi_y, d_Dpsi_z,
                d_vtau, d_occ, d_smgga, Nd, Nband, neg_occfac_dV,
                uvec_inv[0], uvec_inv[1], uvec_inv[2],
                uvec_inv[3], uvec_inv[4], uvec_inv[5],
                uvec_inv[6], uvec_inv[7], uvec_inv[8]);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // Compute tau·vtau dot product (skip when tau_dot_len==0, e.g. spin channel 1)
    if (tau_dot_len > 0) {
        int bs = 256;
        int nblocks = std::min((tau_dot_len + bs - 1) / bs, 256);
        size_t smem = bs * sizeof(double);
        dot_product_reduce_kernel<<<nblocks, bs, smem, stream>>>(
            d_tau, d_vtau_full, d_dot, tau_dot_len);
        CUDA_CHECK(cudaGetLastError());
    }

    // Download results
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(h_stress_mgga, d_smgga, 6 * sizeof(double), cudaMemcpyDeviceToHost, stream));

    double dot_val = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(&dot_val, d_dot, sizeof(double), cudaMemcpyDeviceToHost, stream));
    *h_tau_vtau_dot = dot_val * dV;

    // Cleanup
    cudaFreeAsync(d_Dpsi_x, 0);
    cudaFreeAsync(d_Dpsi_y, 0);
    cudaFreeAsync(d_Dpsi_z, 0);
    ctx.scratch_pool.restore(scratch_cp);
}

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
