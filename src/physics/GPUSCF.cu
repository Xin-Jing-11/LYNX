#ifdef USE_CUDA

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>

#include "physics/SCF.hpp"       // for SCFParams definition
#include "physics/GPUSCF.cuh"
#include "core/GPUContext.cuh"
#include "core/gpu_common.cuh"
#include "core/constants.hpp"

// CPU infrastructure for initial Veff computation
#include "operators/Laplacian.hpp"
#include "operators/Gradient.hpp"
#include "solvers/PoissonSolver.hpp"
#include "solvers/EigenSolver.hpp"
#include "electronic/ElectronDensity.hpp"
#include "electronic/Occupation.hpp"
#include "physics/Electrostatics.hpp"

namespace lynx {

// ============================================================
// Forward declarations for GPU functions
// ============================================================
namespace gpu {

void halo_exchange_gpu(
    const double* d_x, double* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol,
    bool periodic_x, bool periodic_y, bool periodic_z);

void halo_exchange_batched_nomemset_gpu(
    const double* d_x, double* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol,
    bool periodic_x, bool periodic_y, bool periodic_z);

void hamiltonian_apply_local_gpu(
    const double* d_psi, const double* d_Veff, double* d_Hpsi,
    double* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol, double c,
    bool is_orthogonal,
    bool periodic_x, bool periodic_y, bool periodic_z,
    double diag_coeff,
    bool has_xy, bool has_xz, bool has_yz);

void upload_stencil_coefficients(
    const double* D2x, const double* D2y, const double* D2z,
    const double* D1x, const double* D1y, const double* D1z,
    const double* D2xy, const double* D2xz, const double* D2yz,
    int FDn);

void eigensolver_solve_gpu(
    double* d_psi, double* d_eigvals, const double* d_Veff,
    double* d_Y, double* d_Xold, double* d_Xnew,
    double* d_HX, double* d_x_ex,
    double* d_Hs, double* d_Ms,
    int Nd, int Ns,
    double lambda_cutoff, double eigval_min, double eigval_max,
    int cheb_degree, double dV,
    void (*apply_H)(const double*, const double*, double*, double*, int));

void nonlocal_projector_apply_gpu(
    const double* d_psi, double* d_Hpsi,
    const double* d_Chi_flat, const int* d_gpos_flat,
    const int* d_gpos_offsets, const int* d_chi_offsets,
    const int* d_ndc_arr, const int* d_nproj_arr,
    const int* d_IP_displ, const double* d_Gamma,
    double* d_alpha,
    int Nd, int ncol, double dV,
    int n_atoms, int total_nproj,
    int max_ndc, int max_nproj);

void compute_density_gpu(const double* d_psi, const double* d_occ, double* d_rho,
                          int Nd, int Ns, double weight);

void laplacian_orth_v2_gpu(
    const double* d_x_ex, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c, double diag_coeff, int ncol);

void gradient_gpu(
    const double* d_x_ex, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    int direction, int ncol);

void gga_pbe_gpu(const double* d_rho, const double* d_sigma,
                  double* d_exc, double* d_vxc, double* d_v2xc, int N);

void lda_pw_gpu(const double* d_rho, double* d_exc, double* d_vxc, int N);
void lda_pz_gpu(const double* d_rho, double* d_exc, double* d_vxc, int N);

int aar_gpu(
    void (*op_gpu)(const double* d_x, double* d_Ax),
    void (*precond_gpu)(const double* d_r, double* d_f),
    const double* d_b, double* d_x, int N,
    double omega, double beta, int m, int p,
    double tol, int max_iter,
    double* d_r, double* d_f, double* d_Ax,
    double* d_X_hist, double* d_F_hist,
    double* d_x_old, double* d_f_old);

void compute_force_stress_gpu(
    const double* d_psi, const double* d_occ,
    const double* d_Chi_flat, const int* d_gpos_flat,
    const int* d_gpos_offsets, const int* d_chi_offsets,
    const int* d_ndc_arr, const int* d_nproj_arr,
    const int* d_IP_displ, const double* d_Gamma,
    int n_influence, int total_nproj, int max_ndc, int max_nproj,
    int n_phys_atoms,
    const int* h_IP_displ_phys, const double* h_atom_pos,
    int nx, int ny, int nz, int FDn, int Nd, int Nband,
    double dV, double dx, double dy, double dz,
    int xs, int ys, int zs, double occfac,
    double* h_f_nloc, double* h_stress_k, double* h_stress_nl, double* h_energy_nl);

// Complex eigensolver (k-point)
void eigensolver_solve_z_gpu(
    cuDoubleComplex* d_psi_z, double* d_eigvals, const double* d_Veff,
    cuDoubleComplex* d_Y_z, cuDoubleComplex* d_Xold_z, cuDoubleComplex* d_Xnew_z,
    cuDoubleComplex* d_HX_z, cuDoubleComplex* d_x_ex_z,
    cuDoubleComplex* d_Hs_z, cuDoubleComplex* d_Ms_z,
    int Nd, int Ns,
    double lambda_cutoff, double eigval_min, double eigval_max,
    int cheb_degree, double dV,
    void (*apply_H_z)(const cuDoubleComplex*, const double*, cuDoubleComplex*, cuDoubleComplex*, int));

void compute_density_z_gpu(const cuDoubleComplex* d_psi, const double* d_occ,
                            double* d_rho, int Nd, int Ns, double weight);

// Complex operators (k-point)
void halo_exchange_z_gpu(
    const cuDoubleComplex* d_x, cuDoubleComplex* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol,
    bool periodic_x, bool periodic_y, bool periodic_z,
    double kx_Lx, double ky_Ly, double kz_Lz);

void hamiltonian_apply_local_z_gpu(
    const cuDoubleComplex* d_psi, const double* d_Veff, cuDoubleComplex* d_Hpsi,
    cuDoubleComplex* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol, double c,
    bool is_orthogonal,
    bool periodic_x, bool periodic_y, bool periodic_z,
    double diag_coeff,
    bool has_xy, bool has_xz, bool has_yz,
    double kx_Lx, double ky_Ly, double kz_Lz);

void nonlocal_projector_apply_z_gpu(
    const cuDoubleComplex* d_psi, cuDoubleComplex* d_Hpsi,
    const double* d_Chi_flat, const int* d_gpos_flat,
    const int* d_gpos_offsets, const int* d_chi_offsets,
    const int* d_ndc_arr, const int* d_nproj_arr,
    const int* d_IP_displ, const double* d_Gamma,
    cuDoubleComplex* d_alpha,
    const double* d_bloch_fac,
    int Nd, int ncol, double dV,
    int n_atoms, int total_nproj,
    int max_ndc, int max_nproj);

// Spin XC
void lda_pw_spin_gpu(const double* d_rho_up, const double* d_rho_dn,
                      double* d_exc, double* d_vxc_up, double* d_vxc_dn, int N);
void gga_pbe_spin_gpu(const double* d_rho, const double* d_sigma,
                       double* d_exc, double* d_vxc, double* d_v2xc, int N);

// SOC operators
void spinor_offdiag_veff_gpu(
    cuDoubleComplex* d_Hpsi, const cuDoubleComplex* d_psi,
    const double* d_V_ud_re, const double* d_V_ud_im,
    int Nd_d, int ncol);

void soc_apply_z_gpu(
    const cuDoubleComplex* d_psi, cuDoubleComplex* d_Hpsi,
    const double* d_Chi_soc_flat, const int* d_gpos_flat,
    const int* d_gpos_offsets, const int* d_chi_soc_offsets,
    const int* d_ndc_arr, const int* d_nproj_soc_arr,
    const int* d_IP_displ_soc,
    const double* d_Gamma_soc,
    const int* d_proj_l, const int* d_proj_m,
    const double* d_bloch_fac,
    cuDoubleComplex* d_alpha_up, cuDoubleComplex* d_alpha_dn,
    int Nd_d, int ncol, double dV,
    int n_influence, int total_soc_nproj,
    int max_ndc_soc, int max_nproj_soc);

void spinor_density_gpu(
    const cuDoubleComplex* d_psi, const double* d_occ,
    double* d_rho, double* d_mag_x, double* d_mag_y, double* d_mag_z,
    int Nd_d, int Nband, double weight);

} // namespace gpu

// ============================================================
// SOC helper kernels (Veff spinor construction)
// ============================================================

// Convert noncollinear (rho, mx, my, mz) to collinear (rho_up, rho_dn) for XC
// |m| = sqrt(mx^2 + my^2 + mz^2)
// rho_up = 0.5*(rho + |m|), rho_dn = 0.5*(rho - |m|)
__global__ void mag_to_collinear_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ mag_x,
    const double* __restrict__ mag_y,
    const double* __restrict__ mag_z,
    double* __restrict__ rho_up,
    double* __restrict__ rho_dn,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double mx = mag_x[i], my = mag_y[i], mz = mag_z[i];
        double mnorm = sqrt(mx*mx + my*my + mz*mz);
        double r = rho[i];
        rho_up[i] = fmax(0.5 * (r + mnorm), 1e-14);
        rho_dn[i] = fmax(0.5 * (r - mnorm), 1e-14);
    }
}

// Construct 4-component Veff_spinor from XC output and magnetization
// V_uu = 0.5*(Vxc_up+Vxc_dn) + 0.5*(Vxc_up-Vxc_dn)*mz/|m| + phi
// V_dd = 0.5*(Vxc_up+Vxc_dn) - 0.5*(Vxc_up-Vxc_dn)*mz/|m| + phi
// V_ud_re = 0.5*(Vxc_up-Vxc_dn) * mx/|m|
// V_ud_im = -0.5*(Vxc_up-Vxc_dn) * my/|m|
__global__ void veff_spinor_from_xc_kernel(
    const double* __restrict__ Vxc_up,
    const double* __restrict__ Vxc_dn,
    const double* __restrict__ phi,
    const double* __restrict__ mag_x,
    const double* __restrict__ mag_y,
    const double* __restrict__ mag_z,
    double* __restrict__ Veff_spinor,  // [V_uu(Nd) | V_dd(Nd) | V_ud_re(Nd) | V_ud_im(Nd)]
    int Nd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Nd) {
        double vup = Vxc_up[i], vdn = Vxc_dn[i];
        double vavg = 0.5 * (vup + vdn);
        double vdiff = 0.5 * (vup - vdn);
        double p = phi[i];
        double mx = mag_x[i], my = mag_y[i], mz = mag_z[i];
        double mnorm = sqrt(mx*mx + my*my + mz*mz);

        if (mnorm > 1e-16) {
            double inv_m = 1.0 / mnorm;
            Veff_spinor[i]          = vavg + vdiff * mz * inv_m + p;  // V_uu
            Veff_spinor[Nd + i]     = vavg - vdiff * mz * inv_m + p;  // V_dd
            Veff_spinor[2*Nd + i]   = vdiff * mx * inv_m;              // V_ud_re
            Veff_spinor[3*Nd + i]   = -vdiff * my * inv_m;             // V_ud_im
        } else {
            // Zero magnetization: diagonal only
            Veff_spinor[i]          = vavg + p;     // V_uu
            Veff_spinor[Nd + i]     = vavg + p;     // V_dd
            Veff_spinor[2*Nd + i]   = 0.0;          // V_ud_re
            Veff_spinor[3*Nd + i]   = 0.0;          // V_ud_im
        }
    }
}

// ============================================================
// GPU kernels for SCF utility operations
// ============================================================

// Veff = Vxc + phi
__global__ void veff_combine_kernel(
    const double* __restrict__ vxc,
    const double* __restrict__ phi,
    double* __restrict__ veff,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) veff[i] = vxc[i] + phi[i];
}

// sigma = |grad(rho)|^2 for orthogonal cells
__global__ void sigma_kernel(
    const double* __restrict__ Drho_x,
    const double* __restrict__ Drho_y,
    const double* __restrict__ Drho_z,
    double* __restrict__ sigma, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double dx = Drho_x[i], dy = Drho_y[i], dz = Drho_z[i];
        sigma[i] = dx*dx + dy*dy + dz*dz;
    }
}

// f[i] *= v2xc[i]
__global__ void v2xc_scale_kernel(
    double* __restrict__ f,
    const double* __restrict__ v2xc, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) f[i] *= v2xc[i];
}

// rho_xc = max(rho + rho_core, 1e-14)
__global__ void nlcc_add_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ rho_core,
    double* __restrict__ rho_xc, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double val = rho[i] + rho_core[i];
        rho_xc[i] = (val > 1e-14) ? val : 1e-14;
    }
}

// Vxc[i] -= DDrho[i]
__global__ void divergence_sub_kernel(
    double* __restrict__ Vxc,
    const double* __restrict__ DDrho, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) Vxc[i] -= DDrho[i];
}

// f[i] = scale * r[i]
__global__ void jacobi_scale_kernel(
    const double* __restrict__ r,
    double* __restrict__ f,
    double scale, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) f[i] = scale * r[i];
}

// rhs = fourpi * (rho + pseudocharge)
__global__ void poisson_rhs_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ pseudocharge,
    double* __restrict__ rhs,
    double fourpi, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) rhs[i] = fourpi * (rho[i] + pseudocharge[i]);
}

// x[i] -= mean
__global__ void mean_subtract_kernel(double* __restrict__ x, double mean, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] -= mean;
}

// f[i] = g[i] - x[i]
__global__ void mix_residual_kernel(
    const double* __restrict__ g,
    const double* __restrict__ x,
    double* __restrict__ f, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) f[i] = g[i] - x[i];
}

// X(:,col) = x - x_old, F(:,col) = f - f_old
__global__ void mix_store_history_kernel(
    const double* __restrict__ x,
    const double* __restrict__ x_old,
    const double* __restrict__ f,
    const double* __restrict__ f_old,
    double* __restrict__ X_hist,
    double* __restrict__ F_hist,
    int col, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        X_hist[col * N + i] = x[i] - x_old[i];
        F_hist[col * N + i] = f[i] - f_old[i];
    }
}

// x[i] = max(x[i], min_val)
__global__ void clamp_min_kernel(double* __restrict__ x, double min_val, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (x[i] < min_val) x[i] = min_val;
    }
}

// x[i] *= scale
__global__ void scale_kernel(double* __restrict__ x, double scale, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] *= scale;
}

// Veff_s = Vxc_s + phi (spin-resolved: Vxc at offset, phi shared)
__global__ void veff_combine_spin_kernel(
    const double* __restrict__ vxc_s,
    const double* __restrict__ phi,
    double* __restrict__ veff_s,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) veff_s[i] = vxc_s[i] + phi[i];
}

// out[i] = a[i] + b[i]
__global__ void add_kernel(const double* __restrict__ a, const double* __restrict__ b,
                           double* __restrict__ out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i] + b[i];
}

// out[i] = a[i] - b[i]
__global__ void sub_kernel(const double* __restrict__ a, const double* __restrict__ b,
                           double* __restrict__ out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i] - b[i];
}

// out_up[i] = 0.5*(total[i] + mag[i]), out_dn[i] = 0.5*(total[i] - mag[i])
__global__ void unpack_spin_kernel(
    const double* __restrict__ total,
    const double* __restrict__ mag,
    double* __restrict__ out_up,
    double* __restrict__ out_dn,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double t = total[i], m = mag[i];
        out_up[i] = 0.5 * (t + m);
        out_dn[i] = 0.5 * (t - m);
    }
}

// sigma = |grad|^2 for 3 columns (total, up, down), each Nd elements
// Drho_x/y/z layout: [total(Nd) | up(Nd) | down(Nd)]
__global__ void sigma_3col_kernel(
    const double* __restrict__ Drho_x,
    const double* __restrict__ Drho_y,
    const double* __restrict__ Drho_z,
    double* __restrict__ sigma, int N, int ncol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_N = N * ncol;
    if (idx < total_N) {
        double dx = Drho_x[idx], dy = Drho_y[idx], dz = Drho_z[idx];
        sigma[idx] = dx*dx + dy*dy + dz*dz;
    }
}

// Scale Drho by v2xc for 3 columns (spin GGA divergence correction)
__global__ void v2xc_scale_3col_kernel(
    double* __restrict__ f,
    const double* __restrict__ v2xc, int N, int ncol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_N = N * ncol;
    if (idx < total_N) f[idx] *= v2xc[idx];
}

// Spin GGA divergence correction: add to Vxc_up and Vxc_dn
// col0 = correlation (total), col1 = exchange up, col2 = exchange down
// Vxc_up += -(DDrho[col0] + DDrho[col1]), Vxc_dn += -(DDrho[col0] + DDrho[col2])
__global__ void spin_divergence_add_kernel(
    double* __restrict__ Vxc_up,
    double* __restrict__ Vxc_dn,
    const double* __restrict__ DDrho,  // [3*Nd]
    int Nd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Nd) {
        Vxc_up[i] -= DDrho[i] + DDrho[Nd + i];
        Vxc_dn[i] -= DDrho[i] + DDrho[2*Nd + i];
    }
}

// rho_xc for spin: rho_xc layout [total|up|dn], add rho_core to each
__global__ void nlcc_add_spin_kernel(
    const double* __restrict__ rho_up,
    const double* __restrict__ rho_dn,
    const double* __restrict__ rho_core,
    double* __restrict__ rho_xc, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double core_half = 0.5 * rho_core[i];
        double rup = rho_up[i] + core_half;
        double rdn = rho_dn[i] + core_half;
        double rtot = rup + rdn;
        rho_xc[i]       = (rtot > 1e-14) ? rtot : 1e-14;
        rho_xc[N + i]   = (rup > 1e-14) ? rup : 1e-14;
        rho_xc[2*N + i] = (rdn > 1e-14) ? rdn : 1e-14;
    }
}

// Assemble rho_xc = [total | up | dn] without NLCC
__global__ void rho_xc_spin_kernel(
    const double* __restrict__ rho_up,
    const double* __restrict__ rho_dn,
    double* __restrict__ rho_xc, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double rup = rho_up[i], rdn = rho_dn[i];
        double rtot = rup + rdn;
        rho_xc[i]       = (rtot > 1e-14) ? rtot : 1e-14;
        rho_xc[N + i]   = (rup > 1e-14) ? rup : 1e-14;
        rho_xc[2*N + i] = (rdn > 1e-14) ? rdn : 1e-14;
    }
}

// Block-reduce sum
__global__ void sum_reduce_kernel(const double* __restrict__ x, double* __restrict__ out, int N)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double sum = 0.0;
    if (i < N) sum += x[i];
    if (i + blockDim.x < N) sum += x[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

// ============================================================
// Static instance pointer for callback trampolines
// ============================================================
GPUSCFRunner* GPUSCFRunner::s_instance_ = nullptr;

// ============================================================
// GPUNonlocalData::setup — flatten CPU NonlocalProjector for GPU
// ============================================================
void GPUSCFRunner::GPUNonlocalData::setup(
    const NonlocalProjector& vnl,
    const Crystal& crystal,
    const std::vector<AtomNlocInfluence>& nloc_influence,
    int Nband)
{
    if (!vnl.is_setup() || vnl.total_nproj() == 0) return;

    int ntypes = crystal.n_types();

    // Build IP_displ_global for physical atoms
    int n_phys = crystal.n_atom_total();
    std::vector<int> IP_displ_global(n_phys + 1, 0);
    {
        int idx = 0;
        for (int it = 0; it < ntypes; it++) {
            int nproj = crystal.types()[it].psd().nproj_per_atom();
            int nat = crystal.types()[it].n_atoms();
            for (int ia = 0; ia < nat; ia++) {
                IP_displ_global[idx + 1] = IP_displ_global[idx] + nproj;
                idx++;
            }
        }
    }
    total_phys_nproj = IP_displ_global[n_phys];

    // Count influence atoms
    n_influence = 0;
    for (int it = 0; it < ntypes; it++)
        n_influence += nloc_influence[it].n_atom;

    // Flatten all per-atom data
    std::vector<int> h_ndc_arr, h_nproj_arr, h_IP_displ_arr;
    std::vector<int> h_gpos_offsets(1, 0), h_chi_offsets(1, 0);
    std::vector<int> h_gpos_flat;
    std::vector<double> h_Chi_flat;

    max_ndc = 0;
    max_nproj = 0;

    const auto& Chi = vnl.Chi();

    for (int it = 0; it < ntypes; it++) {
        const auto& inf = nloc_influence[it];
        int nproj = crystal.types()[it].psd().nproj_per_atom();

        for (int iat = 0; iat < inf.n_atom; iat++) {
            int ndc = inf.ndc[iat];
            int global_atom = inf.atom_index[iat];

            h_ndc_arr.push_back(ndc);
            h_nproj_arr.push_back(nproj);
            h_IP_displ_arr.push_back(IP_displ_global[global_atom]);

            h_gpos_offsets.push_back(h_gpos_offsets.back() + ndc);
            h_chi_offsets.push_back(h_chi_offsets.back() + ndc * nproj);

            // Copy grid positions
            for (int ig = 0; ig < ndc; ig++)
                h_gpos_flat.push_back(inf.grid_pos[iat][ig]);

            // Copy Chi data (ndc x nproj, column-major, handle ld padding)
            if (ndc > 0 && nproj > 0) {
                const double* chi_data = Chi[it][iat].data();
                int chi_ld = Chi[it][iat].ld();
                for (int jp = 0; jp < nproj; jp++)
                    for (int ig = 0; ig < ndc; ig++)
                        h_Chi_flat.push_back(chi_data[ig + jp * chi_ld]);
            }

            max_ndc = std::max(max_ndc, ndc);
            max_nproj = std::max(max_nproj, nproj);
        }
    }

    // Build Gamma indexed by physical atom projectors
    std::vector<double> h_Gamma(total_phys_nproj, 0.0);
    {
        int phys_idx = 0;
        for (int it = 0; it < ntypes; it++) {
            const auto& psd = crystal.types()[it].psd();
            int nat = crystal.types()[it].n_atoms();
            for (int ia = 0; ia < nat; ia++) {
                int jp = 0;
                for (int l = 0; l <= psd.lmax(); l++) {
                    if (l == psd.lloc()) continue;
                    for (int p = 0; p < psd.ppl()[l]; p++) {
                        double gamma = psd.Gamma()[l][p];
                        for (int m = -l; m <= l; m++) {
                            h_Gamma[IP_displ_global[phys_idx] + jp] = gamma;
                            jp++;
                        }
                    }
                }
                phys_idx++;
            }
        }
    }

    // Upload everything to GPU
    int total_gpos = h_gpos_offsets.back();
    int total_chi = h_chi_offsets.back();

    CUDA_CHECK(cudaMallocAsync(&d_Chi_flat, std::max(1, total_chi) * sizeof(double), 0));
    CUDA_CHECK(cudaMallocAsync(&d_gpos_flat, std::max(1, total_gpos) * sizeof(int), 0));
    CUDA_CHECK(cudaMallocAsync(&d_gpos_offsets, (n_influence + 1) * sizeof(int), 0));
    CUDA_CHECK(cudaMallocAsync(&d_chi_offsets, (n_influence + 1) * sizeof(int), 0));
    CUDA_CHECK(cudaMallocAsync(&d_ndc_arr, n_influence * sizeof(int), 0));
    CUDA_CHECK(cudaMallocAsync(&d_nproj_arr, n_influence * sizeof(int), 0));
    CUDA_CHECK(cudaMallocAsync(&d_IP_displ, n_influence * sizeof(int), 0));
    CUDA_CHECK(cudaMallocAsync(&d_Gamma, total_phys_nproj * sizeof(double), 0));
    CUDA_CHECK(cudaMallocAsync(&d_alpha, (size_t)total_phys_nproj * Nband * sizeof(double), 0));

    if (total_chi > 0)
        CUDA_CHECK(cudaMemcpy(d_Chi_flat, h_Chi_flat.data(), total_chi * sizeof(double), cudaMemcpyHostToDevice));
    if (total_gpos > 0)
        CUDA_CHECK(cudaMemcpy(d_gpos_flat, h_gpos_flat.data(), total_gpos * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gpos_offsets, h_gpos_offsets.data(), (n_influence + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chi_offsets, h_chi_offsets.data(), (n_influence + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ndc_arr, h_ndc_arr.data(), n_influence * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nproj_arr, h_nproj_arr.data(), n_influence * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_IP_displ, h_IP_displ_arr.data(), n_influence * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Gamma, h_Gamma.data(), total_phys_nproj * sizeof(double), cudaMemcpyHostToDevice));

    printf("GPU Vnl: %d influence atoms, %d phys projectors, max_ndc=%d, max_nproj=%d\n",
           n_influence, total_phys_nproj, max_ndc, max_nproj);
    printf("GPU Vnl: Chi=%.1f KB, gpos=%.1f KB, alpha=%.1f KB\n",
           total_chi * 8.0 / 1024, total_gpos * 4.0 / 1024,
           (double)total_phys_nproj * Nband * 8.0 / 1024);
}

// ============================================================
// GPUNonlocalData::free
// ============================================================
void GPUSCFRunner::GPUNonlocalData::free() {
    if (d_Chi_flat) cudaFreeAsync(d_Chi_flat, 0);
    if (d_gpos_flat) cudaFreeAsync(d_gpos_flat, 0);
    if (d_gpos_offsets) cudaFreeAsync(d_gpos_offsets, 0);
    if (d_chi_offsets) cudaFreeAsync(d_chi_offsets, 0);
    if (d_ndc_arr) cudaFreeAsync(d_ndc_arr, 0);
    if (d_nproj_arr) cudaFreeAsync(d_nproj_arr, 0);
    if (d_IP_displ) cudaFreeAsync(d_IP_displ, 0);
    if (d_Gamma) cudaFreeAsync(d_Gamma, 0);
    if (d_alpha) cudaFreeAsync(d_alpha, 0);
    d_Chi_flat = nullptr;
}

// ============================================================
// GPUSOCData::setup_soc — flatten CPU SOC projector data for GPU
// ============================================================
void GPUSCFRunner::GPUSOCData::setup_soc(
    const NonlocalProjector& vnl,
    const Crystal& crystal,
    const std::vector<AtomNlocInfluence>& nloc_influence,
    int Nband)
{
    if (!vnl.has_soc()) return;

    int ntypes = crystal.n_types();

    // Build global SOC IP_displ for physical atoms
    int n_phys = crystal.n_atom_total();
    std::vector<int> IP_displ_global(n_phys + 1, 0);
    {
        int idx = 0;
        for (int it = 0; it < ntypes; it++) {
            const auto& psd = crystal.types()[it].psd();
            int nproj_soc = 0;
            if (psd.has_soc()) {
                for (int l = 1; l <= psd.lmax(); ++l)
                    nproj_soc += psd.ppl_soc()[l] * (2 * l + 1);
            }
            int nat = crystal.types()[it].n_atoms();
            for (int ia = 0; ia < nat; ia++) {
                IP_displ_global[idx + 1] = IP_displ_global[idx] + nproj_soc;
                idx++;
            }
        }
    }
    total_soc_nproj = IP_displ_global[n_phys];
    if (total_soc_nproj == 0) return;

    // Count influence atoms with SOC data
    n_influence_soc = 0;
    for (int it = 0; it < ntypes; it++)
        n_influence_soc += nloc_influence[it].n_atom;

    // Flatten per-atom SOC data
    std::vector<int> h_ndc_arr, h_nproj_soc_arr, h_IP_displ_arr;
    std::vector<int> h_gpos_offsets(1, 0), h_chi_soc_offsets(1, 0);
    std::vector<int> h_gpos_flat;
    std::vector<double> h_Chi_soc_flat;

    max_ndc_soc = 0;
    max_nproj_soc = 0;

    const auto& Chi_soc = vnl.Chi_soc();

    for (int it = 0; it < ntypes; it++) {
        const auto& inf = nloc_influence[it];
        const auto& psd = crystal.types()[it].psd();

        int nproj_soc = 0;
        if (psd.has_soc()) {
            for (int l = 1; l <= psd.lmax(); ++l)
                nproj_soc += psd.ppl_soc()[l] * (2 * l + 1);
        }

        for (int iat = 0; iat < inf.n_atom; iat++) {
            int ndc = inf.ndc[iat];
            int global_atom = inf.atom_index[iat];

            h_ndc_arr.push_back(ndc);
            h_nproj_soc_arr.push_back(nproj_soc);
            h_IP_displ_arr.push_back(IP_displ_global[global_atom]);

            h_gpos_offsets.push_back(h_gpos_offsets.back() + ndc);
            h_chi_soc_offsets.push_back(h_chi_soc_offsets.back() + ndc * nproj_soc);

            // Copy grid positions
            for (int ig = 0; ig < ndc; ig++)
                h_gpos_flat.push_back(inf.grid_pos[iat][ig]);

            // Copy Chi_soc data (ndc x nproj_soc, column-major)
            if (ndc > 0 && nproj_soc > 0 && psd.has_soc()) {
                const double* chi_data = Chi_soc[it][iat].data();
                int chi_ld = Chi_soc[it][iat].ld();
                for (int jp = 0; jp < nproj_soc; jp++)
                    for (int ig = 0; ig < ndc; ig++)
                        h_Chi_soc_flat.push_back(chi_data[ig + jp * chi_ld]);
            } else {
                // Pad with zeros for atoms without SOC
                for (int jp = 0; jp < nproj_soc; jp++)
                    for (int ig = 0; ig < ndc; ig++)
                        h_Chi_soc_flat.push_back(0.0);
            }

            max_ndc_soc = std::max(max_ndc_soc, ndc);
            max_nproj_soc = std::max(max_nproj_soc, nproj_soc);
        }
    }

    // Build Gamma_soc indexed by physical atom projectors
    std::vector<double> h_Gamma_soc(total_soc_nproj, 0.0);
    {
        int phys_idx = 0;
        for (int it = 0; it < ntypes; it++) {
            const auto& psd = crystal.types()[it].psd();
            int nat = crystal.types()[it].n_atoms();
            for (int ia = 0; ia < nat; ia++) {
                if (psd.has_soc()) {
                    int jp = 0;
                    for (int l = 1; l <= psd.lmax(); ++l) {
                        for (int p = 0; p < psd.ppl_soc()[l]; ++p) {
                            double gamma = psd.Gamma_soc()[l][p];
                            for (int m = -l; m <= l; ++m) {
                                h_Gamma_soc[IP_displ_global[phys_idx] + jp] = gamma;
                                jp++;
                            }
                        }
                    }
                }
                phys_idx++;
            }
        }
    }

    // Build proj_l, proj_m arrays (global projector index -> l, m)
    std::vector<int> h_proj_l(total_soc_nproj, 0);
    std::vector<int> h_proj_m(total_soc_nproj, 0);
    {
        int phys_idx = 0;
        for (int it = 0; it < ntypes; it++) {
            const auto& psd = crystal.types()[it].psd();
            const auto& spi = vnl.soc_proj_info()[it];
            int nat = crystal.types()[it].n_atoms();
            for (int ia = 0; ia < nat; ia++) {
                int ip_off = IP_displ_global[phys_idx];
                for (int jp = 0; jp < (int)spi.size(); jp++) {
                    h_proj_l[ip_off + jp] = spi[jp].l;
                    h_proj_m[ip_off + jp] = spi[jp].m;
                }
                phys_idx++;
            }
        }
    }

    // Upload everything to GPU
    int total_gpos = h_gpos_offsets.back();
    int total_chi_soc = h_chi_soc_offsets.back();

    CUDA_CHECK(cudaMallocAsync(&d_Chi_soc_flat, std::max(1, total_chi_soc) * sizeof(double), 0));
    CUDA_CHECK(cudaMallocAsync(&d_gpos_offsets_soc, (n_influence_soc + 1) * sizeof(int), 0));
    CUDA_CHECK(cudaMallocAsync(&d_chi_soc_offsets, (n_influence_soc + 1) * sizeof(int), 0));
    CUDA_CHECK(cudaMallocAsync(&d_ndc_arr_soc, n_influence_soc * sizeof(int), 0));
    CUDA_CHECK(cudaMallocAsync(&d_nproj_soc_arr, n_influence_soc * sizeof(int), 0));
    CUDA_CHECK(cudaMallocAsync(&d_IP_displ_soc, n_influence_soc * sizeof(int), 0));
    {
        double gmax = 0;
        for (int i = 0; i < total_soc_nproj; i++)
            gmax = std::max(gmax, std::abs(h_Gamma_soc[i]));
        printf("GPU SOC: Gamma_soc max=%.6e, nproj=%d\n", gmax, total_soc_nproj);
        for (int i = 0; i < std::min(10, total_soc_nproj); i++)
            printf("  Gamma_soc[%d]=%.6e\n", i, h_Gamma_soc[i]);
    }
    CUDA_CHECK(cudaMallocAsync(&d_Gamma_soc, total_soc_nproj * sizeof(double), 0));
    CUDA_CHECK(cudaMallocAsync(&d_proj_l, total_soc_nproj * sizeof(int), 0));
    CUDA_CHECK(cudaMallocAsync(&d_proj_m, total_soc_nproj * sizeof(int), 0));
    CUDA_CHECK(cudaMallocAsync(&d_alpha_soc_up,
        (size_t)total_soc_nproj * Nband * sizeof(cuDoubleComplex), 0));
    CUDA_CHECK(cudaMallocAsync(&d_alpha_soc_dn,
        (size_t)total_soc_nproj * Nband * sizeof(cuDoubleComplex), 0));

    if (total_chi_soc > 0)
        CUDA_CHECK(cudaMemcpy(d_Chi_soc_flat, h_Chi_soc_flat.data(), total_chi_soc * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gpos_offsets_soc, h_gpos_offsets.data(), (n_influence_soc + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chi_soc_offsets, h_chi_soc_offsets.data(), (n_influence_soc + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ndc_arr_soc, h_ndc_arr.data(), n_influence_soc * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nproj_soc_arr, h_nproj_soc_arr.data(), n_influence_soc * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_IP_displ_soc, h_IP_displ_arr.data(), n_influence_soc * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Gamma_soc, h_Gamma_soc.data(), total_soc_nproj * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_proj_l, h_proj_l.data(), total_soc_nproj * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_proj_m, h_proj_m.data(), total_soc_nproj * sizeof(int), cudaMemcpyHostToDevice));

    printf("GPU SOC: %d influence atoms, %d SOC projectors, max_ndc=%d, max_nproj_soc=%d\n",
           n_influence_soc, total_soc_nproj, max_ndc_soc, max_nproj_soc);
    printf("GPU SOC: Chi_soc=%.1f KB, alpha_soc=%.1f KB (x2)\n",
           total_chi_soc * 8.0 / 1024,
           (double)total_soc_nproj * Nband * 16.0 / 1024);
}

// ============================================================
// GPUSOCData::free_soc
// ============================================================
void GPUSCFRunner::GPUSOCData::free_soc() {
    if (d_Chi_soc_flat)     cudaFreeAsync(d_Chi_soc_flat, 0);
    if (d_gpos_offsets_soc) cudaFreeAsync(d_gpos_offsets_soc, 0);
    if (d_chi_soc_offsets)  cudaFreeAsync(d_chi_soc_offsets, 0);
    if (d_ndc_arr_soc)      cudaFreeAsync(d_ndc_arr_soc, 0);
    if (d_nproj_soc_arr)    cudaFreeAsync(d_nproj_soc_arr, 0);
    if (d_IP_displ_soc)     cudaFreeAsync(d_IP_displ_soc, 0);
    if (d_Gamma_soc)        cudaFreeAsync(d_Gamma_soc, 0);
    if (d_proj_l)           cudaFreeAsync(d_proj_l, 0);
    if (d_proj_m)           cudaFreeAsync(d_proj_m, 0);
    if (d_alpha_soc_up)     cudaFreeAsync(d_alpha_soc_up, 0);
    if (d_alpha_soc_dn)     cudaFreeAsync(d_alpha_soc_dn, 0);
    d_Chi_soc_flat = nullptr;
    d_gpos_offsets_soc = nullptr;
    d_chi_soc_offsets = nullptr;
    d_ndc_arr_soc = nullptr;
    d_nproj_soc_arr = nullptr;
    d_IP_displ_soc = nullptr;
    d_Gamma_soc = nullptr;
    d_proj_l = nullptr;
    d_proj_m = nullptr;
    d_alpha_soc_up = nullptr;
    d_alpha_soc_dn = nullptr;
}

// ============================================================
// Static callback: Hamiltonian apply (H*psi on GPU)
// ============================================================
void GPUSCFRunner::hamiltonian_apply_cb(
    const double* d_psi, const double* d_Veff,
    double* d_Hpsi, double* d_x_ex, int ncol)
{
    auto* s = s_instance_;

    // GPU local part: -0.5*Lap + Veff
    gpu::hamiltonian_apply_local_gpu(
        d_psi, d_Veff, d_Hpsi, d_x_ex,
        s->nx_, s->ny_, s->nz_, s->FDn_, ncol, 0.0,
        s->is_orth_, true, true, true,
        s->diag_coeff_ham_,
        s->has_mixed_deriv_, s->has_mixed_deriv_, s->has_mixed_deriv_);

    // GPU nonlocal part: Vnl*psi -> add to Hpsi
    if (s->gpu_vnl_.total_phys_nproj > 0) {
        gpu::nonlocal_projector_apply_gpu(
            d_psi, d_Hpsi,
            s->gpu_vnl_.d_Chi_flat, s->gpu_vnl_.d_gpos_flat,
            s->gpu_vnl_.d_gpos_offsets, s->gpu_vnl_.d_chi_offsets,
            s->gpu_vnl_.d_ndc_arr, s->gpu_vnl_.d_nproj_arr,
            s->gpu_vnl_.d_IP_displ, s->gpu_vnl_.d_Gamma,
            s->gpu_vnl_.d_alpha,
            s->Nd_, ncol, s->dV_,
            s->gpu_vnl_.n_influence, s->gpu_vnl_.total_phys_nproj,
            s->gpu_vnl_.max_ndc, s->gpu_vnl_.max_nproj);
    }
}

// ============================================================
// Static callback: Complex Hamiltonian apply (H*psi for k-point)
// ============================================================
void GPUSCFRunner::hamiltonian_apply_z_cb(
    const cuDoubleComplex* d_psi, const double* d_Veff,
    cuDoubleComplex* d_Hpsi, cuDoubleComplex* d_x_ex, int ncol)
{
    auto* s = s_instance_;

    // GPU local part: -0.5*Lap + Veff (complex, with Bloch phases)
    gpu::hamiltonian_apply_local_z_gpu(
        d_psi, d_Veff, d_Hpsi, d_x_ex,
        s->nx_, s->ny_, s->nz_, s->FDn_, ncol, 0.0,
        s->is_orth_, true, true, true,
        s->diag_coeff_ham_,
        s->has_mixed_deriv_, s->has_mixed_deriv_, s->has_mixed_deriv_,
        s->kxLx_, s->kyLy_, s->kzLz_);

    // GPU nonlocal part with Bloch phases
    if (s->gpu_vnl_.total_phys_nproj > 0 && s->d_bloch_fac_) {
        gpu::nonlocal_projector_apply_z_gpu(
            d_psi, d_Hpsi,
            s->gpu_vnl_.d_Chi_flat, s->gpu_vnl_.d_gpos_flat,
            s->gpu_vnl_.d_gpos_offsets, s->gpu_vnl_.d_chi_offsets,
            s->gpu_vnl_.d_ndc_arr, s->gpu_vnl_.d_nproj_arr,
            s->gpu_vnl_.d_IP_displ, s->gpu_vnl_.d_Gamma,
            static_cast<cuDoubleComplex*>(s->d_alpha_z_),
            s->d_bloch_fac_,
            s->Nd_, ncol, s->dV_,
            s->gpu_vnl_.n_influence, s->gpu_vnl_.total_phys_nproj,
            s->gpu_vnl_.max_ndc, s->gpu_vnl_.max_nproj);
    }
}

// ============================================================
// Static callback: Spinor Hamiltonian (SOC) for 2-component spinors
// psi/Hpsi layout: [spin-up(Nd_d) | spin-down(Nd_d)] per band
// Veff layout: [V_uu(Nd_d) | V_dd(Nd_d) | Re(V_ud)(Nd_d) | Im(V_ud)(Nd_d)]
// ============================================================
void GPUSCFRunner::hamiltonian_apply_spinor_z_cb(
    const cuDoubleComplex* d_psi, const double* d_Veff,
    cuDoubleComplex* d_Hpsi, cuDoubleComplex* d_x_ex, int ncol)
{
    auto* s = s_instance_;
    int Nd_d = s->Nd_;
    int Nd_d_spinor = 2 * Nd_d;

    const double* V_uu = d_Veff;
    const double* V_dd = d_Veff + Nd_d;
    const double* V_ud_re = d_Veff + 2 * Nd_d;
    const double* V_ud_im = d_Veff + 3 * Nd_d;

    // Process all bands at once per spinor component using strided layout.
    // psi layout: [up0(Nd)|dn0(Nd)|up1(Nd)|dn1(Nd)|...] with stride Nd_d_spinor per band
    // We need to apply H to all up components and all dn components separately.
    // Since halo_exchange_z_gpu/laplacian handles multi-column with stride Nd,
    // we need to extract up/dn into contiguous arrays, apply H, then scatter back.
    // For efficiency, process per-band but WITHOUT debug syncs.
    for (int n = 0; n < ncol; ++n) {
        const cuDoubleComplex* psi_up = d_psi + n * Nd_d_spinor;
        const cuDoubleComplex* psi_dn = psi_up + Nd_d;
        cuDoubleComplex* Hpsi_up = d_Hpsi + n * Nd_d_spinor;
        cuDoubleComplex* Hpsi_dn = Hpsi_up + Nd_d;

        gpu::hamiltonian_apply_local_z_gpu(
            psi_up, V_uu, Hpsi_up, d_x_ex,
            s->nx_, s->ny_, s->nz_, s->FDn_, 1, 0.0,
            s->is_orth_, true, true, true,
            s->diag_coeff_ham_,
            s->has_mixed_deriv_, s->has_mixed_deriv_, s->has_mixed_deriv_,
            s->kxLx_, s->kyLy_, s->kzLz_);

        gpu::hamiltonian_apply_local_z_gpu(
            psi_dn, V_dd, Hpsi_dn, d_x_ex,
            s->nx_, s->ny_, s->nz_, s->FDn_, 1, 0.0,
            s->is_orth_, true, true, true,
            s->diag_coeff_ham_,
            s->has_mixed_deriv_, s->has_mixed_deriv_, s->has_mixed_deriv_,
            s->kxLx_, s->kyLy_, s->kzLz_);

        if (s->gpu_vnl_.total_phys_nproj > 0 && s->d_bloch_fac_) {
            gpu::nonlocal_projector_apply_z_gpu(
                psi_up, Hpsi_up,
                s->gpu_vnl_.d_Chi_flat, s->gpu_vnl_.d_gpos_flat,
                s->gpu_vnl_.d_gpos_offsets, s->gpu_vnl_.d_chi_offsets,
                s->gpu_vnl_.d_ndc_arr, s->gpu_vnl_.d_nproj_arr,
                s->gpu_vnl_.d_IP_displ, s->gpu_vnl_.d_Gamma,
                static_cast<cuDoubleComplex*>(s->d_alpha_z_),
                s->d_bloch_fac_,
                Nd_d, 1, s->dV_,
                s->gpu_vnl_.n_influence, s->gpu_vnl_.total_phys_nproj,
                s->gpu_vnl_.max_ndc, s->gpu_vnl_.max_nproj);

            gpu::nonlocal_projector_apply_z_gpu(
                psi_dn, Hpsi_dn,
                s->gpu_vnl_.d_Chi_flat, s->gpu_vnl_.d_gpos_flat,
                s->gpu_vnl_.d_gpos_offsets, s->gpu_vnl_.d_chi_offsets,
                s->gpu_vnl_.d_ndc_arr, s->gpu_vnl_.d_nproj_arr,
                s->gpu_vnl_.d_IP_displ, s->gpu_vnl_.d_Gamma,
                static_cast<cuDoubleComplex*>(s->d_alpha_z_),
                s->d_bloch_fac_,
                Nd_d, 1, s->dV_,
                s->gpu_vnl_.n_influence, s->gpu_vnl_.total_phys_nproj,
                s->gpu_vnl_.max_ndc, s->gpu_vnl_.max_nproj);
        }
    }

    // Off-diagonal Veff: Hpsi_up += V_ud * psi_dn, Hpsi_dn += conj(V_ud) * psi_up
    gpu::spinor_offdiag_veff_gpu(d_Hpsi, d_psi, V_ud_re, V_ud_im, Nd_d, ncol);

    // SOC terms (Term 1 + Term 2)
    if (s->has_soc_ && s->gpu_soc_.total_soc_nproj > 0) {
        gpu::soc_apply_z_gpu(
            d_psi, d_Hpsi,
            s->gpu_soc_.d_Chi_soc_flat, s->gpu_vnl_.d_gpos_flat,
            s->gpu_soc_.d_gpos_offsets_soc, s->gpu_soc_.d_chi_soc_offsets,
            s->gpu_soc_.d_ndc_arr_soc, s->gpu_soc_.d_nproj_soc_arr,
            s->gpu_soc_.d_IP_displ_soc,
            s->gpu_soc_.d_Gamma_soc,
            s->gpu_soc_.d_proj_l, s->gpu_soc_.d_proj_m,
            s->d_bloch_fac_,
            static_cast<cuDoubleComplex*>(s->gpu_soc_.d_alpha_soc_up),
            static_cast<cuDoubleComplex*>(s->gpu_soc_.d_alpha_soc_dn),
            Nd_d, ncol, s->dV_,
            s->gpu_soc_.n_influence_soc, s->gpu_soc_.total_soc_nproj,
            s->gpu_soc_.max_ndc_soc, s->gpu_soc_.max_nproj_soc);
    }
}

// ============================================================
// Static callback: Poisson operator (-Lap * x)
// ============================================================
void GPUSCFRunner::poisson_op_cb(const double* d_x, double* d_Ax) {
    auto* s = s_instance_;
    auto& ctx = gpu::GPUContext::instance();

    gpu::halo_exchange_gpu(d_x, ctx.buf.aar_x_ex,
        s->nx_, s->ny_, s->nz_, s->FDn_, 1, true, true, true);
    int nx_ex = s->nx_ + 2 * s->FDn_, ny_ex = s->ny_ + 2 * s->FDn_;
    gpu::laplacian_orth_v2_gpu(ctx.buf.aar_x_ex, nullptr, d_Ax,
        s->nx_, s->ny_, s->nz_, s->FDn_, nx_ex, ny_ex,
        -1.0, 0.0, 0.0, s->poisson_diag_, 1);
}

// ============================================================
// Static callback: Poisson Jacobi preconditioner
// ============================================================
void GPUSCFRunner::poisson_precond_cb(const double* d_r, double* d_f) {
    auto* s = s_instance_;
    int bs = 256;
    jacobi_scale_kernel<<<gpu::ceildiv(s->Nd_, bs), bs>>>(
        d_r, d_f, s->jacobi_m_inv_, s->Nd_);
}

// ============================================================
// Static callback: Kerker operator (-Lap + kTF^2) * x
// ============================================================
void GPUSCFRunner::kerker_op_cb(const double* d_x, double* d_Ax) {
    auto* s = s_instance_;
    auto& ctx = gpu::GPUContext::instance();

    gpu::halo_exchange_gpu(d_x, ctx.buf.aar_x_ex,
        s->nx_, s->ny_, s->nz_, s->FDn_, 1, true, true, true);
    int nx_ex = s->nx_ + 2 * s->FDn_, ny_ex = s->ny_ + 2 * s->FDn_;
    constexpr double kTF2 = 1.0;
    gpu::laplacian_orth_v2_gpu(ctx.buf.aar_x_ex, nullptr, d_Ax,
        s->nx_, s->ny_, s->nz_, s->FDn_, nx_ex, ny_ex,
        -1.0, 0.0, kTF2, s->kerker_diag_, 1);
}

// ============================================================
// Static callback: Kerker Jacobi preconditioner
// ============================================================
void GPUSCFRunner::kerker_precond_cb(const double* d_r, double* d_f) {
    auto* s = s_instance_;
    int bs = 256;
    jacobi_scale_kernel<<<gpu::ceildiv(s->Nd_, bs), bs>>>(
        d_r, d_f, s->kerker_m_inv_, s->Nd_);
}

// ============================================================
// GPU sum (small arrays — download + CPU sum)
// ============================================================
double GPUSCFRunner::gpu_sum(const double* d_x, int N) {
    std::vector<double> h(N);
    CUDA_CHECK(cudaMemcpy(h.data(), d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
    double s = 0;
    for (int i = 0; i < N; i++) s += h[i];
    return s;
}

// ============================================================
// GPU XC evaluate (GGA PBE or LDA, orthogonal, non-spin)
// ============================================================
void GPUSCFRunner::gpu_xc_evaluate(double* d_rho, double* d_exc, double* d_Vxc, int Nd) {
    auto& ctx = gpu::GPUContext::instance();
    int bs = 256;
    int grid_sz = gpu::ceildiv(Nd, bs);
    int nx_ex = nx_ + 2 * FDn_, ny_ex = ny_ + 2 * FDn_;

    if (is_gga_) {
        // GGA path (PBE)
        double* d_Drho_x = ctx.buf.grad_rho;
        double* d_Drho_y = ctx.buf.grad_rho + Nd;
        double* d_Drho_z = ctx.buf.grad_rho + 2 * Nd;
        double* d_sigma  = ctx.buf.aar_r;     // reuse (not in AAR yet)
        double* d_v2xc   = ctx.buf.Dxcdgrho;
        double* d_x_ex   = ctx.buf.aar_x_ex;

        // Prepare rho_xc (NLCC: rho + rho_core)
        double* d_rho_xc;
        if (has_nlcc_ && d_rho_core_) {
            d_rho_xc = ctx.buf.b;  // reuse Poisson RHS buffer
            nlcc_add_kernel<<<grid_sz, bs>>>(d_rho, d_rho_core_, d_rho_xc, Nd);
        } else {
            d_rho_xc = d_rho;
        }

        // Gradient of rho_xc
        gpu::halo_exchange_gpu(d_rho_xc, d_x_ex, nx_, ny_, nz_, FDn_, 1, true, true, true);
        gpu::gradient_gpu(d_x_ex, d_Drho_x, nx_, ny_, nz_, FDn_, nx_ex, ny_ex, 0, 1);
        gpu::gradient_gpu(d_x_ex, d_Drho_y, nx_, ny_, nz_, FDn_, nx_ex, ny_ex, 1, 1);
        gpu::gradient_gpu(d_x_ex, d_Drho_z, nx_, ny_, nz_, FDn_, nx_ex, ny_ex, 2, 1);

        // sigma = |nabla rho|^2
        sigma_kernel<<<grid_sz, bs>>>(d_Drho_x, d_Drho_y, d_Drho_z, d_sigma, Nd);

        // Fused PBE kernel: (rho_xc, sigma) -> (exc, Vxc, v2xc)
        gpu::gga_pbe_gpu(d_rho_xc, d_sigma, d_exc, d_Vxc, d_v2xc, Nd);

        // Divergence correction: Vxc += -div(v2xc * nabla rho)
        // Scale gradients by v2xc (in place)
        v2xc_scale_kernel<<<grid_sz, bs>>>(d_Drho_x, d_v2xc, Nd);
        v2xc_scale_kernel<<<grid_sz, bs>>>(d_Drho_y, d_v2xc, Nd);
        v2xc_scale_kernel<<<grid_sz, bs>>>(d_Drho_z, d_v2xc, Nd);

        // Process each direction: halo -> gradient -> subtract from Vxc
        double* d_DDrho = d_sigma;  // reuse sigma buffer

        // x-direction
        gpu::halo_exchange_gpu(d_Drho_x, d_x_ex, nx_, ny_, nz_, FDn_, 1, true, true, true);
        gpu::gradient_gpu(d_x_ex, d_DDrho, nx_, ny_, nz_, FDn_, nx_ex, ny_ex, 0, 1);
        divergence_sub_kernel<<<grid_sz, bs>>>(d_Vxc, d_DDrho, Nd);

        // y-direction
        gpu::halo_exchange_gpu(d_Drho_y, d_x_ex, nx_, ny_, nz_, FDn_, 1, true, true, true);
        gpu::gradient_gpu(d_x_ex, d_DDrho, nx_, ny_, nz_, FDn_, nx_ex, ny_ex, 1, 1);
        divergence_sub_kernel<<<grid_sz, bs>>>(d_Vxc, d_DDrho, Nd);

        // z-direction
        gpu::halo_exchange_gpu(d_Drho_z, d_x_ex, nx_, ny_, nz_, FDn_, 1, true, true, true);
        gpu::gradient_gpu(d_x_ex, d_DDrho, nx_, ny_, nz_, FDn_, nx_ex, ny_ex, 2, 1);
        divergence_sub_kernel<<<grid_sz, bs>>>(d_Vxc, d_DDrho, Nd);
    } else {
        // LDA path
        double* d_rho_xc;
        if (has_nlcc_ && d_rho_core_) {
            d_rho_xc = ctx.buf.b;
            nlcc_add_kernel<<<grid_sz, bs>>>(d_rho, d_rho_core_, d_rho_xc, Nd);
        } else {
            d_rho_xc = d_rho;
        }
        // Use LDA_PW by default (xc_type_ distinguished in run())
        gpu::lda_pw_gpu(d_rho_xc, d_exc, d_Vxc, Nd);
    }
}

// ============================================================
// GPU XC evaluate spin-polarized (Nspin=2)
// rho_up = d_rho, rho_dn = d_rho + Nd
// d_Vxc layout: [up(Nd) | down(Nd)]
// ============================================================
void GPUSCFRunner::gpu_xc_evaluate_spin(double* d_rho, double* d_exc, double* d_Vxc, int Nd) {
    auto& ctx = gpu::GPUContext::instance();
    int bs = 256;
    int grid_sz = gpu::ceildiv(Nd, bs);
    int nx_ex = nx_ + 2 * FDn_, ny_ex = ny_ + 2 * FDn_;

    double* d_rho_up = d_rho;
    double* d_rho_dn = d_rho + Nd;

    if (is_gga_) {
        // Spin GGA path
        // Need workspace for rho_xc[3*Nd], sigma[3*Nd], Drho_x/y/z[3*Nd], v2xc[3*Nd]
        auto& sp = ctx.scratch_pool;
        size_t sp_cp = sp.checkpoint();

        double* d_rho_xc   = sp.alloc<double>(3 * Nd);  // [total|up|dn]
        double* d_sigma     = sp.alloc<double>(3 * Nd);
        double* d_Drho_x    = sp.alloc<double>(3 * Nd);
        double* d_Drho_y    = sp.alloc<double>(3 * Nd);
        double* d_Drho_z    = sp.alloc<double>(3 * Nd);
        double* d_v2xc      = ctx.buf.Dxcdgrho;  // [3*Nd]
        double* d_x_ex_tmp  = ctx.buf.aar_x_ex;

        // Build rho_xc = [total|up|dn] with NLCC
        if (has_nlcc_ && d_rho_core_) {
            nlcc_add_spin_kernel<<<grid_sz, bs>>>(d_rho_up, d_rho_dn, d_rho_core_, d_rho_xc, Nd);
        } else {
            rho_xc_spin_kernel<<<grid_sz, bs>>>(d_rho_up, d_rho_dn, d_rho_xc, Nd);
        }

        // Gradient of all 3 density columns
        for (int col = 0; col < 3; col++) {
            gpu::halo_exchange_gpu(d_rho_xc + col * Nd, d_x_ex_tmp,
                                   nx_, ny_, nz_, FDn_, 1, true, true, true);
            gpu::gradient_gpu(d_x_ex_tmp, d_Drho_x + col * Nd,
                              nx_, ny_, nz_, FDn_, nx_ex, ny_ex, 0, 1);
            gpu::gradient_gpu(d_x_ex_tmp, d_Drho_y + col * Nd,
                              nx_, ny_, nz_, FDn_, nx_ex, ny_ex, 1, 1);
            gpu::gradient_gpu(d_x_ex_tmp, d_Drho_z + col * Nd,
                              nx_, ny_, nz_, FDn_, nx_ex, ny_ex, 2, 1);
        }

        // sigma for 3 columns
        int grid3 = gpu::ceildiv(3 * Nd, bs);
        sigma_3col_kernel<<<grid3, bs>>>(d_Drho_x, d_Drho_y, d_Drho_z, d_sigma, Nd, 3);

        // Fused spin PBE kernel: (rho_xc[3*Nd], sigma[3*Nd]) -> (exc, Vxc[2*Nd], v2xc[3*Nd])
        gpu::gga_pbe_spin_gpu(d_rho_xc, d_sigma, d_exc, d_Vxc, d_v2xc, Nd);

        // Divergence correction for 3 columns
        // Scale: Drho_dir[col] *= v2xc[col]
        v2xc_scale_3col_kernel<<<grid3, bs>>>(d_Drho_x, d_v2xc, Nd, 3);
        v2xc_scale_3col_kernel<<<grid3, bs>>>(d_Drho_y, d_v2xc, Nd, 3);
        v2xc_scale_3col_kernel<<<grid3, bs>>>(d_Drho_z, d_v2xc, Nd, 3);

        // For each direction and each column, compute divergence and accumulate
        // DDrho accumulates all 3 directions per column
        double* d_DDrho = d_sigma;  // reuse sigma buffer [3*Nd]
        CUDA_CHECK(cudaMemset(d_DDrho, 0, 3 * Nd * sizeof(double)));

        // x-direction: halo + gradient for each column -> DDrho
        for (int col = 0; col < 3; col++) {
            double* d_DDcol = sp.alloc<double>(Nd);
            gpu::halo_exchange_gpu(d_Drho_x + col * Nd, d_x_ex_tmp,
                                   nx_, ny_, nz_, FDn_, 1, true, true, true);
            gpu::gradient_gpu(d_x_ex_tmp, d_DDcol, nx_, ny_, nz_, FDn_, nx_ex, ny_ex, 0, 1);
            // Accumulate into DDrho
            double one = 1.0;
            cublasDaxpy(ctx.cublas, Nd, &one, d_DDcol, 1, d_DDrho + col * Nd, 1);
            sp.restore(sp.checkpoint() - Nd * sizeof(double));  // free d_DDcol
        }

        // y-direction
        for (int col = 0; col < 3; col++) {
            double* d_DDcol = sp.alloc<double>(Nd);
            gpu::halo_exchange_gpu(d_Drho_y + col * Nd, d_x_ex_tmp,
                                   nx_, ny_, nz_, FDn_, 1, true, true, true);
            gpu::gradient_gpu(d_x_ex_tmp, d_DDcol, nx_, ny_, nz_, FDn_, nx_ex, ny_ex, 1, 1);
            double one = 1.0;
            cublasDaxpy(ctx.cublas, Nd, &one, d_DDcol, 1, d_DDrho + col * Nd, 1);
            sp.restore(sp.checkpoint() - Nd * sizeof(double));
        }

        // z-direction
        for (int col = 0; col < 3; col++) {
            double* d_DDcol = sp.alloc<double>(Nd);
            gpu::halo_exchange_gpu(d_Drho_z + col * Nd, d_x_ex_tmp,
                                   nx_, ny_, nz_, FDn_, 1, true, true, true);
            gpu::gradient_gpu(d_x_ex_tmp, d_DDcol, nx_, ny_, nz_, FDn_, nx_ex, ny_ex, 2, 1);
            double one = 1.0;
            cublasDaxpy(ctx.cublas, Nd, &one, d_DDcol, 1, d_DDrho + col * Nd, 1);
            sp.restore(sp.checkpoint() - Nd * sizeof(double));
        }

        // Apply spin divergence correction
        spin_divergence_add_kernel<<<grid_sz, bs>>>(d_Vxc, d_Vxc + Nd, d_DDrho, Nd);

        sp.restore(sp_cp);
    } else {
        // LDA spin path
        double* d_rho_xc_up, *d_rho_xc_dn;
        if (has_nlcc_ && d_rho_core_) {
            auto& sp = ctx.scratch_pool;
            size_t sp_cp = sp.checkpoint();
            d_rho_xc_up = sp.alloc<double>(Nd);
            d_rho_xc_dn = sp.alloc<double>(Nd);
            // rho_xc_up = max(rho_up + 0.5*rho_core, 1e-14)
            // Use nlcc_add_spin_kernel to build [total|up|dn], then just pass up/dn
            double* d_rho_xc_3 = sp.alloc<double>(3 * Nd);
            nlcc_add_spin_kernel<<<grid_sz, bs>>>(d_rho_up, d_rho_dn, d_rho_core_, d_rho_xc_3, Nd);
            d_rho_xc_up = d_rho_xc_3 + Nd;
            d_rho_xc_dn = d_rho_xc_3 + 2 * Nd;
            gpu::lda_pw_spin_gpu(d_rho_xc_up, d_rho_xc_dn, d_exc, d_Vxc, d_Vxc + Nd, Nd);
            sp.restore(sp_cp);
        } else {
            gpu::lda_pw_spin_gpu(d_rho_up, d_rho_dn, d_exc, d_Vxc, d_Vxc + Nd, Nd);
        }
    }
}

// ============================================================
// Setup Bloch factors for k-point nonlocal projector
// ============================================================
void GPUSCFRunner::setup_bloch_factors(
    const std::vector<AtomNlocInfluence>& nloc_influence,
    const Crystal& crystal,
    const Vec3& kpt_cart)
{
    int n_influence = gpu_vnl_.n_influence;
    if (n_influence == 0) return;

    // Allocate device buffer if needed
    if (!d_bloch_fac_) {
        CUDA_CHECK(cudaMallocAsync(&d_bloch_fac_, n_influence * 2 * sizeof(double), 0));
    }

    // Compute cos/sin of -k.R_image for each influence atom
    std::vector<double> h_bloch(n_influence * 2);
    int idx = 0;
    int ntypes = crystal.n_types();
    for (int it = 0; it < ntypes; it++) {
        const auto& inf = nloc_influence[it];
        for (int iat = 0; iat < inf.n_atom; iat++) {
            const Vec3& shift = inf.image_shift[iat];
            double theta = -(kpt_cart.x * shift.x + kpt_cart.y * shift.y + kpt_cart.z * shift.z);
            h_bloch[idx * 2]     = std::cos(theta);
            h_bloch[idx * 2 + 1] = std::sin(theta);
            idx++;
        }
    }

    CUDA_CHECK(cudaMemcpy(d_bloch_fac_, h_bloch.data(),
                           n_influence * 2 * sizeof(double), cudaMemcpyHostToDevice));
}

// ============================================================
// GPU Poisson solver
// ============================================================
int GPUSCFRunner::gpu_poisson_solve(double* d_rho, double* d_phi,
                                     double* d_rhs, int Nd, double tol) {
    auto& ctx = gpu::GPUContext::instance();
    int bs = 256;
    int grid_sz = gpu::ceildiv(Nd, bs);

    // RHS = 4*pi*(rho + pseudocharge)
    poisson_rhs_kernel<<<grid_sz, bs>>>(d_rho, d_pseudocharge_, d_rhs,
                                         4.0 * constants::PI, Nd);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Mean-subtract RHS
    double rhs_mean = gpu_sum(d_rhs, Nd) / Nd;
    mean_subtract_kernel<<<grid_sz, bs>>>(d_rhs, rhs_mean, Nd);

    // Allocate x_old and f_old from scratch pool
    auto& sp = ctx.scratch_pool;
    size_t sp_cp = sp.checkpoint();
    double* d_xold = sp.alloc<double>(Nd);
    double* d_fold = sp.alloc<double>(Nd);

    int iters = gpu::aar_gpu(
        poisson_op_cb, poisson_precond_cb,
        d_rhs, d_phi, Nd,
        0.6, 0.6, 7, 6, tol, 3000,
        ctx.buf.aar_r, ctx.buf.aar_f, ctx.buf.aar_Ax,
        ctx.buf.aar_X, ctx.buf.aar_F,
        d_xold, d_fold);

    sp.restore(sp_cp);

    // Mean-subtract phi
    CUDA_CHECK(cudaDeviceSynchronize());
    double phi_mean = gpu_sum(d_phi, Nd) / Nd;
    mean_subtract_kernel<<<grid_sz, bs>>>(d_phi, phi_mean, Nd);

    return iters;
}

// ============================================================
// GPU Pulay + Kerker mixer
// ============================================================
void GPUSCFRunner::gpu_pulay_mix(double* d_x, const double* d_g,
                                  int Nd, int m_depth, double beta_mix) {
    auto& ctx = gpu::GPUContext::instance();
    int bs = 256;
    int grid_sz = gpu::ceildiv(Nd, bs);

    double* d_fk   = ctx.buf.mix_fk;
    double* d_xkm1 = ctx.buf.mix_xkm1;
    double* d_R    = ctx.buf.mix_R;
    double* d_F    = ctx.buf.mix_F;

    // Save old f_k -> f_km1
    if (mix_iter_ > 0) {
        CUDA_CHECK(cudaMemcpy(d_mix_fkm1_, d_fk, Nd * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    // f_k = g - x
    mix_residual_kernel<<<grid_sz, bs>>>(d_g, d_x, d_fk, Nd);

    // Store history
    if (mix_iter_ > 0) {
        int i_hist = (mix_iter_ - 1) % m_depth;
        mix_store_history_kernel<<<grid_sz, bs>>>(
            d_x, d_xkm1, d_fk, d_mix_fkm1_, d_R, d_F, i_hist, Nd);
    }

    // Allocate workspace from scratch pool
    auto& sp = ctx.scratch_pool;
    size_t sp_cp = sp.checkpoint();
    double* d_x_wavg = sp.alloc<double>(Nd);
    double* d_f_wavg = sp.alloc<double>(Nd);

    if (mix_iter_ > 0) {
        int cols = std::min(mix_iter_, m_depth);

        // Build F^T*F and F^T*f_k using cuBLAS ddot
        std::vector<double> h_FtF(cols * cols);
        std::vector<double> h_Ftf(cols);

        CUDA_CHECK(cudaDeviceSynchronize());
        for (int ii = 0; ii < cols; ii++) {
            double* Fi = d_F + ii * Nd;
            cublasDdot(ctx.cublas, Nd, Fi, 1, d_fk, 1, &h_Ftf[ii]);
            for (int jj = 0; jj <= ii; jj++) {
                double* Fj = d_F + jj * Nd;
                cublasDdot(ctx.cublas, Nd, Fi, 1, Fj, 1, &h_FtF[ii * cols + jj]);
                h_FtF[jj * cols + ii] = h_FtF[ii * cols + jj];
            }
        }

        // Solve Gamma on CPU (tiny matrix)
        std::vector<double> Gamma(cols, 0.0);
        {
            std::vector<double> A(h_FtF);
            std::vector<double> b(h_Ftf);
            for (int k = 0; k < cols; k++) {
                int pivot = k;
                for (int i = k+1; i < cols; i++)
                    if (std::abs(A[i*cols+k]) > std::abs(A[pivot*cols+k])) pivot = i;
                if (pivot != k) {
                    for (int j = 0; j < cols; j++) std::swap(A[k*cols+j], A[pivot*cols+j]);
                    std::swap(b[k], b[pivot]);
                }
                double d = A[k*cols+k];
                if (std::abs(d) < 1e-14) continue;
                for (int i = k+1; i < cols; i++) {
                    double fac = A[i*cols+k] / d;
                    for (int j = k+1; j < cols; j++) A[i*cols+j] -= fac * A[k*cols+j];
                    b[i] -= fac * b[k];
                }
            }
            for (int k = cols-1; k >= 0; k--) {
                if (std::abs(A[k*cols+k]) < 1e-14) continue;
                Gamma[k] = b[k];
                for (int j = k+1; j < cols; j++) Gamma[k] -= A[k*cols+j] * Gamma[j];
                Gamma[k] /= A[k*cols+k];
            }
        }

        // x_wavg = x - R * Gamma, f_wavg = f_k - F * Gamma
        CUDA_CHECK(cudaMemcpy(d_x_wavg, d_x, Nd * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_f_wavg, d_fk, Nd * sizeof(double), cudaMemcpyDeviceToDevice));

        for (int j = 0; j < cols; j++) {
            double neg_gj = -Gamma[j];
            cublasDaxpy(ctx.cublas, Nd, &neg_gj, d_R + j * Nd, 1, d_x_wavg, 1);
            cublasDaxpy(ctx.cublas, Nd, &neg_gj, d_F + j * Nd, 1, d_f_wavg, 1);
        }
    } else {
        CUDA_CHECK(cudaMemcpy(d_x_wavg, d_x, Nd * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_f_wavg, d_fk, Nd * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    // Kerker preconditioner: Pf = kerker(f_wavg, beta_mix)
    double* d_Pf = sp.alloc<double>(Nd);
    CUDA_CHECK(cudaMemset(d_Pf, 0, Nd * sizeof(double)));

    // Kerker step 1: Lf = (Lap - idiemac*kTF^2) * f_wavg
    double* d_Lf = sp.alloc<double>(Nd);
    {
        constexpr double idiemac_kTF2 = 0.1;  // idiemac * kTF^2
        int nx_ex = nx_ + 2 * FDn_, ny_ex = ny_ + 2 * FDn_;
        gpu::halo_exchange_gpu(d_f_wavg, ctx.buf.aar_x_ex,
            nx_, ny_, nz_, FDn_, 1, true, true, true);
        gpu::laplacian_orth_v2_gpu(ctx.buf.aar_x_ex, nullptr, d_Lf,
            nx_, ny_, nz_, FDn_, nx_ex, ny_ex,
            1.0, 0.0, -idiemac_kTF2, kerker_rhs_diag_, 1);
    }

    // Kerker step 2: Solve (-Lap + kTF^2)*Pf = Lf via AAR
    {
        double* d_kr    = sp.alloc<double>(Nd);
        double* d_kf    = sp.alloc<double>(Nd);
        double* d_kAx   = sp.alloc<double>(Nd);
        double* d_kX    = sp.alloc<double>(Nd * 7);
        double* d_kF    = sp.alloc<double>(Nd * 7);
        double* d_kxold = sp.alloc<double>(Nd);
        double* d_kfold = sp.alloc<double>(Nd);

        gpu::aar_gpu(
            kerker_op_cb, kerker_precond_cb,
            d_Lf, d_Pf, Nd,
            0.6, 0.6, 7, 6, precond_tol_, 1000,
            d_kr, d_kf, d_kAx, d_kX, d_kF, d_kxold, d_kfold);
    }

    // Kerker step 3: Pf *= -beta_mix
    {
        double neg_beta = -beta_mix;
        cublasDscal(ctx.cublas, Nd, &neg_beta, d_Pf, 1);
    }

    // Save x_km1 = x (before update)
    CUDA_CHECK(cudaMemcpy(d_xkm1, d_x, Nd * sizeof(double), cudaMemcpyDeviceToDevice));

    // x_{k+1} = x_wavg + Pf
    CUDA_CHECK(cudaMemcpy(d_x, d_x_wavg, Nd * sizeof(double), cudaMemcpyDeviceToDevice));
    {
        double one = 1.0;
        cublasDaxpy(ctx.cublas, Nd, &one, d_Pf, 1, d_x, 1);
    }

    sp.restore(sp_cp);
    mix_iter_++;
}

// ============================================================
// Main run() method — GPU SCF loop
// ============================================================
double GPUSCFRunner::run(
    Wavefunction& wfn,
    const SCFParams& params,
    const FDGrid& grid,
    const Domain& domain,
    const FDStencil& stencil,
    const Hamiltonian& hamiltonian,
    const HaloExchange& halo,
    const NonlocalProjector* vnl,
    const Crystal& crystal,
    const std::vector<AtomNlocInfluence>& nloc_influence,
    const MPIComm& bandcomm,
    int Nelectron,
    int Natom,
    const double* rho_init,
    const double* rho_b,
    double Eself,
    double Ec,
    XCType xc_type,
    const double* rho_core,
    bool is_gga,
    int Nspin,
    bool is_kpt,
    const KPoints* kpoints,
    const std::vector<double>& kpt_weights_in,
    int Nspin_local,
    int spin_start,
    int kpt_start,
    const double* rho_up_init,
    const double* rho_dn_init,
    bool is_soc)
{
    // Store parameters as members
    nx_ = grid.Nx();
    ny_ = grid.Ny();
    nz_ = grid.Nz();
    FDn_ = stencil.FDn();
    Nd_ = domain.Nd_d();
    dV_ = grid.dV();
    is_gga_ = is_gga;
    is_orth_ = grid.lattice().is_orthogonal();
    has_mixed_deriv_ = !is_orth_;  // non-orth has xy/xz/yz mixed derivatives
    has_nlcc_ = (rho_core != nullptr);
    converged_ = false;
    mix_iter_ = 0;
    Ef_ = 0.0;
    Ef_prev_ = 0.0;
    Nspin_ = Nspin;
    Nspin_local_ = Nspin_local;
    spin_start_ = spin_start;
    is_kpt_ = is_kpt;
    kpt_start_ = kpt_start;
    kpoints_ = kpoints;

    // SOC detection: requires k-point mode and SOC pseudopotentials
    has_soc_ = is_soc && is_kpt_ && vnl && vnl->has_soc();
    vnl_ptr_ = vnl;

    int Nband = wfn.Nband();

    // Compute stencil-derived coefficients
    const double* D2x = stencil.D2_coeff_x();
    const double* D2y = stencil.D2_coeff_y();
    const double* D2z = stencil.D2_coeff_z();
    double D2sum = D2x[0] + D2y[0] + D2z[0];

    diag_coeff_ham_ = -0.5 * D2sum;
    poisson_diag_   = -1.0 * D2sum;
    jacobi_m_inv_   = -1.0 / D2sum;
    kerker_diag_    = -1.0 * D2sum + 1.0;
    kerker_m_inv_   = -1.0 / (D2sum - 1.0);
    kerker_rhs_diag_ = 1.0 * D2sum + (-0.1);

    // TOL_PRECOND = h_eff^2 * 1e-3
    double h_eff = grid.dx();
    precond_tol_ = h_eff * h_eff * 1e-3;

    // Upload FD stencil coefficients to GPU constant memory
    gpu::upload_stencil_coefficients(
        stencil.D2_coeff_x(), stencil.D2_coeff_y(), stencil.D2_coeff_z(),
        stencil.D1_coeff_x(), stencil.D1_coeff_y(), stencil.D1_coeff_z(),
        stencil.D2_coeff_xy(), stencil.D2_coeff_xz(), stencil.D2_coeff_yz(),
        FDn_);

    // ============================================================
    // GPU memory allocation + data upload
    // ============================================================
    auto& ctx = gpu::GPUContext::instance();
    int mix_ncol = has_soc_ ? 4 : ((Nspin >= 2) ? 2 : 1);  // SOC: [rho|mx|my|mz]
    int Nspin_buf = has_soc_ ? 2 : Nspin;  // SOC uses spin-polarized XC internally
    ctx.init_scf_buffers(Nd_, nx_, ny_, nz_, FDn_,
                          Nband, Nband, Nspin_buf,
                          7, 7, mix_ncol,
                          0, 0, 0,
                          is_gga_, false, is_kpt);

    // Upload nonlocal projector data to GPU (once)
    if (vnl) {
        gpu_vnl_.setup(*vnl, crystal, nloc_influence, Nband);
        // Upload SOC projector data if needed
        if (has_soc_) {
            gpu_soc_.setup_soc(*vnl, crystal, nloc_influence, Nband);
        }
    }

    // Upload NLCC core density to GPU
    if (has_nlcc_) {
        CUDA_CHECK(cudaMallocAsync(&d_rho_core_, Nd_ * sizeof(double), 0));
        CUDA_CHECK(cudaMemcpy(d_rho_core_, rho_core, Nd_ * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Upload pseudocharge to GPU
    CUDA_CHECK(cudaMallocAsync(&d_pseudocharge_, Nd_ * sizeof(double), 0));
    CUDA_CHECK(cudaMemcpy(d_pseudocharge_, rho_b, Nd_ * sizeof(double), cudaMemcpyHostToDevice));

    // Allocate mixer persistent buffer
    int mix_N = has_soc_ ? Nd_ * 4 : Nd_ * ((Nspin >= 2) ? 2 : 1);
    CUDA_CHECK(cudaMallocAsync(&d_mix_fkm1_, mix_N * sizeof(double), 0));

    // d_Y must be separate from d_Hpsi (used as d_HX inside eigensolver)
    // For SOC: spinor dimension is 2*Nd_d, complex
    // For k-point: complex-sized (2x) since Y is used as cuDoubleComplex*
    int Nd_eigensolver = has_soc_ ? 2 * Nd_ : Nd_;
    size_t y_elem_size = is_kpt ? sizeof(cuDoubleComplex) : sizeof(double);
    CUDA_CHECK(cudaMallocAsync(&d_Y_, (size_t)Nd_eigensolver * Nband * y_elem_size, 0));

    // Device pointers (aliases into ctx.buf)
    double* d_psi     = ctx.buf.psi;
    double* d_Hpsi    = ctx.buf.Hpsi;
    double* d_Veff    = ctx.buf.Veff;
    double* d_rho     = ctx.buf.rho;
    double* d_rho_new = ctx.buf.rho_total;
    double* d_phi     = ctx.buf.phi;
    double* d_exc     = ctx.buf.exc;
    double* d_Vxc     = (Nspin >= 2 || has_soc_) ? ctx.buf.Vxc : ctx.buf.Vc;
    double* d_eigvals = ctx.buf.eigenvalues;
    double* d_occ     = ctx.buf.occupations;
    double* d_Xold    = ctx.buf.Xold;
    double* d_Xnew    = ctx.buf.Xnew;
    double* d_x_ex    = ctx.buf.x_ex;
    double* d_Hs      = ctx.buf.Hs;
    double* d_Ms      = ctx.buf.Ms;

    // ============================================================
    // Upload initial density
    // ============================================================
    if (Nspin == 2 && rho_up_init && rho_dn_init) {
        // Spin-polarized: d_rho = [up(Nd) | down(Nd)]
        CUDA_CHECK(cudaMemcpy(d_rho, rho_up_init, Nd_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rho + Nd_, rho_dn_init, Nd_ * sizeof(double), cudaMemcpyHostToDevice));
        // Also upload total density for Poisson
        CUDA_CHECK(cudaMemcpy(d_rho_new, rho_init, Nd_ * sizeof(double), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_rho, rho_init, Nd_ * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Number of k-points
    int Nkpts = is_kpt_ ? (int)kpt_weights_in.size() : 1;

    // Spin-resolved per-spin device pointers
    double* d_psi_arr[2]     = { d_psi, ctx.buf.psi_s1 };
    double* d_Hpsi_arr[2]    = { d_Hpsi, ctx.buf.Hpsi_s1 };
    double* d_eigvals_arr[2] = { d_eigvals, ctx.buf.eigenvalues_s1 };
    double* d_occ_arr[2]     = { d_occ, ctx.buf.occupations_s1 };
    double* d_Y_arr[2]       = { d_Y_, d_Y_s1_ };

    // Complex buffer aliases
    cuDoubleComplex* d_psi_z  = static_cast<cuDoubleComplex*>(ctx.buf.psi_z);
    cuDoubleComplex* d_Hpsi_z = static_cast<cuDoubleComplex*>(ctx.buf.Hpsi_z);
    cuDoubleComplex* d_Xold_z = static_cast<cuDoubleComplex*>(ctx.buf.Xold_z);
    cuDoubleComplex* d_Xnew_z = static_cast<cuDoubleComplex*>(ctx.buf.Xnew_z);
    cuDoubleComplex* d_x_ex_z = static_cast<cuDoubleComplex*>(ctx.buf.x_ex_z);
    cuDoubleComplex* d_Hs_z   = static_cast<cuDoubleComplex*>(ctx.buf.Hs_z);
    cuDoubleComplex* d_Ms_z   = static_cast<cuDoubleComplex*>(ctx.buf.Ms_z);

    // Allocate complex alpha if k-point
    if (is_kpt_ && gpu_vnl_.total_phys_nproj > 0) {
        CUDA_CHECK(cudaMallocAsync(&d_alpha_z_,
            (size_t)gpu_vnl_.total_phys_nproj * Nband * sizeof(cuDoubleComplex), 0));
    }

    // Allocate d_Y_s1_ if spin-polarized (non-SOC)
    if (Nspin >= 2 && !has_soc_ && !d_Y_s1_) {
        CUDA_CHECK(cudaMallocAsync(&d_Y_s1_, (size_t)Nd_ * Nband * y_elem_size, 0));
        d_Y_arr[1] = d_Y_s1_;
    }

    // SOC-specific: allocate spinor psi buffer (2*Nd_d complex per band)
    // We reuse d_psi_z but need to ensure it can hold the spinor dimension
    cuDoubleComplex* d_psi_spinor = nullptr;
    cuDoubleComplex* d_Hpsi_spinor = nullptr;
    cuDoubleComplex* d_Xold_spinor = nullptr;
    cuDoubleComplex* d_Xnew_spinor = nullptr;
    cuDoubleComplex* d_x_ex_spinor = nullptr;
    cuDoubleComplex* d_Hs_spinor = nullptr;
    cuDoubleComplex* d_Ms_spinor = nullptr;
    double* d_Veff_spinor = nullptr;
    double* d_mag_x = nullptr, *d_mag_y = nullptr, *d_mag_z = nullptr;
    double* d_rho_soc = nullptr;

    if (has_soc_) {
        int Nd_spinor = 2 * Nd_;
        // Allocate spinor-sized complex buffers
        CUDA_CHECK(cudaMallocAsync(&d_psi_spinor, (size_t)Nd_spinor * Nband * sizeof(cuDoubleComplex), 0));
        CUDA_CHECK(cudaMallocAsync(&d_Hpsi_spinor, (size_t)Nd_spinor * Nband * sizeof(cuDoubleComplex), 0));
        CUDA_CHECK(cudaMallocAsync(&d_Xold_spinor, (size_t)Nd_spinor * Nband * sizeof(cuDoubleComplex), 0));
        CUDA_CHECK(cudaMallocAsync(&d_Xnew_spinor, (size_t)Nd_spinor * Nband * sizeof(cuDoubleComplex), 0));
        int nx_ex = nx_ + 2 * FDn_, ny_ex = ny_ + 2 * FDn_, nz_ex = nz_ + 2 * FDn_;
        int Nd_ex = nx_ex * ny_ex * nz_ex;
        CUDA_CHECK(cudaMallocAsync(&d_x_ex_spinor, (size_t)Nd_ex * sizeof(cuDoubleComplex), 0));
        CUDA_CHECK(cudaMallocAsync(&d_Hs_spinor, (size_t)Nband * Nband * sizeof(cuDoubleComplex), 0));
        CUDA_CHECK(cudaMallocAsync(&d_Ms_spinor, (size_t)Nband * Nband * sizeof(cuDoubleComplex), 0));

        // 4-component Veff: [V_uu | V_dd | V_ud_re | V_ud_im]
        CUDA_CHECK(cudaMallocAsync(&d_Veff_spinor, 4 * Nd_ * sizeof(double), 0));

        // Noncollinear density arrays: rho, mag_x, mag_y, mag_z
        CUDA_CHECK(cudaMallocAsync(&d_rho_soc, Nd_ * sizeof(double), 0));
        CUDA_CHECK(cudaMallocAsync(&d_mag_x, Nd_ * sizeof(double), 0));
        CUDA_CHECK(cudaMallocAsync(&d_mag_y, Nd_ * sizeof(double), 0));
        CUDA_CHECK(cudaMallocAsync(&d_mag_z, Nd_ * sizeof(double), 0));
    }

    // Randomize wavefunctions on CPU, upload to GPU
    if (has_soc_) {
        // SOC: single spin channel (s=0), spinor randomization
        wfn.randomize_kpt(0, 0, 42);
        // The complex spinor psi will be uploaded per-kpt inside the SCF loop
    } else {
        for (int s = 0; s < Nspin_local_; s++) {
            if (is_kpt_) {
                wfn.randomize_kpt(s, 0, 42 + s);
            } else {
                wfn.randomize(s, 0, 1 + s);
                std::vector<double> h_psi(Nd_ * Nband);
                for (int j = 0; j < Nband; j++)
                    std::memcpy(h_psi.data() + j * Nd_, wfn.psi(s, 0).col(j), Nd_ * sizeof(double));
                int s_glob = spin_start_ + s;
                CUDA_CHECK(cudaMemcpy(d_psi_arr[s_glob], h_psi.data(),
                                       (size_t)Nd_ * Nband * sizeof(double), cudaMemcpyHostToDevice));
            }
        }
    }

    // ============================================================
    // Initial Veff (CPU path — runs once)
    // ============================================================
    {
        std::vector<double> h_rho(rho_init, rho_init + Nd_);
        int vxc_size = Nd_ * Nspin;
        int veff_size = Nd_ * Nspin;
        std::vector<double> h_Vxc(vxc_size), h_exc(Nd_), h_phi(Nd_), h_Veff(veff_size);

        // XC on CPU — initial only
        {
            Gradient gradient_cpu(stencil, domain);
            XCFunctional xcfunc;
            xcfunc.setup(xc_type, domain, grid, &gradient_cpu, &halo);

            if (Nspin == 2 && rho_up_init && rho_dn_init) {
                // Spin-polarized: build rho_xc = [total | up | down]
                std::vector<double> rho_xc(3 * Nd_);
                for (int i = 0; i < Nd_; i++) {
                    double rup = rho_up_init[i] + (has_nlcc_ ? 0.5 * rho_core[i] : 0.0);
                    double rdn = rho_dn_init[i] + (has_nlcc_ ? 0.5 * rho_core[i] : 0.0);
                    rho_xc[i] = std::max(rup + rdn, 1e-14);
                    rho_xc[Nd_ + i] = std::max(rup, 1e-14);
                    rho_xc[2*Nd_ + i] = std::max(rdn, 1e-14);
                }
                xcfunc.evaluate_spin(rho_xc.data(), h_Vxc.data(), h_exc.data(), Nd_);
            } else {
                std::vector<double> rho_xc(Nd_);
                for (int i = 0; i < Nd_; i++) {
                    rho_xc[i] = h_rho[i] + (has_nlcc_ ? rho_core[i] : 0.0);
                    if (rho_xc[i] < 1e-14) rho_xc[i] = 1e-14;
                }
                xcfunc.evaluate(rho_xc.data(), h_Vxc.data(), h_exc.data(), Nd_);
            }
        }
        CUDA_CHECK(cudaMemcpy(d_exc, h_exc.data(), Nd_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Vxc, h_Vxc.data(), vxc_size * sizeof(double), cudaMemcpyHostToDevice));

        // Poisson on CPU — initial only (uses total density)
        std::vector<double> rhs(Nd_);
        for (int i = 0; i < Nd_; i++)
            rhs[i] = 4.0 * constants::PI * (h_rho[i] + rho_b[i]);
        double rhs_sum = 0;
        for (int i = 0; i < Nd_; i++) rhs_sum += rhs[i];
        double rhs_mean = rhs_sum / grid.Nd();
        for (int i = 0; i < Nd_; i++) rhs[i] -= rhs_mean;

        Laplacian laplacian(stencil, domain);
        PoissonSolver poisson_init;
        poisson_init.setup(laplacian, stencil, domain, grid, halo);
        std::fill(h_phi.begin(), h_phi.end(), 0.0);

        double poisson_tol = (params.poisson_tol > 0) ? params.poisson_tol : params.tol * 0.01;
        poisson_init.solve(rhs.data(), h_phi.data(), poisson_tol);

        double phi_sum = 0;
        for (int i = 0; i < Nd_; i++) phi_sum += h_phi[i];
        double phi_mean = phi_sum / grid.Nd();
        for (int i = 0; i < Nd_; i++) h_phi[i] -= phi_mean;

        // Veff per spin: Veff_s = Vxc_s + phi
        for (int s = 0; s < Nspin; s++) {
            for (int i = 0; i < Nd_; i++)
                h_Veff[s * Nd_ + i] = h_Vxc[s * Nd_ + i] + h_phi[i];
        }

        {
            double vxc_sum = 0, veff_sum = 0, phi_s = 0, exc_s = 0;
            for (int i = 0; i < Nd_; i++) { vxc_sum += h_Vxc[i]; veff_sum += h_Veff[i]; phi_s += h_phi[i]; exc_s += h_exc[i]; }
            printf("GPUSCF init: Vxc_sum=%.6e phi_sum=%.6e Veff_sum=%.6e exc_sum=%.6e\n",
                   vxc_sum, phi_s, veff_sum, exc_s);
            printf("GPUSCF init: Veff[0]=%.15e Veff[100]=%.15e Veff[1000]=%.15e\n",
                   h_Veff[0], h_Veff[std::min(100,Nd_-1)], h_Veff[std::min(1000,Nd_-1)]);
        }

        CUDA_CHECK(cudaMemcpy(d_Veff, h_Veff.data(), veff_size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_phi, h_phi.data(), Nd_ * sizeof(double), cudaMemcpyHostToDevice));

        // SOC: build initial Veff_spinor from scalar Veff (uniform rho, zero magnetization)
        // V_uu = V_dd = Vxc + phi, V_ud = 0
        if (has_soc_) {
            std::vector<double> h_Veff_spinor(4 * Nd_, 0.0);
            for (int i = 0; i < Nd_; i++) {
                double v = h_Vxc[i] + h_phi[i];  // Vxc is non-spin here (initial)
                h_Veff_spinor[i]           = v;   // V_uu
                h_Veff_spinor[Nd_ + i]     = v;   // V_dd
                // V_ud_re and V_ud_im = 0 (already zero-initialized)
            }
            CUDA_CHECK(cudaMemcpy(d_Veff_spinor, h_Veff_spinor.data(),
                                   4 * Nd_ * sizeof(double), cudaMemcpyHostToDevice));

            // Initialize noncollinear density: rho = total, mx = my = mz = 0
            CUDA_CHECK(cudaMemcpy(d_rho_soc, rho_init, Nd_ * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_mag_x, 0, Nd_ * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_mag_y, 0, Nd_ * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_mag_z, 0, Nd_ * sizeof(double)));

            // Also set d_rho for spin XC: [rho_up | rho_dn] = [0.5*rho | 0.5*rho]
            {
                int bs = 256;
                int gs = gpu::ceildiv(Nd_, bs);
                scale_kernel<<<gs, bs>>>(d_rho_soc, 1.0, Nd_);  // noop, just ensure sync
                // d_rho holds [up|dn] for XC
                std::vector<double> h_rho_half(Nd_);
                for (int i = 0; i < Nd_; i++) h_rho_half[i] = 0.5 * rho_init[i];
                CUDA_CHECK(cudaMemcpy(d_rho, h_rho_half.data(), Nd_ * sizeof(double), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_rho + Nd_, h_rho_half.data(), Nd_ * sizeof(double), cudaMemcpyHostToDevice));
            }
        }
    }

    // ============================================================
    // Lanczos spectrum bounds (CPU — runs once)
    // ============================================================
    EigenSolver eigsolver;
    eigsolver.setup(hamiltonian, halo, domain, bandcomm, Nband);

    // Per-spin spectral bounds
    int Nspin_lanczos = has_soc_ ? 1 : Nspin;
    std::vector<double> eigval_min_s(std::max(Nspin, 1), 0.0), eigval_max_s(std::max(Nspin, 1), 0.0);
    {
        if (has_soc_) {
            // SOC: use spinor Lanczos on CPU (4-component Veff_spinor)
            std::vector<double> h_Veff_spinor(4 * Nd_);
            CUDA_CHECK(cudaMemcpy(h_Veff_spinor.data(), d_Veff_spinor,
                                   4 * Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
            Vec3 kpt0_est = kpoints_->kpts_cart()[kpt_start_];
            Vec3 cell_lengths = grid.lattice().lengths();
            eigsolver.lanczos_bounds_spinor_kpt(h_Veff_spinor.data(), Nd_,
                                                 kpt0_est, cell_lengths,
                                                 eigval_min_s[0], eigval_max_s[0]);
        } else {
            std::vector<double> h_Veff(Nd_);
            CUDA_CHECK(cudaMemcpy(h_Veff.data(), d_Veff, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
            if (is_kpt_) {
                Vec3 kpt0_est = kpoints_->kpts_cart()[kpt_start_];
                Vec3 cell_lengths = grid.lattice().lengths();
                eigsolver.lanczos_bounds_kpt(h_Veff.data(), Nd_, kpt0_est, cell_lengths,
                                              eigval_min_s[0], eigval_max_s[0]);
            } else {
                eigsolver.lanczos_bounds(h_Veff.data(), Nd_, eigval_min_s[0], eigval_max_s[0]);
            }
            for (int s = 1; s < Nspin; s++) {
                eigval_min_s[s] = eigval_min_s[0];
                eigval_max_s[s] = eigval_max_s[0];
            }
        }
    }
    // For SOC: Lanczos on CPU may underestimate eigmax because the spinor Hamiltonian
    // includes SOC terms that aren't captured by short Lanczos runs.
    // Use a power-iteration-like estimate: eigmax ≈ ||H*x||/||x|| for random x on GPU.
    if (has_soc_) {
        // Use GPU to estimate eigmax via Rayleigh quotient
        auto& ctx = gpu::GPUContext::instance();
        int Nd_spinor = 2 * Nd_;
        // Set up callback state and Bloch phases for first k-point
        s_instance_ = this;
        Vec3 kpt0_est = kpoints_->kpts_cart()[kpt_start_];
        Vec3 cell_len = grid.lattice().lengths();
        kxLx_ = kpt0_est.x * cell_len.x;
        kyLy_ = kpt0_est.y * cell_len.y;
        kzLz_ = kpt0_est.z * cell_len.z;
        setup_bloch_factors(nloc_influence, crystal, kpt0_est);
        // Randomize spinor psi on GPU
        wfn.randomize_kpt(0, 0, 42);
        {
            size_t psi_bytes = 2 * (size_t)Nd_spinor * Nband * sizeof(double);
            std::vector<double> h_psi_z(2 * Nd_spinor * Nband);
            std::memcpy(h_psi_z.data(), wfn.psi_kpt(0, 0).data(), psi_bytes);
            CUDA_CHECK(cudaMemcpy(d_psi_spinor, h_psi_z.data(), psi_bytes, cudaMemcpyHostToDevice));
        }
        cuDoubleComplex* d_test = d_psi_spinor;
        cuDoubleComplex* d_Htest = d_Hpsi_spinor;
        // For SOC, the Lanczos may significantly underestimate eigmax.
        // Use a safe bound: compute from stencil coefficients + SOC Gamma.
        // The kinetic eigmax ~ |diag_coeff_ham_| * 2, SOC adds Gamma_soc * ndc * dV
        double kinetic_bound = std::abs(diag_coeff_ham_) * 2.0;
        double soc_bound = 0.5 * 5.0 * 50.0;  // conservative: 0.5 * max_Gamma * max_factor
        double eigmax_safe = std::max(kinetic_bound + soc_bound, eigval_max_s[0]) * 1.2;
        printf("SOC eigmax: Lanczos=%.2f, kinetic=%.2f, using=%.2f\n",
               eigval_max_s[0], kinetic_bound, eigmax_safe);
        eigval_max_s[0] = eigmax_safe;
    }
    double lambda_cutoff = 0.5 * (eigval_min_s[0] + eigval_max_s[0]);
    printf("Lanczos: eigmin=%.6e, eigmax=%.6e, lambda_cutoff=%.6e\n",
           eigval_min_s[0], eigval_max_s[0], lambda_cutoff);

    // Auto Chebyshev degree
    int cheb_degree = params.cheb_degree;
    if (cheb_degree <= 0) {
        double p3 = -700.0/3, p2 = 1240.0/3, p1 = -773.0/3, p0 = 1078.0/15;
        cheb_degree = (h_eff > 0.7) ? 14 : (int)std::round(((p3*h_eff + p2)*h_eff + p1)*h_eff + p0);
    }
    printf("Chebyshev degree: %d (h_eff=%.6f)\n", cheb_degree, h_eff);

    // Smearing parameters: params.elec_temp is in Kelvin (or -1 for auto)
    double elec_temp_K = params.elec_temp;
    if (elec_temp_K <= 0) {
        // Auto: 0.2 eV for Gaussian, 0.1 eV for FD
        double smearing_eV = (params.smearing == SmearingType::GaussianSmearing) ? 0.2 : 0.1;
        double beta_au = constants::EH / smearing_eV;
        elec_temp_K = 1.0 / (constants::KB * beta_au);
    }
    double kBT = elec_temp_K * constants::KB;
    double beta_smearing = 1.0 / kBT;

    // SCF parameters
    int max_iter = params.max_iter;
    double scf_tol = params.tol;
    int rho_trigger = params.rho_trigger;
    int nchefsi_per_iter = params.nchefsi;
    double mixing_param = params.mixing_param;
    int mixing_history = params.mixing_history;
    double poisson_tol = (params.poisson_tol > 0) ? params.poisson_tol : scf_tol * 0.01;

    // ============================================================
    // Set static callback state
    // ============================================================
    s_instance_ = this;

    // ============================================================
    // H*psi validation for SOC spinor
    // ============================================================
    if (has_soc_ && d_psi_spinor && d_Hpsi_spinor && d_Veff_spinor) {
        using Complex = std::complex<double>;
        int Nd_spinor = 2 * Nd_;
        Vec3 kpt0v = kpoints_->kpts_cart()[kpt_start_];
        Vec3 cell_len_v = grid.lattice().lengths();
        kxLx_ = kpt0v.x * cell_len_v.x;
        kyLy_ = kpt0v.y * cell_len_v.y;
        kzLz_ = kpt0v.z * cell_len_v.z;
        setup_bloch_factors(nloc_influence, crystal, kpt0v);

        // Upload random spinor psi
        wfn.randomize_kpt(0, 0, 99);
        {
            size_t bytes = 2 * (size_t)Nd_spinor * sizeof(double);
            std::vector<double> h_psi(2 * Nd_spinor);
            auto* pc = wfn.psi_kpt(0, 0).col(0);
            std::memcpy(h_psi.data(), pc, bytes);
            CUDA_CHECK(cudaMemcpy(d_psi_spinor, h_psi.data(), bytes, cudaMemcpyHostToDevice));
        }

        // === Staged GPU vs CPU comparison ===
        const double* V_uu = d_Veff_spinor;
        const double* V_dd = d_Veff_spinor + Nd_;

        // Stage 1: LOCAL only (kinetic + Veff per spinor)
        {
            cuDoubleComplex* psi_up = d_psi_spinor;
            cuDoubleComplex* psi_dn = d_psi_spinor + Nd_;
            cuDoubleComplex* Hpsi_up = d_Hpsi_spinor;
            cuDoubleComplex* Hpsi_dn = d_Hpsi_spinor + Nd_;
            gpu::hamiltonian_apply_local_z_gpu(psi_up, V_uu, Hpsi_up, d_x_ex_spinor,
                nx_, ny_, nz_, FDn_, 1, 0.0, is_orth_, true, true, true,
                diag_coeff_ham_, has_mixed_deriv_, has_mixed_deriv_, has_mixed_deriv_,
                kxLx_, kyLy_, kzLz_);
            gpu::hamiltonian_apply_local_z_gpu(psi_dn, V_dd, Hpsi_dn, d_x_ex_spinor,
                nx_, ny_, nz_, FDn_, 1, 0.0, is_orth_, true, true, true,
                diag_coeff_ham_, has_mixed_deriv_, has_mixed_deriv_, has_mixed_deriv_,
                kxLx_, kyLy_, kzLz_);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        std::vector<Complex> gpu_local(Nd_spinor);
        CUDA_CHECK(cudaMemcpy(gpu_local.data(), d_Hpsi_spinor, Nd_spinor * sizeof(Complex), cudaMemcpyDeviceToHost));

        // CPU local only
        std::vector<double> h_Veff_sp(4 * Nd_);
        CUDA_CHECK(cudaMemcpy(h_Veff_sp.data(), d_Veff_spinor, 4 * Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
        const Complex* psi_ptr = reinterpret_cast<const Complex*>(wfn.psi_kpt(0, 0).col(0));
        std::vector<Complex> cpu_local(Nd_spinor, Complex(0));
        // Use CPU Hamiltonian for local only (apply_local_kpt per spinor)
        hamiltonian.apply_local_kpt(psi_ptr, h_Veff_sp.data(), cpu_local.data(), 1, kpt0v, cell_len_v);
        hamiltonian.apply_local_kpt(psi_ptr + Nd_, h_Veff_sp.data() + Nd_, cpu_local.data() + Nd_, 1, kpt0v, cell_len_v);

        double max_err_local = 0, norm_local = 0;
        for (int i = 0; i < Nd_spinor; i++) {
            max_err_local = std::max(max_err_local, std::abs(gpu_local[i] - cpu_local[i]));
            norm_local += std::norm(cpu_local[i]);
        }
        double rel_err_local = max_err_local / std::sqrt(norm_local / Nd_spinor);
        printf("SOC H*psi LOCAL: max_err=%.3e, rel_err=%.3e %s\n",
               max_err_local, rel_err_local, rel_err_local < 1e-10 ? "[PASS]" : "[FAIL]");

        // Stage 2: FULL H (local + Vnl + V_ud + SOC)
        hamiltonian_apply_spinor_z_cb(d_psi_spinor, d_Veff_spinor, d_Hpsi_spinor, d_x_ex_spinor, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<Complex> gpu_Hpsi(Nd_spinor);
        CUDA_CHECK(cudaMemcpy(gpu_Hpsi.data(), d_Hpsi_spinor, Nd_spinor * sizeof(Complex), cudaMemcpyDeviceToHost));

        std::vector<Complex> cpu_Hpsi(Nd_spinor, Complex(0));
        const_cast<NonlocalProjector*>(vnl)->set_kpoint(kpt0v);
        hamiltonian.apply_spinor_kpt(psi_ptr, h_Veff_sp.data(), cpu_Hpsi.data(), 1, Nd_,
                                      kpt0v, cell_len_v);

        double max_err = 0, norm_cpu = 0;
        for (int i = 0; i < Nd_spinor; i++) {
            max_err = std::max(max_err, std::abs(gpu_Hpsi[i] - cpu_Hpsi[i]));
            norm_cpu += std::norm(cpu_Hpsi[i]);
        }
        double rel_err = max_err / std::sqrt(norm_cpu / Nd_spinor);
        printf("SOC H*psi FULL:  max_err=%.3e, rel_err=%.3e %s\n",
               max_err, rel_err, rel_err < 1e-10 ? "[PASS]" : "[FAIL]");
    }

    // ============================================================
    // GPU SCF Loop
    // ============================================================
    printf("\n--- GPU SCF Loop (Nspin=%d, Nkpts=%d, %s%s) ---\n",
           Nspin, Nkpts, is_kpt_ ? "k-point" : "gamma",
           has_soc_ ? ", SOC" : "");

    std::vector<double> h_eigvals(Nband), h_occ(Nband);
    std::vector<double> kpt_weights = kpt_weights_in;
    double Ef = 0.0;

    // Host-side density for energy evaluation
    std::vector<double> h_rho_core_vec;
    if (has_nlcc_) {
        h_rho_core_vec.assign(rho_core, rho_core + Nd_);
    }

    // Cell lengths for k-point Bloch phases
    Vec3 cell_lengths = {0, 0, 0};
    if (is_kpt_ && kpoints_) {
        cell_lengths = grid.lattice().lengths();
    }

    // Density weight: occfac * wk
    // For Nspin=1: occfac=2; for Nspin=2: occfac=1 (each spin contributes separately)
    double occfac = (Nspin == 1) ? 2.0 : 1.0;

    for (int scf_iter = 0; scf_iter < max_iter; scf_iter++) {
        int nchefsi = (scf_iter == 0) ? rho_trigger : nchefsi_per_iter;

        // Step 1: Eigensolver (nchefsi passes, loop over spin and k-points)
        for (int pass = 0; pass < nchefsi; pass++) {
          if (has_soc_) {
            // ---- SOC spinor eigensolver ----
            int Nd_spinor = 2 * Nd_;
            for (int k = 0; k < Nkpts; k++) {
                int k_glob = kpt_start_ + k;
                Vec3 kpt = kpoints_->kpts_cart()[k_glob];

                kxLx_ = kpt.x * cell_lengths.x;
                kyLy_ = kpt.y * cell_lengths.y;
                kzLz_ = kpt.z * cell_lengths.z;
                setup_bloch_factors(nloc_influence, crystal, kpt);

                // Upload spinor psi from CPU if first pass of first iter
                if (scf_iter == 0 && pass == 0) {
                    wfn.randomize_kpt(0, k, 42 + k);
                    size_t psi_bytes = 2 * (size_t)Nd_spinor * Nband * sizeof(double);
                    std::vector<double> h_psi_z(2 * Nd_spinor * Nband);
                    auto* psi_c = wfn.psi_kpt(0, k).data();
                    std::memcpy(h_psi_z.data(), psi_c, psi_bytes);
                    CUDA_CHECK(cudaMemcpy(d_psi_spinor, h_psi_z.data(),
                        psi_bytes, cudaMemcpyHostToDevice));
                }

                // Spinor eigensolver: Nd = 2*Nd_d, uses hamiltonian_apply_spinor_z_cb
                gpu::eigensolver_solve_z_gpu(
                    d_psi_spinor, d_eigvals, d_Veff_spinor,
                    reinterpret_cast<cuDoubleComplex*>(d_Y_),
                    d_Xold_spinor, d_Xnew_spinor,
                    d_Hpsi_spinor, d_x_ex_spinor,
                    d_Hs_spinor, d_Ms_spinor,
                    Nd_spinor, Nband,
                    lambda_cutoff, eigval_min_s[0], eigval_max_s[0],
                    cheb_degree, dV_,
                    hamiltonian_apply_spinor_z_cb);

                // Download eigenvalues + spinor psi
                CUDA_CHECK(cudaDeviceSynchronize());
                {
                    CUDA_CHECK(cudaMemcpy(h_eigvals.data(), d_eigvals,
                                           Nband * sizeof(double), cudaMemcpyDeviceToHost));
                    if (scf_iter == 0 && k == 0) {
                        printf("  [SOC eigvals] eig[0]=%.6e eig[%d]=%.6e\n",
                               h_eigvals[0], Nband-1, h_eigvals[Nband-1]);
                    }
                    for (int n = 0; n < Nband; n++)
                        wfn.eigenvalues(0, k)(n) = h_eigvals[n];
                }
                {
                    size_t psi_bytes = 2 * (size_t)Nd_spinor * Nband * sizeof(double);
                    std::vector<double> h_psi_z(2 * Nd_spinor * Nband);
                    CUDA_CHECK(cudaMemcpy(h_psi_z.data(), d_psi_spinor,
                        psi_bytes, cudaMemcpyDeviceToHost));
                    if (scf_iter == 0 && k == 0) {
                        printf("  [SOC psi] psi[0..3]=%.3e %.3e %.3e %.3e\n",
                               h_psi_z[0], h_psi_z[1], h_psi_z[2], h_psi_z[3]);
                    }
                    std::memcpy(wfn.psi_kpt(0, k).data(), h_psi_z.data(), psi_bytes);
                }
            }
          } else {
            for (int s = 0; s < Nspin_local_; s++) {
                int s_glob = spin_start_ + s;
                double* d_Veff_s = d_Veff + s_glob * Nd_;

                if (is_kpt_) {
                    // Complex k-point path
                    for (int k = 0; k < Nkpts; k++) {
                        int k_glob = kpt_start_ + k;
                        Vec3 kpt = kpoints_->kpts_cart()[k_glob];

                        // Set Bloch phases for this k-point
                        kxLx_ = kpt.x * cell_lengths.x;
                        kyLy_ = kpt.y * cell_lengths.y;
                        kzLz_ = kpt.z * cell_lengths.z;
                        setup_bloch_factors(nloc_influence, crystal, kpt);

                        // Upload complex psi from CPU if first pass of first iter
                        if (scf_iter == 0 && pass == 0) {
                            wfn.randomize_kpt(s, k, 42 + s * Nkpts + k);
                            std::vector<double> h_psi_z(2 * Nd_ * Nband);
                            auto* psi_c = wfn.psi_kpt(s, k).data();
                            // cuDoubleComplex layout matches std::complex<double>
                            std::memcpy(h_psi_z.data(), psi_c, 2 * Nd_ * Nband * sizeof(double));
                            CUDA_CHECK(cudaMemcpy(d_psi_z, h_psi_z.data(),
                                2 * (size_t)Nd_ * Nband * sizeof(double), cudaMemcpyHostToDevice));
                        }

                        gpu::eigensolver_solve_z_gpu(
                            d_psi_z, d_eigvals_arr[s_glob], d_Veff_s,
                            reinterpret_cast<cuDoubleComplex*>(d_Y_arr[s_glob]),
                            d_Xold_z, d_Xnew_z,
                            d_Hpsi_z, d_x_ex_z,
                            d_Hs_z, d_Ms_z,
                            Nd_, Nband,
                            lambda_cutoff, eigval_min_s[s_glob], eigval_max_s[s_glob],
                            cheb_degree, dV_,
                            hamiltonian_apply_z_cb);

                        // Download eigenvalues + complex psi back to CPU wfn
                        CUDA_CHECK(cudaDeviceSynchronize());
                        {
                            // Store eigenvalues per (s, k)
                            CUDA_CHECK(cudaMemcpy(h_eigvals.data(), d_eigvals_arr[s_glob],
                                                   Nband * sizeof(double), cudaMemcpyDeviceToHost));
                            for (int n = 0; n < Nband; n++)
                                wfn.eigenvalues(s, k)(n) = h_eigvals[n];
                        }
                        {
                            std::vector<double> h_psi_z(2 * Nd_ * Nband);
                            CUDA_CHECK(cudaMemcpy(h_psi_z.data(), d_psi_z,
                                2 * (size_t)Nd_ * Nband * sizeof(double), cudaMemcpyDeviceToHost));
                            std::memcpy(wfn.psi_kpt(s, k).data(), h_psi_z.data(),
                                        2 * Nd_ * Nband * sizeof(double));
                        }
                    }
                } else {
                    // Real gamma-point path
                    gpu::eigensolver_solve_gpu(
                        d_psi_arr[s_glob], d_eigvals_arr[s_glob], d_Veff_s,
                        d_Y_arr[s_glob], d_Xold, d_Xnew,
                        d_Hpsi_arr[s_glob], d_x_ex,
                        d_Hs, d_Ms,
                        Nd_, Nband,
                        lambda_cutoff, eigval_min_s[s_glob], eigval_max_s[s_glob],
                        cheb_degree, dV_,
                        hamiltonian_apply_cb);
                }
            }
          } // end non-SOC eigensolver branch

            // Update spectral bounds from eigenvalues (per-spin)
            CUDA_CHECK(cudaDeviceSynchronize());
            if (has_soc_) {
                // SOC: single spin channel
                const double* eig0 = wfn.eigenvalues(0, 0).data();
                if (scf_iter > 0)
                    eigval_min_s[0] = eig0[0];
                lambda_cutoff = eig0[Nband - 1] + 0.1;
            } else {
                double eig_last_max = -1e30;
                for (int s = 0; s < Nspin_local_; s++) {
                    int s_glob = spin_start_ + s;
                    if (!is_kpt_) {
                        // Gamma: download eigenvalues from device
                        CUDA_CHECK(cudaMemcpy(h_eigvals.data(), d_eigvals_arr[s_glob],
                                               Nband * sizeof(double), cudaMemcpyDeviceToHost));
                        for (int n = 0; n < Nband; n++)
                            wfn.eigenvalues(s, 0)(n) = h_eigvals[n];
                    }
                    // For k-points, eigenvalues already stored per (s,k) in inner loop

                    // Use first k-point's eigenvalues for spectral bounds
                    const double* eig0 = wfn.eigenvalues(s, 0).data();
                    if (scf_iter > 0)
                        eigval_min_s[s_glob] = eig0[0];
                    if (eig0[Nband - 1] > eig_last_max)
                        eig_last_max = eig0[Nband - 1];
                }
                lambda_cutoff = eig_last_max + 0.1;
            }

            // Compute occupations on CPU (lightweight)
            Ef = Occupation::compute(wfn, Nelectron, beta_smearing,
                                    params.smearing,
                                    kpt_weights, bandcomm, bandcomm, 0);
        }

        // Upload occupations to GPU
        if (has_soc_) {
            for (int n = 0; n < Nband; n++)
                h_occ[n] = wfn.occupations(0, 0)(n);
            CUDA_CHECK(cudaMemcpy(d_occ, h_occ.data(), Nband * sizeof(double), cudaMemcpyHostToDevice));
        } else {
            for (int s = 0; s < Nspin_local_; s++) {
                int s_glob = spin_start_ + s;
                for (int n = 0; n < Nband; n++)
                    h_occ[n] = wfn.occupations(s, 0)(n);
                CUDA_CHECK(cudaMemcpy(d_occ_arr[s_glob], h_occ.data(),
                                       Nband * sizeof(double), cudaMemcpyHostToDevice));
            }
        }

        // Step 2: Compute new density on GPU
        double* d_rho_new_total = d_rho_new;  // total density (Nd)
        CUDA_CHECK(cudaMemset(d_rho_new_total, 0, Nd_ * sizeof(double)));

        // For spin-polarized / SOC: also need per-spin new density
        auto& sp_dens = ctx.scratch_pool;
        size_t sp_dens_cp = sp_dens.checkpoint();
        double* d_rho_new_up = nullptr;
        double* d_rho_new_dn = nullptr;

        // SOC-specific scratch for new noncollinear density
        double* d_rho_new_soc = nullptr;
        double* d_mag_x_new = nullptr, *d_mag_y_new = nullptr, *d_mag_z_new = nullptr;

        if (has_soc_) {
            // SOC: use spinor_density_gpu to compute (rho, mx, my, mz)
            d_rho_new_soc = sp_dens.alloc<double>(Nd_);
            d_mag_x_new   = sp_dens.alloc<double>(Nd_);
            d_mag_y_new   = sp_dens.alloc<double>(Nd_);
            d_mag_z_new   = sp_dens.alloc<double>(Nd_);
            CUDA_CHECK(cudaMemset(d_rho_new_soc, 0, Nd_ * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_mag_x_new, 0, Nd_ * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_mag_y_new, 0, Nd_ * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_mag_z_new, 0, Nd_ * sizeof(double)));

            int Nd_spinor = 2 * Nd_;
            for (int k = 0; k < Nkpts; k++) {
                double wk = kpt_weights[k];
                // Upload spinor psi for this k-point
                {
                    size_t psi_bytes = 2 * (size_t)Nd_spinor * Nband * sizeof(double);
                    std::vector<double> h_psi_z(2 * Nd_spinor * Nband);
                    std::memcpy(h_psi_z.data(), wfn.psi_kpt(0, k).data(), psi_bytes);
                    CUDA_CHECK(cudaMemcpy(d_psi_spinor, h_psi_z.data(),
                        psi_bytes, cudaMemcpyHostToDevice));
                }
                // Upload per-k occupations
                {
                    std::vector<double> h_occ_k(Nband);
                    for (int n = 0; n < Nband; n++)
                        h_occ_k[n] = wfn.occupations(0, k)(n);
                    CUDA_CHECK(cudaMemcpy(d_occ, h_occ_k.data(),
                                           Nband * sizeof(double), cudaMemcpyHostToDevice));
                }
                // SOC density uses occfac=1 (no spin degeneracy since both spins in spinor)
                gpu::spinor_density_gpu(d_psi_spinor, d_occ,
                                         d_rho_new_soc, d_mag_x_new, d_mag_y_new, d_mag_z_new,
                                         Nd_, Nband, 1.0 * wk);
            }
            // Debug: check density values and occ/psi norms
            {
                double rho_sum = gpu_sum(d_rho_new_soc, Nd_);
                double occ_sum = gpu_sum(d_occ, Nband);
                // Check first few psi values
                std::vector<double> psi_check(20);
                CUDA_CHECK(cudaMemcpy(psi_check.data(), (double*)d_psi_spinor,
                    20 * sizeof(double), cudaMemcpyDeviceToHost));
                printf("  [SOC dbg] rho_sum=%.6e occ_sum=%.6e psi[0..3]=%.3e %.3e %.3e %.3e\n",
                       rho_sum, occ_sum, psi_check[0], psi_check[1], psi_check[2], psi_check[3]);
            }
            // Copy total density for SCF error
            CUDA_CHECK(cudaMemcpy(d_rho_new_total, d_rho_new_soc, Nd_ * sizeof(double), cudaMemcpyDeviceToDevice));
        } else if (Nspin >= 2) {
            d_rho_new_up = sp_dens.alloc<double>(Nd_);
            d_rho_new_dn = sp_dens.alloc<double>(Nd_);
            CUDA_CHECK(cudaMemset(d_rho_new_up, 0, Nd_ * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_rho_new_dn, 0, Nd_ * sizeof(double)));

            for (int s = 0; s < Nspin_local_; s++) {
                int s_glob = spin_start_ + s;
                double* d_rho_spin_target = (s_glob == 0 ? d_rho_new_up : d_rho_new_dn);

                if (is_kpt_) {
                    for (int k = 0; k < Nkpts; k++) {
                        double wk = kpt_weights[k];
                        {
                            std::vector<double> h_psi_z(2 * Nd_ * Nband);
                            std::memcpy(h_psi_z.data(), wfn.psi_kpt(s, k).data(),
                                        2 * Nd_ * Nband * sizeof(double));
                            CUDA_CHECK(cudaMemcpy(d_psi_z, h_psi_z.data(),
                                2 * (size_t)Nd_ * Nband * sizeof(double), cudaMemcpyHostToDevice));
                        }
                        gpu::compute_density_z_gpu(d_psi_z, d_occ_arr[s_glob],
                                                   d_rho_spin_target, Nd_, Nband, occfac * wk);
                    }
                } else {
                    gpu::compute_density_gpu(d_psi_arr[s_glob], d_occ_arr[s_glob],
                                              d_rho_spin_target, Nd_, Nband, occfac);
                }
            }

            int bs = 256;
            int gs = gpu::ceildiv(Nd_, bs);
            add_kernel<<<gs, bs>>>(d_rho_new_up, d_rho_new_dn, d_rho_new_total, Nd_);
        } else {
            for (int s = 0; s < Nspin_local_; s++) {
                int s_glob = spin_start_ + s;
                if (is_kpt_) {
                    for (int k = 0; k < Nkpts; k++) {
                        double wk = kpt_weights[k];
                        {
                            std::vector<double> h_psi_z(2 * Nd_ * Nband);
                            std::memcpy(h_psi_z.data(), wfn.psi_kpt(s, k).data(),
                                        2 * Nd_ * Nband * sizeof(double));
                            CUDA_CHECK(cudaMemcpy(d_psi_z, h_psi_z.data(),
                                2 * (size_t)Nd_ * Nband * sizeof(double), cudaMemcpyHostToDevice));
                        }
                        gpu::compute_density_z_gpu(d_psi_z, d_occ_arr[s_glob],
                                                   d_rho_new_total, Nd_, Nband, occfac * wk);
                    }
                } else {
                    gpu::compute_density_gpu(d_psi_arr[s_glob], d_occ_arr[s_glob],
                                              d_rho_new_total, Nd_, Nband, occfac);
                }
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 3: Compute energy (D2H for energy components)
        {
            // For SOC energy: use Nspin=2 with d_rho = [up|dn] from collinear conversion
            int Nspin_energy = has_soc_ ? 2 : Nspin;
            int veff_size = Nd_ * Nspin_energy;
            int vxc_size = Nd_ * Nspin_energy;
            std::vector<double> h_rho_in(Nd_), h_Veff_e(veff_size), h_phi_e(Nd_), h_exc_e(Nd_), h_Vxc_e(vxc_size);

            // For energy, use total input density
            if (has_soc_) {
                // SOC: total density is d_rho_soc
                CUDA_CHECK(cudaMemcpy(h_rho_in.data(), d_rho_soc, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
            } else if (Nspin >= 2) {
                // Compute total from per-spin
                auto& sp_e = ctx.scratch_pool;
                size_t sp_e_cp = sp_e.checkpoint();
                double* d_rho_in_total = sp_e.alloc<double>(Nd_);
                int bs = 256;
                int gs = gpu::ceildiv(Nd_, bs);
                add_kernel<<<gs, bs>>>(d_rho, d_rho + Nd_, d_rho_in_total, Nd_);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_rho_in.data(), d_rho_in_total, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
                sp_e.restore(sp_e_cp);
            } else {
                CUDA_CHECK(cudaMemcpy(h_rho_in.data(), d_rho, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
            }
            // Download Veff for energy (for SOC, use d_Veff which holds [Vxc_up|Vxc_dn] from spin XC + phi)
            CUDA_CHECK(cudaMemcpy(h_Veff_e.data(), d_Veff, veff_size * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_phi_e.data(), d_phi, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_exc_e.data(), d_exc, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_Vxc_e.data(), d_Vxc, vxc_size * sizeof(double), cudaMemcpyDeviceToHost));

            // Download psi for Eband computation (only spin-0, kpt-0 for gamma)
            if (!is_kpt_ && !has_soc_) {
                for (int s = 0; s < Nspin_local_; s++) {
                    int s_glob = spin_start_ + s;
                    std::vector<double> h_psi_tmp(Nd_ * Nband);
                    CUDA_CHECK(cudaMemcpy(h_psi_tmp.data(), d_psi_arr[s_glob],
                        (size_t)Nd_ * Nband * sizeof(double), cudaMemcpyDeviceToHost));
                    for (int j = 0; j < Nband; j++)
                        std::memcpy(wfn.psi(s, 0).col(j), h_psi_tmp.data() + j * Nd_, Nd_ * sizeof(double));
                }
            }

            ElectronDensity dens_in;
            dens_in.allocate(Nd_, Nspin_energy);
            std::memcpy(dens_in.rho_total().data(), h_rho_in.data(), Nd_ * sizeof(double));
            if (has_soc_ || Nspin >= 2) {
                std::vector<double> h_rho_up(Nd_), h_rho_dn(Nd_);
                CUDA_CHECK(cudaMemcpy(h_rho_up.data(), d_rho, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_rho_dn.data(), d_rho + Nd_, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
                std::memcpy(dens_in.rho(0).data(), h_rho_up.data(), Nd_ * sizeof(double));
                std::memcpy(dens_in.rho(1).data(), h_rho_dn.data(), Nd_ * sizeof(double));
            } else {
                std::memcpy(dens_in.rho(0).data(), h_rho_in.data(), Nd_ * sizeof(double));
            }

            energy_ = Energy::compute_all(
                wfn, dens_in, h_Veff_e.data(), h_phi_e.data(),
                h_exc_e.data(), h_Vxc_e.data(), rho_b,
                Eself, Ec, beta_smearing,
                params.smearing,
                kpt_weights, Nd_, dV_,
                has_nlcc_ ? h_rho_core_vec.data() : nullptr, Ef_prev_, kpt_start_,
                nullptr, nullptr, Nspin_energy);

            // SCF error: ||rho_out_total - rho_in_total|| / ||rho_out_total||
            std::vector<double> h_rho_new(Nd_);
            CUDA_CHECK(cudaMemcpy(h_rho_new.data(), d_rho_new_total, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));

            double sum_sq_diff = 0, sum_sq_out = 0;
            for (int i = 0; i < Nd_; i++) {
                double diff = h_rho_new[i] - h_rho_in[i];
                sum_sq_diff += diff * diff;
                sum_sq_out += h_rho_new[i] * h_rho_new[i];
            }
            double scf_error = (sum_sq_out > 0) ? std::sqrt(sum_sq_diff / sum_sq_out) : 0;

            if (has_soc_) {
                // SOC: report total |m|
                double mag_sum = 0;
                if (d_mag_x_new && d_mag_y_new && d_mag_z_new) {
                    std::vector<double> h_mx(Nd_), h_my(Nd_), h_mz(Nd_);
                    CUDA_CHECK(cudaMemcpy(h_mx.data(), d_mag_x_new, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_my.data(), d_mag_y_new, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_mz.data(), d_mag_z_new, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
                    for (int i = 0; i < Nd_; i++)
                        mag_sum += std::sqrt(h_mx[i]*h_mx[i] + h_my[i]*h_my[i] + h_mz[i]*h_mz[i]);
                    mag_sum *= dV_;
                }
                printf("SCF iter %3d: Etot = %18.10f Ha, SCF error = %10.3e, |m| = %8.4f (SOC)\n",
                       scf_iter + 1, energy_.Etotal, scf_error, mag_sum);
            } else if (Nspin == 2) {
                // Compute magnetization for reporting
                double mag_sum = 0;
                if (d_rho_new_up && d_rho_new_dn) {
                    std::vector<double> h_ru(Nd_), h_rd(Nd_);
                    CUDA_CHECK(cudaMemcpy(h_ru.data(), d_rho_new_up, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_rd.data(), d_rho_new_dn, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
                    for (int i = 0; i < Nd_; i++) mag_sum += h_ru[i] - h_rd[i];
                    mag_sum *= dV_;
                }
                printf("SCF iter %3d: Etot = %18.10f Ha, SCF error = %10.3e, mag = %8.4f\n",
                       scf_iter + 1, energy_.Etotal, scf_error, mag_sum);
            } else {
                printf("SCF iter %3d: Etot = %18.10f Ha, SCF error = %10.3e\n",
                       scf_iter + 1, energy_.Etotal, scf_error);
            }

            if (scf_iter >= params.min_iter && scf_error < scf_tol) {
                converged_ = true;
                Ef_ = Ef;
                printf("\nGPU SCF converged after %d iterations.\n", scf_iter + 1);
                printf("Final energy: %.10f Ha\n", energy_.Etotal);
                sp_dens.restore(sp_dens_cp);
                break;
            }

            Ef_prev_ = Ef;
        }

        if (converged_) {
            sp_dens.restore(sp_dens_cp);
            break;
        }

        // Step 4: Mix density
        if (has_soc_) {
            // SOC: mix packed [rho | mx | my | mz] (4*Nd)
            auto& sp_mix = ctx.scratch_pool;
            double* d_dens_in  = sp_mix.alloc<double>(4 * Nd_);
            double* d_dens_out = sp_mix.alloc<double>(4 * Nd_);
            int bs = 256;
            int gs = gpu::ceildiv(Nd_, bs);

            // Pack input: [rho | mx | my | mz] from current density
            CUDA_CHECK(cudaMemcpy(d_dens_in, d_rho_soc, Nd_ * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_dens_in + Nd_, d_mag_x, Nd_ * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_dens_in + 2*Nd_, d_mag_y, Nd_ * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_dens_in + 3*Nd_, d_mag_z, Nd_ * sizeof(double), cudaMemcpyDeviceToDevice));

            // Pack output: [rho | mx | my | mz] from new density
            CUDA_CHECK(cudaMemcpy(d_dens_out, d_rho_new_soc, Nd_ * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_dens_out + Nd_, d_mag_x_new, Nd_ * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_dens_out + 2*Nd_, d_mag_y_new, Nd_ * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_dens_out + 3*Nd_, d_mag_z_new, Nd_ * sizeof(double), cudaMemcpyDeviceToDevice));

            // Mix on packed 4*Nd array
            gpu_pulay_mix(d_dens_in, d_dens_out, 4 * Nd_, mixing_history, mixing_param);

            // Unpack mixed density back
            CUDA_CHECK(cudaMemcpy(d_rho_soc, d_dens_in, Nd_ * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_mag_x, d_dens_in + Nd_, Nd_ * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_mag_y, d_dens_in + 2*Nd_, Nd_ * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_mag_z, d_dens_in + 3*Nd_, Nd_ * sizeof(double), cudaMemcpyDeviceToDevice));

            // Clamp rho and renormalize
            clamp_min_kernel<<<gs, bs>>>(d_rho_soc, 0.0, Nd_);
            CUDA_CHECK(cudaDeviceSynchronize());
            double rho_sum = gpu_sum(d_rho_soc, Nd_);
            double Ne_current = rho_sum * dV_;
            if (Ne_current > 1e-10) {
                double sc = (double)Nelectron / Ne_current;
                scale_kernel<<<gs, bs>>>(d_rho_soc, sc, Nd_);
                scale_kernel<<<gs, bs>>>(d_mag_x, sc, Nd_);
                scale_kernel<<<gs, bs>>>(d_mag_y, sc, Nd_);
                scale_kernel<<<gs, bs>>>(d_mag_z, sc, Nd_);
            }
        } else if (Nspin == 1) {
            // Simple total density mixing
            gpu_pulay_mix(d_rho, d_rho_new_total, Nd_, mixing_history, mixing_param);

            // Clamp + normalize
            {
                int bs = 256;
                int gs = gpu::ceildiv(Nd_, bs);
                clamp_min_kernel<<<gs, bs>>>(d_rho, 0.0, Nd_);
                CUDA_CHECK(cudaDeviceSynchronize());

                double rho_sum = gpu_sum(d_rho, Nd_);
                double Ne_current = rho_sum * dV_;
                if (Ne_current > 1e-10) {
                    double sc = (double)Nelectron / Ne_current;
                    scale_kernel<<<gs, bs>>>(d_rho, sc, Nd_);
                }
            }
        } else {
            // Spin-polarized: mix packed [total | magnetization] (2*Nd)
            auto& sp_mix = ctx.scratch_pool;
            // Note: sp_dens still has d_rho_new_up/dn allocated
            double* d_dens_in  = sp_mix.alloc<double>(2 * Nd_);
            double* d_dens_out = sp_mix.alloc<double>(2 * Nd_);

            // Pack input: [total | mag]
            int bs = 256;
            int gs = gpu::ceildiv(Nd_, bs);
            add_kernel<<<gs, bs>>>(d_rho, d_rho + Nd_, d_dens_in, Nd_);  // total
            sub_kernel<<<gs, bs>>>(d_rho, d_rho + Nd_, d_dens_in + Nd_, Nd_);  // mag = up - dn

            // Pack output: [total | mag]
            add_kernel<<<gs, bs>>>(d_rho_new_up, d_rho_new_dn, d_dens_out, Nd_);  // total
            sub_kernel<<<gs, bs>>>(d_rho_new_up, d_rho_new_dn, d_dens_out + Nd_, Nd_);  // mag

            // Mix on packed 2*Nd array
            gpu_pulay_mix(d_dens_in, d_dens_out, 2 * Nd_, mixing_history, mixing_param);

            // Unpack: rho_up = 0.5*(total + mag), rho_dn = 0.5*(total - mag)
            unpack_spin_kernel<<<gs, bs>>>(d_dens_in, d_dens_in + Nd_, d_rho, d_rho + Nd_, Nd_);

            // Clamp and renormalize
            clamp_min_kernel<<<gs, bs>>>(d_rho, 0.0, Nd_);
            clamp_min_kernel<<<gs, bs>>>(d_rho + Nd_, 0.0, Nd_);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Renormalize total
            double rho_up_sum = gpu_sum(d_rho, Nd_);
            double rho_dn_sum = gpu_sum(d_rho + Nd_, Nd_);
            double Ne_current = (rho_up_sum + rho_dn_sum) * dV_;
            if (Ne_current > 1e-10) {
                double sc = (double)Nelectron / Ne_current;
                scale_kernel<<<gs, bs>>>(d_rho, sc, Nd_);
                scale_kernel<<<gs, bs>>>(d_rho + Nd_, sc, Nd_);
            }
        }

        sp_dens.restore(sp_dens_cp);

        // Step 5: XC + Step 6: Poisson + Step 7: Veff
        if (has_soc_) {
            int bs = 256;
            int gs = gpu::ceildiv(Nd_, bs);

            // SOC Step 5a: Convert noncollinear to collinear for XC
            // d_rho = [rho_up | rho_dn] from (rho, mx, my, mz)
            mag_to_collinear_kernel<<<gs, bs>>>(d_rho_soc, d_mag_x, d_mag_y, d_mag_z,
                                                  d_rho, d_rho + Nd_, Nd_);
            CUDA_CHECK(cudaDeviceSynchronize());

            // SOC Step 5b: Spin-polarized XC on collinear densities
            gpu_xc_evaluate_spin(d_rho, d_exc, d_Vxc, Nd_);

            // SOC Step 6: Poisson on total density
            gpu_poisson_solve(d_rho_soc, d_phi, ctx.buf.b, Nd_, poisson_tol);

            // SOC Step 7a: Build Veff for energy = [Vxc_up + phi | Vxc_dn + phi]
            veff_combine_spin_kernel<<<gs, bs>>>(d_Vxc, d_phi, d_Veff, Nd_);
            veff_combine_spin_kernel<<<gs, bs>>>(d_Vxc + Nd_, d_phi, d_Veff + Nd_, Nd_);

            // SOC Step 7b: Build Veff_spinor from XC + magnetization direction
            veff_spinor_from_xc_kernel<<<gs, bs>>>(d_Vxc, d_Vxc + Nd_, d_phi,
                                                     d_mag_x, d_mag_y, d_mag_z,
                                                     d_Veff_spinor, Nd_);
        } else {
            // Non-SOC XC
            if (Nspin >= 2) {
                gpu_xc_evaluate_spin(d_rho, d_exc, d_Vxc, Nd_);
            } else {
                gpu_xc_evaluate(d_rho, d_exc, d_Vxc, Nd_);
            }

            // Non-SOC Poisson
            if (Nspin >= 2) {
                int bs = 256;
                int gs = gpu::ceildiv(Nd_, bs);
                add_kernel<<<gs, bs>>>(d_rho, d_rho + Nd_, d_rho_new, Nd_);
                gpu_poisson_solve(d_rho_new, d_phi, ctx.buf.b, Nd_, poisson_tol);
            } else {
                gpu_poisson_solve(d_rho, d_phi, ctx.buf.b, Nd_, poisson_tol);
            }

            // Non-SOC Veff
            {
                int bs = 256;
                int gs = gpu::ceildiv(Nd_, bs);
                for (int s = 0; s < Nspin; s++) {
                    veff_combine_spin_kernel<<<gs, bs>>>(
                        d_Vxc + s * Nd_, d_phi, d_Veff + s * Nd_, Nd_);
                }
            }
        }
    }

    if (!converged_) {
        printf("WARNING: GPU SCF did not converge within %d iterations.\n", max_iter);
    }

    // Free SOC-specific device buffers
    if (has_soc_) {
        if (d_psi_spinor)   cudaFreeAsync(d_psi_spinor, 0);
        if (d_Hpsi_spinor)  cudaFreeAsync(d_Hpsi_spinor, 0);
        if (d_Xold_spinor)  cudaFreeAsync(d_Xold_spinor, 0);
        if (d_Xnew_spinor)  cudaFreeAsync(d_Xnew_spinor, 0);
        if (d_x_ex_spinor)  cudaFreeAsync(d_x_ex_spinor, 0);
        if (d_Hs_spinor)    cudaFreeAsync(d_Hs_spinor, 0);
        if (d_Ms_spinor)    cudaFreeAsync(d_Ms_spinor, 0);
        if (d_Veff_spinor)  cudaFreeAsync(d_Veff_spinor, 0);
        if (d_rho_soc)      cudaFreeAsync(d_rho_soc, 0);
        if (d_mag_x)        cudaFreeAsync(d_mag_x, 0);
        if (d_mag_y)        cudaFreeAsync(d_mag_y, 0);
        if (d_mag_z)        cudaFreeAsync(d_mag_z, 0);
    }

    Ef_ = Ef;
    return energy_.Etotal;
}

// ============================================================
// Download results back to CPU
// ============================================================
void GPUSCFRunner::download_results(double* phi, double* Vxc, double* exc,
                                     double* Veff, double* Dxcdgrho,
                                     double* rho, Wavefunction& wfn)
{
    auto& ctx = gpu::GPUContext::instance();
    int Nband = wfn.Nband();
    int vxc_size = Nd_ * Nspin_;
    int veff_size = Nd_ * Nspin_;
    int dxc_ncol = is_gga_ ? ((Nspin_ == 2) ? 3 : 1) : 0;

    if (phi)
        CUDA_CHECK(cudaMemcpy(phi, ctx.buf.phi, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
    if (Vxc)
        CUDA_CHECK(cudaMemcpy(Vxc, ctx.buf.Vc, vxc_size * sizeof(double), cudaMemcpyDeviceToHost));
    if (exc)
        CUDA_CHECK(cudaMemcpy(exc, ctx.buf.exc, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
    if (Veff)
        CUDA_CHECK(cudaMemcpy(Veff, ctx.buf.Veff, veff_size * sizeof(double), cudaMemcpyDeviceToHost));

    // Download total density
    if (rho) {
        if (Nspin_ >= 2) {
            // Compute total = up + dn, download
            int bs = 256;
            int gs = gpu::ceildiv(Nd_, bs);
            double* d_temp = ctx.buf.rho_total;
            add_kernel<<<gs, bs>>>(ctx.buf.rho, ctx.buf.rho + Nd_, d_temp, Nd_);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(rho, d_temp, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
        } else {
            CUDA_CHECK(cudaMemcpy(rho, ctx.buf.rho, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }

    // Download Dxcdgrho if GGA
    if (Dxcdgrho && dxc_ncol > 0 && ctx.buf.Dxcdgrho) {
        CUDA_CHECK(cudaMemcpy(Dxcdgrho, ctx.buf.Dxcdgrho,
                               Nd_ * dxc_ncol * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Download wavefunctions (real for gamma, complex already in wfn for k-point)
    if (!is_kpt_) {
        double* d_psi_arr[2] = { ctx.buf.psi, ctx.buf.psi_s1 };
        for (int s = 0; s < Nspin_local_; s++) {
            int s_glob = spin_start_ + s;
            std::vector<double> h_psi(Nd_ * Nband);
            CUDA_CHECK(cudaMemcpy(h_psi.data(), d_psi_arr[s_glob],
                (size_t)Nd_ * Nband * sizeof(double), cudaMemcpyDeviceToHost));
            for (int j = 0; j < Nband; j++)
                std::memcpy(wfn.psi(s, 0).col(j), h_psi.data() + j * Nd_, Nd_ * sizeof(double));
        }
    }
    // For k-point, complex psi is already downloaded into wfn during SCF loop
}

// ============================================================
// GPU Force/Stress computation
// ============================================================
void GPUSCFRunner::compute_force_stress(
    const Wavefunction& wfn,
    const Crystal& crystal,
    const std::vector<AtomNlocInfluence>& nloc_influence,
    const Domain& domain,
    const FDGrid& grid,
    double* f_nloc,
    double* stress_k,
    double* stress_nl,
    double* energy_nl)
{
    auto& ctx = gpu::GPUContext::instance();
    int Nband = wfn.Nband();
    int n_phys = crystal.n_atom_total();
    int ntypes = crystal.n_types();

    // Build physical atom IP_displ (CSR prefix sum)
    std::vector<int> IP_displ_phys(n_phys + 1, 0);
    {
        int idx = 0;
        for (int it = 0; it < ntypes; it++) {
            int nproj = crystal.types()[it].psd().nproj_per_atom();
            int nat = crystal.types()[it].n_atoms();
            for (int ia = 0; ia < nat; ia++) {
                IP_displ_phys[idx + 1] = IP_displ_phys[idx] + nproj;
                idx++;
            }
        }
    }

    // Build atom positions for influence atoms
    std::vector<double> atom_pos;
    for (int it = 0; it < ntypes; it++) {
        const auto& inf = nloc_influence[it];
        for (int iat = 0; iat < inf.n_atom; iat++) {
            atom_pos.push_back(inf.coords[iat].x);
            atom_pos.push_back(inf.coords[iat].y);
            atom_pos.push_back(inf.coords[iat].z);
        }
    }

    double occfac = 2.0;  // Nspin=1

    gpu::compute_force_stress_gpu(
        ctx.buf.psi, ctx.buf.occupations,
        gpu_vnl_.d_Chi_flat, gpu_vnl_.d_gpos_flat,
        gpu_vnl_.d_gpos_offsets, gpu_vnl_.d_chi_offsets,
        gpu_vnl_.d_ndc_arr, gpu_vnl_.d_nproj_arr,
        gpu_vnl_.d_IP_displ, gpu_vnl_.d_Gamma,
        gpu_vnl_.n_influence, gpu_vnl_.total_phys_nproj,
        gpu_vnl_.max_ndc, gpu_vnl_.max_nproj,
        n_phys,
        IP_displ_phys.data(),
        atom_pos.data(),
        nx_, ny_, nz_, FDn_, Nd_, Nband,
        dV_, grid.dx(), grid.dy(), grid.dz(),
        domain.vertices().xs, domain.vertices().ys, domain.vertices().zs,
        occfac,
        f_nloc, stress_k, stress_nl, energy_nl);

    // Normalize stress by cell volume (matching CPU Stress.cpp)
    const auto& lat = grid.lattice();
    Vec3 L = lat.lengths();
    double Jacbdet = lat.jacobian() / (L.x * L.y * L.z);
    double cell_measure = Jacbdet;
    if (grid.bcx() == BCType::Periodic) cell_measure *= L.x;
    if (grid.bcy() == BCType::Periodic) cell_measure *= L.y;
    if (grid.bcz() == BCType::Periodic) cell_measure *= L.z;
    for (int i = 0; i < 6; i++) {
        stress_k[i] /= cell_measure;
        stress_nl[i] /= cell_measure;
    }
}

// ============================================================
// Cleanup
// ============================================================
void GPUSCFRunner::cleanup() {
    gpu_vnl_.free();
    gpu_soc_.free_soc();
    if (d_Y_) { cudaFreeAsync(d_Y_, 0); d_Y_ = nullptr; }
    if (d_Y_s1_) { cudaFreeAsync(d_Y_s1_, 0); d_Y_s1_ = nullptr; }
    if (d_rho_core_) { cudaFreeAsync(d_rho_core_, 0); d_rho_core_ = nullptr; }
    if (d_pseudocharge_) { cudaFreeAsync(d_pseudocharge_, 0); d_pseudocharge_ = nullptr; }
    if (d_mix_fkm1_) { cudaFreeAsync(d_mix_fkm1_, 0); d_mix_fkm1_ = nullptr; }
    if (d_bloch_fac_) { cudaFreeAsync(d_bloch_fac_, 0); d_bloch_fac_ = nullptr; }
    if (d_alpha_z_) { cudaFreeAsync(d_alpha_z_, 0); d_alpha_z_ = nullptr; }

    if (s_instance_ == this) s_instance_ = nullptr;
}

GPUSCFRunner::~GPUSCFRunner() {
    cleanup();
}

} // namespace lynx

#endif // USE_CUDA
