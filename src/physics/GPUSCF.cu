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

namespace sparc {

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

} // namespace gpu

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
        true, true, true, true,
        s->diag_coeff_ham_,
        false, false, false);

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
        true, true, true, true,
        s->diag_coeff_ham_,
        false, false, false,
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
    const double* rho_dn_init)
{
    // Store parameters as members
    nx_ = grid.Nx();
    ny_ = grid.Ny();
    nz_ = grid.Nz();
    FDn_ = stencil.FDn();
    Nd_ = domain.Nd_d();
    dV_ = grid.dV();
    is_gga_ = is_gga;
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
        nullptr, nullptr, nullptr, FDn_);

    // ============================================================
    // GPU memory allocation + data upload
    // ============================================================
    auto& ctx = gpu::GPUContext::instance();
    int mix_ncol = (Nspin >= 2) ? 2 : 1;  // packed [total | mag] for spin
    ctx.init_scf_buffers(Nd_, nx_, ny_, nz_, FDn_,
                          Nband, Nband, Nspin,
                          7, 7, mix_ncol,
                          0, 0, 0,
                          is_gga_, false, is_kpt);

    // Upload nonlocal projector data to GPU (once)
    if (vnl) {
        gpu_vnl_.setup(*vnl, crystal, nloc_influence, Nband);
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
    int mix_N = Nd_ * ((Nspin >= 2) ? 2 : 1);
    CUDA_CHECK(cudaMallocAsync(&d_mix_fkm1_, mix_N * sizeof(double), 0));

    // d_Y must be separate from d_Hpsi (used as d_HX inside eigensolver)
    // For k-point, allocate complex-sized (2x) since Y is used as cuDoubleComplex*
    size_t y_elem_size = is_kpt ? sizeof(cuDoubleComplex) : sizeof(double);
    CUDA_CHECK(cudaMallocAsync(&d_Y_, (size_t)Nd_ * Nband * y_elem_size, 0));

    // Device pointers (aliases into ctx.buf)
    double* d_psi     = ctx.buf.psi;
    double* d_Hpsi    = ctx.buf.Hpsi;
    double* d_Veff    = ctx.buf.Veff;
    double* d_rho     = ctx.buf.rho;
    double* d_rho_new = ctx.buf.rho_total;
    double* d_phi     = ctx.buf.phi;
    double* d_exc     = ctx.buf.exc;
    double* d_Vxc     = ctx.buf.Vc;
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

    // Allocate d_Y_s1_ if spin-polarized
    if (Nspin >= 2 && !d_Y_s1_) {
        CUDA_CHECK(cudaMallocAsync(&d_Y_s1_, (size_t)Nd_ * Nband * y_elem_size, 0));
        d_Y_arr[1] = d_Y_s1_;
    }

    // Randomize wavefunctions on CPU, upload to GPU
    for (int s = 0; s < Nspin_local_; s++) {
        if (is_kpt_) {
            // For k-point, randomize complex psi for first k-point, upload
            wfn.randomize_kpt(s, 0, 42 + s);
            // The complex psi will be uploaded per-kpt inside the SCF loop
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
    }

    // ============================================================
    // Lanczos spectrum bounds (CPU — runs once)
    // ============================================================
    EigenSolver eigsolver;
    eigsolver.setup(hamiltonian, halo, domain, bandcomm, Nband);

    // Per-spin spectral bounds
    std::vector<double> eigval_min_s(Nspin, 0.0), eigval_max_s(Nspin, 0.0);
    {
        std::vector<double> h_Veff(Nd_);
        CUDA_CHECK(cudaMemcpy(h_Veff.data(), d_Veff, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
        eigsolver.lanczos_bounds(h_Veff.data(), Nd_, eigval_min_s[0], eigval_max_s[0]);
        for (int s = 1; s < Nspin; s++) {
            eigval_min_s[s] = eigval_min_s[0];
            eigval_max_s[s] = eigval_max_s[0];
        }
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
    // GPU SCF Loop
    // ============================================================
    printf("\n--- GPU SCF Loop (Nspin=%d, Nkpts=%d, %s) ---\n",
           Nspin, Nkpts, is_kpt_ ? "k-point" : "gamma");

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

            // Update spectral bounds from eigenvalues (per-spin)
            CUDA_CHECK(cudaDeviceSynchronize());
            {
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
        for (int s = 0; s < Nspin_local_; s++) {
            int s_glob = spin_start_ + s;
            for (int n = 0; n < Nband; n++)
                h_occ[n] = wfn.occupations(s, 0)(n);
            CUDA_CHECK(cudaMemcpy(d_occ_arr[s_glob], h_occ.data(),
                                   Nband * sizeof(double), cudaMemcpyHostToDevice));
        }

        // Step 2: Compute new density on GPU
        // For Nspin=2: accumulate into d_rho_new as total density for SCF error
        // Also compute per-spin density in a scratch buffer for mixing
        double* d_rho_new_total = d_rho_new;  // total density (Nd)
        CUDA_CHECK(cudaMemset(d_rho_new_total, 0, Nd_ * sizeof(double)));

        // For spin-polarized: also need per-spin new density
        auto& sp_dens = ctx.scratch_pool;
        size_t sp_dens_cp = sp_dens.checkpoint();
        double* d_rho_new_up = nullptr;
        double* d_rho_new_dn = nullptr;
        if (Nspin >= 2) {
            d_rho_new_up = sp_dens.alloc<double>(Nd_);
            d_rho_new_dn = sp_dens.alloc<double>(Nd_);
            CUDA_CHECK(cudaMemset(d_rho_new_up, 0, Nd_ * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_rho_new_dn, 0, Nd_ * sizeof(double)));
        }

        for (int s = 0; s < Nspin_local_; s++) {
            int s_glob = spin_start_ + s;
            double* d_rho_spin_target = (Nspin >= 2) ? (s_glob == 0 ? d_rho_new_up : d_rho_new_dn) : d_rho_new_total;

            if (is_kpt_) {
                for (int k = 0; k < Nkpts; k++) {
                    double wk = kpt_weights[k];
                    // Upload complex psi for this (s,k)
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

        // Compute total density from spin components
        if (Nspin >= 2) {
            int bs = 256;
            int gs = gpu::ceildiv(Nd_, bs);
            add_kernel<<<gs, bs>>>(d_rho_new_up, d_rho_new_dn, d_rho_new_total, Nd_);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 3: Compute energy (D2H for energy components)
        {
            int veff_size = Nd_ * Nspin;
            int vxc_size = Nd_ * Nspin;
            std::vector<double> h_rho_in(Nd_), h_Veff_e(veff_size), h_phi_e(Nd_), h_exc_e(Nd_), h_Vxc_e(vxc_size);

            // For energy, use total input density
            if (Nspin >= 2) {
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
            CUDA_CHECK(cudaMemcpy(h_Veff_e.data(), d_Veff, veff_size * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_phi_e.data(), d_phi, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_exc_e.data(), d_exc, Nd_ * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_Vxc_e.data(), d_Vxc, vxc_size * sizeof(double), cudaMemcpyDeviceToHost));

            // Download psi for Eband computation (only spin-0, kpt-0 for gamma)
            if (!is_kpt_) {
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
            dens_in.allocate(Nd_, Nspin);
            std::memcpy(dens_in.rho_total().data(), h_rho_in.data(), Nd_ * sizeof(double));
            if (Nspin >= 2) {
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
                nullptr, nullptr, Nspin);

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

            if (Nspin == 2) {
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
        if (Nspin == 1) {
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

        // Step 5: XC
        if (Nspin >= 2) {
            gpu_xc_evaluate_spin(d_rho, d_exc, d_Vxc, Nd_);
        } else {
            gpu_xc_evaluate(d_rho, d_exc, d_Vxc, Nd_);
        }

        // Step 6: Poisson solver (always uses total density)
        if (Nspin >= 2) {
            // Compute total density into d_rho_new (reuse as temp)
            int bs = 256;
            int gs = gpu::ceildiv(Nd_, bs);
            add_kernel<<<gs, bs>>>(d_rho, d_rho + Nd_, d_rho_new, Nd_);
            gpu_poisson_solve(d_rho_new, d_phi, ctx.buf.b, Nd_, poisson_tol);
        } else {
            gpu_poisson_solve(d_rho, d_phi, ctx.buf.b, Nd_, poisson_tol);
        }

        // Step 7: Veff = Vxc_s + phi for each spin
        {
            int bs = 256;
            int gs = gpu::ceildiv(Nd_, bs);
            for (int s = 0; s < Nspin; s++) {
                veff_combine_spin_kernel<<<gs, bs>>>(
                    d_Vxc + s * Nd_, d_phi, d_Veff + s * Nd_, Nd_);
            }
        }
    }

    if (!converged_) {
        printf("WARNING: GPU SCF did not converge within %d iterations.\n", max_iter);
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

} // namespace sparc

#endif // USE_CUDA
