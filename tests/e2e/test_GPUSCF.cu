// GPU SCF End-to-End Test
// Runs full SCF loop on GPU for BaTiO3, verifies energy matches CPU reference.
// Fully GPU-resident: psi, rho, Veff, phi, exc, Vxc on device.
// GPU: eigensolver (CheFSI), density, Hamiltonian (local+Vnl), Veff combine,
//      XC (GGA_PBE + NLCC), Poisson solver (AAR), Pulay+Kerker mixer.
// CPU fallback: occupation (tiny), energy evaluation (diagnostic).

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>
#include <numeric>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "core/GPUContext.cuh"
#include "core/gpu_common.cuh"
#include "core/constants.hpp"

// CPU infrastructure for setup
#include "io/InputParser.hpp"
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include "core/Domain.hpp"
#include "operators/FDStencil.hpp"
#include "operators/Laplacian.hpp"
#include "operators/Gradient.hpp"
#include "operators/Hamiltonian.hpp"
#include "operators/NonlocalProjector.hpp"
#include "atoms/Crystal.hpp"
#include "atoms/AtomType.hpp"
#include "electronic/Wavefunction.hpp"
#include "electronic/Occupation.hpp"
#include "solvers/LinearSolver.hpp"
#include "solvers/EigenSolver.hpp"
#include "solvers/PoissonSolver.hpp"
#include "solvers/Mixer.hpp"
#include "xc/XCFunctional.hpp"
#include "physics/Electrostatics.hpp"
#include "physics/Energy.hpp"
#include "physics/Forces.hpp"
#include "physics/Stress.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"
#include "core/LynxContext.hpp"
#include "core/ParameterDefaults.hpp"

using namespace lynx;

// ============================================================
// Forward declarations for GPU functions
// ============================================================
namespace lynx { namespace gpu {

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

}} // namespace lynx::gpu

// ============================================================
// GPU nonlocal projector data — flattened from CPU NonlocalProjector
// Uploaded once during setup, used every H*psi call.
// ============================================================
struct GPUNonlocalData {
    // Device arrays (persistent, uploaded once)
    double* d_Chi_flat = nullptr;
    int*    d_gpos_flat = nullptr;
    int*    d_gpos_offsets = nullptr;
    int*    d_chi_offsets = nullptr;
    int*    d_ndc_arr = nullptr;
    int*    d_nproj_arr = nullptr;
    int*    d_IP_displ = nullptr;
    double* d_Gamma = nullptr;
    double* d_alpha = nullptr;      // workspace [total_phys_nproj * Nband]

    int n_influence = 0;            // total influence atoms across all types
    int total_phys_nproj = 0;       // projectors indexed by physical atom
    int max_ndc = 0;
    int max_nproj = 0;

    void setup(const NonlocalProjector& vnl,
               const Crystal& crystal,
               const std::vector<AtomNlocInfluence>& nloc_influence,
               int Nband) {
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

        CUDA_CHECK(cudaMalloc(&d_Chi_flat, std::max(1, total_chi) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_gpos_flat, std::max(1, total_gpos) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_gpos_offsets, (n_influence + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_chi_offsets, (n_influence + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ndc_arr, n_influence * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_nproj_arr, n_influence * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_IP_displ, n_influence * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_Gamma, total_phys_nproj * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_alpha, (size_t)total_phys_nproj * Nband * sizeof(double)));

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

    void free() {
        if (d_Chi_flat) cudaFree(d_Chi_flat);
        if (d_gpos_flat) cudaFree(d_gpos_flat);
        if (d_gpos_offsets) cudaFree(d_gpos_offsets);
        if (d_chi_offsets) cudaFree(d_chi_offsets);
        if (d_ndc_arr) cudaFree(d_ndc_arr);
        if (d_nproj_arr) cudaFree(d_nproj_arr);
        if (d_IP_displ) cudaFree(d_IP_displ);
        if (d_Gamma) cudaFree(d_Gamma);
        if (d_alpha) cudaFree(d_alpha);
        d_Chi_flat = nullptr;
    }
};

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
// Global state for GPU operator callbacks
// ============================================================
static int g_nx, g_ny, g_nz, g_FDn;
static double g_diag_coeff_ham;
static int g_Nd = 0;
static double g_dV = 0.0;
static double Ef_ = 0.0;
static GPUNonlocalData* g_gpu_vnl = nullptr;

// Poisson solver callback state
static double* g_aar_x_ex = nullptr;
static double g_poisson_diag = 0.0;     // diag for -Lap (a=-1)
static double g_jacobi_m_inv = 0.0;     // -1/(D2x[0]+D2y[0]+D2z[0])

// Kerker preconditioner callback state
static double g_kerker_diag = 0.0;      // diag for (-Lap + kTF²)
static double g_kerker_m_inv = 0.0;     // -1/(D2[0] - kTF²)
static double g_kerker_rhs_diag = 0.0;  // diag for (Lap - idiemac*kTF²)
static double g_precond_tol = 0.0;

// XC / NLCC state
static double* g_d_rho_core = nullptr;
static double* g_d_pseudocharge = nullptr;

// Mixer state
static int g_mix_iter = 0;
static double* g_d_mix_fkm1 = nullptr;

// Hamiltonian apply callback for CheFSI — fully on GPU
static void gpu_hamiltonian_apply(
    const double* d_psi, const double* d_Veff,
    double* d_Hpsi, double* d_x_ex, int ncol)
{
    // GPU local part: -0.5*Lap + Veff
    lynx::gpu::hamiltonian_apply_local_gpu(
        d_psi, d_Veff, d_Hpsi, d_x_ex,
        g_nx, g_ny, g_nz, g_FDn, ncol, 0.0,
        true, true, true, true,
        g_diag_coeff_ham,
        false, false, false);

    // GPU nonlocal part: Vnl*psi → add to Hpsi (no D2H/H2D!)
    if (g_gpu_vnl && g_gpu_vnl->total_phys_nproj > 0) {
        lynx::gpu::nonlocal_projector_apply_gpu(
            d_psi, d_Hpsi,
            g_gpu_vnl->d_Chi_flat, g_gpu_vnl->d_gpos_flat,
            g_gpu_vnl->d_gpos_offsets, g_gpu_vnl->d_chi_offsets,
            g_gpu_vnl->d_ndc_arr, g_gpu_vnl->d_nproj_arr,
            g_gpu_vnl->d_IP_displ, g_gpu_vnl->d_Gamma,
            g_gpu_vnl->d_alpha,
            g_Nd, ncol, g_dV,
            g_gpu_vnl->n_influence, g_gpu_vnl->total_phys_nproj,
            g_gpu_vnl->max_ndc, g_gpu_vnl->max_nproj);
    }
}

// ============================================================
// GPU Poisson operator: -Lap * x
// ============================================================
static void poisson_op_gpu(const double* d_x, double* d_Ax) {
    lynx::gpu::halo_exchange_gpu(d_x, g_aar_x_ex,
        g_nx, g_ny, g_nz, g_FDn, 1, true, true, true);
    int nx_ex = g_nx + 2*g_FDn, ny_ex = g_ny + 2*g_FDn;
    lynx::gpu::laplacian_orth_v2_gpu(g_aar_x_ex, nullptr, d_Ax,
        g_nx, g_ny, g_nz, g_FDn, nx_ex, ny_ex,
        -1.0, 0.0, 0.0, g_poisson_diag, 1);
}

// GPU Poisson Jacobi preconditioner: f = m_inv * r
static void poisson_precond_gpu(const double* d_r, double* d_f) {
    int bs = 256;
    jacobi_scale_kernel<<<lynx::gpu::ceildiv(g_Nd, bs), bs>>>(
        d_r, d_f, g_jacobi_m_inv, g_Nd);
}

// ============================================================
// GPU Kerker operator: (-Lap + kTF²) * x
// ============================================================
static void kerker_op_gpu(const double* d_x, double* d_Ax) {
    lynx::gpu::halo_exchange_gpu(d_x, g_aar_x_ex,
        g_nx, g_ny, g_nz, g_FDn, 1, true, true, true);
    int nx_ex = g_nx + 2*g_FDn, ny_ex = g_ny + 2*g_FDn;
    constexpr double kTF2 = 1.0;  // kTF²
    lynx::gpu::laplacian_orth_v2_gpu(g_aar_x_ex, nullptr, d_Ax,
        g_nx, g_ny, g_nz, g_FDn, nx_ex, ny_ex,
        -1.0, 0.0, kTF2, g_kerker_diag, 1);
}

// GPU Kerker Jacobi preconditioner
static void kerker_precond_gpu(const double* d_r, double* d_f) {
    int bs = 256;
    jacobi_scale_kernel<<<lynx::gpu::ceildiv(g_Nd, bs), bs>>>(
        d_r, d_f, g_kerker_m_inv, g_Nd);
}

// ============================================================
// GPU device-side sum (small arrays — download + CPU sum)
// ============================================================
static double gpu_sum(const double* d_x, int N) {
    std::vector<double> h(N);
    CUDA_CHECK(cudaMemcpy(h.data(), d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
    double s = 0;
    for (int i = 0; i < N; i++) s += h[i];
    return s;
}

// ============================================================
// GPU XC evaluate (GGA PBE, orthogonal, non-spin)
// ============================================================
static void gpu_xc_evaluate(double* d_rho, double* d_exc, double* d_Vxc,
                              int Nd, bool has_nlcc) {
    auto& ctx = lynx::gpu::GPUContext::instance();
    int bs = 256;
    int grid_sz = lynx::gpu::ceildiv(Nd, bs);
    int nx_ex = g_nx + 2*g_FDn, ny_ex = g_ny + 2*g_FDn;

    double* d_Drho_x = ctx.buf.grad_rho;
    double* d_Drho_y = ctx.buf.grad_rho + Nd;
    double* d_Drho_z = ctx.buf.grad_rho + 2 * Nd;
    double* d_sigma  = ctx.buf.aar_r;     // reuse (not in AAR yet)
    double* d_v2xc   = ctx.buf.Dxcdgrho;
    double* d_x_ex   = ctx.buf.aar_x_ex;

    // Prepare rho_xc (NLCC: rho + rho_core)
    double* d_rho_xc;
    if (has_nlcc && g_d_rho_core) {
        d_rho_xc = ctx.buf.b;  // reuse Poisson RHS buffer
        nlcc_add_kernel<<<grid_sz, bs>>>(d_rho, g_d_rho_core, d_rho_xc, Nd);
    } else {
        d_rho_xc = d_rho;
    }

    // Gradient of rho_xc
    lynx::gpu::halo_exchange_gpu(d_rho_xc, d_x_ex, g_nx, g_ny, g_nz, g_FDn, 1, true, true, true);
    lynx::gpu::gradient_gpu(d_x_ex, d_Drho_x, g_nx, g_ny, g_nz, g_FDn, nx_ex, ny_ex, 0, 1);
    lynx::gpu::gradient_gpu(d_x_ex, d_Drho_y, g_nx, g_ny, g_nz, g_FDn, nx_ex, ny_ex, 1, 1);
    lynx::gpu::gradient_gpu(d_x_ex, d_Drho_z, g_nx, g_ny, g_nz, g_FDn, nx_ex, ny_ex, 2, 1);

    // sigma = |∇ρ|²
    sigma_kernel<<<grid_sz, bs>>>(d_Drho_x, d_Drho_y, d_Drho_z, d_sigma, Nd);

    // Fused PBE kernel: (rho_xc, sigma) → (exc, Vxc, v2xc)
    lynx::gpu::gga_pbe_gpu(d_rho_xc, d_sigma, d_exc, d_Vxc, d_v2xc, Nd);

    // Divergence correction: Vxc += -div(v2xc * ∇ρ)
    // Scale gradients by v2xc (in place)
    v2xc_scale_kernel<<<grid_sz, bs>>>(d_Drho_x, d_v2xc, Nd);
    v2xc_scale_kernel<<<grid_sz, bs>>>(d_Drho_y, d_v2xc, Nd);
    v2xc_scale_kernel<<<grid_sz, bs>>>(d_Drho_z, d_v2xc, Nd);

    // Process each direction: halo → gradient → subtract from Vxc
    double* d_DDrho = d_sigma;  // reuse sigma buffer (no longer needed)

    // x-direction
    lynx::gpu::halo_exchange_gpu(d_Drho_x, d_x_ex, g_nx, g_ny, g_nz, g_FDn, 1, true, true, true);
    lynx::gpu::gradient_gpu(d_x_ex, d_DDrho, g_nx, g_ny, g_nz, g_FDn, nx_ex, ny_ex, 0, 1);
    divergence_sub_kernel<<<grid_sz, bs>>>(d_Vxc, d_DDrho, Nd);

    // y-direction
    lynx::gpu::halo_exchange_gpu(d_Drho_y, d_x_ex, g_nx, g_ny, g_nz, g_FDn, 1, true, true, true);
    lynx::gpu::gradient_gpu(d_x_ex, d_DDrho, g_nx, g_ny, g_nz, g_FDn, nx_ex, ny_ex, 1, 1);
    divergence_sub_kernel<<<grid_sz, bs>>>(d_Vxc, d_DDrho, Nd);

    // z-direction
    lynx::gpu::halo_exchange_gpu(d_Drho_z, d_x_ex, g_nx, g_ny, g_nz, g_FDn, 1, true, true, true);
    lynx::gpu::gradient_gpu(d_x_ex, d_DDrho, g_nx, g_ny, g_nz, g_FDn, nx_ex, ny_ex, 2, 1);
    divergence_sub_kernel<<<grid_sz, bs>>>(d_Vxc, d_DDrho, Nd);
}

// ============================================================
// GPU Poisson solver
// ============================================================
static int gpu_poisson_solve(double* d_rho, double* d_phi,
                              double* d_rhs, int Nd, double tol) {
    auto& ctx = lynx::gpu::GPUContext::instance();
    int bs = 256;
    int grid_sz = lynx::gpu::ceildiv(Nd, bs);

    // RHS = 4π(rho + pseudocharge)
    poisson_rhs_kernel<<<grid_sz, bs>>>(d_rho, g_d_pseudocharge, d_rhs,
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

    // Set halo workspace for operator callbacks
    g_aar_x_ex = ctx.buf.aar_x_ex;

    int iters = lynx::gpu::aar_gpu(
        poisson_op_gpu, poisson_precond_gpu,
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
static void gpu_pulay_mix(double* d_x, const double* d_g,
                            int Nd, int m_depth, double beta_mix) {
    auto& ctx = lynx::gpu::GPUContext::instance();
    int bs = 256;
    int grid_sz = lynx::gpu::ceildiv(Nd, bs);

    double* d_fk   = ctx.buf.mix_fk;
    double* d_xkm1 = ctx.buf.mix_xkm1;
    double* d_R    = ctx.buf.mix_R;
    double* d_F    = ctx.buf.mix_F;

    // Save old f_k → f_km1
    if (g_mix_iter > 0) {
        CUDA_CHECK(cudaMemcpy(g_d_mix_fkm1, d_fk, Nd * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    // f_k = g - x
    mix_residual_kernel<<<grid_sz, bs>>>(d_g, d_x, d_fk, Nd);

    // Store history
    if (g_mix_iter > 0) {
        int i_hist = (g_mix_iter - 1) % m_depth;
        mix_store_history_kernel<<<grid_sz, bs>>>(
            d_x, d_xkm1, d_fk, g_d_mix_fkm1, d_R, d_F, i_hist, Nd);
    }

    // Allocate workspace from scratch pool
    auto& sp = ctx.scratch_pool;
    size_t sp_cp = sp.checkpoint();
    double* d_x_wavg = sp.alloc<double>(Nd);
    double* d_f_wavg = sp.alloc<double>(Nd);

    if (g_mix_iter > 0) {
        int cols = std::min(g_mix_iter, m_depth);

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

    // Kerker step 1: Lf = (Lap - idiemac*kTF²) * f_wavg
    double* d_Lf = sp.alloc<double>(Nd);
    g_aar_x_ex = ctx.buf.aar_x_ex;
    {
        constexpr double idiemac_kTF2 = 0.1;  // idiemac * kTF²
        int nx_ex = g_nx + 2*g_FDn, ny_ex = g_ny + 2*g_FDn;
        lynx::gpu::halo_exchange_gpu(d_f_wavg, ctx.buf.aar_x_ex,
            g_nx, g_ny, g_nz, g_FDn, 1, true, true, true);
        lynx::gpu::laplacian_orth_v2_gpu(ctx.buf.aar_x_ex, nullptr, d_Lf,
            g_nx, g_ny, g_nz, g_FDn, nx_ex, ny_ex,
            1.0, 0.0, -idiemac_kTF2, g_kerker_rhs_diag, 1);
    }

    // Kerker step 2: Solve (-Lap + kTF²)*Pf = Lf via AAR
    {
        double* d_kr    = sp.alloc<double>(Nd);
        double* d_kf    = sp.alloc<double>(Nd);
        double* d_kAx   = sp.alloc<double>(Nd);
        double* d_kX    = sp.alloc<double>(Nd * 7);
        double* d_kF    = sp.alloc<double>(Nd * 7);
        double* d_kxold = sp.alloc<double>(Nd);
        double* d_kfold = sp.alloc<double>(Nd);

        lynx::gpu::aar_gpu(
            kerker_op_gpu, kerker_precond_gpu,
            d_Lf, d_Pf, Nd,
            0.6, 0.6, 7, 6, g_precond_tol, 1000,
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
    g_mix_iter++;
}

// ============================================================
// Main: GPU SCF for BaTiO3
// ============================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("=== GPU SCF End-to-End Test: BaTiO3 (Fully GPU-Resident) ===\n");
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("GPU: %s\n\n", prop.name);
    }

    // ============================================================
    // Phase 1: CPU setup
    // ============================================================
    std::string json_file = "/home/xx/Desktop/LYNX/examples/BaTiO3.json";
    auto config = InputParser::parse(json_file);
    InputParser::resolve_pseudopotentials(config);
    InputParser::validate(config);

    // Initialize LynxContext for EigenSolver setup
    auto& ctx = LynxContext::instance();
    ctx.reset();
    ctx.initialize(config, MPI_COMM_WORLD);

    Lattice lattice(config.latvec, config.cell_type);
    FDGrid grid(config.Nx, config.Ny, config.Nz, lattice,
                config.bcx, config.bcy, config.bcz);
    FDStencil stencil(config.fd_order, grid, lattice);

    int Nspin = 1;
    DomainVertices verts = {0, config.Nx - 1, 0, config.Ny - 1, 0, config.Nz - 1};
    Domain domain(grid, verts);

    // Load atoms
    std::vector<AtomType> atom_types;
    std::vector<Vec3> all_positions;
    std::vector<int> type_indices;
    int total_Nelectron = 0;

    for (size_t it = 0; it < config.atom_types.size(); ++it) {
        const auto& at_in = config.atom_types[it];
        int n_atoms = (int)at_in.coords.size();
        Pseudopotential psd_tmp;
        psd_tmp.load_psp8(at_in.pseudo_file);
        double Zval = psd_tmp.Zval();
        AtomType atype(at_in.element, 1.0, Zval, n_atoms);
        atype.psd().load_psp8(at_in.pseudo_file);
        for (int ia = 0; ia < n_atoms; ++ia) {
            Vec3 pos = at_in.coords[ia];
            if (at_in.fractional) pos = lattice.frac_to_cart(pos);
            all_positions.push_back(pos);
            type_indices.push_back((int)it);
        }
        total_Nelectron += (int)Zval * n_atoms;
        atom_types.push_back(std::move(atype));
    }

    int Nelectron = total_Nelectron;
    int Natom = (int)all_positions.size();
    Crystal crystal(std::move(atom_types), all_positions, type_indices, lattice);

    double rc_max = 0.0;
    for (int it = 0; it < crystal.n_types(); ++it) {
        const auto& psd = crystal.types()[it].psd();
        for (auto rc : psd.rc()) rc_max = std::max(rc_max, rc);
        if (!psd.radial_grid().empty())
            rc_max = std::max(rc_max, psd.radial_grid().back());
    }
    rc_max += 2.0 * std::max({grid.dx(), grid.dy(), grid.dz()});

    std::vector<AtomInfluence> influence;
    crystal.compute_atom_influence(domain, rc_max, influence);
    std::vector<AtomNlocInfluence> nloc_influence;
    crystal.compute_nloc_influence(domain, nloc_influence);

    Electrostatics elec;
    elec.compute_pseudocharge(crystal, influence, domain, grid, stencil);

    int Nd = domain.Nd_d();
    std::vector<double> h_Vloc(Nd, 0.0);
    elec.compute_Vloc(crystal, influence, domain, grid, h_Vloc.data());
    elec.compute_Ec(h_Vloc.data(), Nd, grid.dV());

    // NLCC core density
    std::vector<double> rho_core(Nd, 0.0);
    bool has_nlcc = elec.compute_core_density(crystal, influence, domain, grid, rho_core.data());

    HaloExchange halo(domain, stencil.FDn());
    Laplacian laplacian(stencil, domain);
    Gradient gradient(stencil, domain);
    NonlocalProjector vnl;
    vnl.setup(crystal, nloc_influence, domain, grid);
    Hamiltonian hamiltonian;
    hamiltonian.setup(stencil, domain, grid, halo, &vnl);

    // XC functional — CPU object for initial Veff only
    XCFunctional xcfunc;
    xcfunc.setup(XCType::GGA_PBE, domain, grid, &gradient, &halo);

    int Nband = 29;  // BaTiO3.json Nstates
    double dV = grid.dV();
    int nx = grid.Nx(), ny = grid.Ny(), nz = grid.Nz();
    int FDn = stencil.FDn();

    printf("BaTiO3: Nd=%d (%dx%dx%d), Nband=%d, FDn=%d, dV=%.10f\n",
           Nd, nx, ny, nz, Nband, FDn, dV);

    // Set global state for GPU callbacks
    g_nx = nx; g_ny = ny; g_nz = nz; g_FDn = FDn;
    g_Nd = Nd;
    g_dV = dV;

    // Stencil coefficients for operator callbacks
    const double* D2x = stencil.D2_coeff_x();
    const double* D2y = stencil.D2_coeff_y();
    const double* D2z = stencil.D2_coeff_z();
    double D2sum = D2x[0] + D2y[0] + D2z[0];

    g_diag_coeff_ham = -0.5 * D2sum;
    g_poisson_diag = -1.0 * D2sum;                     // diag for -Lap
    g_jacobi_m_inv = -1.0 / D2sum;                     // Jacobi for Poisson
    g_kerker_diag = -1.0 * D2sum + 1.0;                // diag for (-Lap + kTF²)
    g_kerker_m_inv = -1.0 / (D2sum - 1.0);             // Jacobi for Kerker
    g_kerker_rhs_diag = 1.0 * D2sum + (-0.1);          // diag for (Lap - 0.1*kTF²)

    // TOL_PRECOND = h_eff² * 1e-3
    double h_eff = grid.dx();
    g_precond_tol = h_eff * h_eff * 1e-3;

    // Upload FD stencil coefficients to GPU constant memory
    lynx::gpu::upload_stencil_coefficients(
        stencil.D2_coeff_x(), stencil.D2_coeff_y(), stencil.D2_coeff_z(),
        stencil.D1_coeff_x(), stencil.D1_coeff_y(), stencil.D1_coeff_z(),
        nullptr, nullptr, nullptr, FDn);

    // ============================================================
    // Phase 2: GPU memory allocation + data upload
    // ============================================================
    auto& gpu_ctx = lynx::gpu::GPUContext::instance();
    gpu_ctx.init_scf_buffers(Nd, nx, ny, nz, FDn,
                          Nband, Nband, Nspin,
                          7, 7, 1,
                          0, 0, 0,
                          true, false);  // is_gga=true

    // Upload nonlocal projector data to GPU (once)
    GPUNonlocalData gpu_vnl_data;
    gpu_vnl_data.setup(vnl, crystal, nloc_influence, Nband);
    g_gpu_vnl = &gpu_vnl_data;

    // Upload NLCC core density to GPU
    if (has_nlcc) {
        CUDA_CHECK(cudaMalloc(&g_d_rho_core, Nd * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(g_d_rho_core, rho_core.data(), Nd * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Upload pseudocharge to GPU
    CUDA_CHECK(cudaMalloc(&g_d_pseudocharge, Nd * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(g_d_pseudocharge, elec.pseudocharge().data(), Nd * sizeof(double), cudaMemcpyHostToDevice));

    // Allocate mixer persistent buffer
    CUDA_CHECK(cudaMalloc(&g_d_mix_fkm1, Nd * sizeof(double)));

    // Validate GPU H*psi vs CPU H*psi before starting SCF
    {
        std::vector<double> h_test(Nd);
        srand(42);
        for (int i = 0; i < Nd; i++) h_test[i] = (rand() / (double)RAND_MAX - 0.5);

        std::vector<double> h_Veff_test(Nd, 0.1);
        std::vector<double> h_Hpsi_cpu(Nd, 0.0);
        hamiltonian.apply(h_test.data(), h_Veff_test.data(), h_Hpsi_cpu.data(), 1);

        double *d_test_psi, *d_test_Hpsi, *d_test_Veff, *d_test_xex;
        int Nd_ex = (nx + 2*FDn) * (ny + 2*FDn) * (nz + 2*FDn);
        CUDA_CHECK(cudaMalloc(&d_test_psi, Nd * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_test_Hpsi, Nd * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_test_Veff, Nd * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_test_xex, Nd_ex * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_test_psi, h_test.data(), Nd * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_test_Veff, h_Veff_test.data(), Nd * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_test_Hpsi, 0, Nd * sizeof(double)));

        gpu_hamiltonian_apply(d_test_psi, d_test_Veff, d_test_Hpsi, d_test_xex, 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> h_Hpsi_gpu(Nd);
        CUDA_CHECK(cudaMemcpy(h_Hpsi_gpu.data(), d_test_Hpsi, Nd * sizeof(double), cudaMemcpyDeviceToHost));

        double max_err = 0, norm_cpu = 0;
        for (int i = 0; i < Nd; i++) {
            max_err = std::max(max_err, std::abs(h_Hpsi_gpu[i] - h_Hpsi_cpu[i]));
            norm_cpu += h_Hpsi_cpu[i] * h_Hpsi_cpu[i];
        }
        double rel_err = max_err / std::sqrt(norm_cpu / Nd);
        printf("H*psi validation (local+Vnl): max_err=%.3e, rel_err=%.3e %s\n",
               max_err, rel_err, rel_err < 1e-10 ? "[PASS]" : "[FAIL]");

        cudaFree(d_test_psi); cudaFree(d_test_Hpsi);
        cudaFree(d_test_Veff); cudaFree(d_test_xex);
    }

    // Device pointers
    double* d_psi = gpu_ctx.buf.psi;
    double* d_Hpsi = gpu_ctx.buf.Hpsi;
    double* d_Veff = gpu_ctx.buf.Veff;
    double* d_rho = gpu_ctx.buf.rho;
    double* d_rho_new = gpu_ctx.buf.rho_total;
    double* d_phi = gpu_ctx.buf.phi;
    double* d_exc = gpu_ctx.buf.exc;
    double* d_Vxc = gpu_ctx.buf.Vc;
    double* d_eigvals = gpu_ctx.buf.eigenvalues;
    double* d_occ = gpu_ctx.buf.occupations;

    // CheFSI workspace
    double* d_Xold = gpu_ctx.buf.Xold;
    double* d_Xnew = gpu_ctx.buf.Xnew;
    double* d_x_ex = gpu_ctx.buf.x_ex;
    double* d_Hs = gpu_ctx.buf.Hs;
    double* d_Ms = gpu_ctx.buf.Ms;

    // d_Y must be separate from d_Hpsi (used as d_HX inside eigensolver)
    double* d_Y;
    CUDA_CHECK(cudaMalloc(&d_Y, (size_t)Nd * Nband * sizeof(double)));

    // ============================================================
    // Phase 3: Initialize
    // ============================================================

    // Initial density: atomic superposition
    std::vector<double> h_rho(Nd, 0.0);
    elec.compute_atomic_density(crystal, influence, domain, grid, h_rho.data(), Nelectron);
    CUDA_CHECK(cudaMemcpy(d_rho, h_rho.data(), Nd * sizeof(double), cudaMemcpyHostToDevice));

    // Randomize wavefunctions on CPU, upload to GPU
    Wavefunction wfn;
    wfn.allocate(Nd, Nband, 1, 1);
    wfn.randomize(0, 0, 1);

    {
        std::vector<double> h_psi(Nd * Nband);
        for (int j = 0; j < Nband; j++)
            std::memcpy(h_psi.data() + j * Nd, wfn.psi(0, 0).col(j), Nd * sizeof(double));
        CUDA_CHECK(cudaMemcpy(d_psi, h_psi.data(), (size_t)Nd * Nband * sizeof(double),
                               cudaMemcpyHostToDevice));
    }

    // ============================================================
    // Phase 4: Initial Veff (CPU path — runs once)
    // ============================================================
    {
        std::vector<double> h_Vxc(Nd), h_exc(Nd), h_phi(Nd), h_Veff(Nd);

        // XC on CPU (GGA_PBE with NLCC) — initial only
        {
            std::vector<double> rho_xc(Nd);
            for (int i = 0; i < Nd; i++) {
                rho_xc[i] = h_rho[i] + (has_nlcc ? rho_core[i] : 0.0);
                if (rho_xc[i] < 1e-14) rho_xc[i] = 1e-14;
            }
            xcfunc.evaluate(rho_xc.data(), h_Vxc.data(), h_exc.data(), Nd);
        }
        CUDA_CHECK(cudaMemcpy(d_exc, h_exc.data(), Nd * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Vxc, h_Vxc.data(), Nd * sizeof(double), cudaMemcpyHostToDevice));

        // Poisson on CPU — initial only
        std::vector<double> rhs(Nd);
        for (int i = 0; i < Nd; i++)
            rhs[i] = 4.0 * constants::PI * (h_rho[i] + elec.pseudocharge().data()[i]);
        double rhs_sum = 0;
        for (int i = 0; i < Nd; i++) rhs_sum += rhs[i];
        double rhs_mean = rhs_sum / grid.Nd();
        for (int i = 0; i < Nd; i++) rhs[i] -= rhs_mean;

        PoissonSolver poisson_init;
        poisson_init.setup(laplacian, stencil, domain, grid, halo);
        std::fill(h_phi.begin(), h_phi.end(), 0.0);
        poisson_init.solve(rhs.data(), h_phi.data(), 1e-6 * 0.01);

        double phi_sum = 0;
        for (int i = 0; i < Nd; i++) phi_sum += h_phi[i];
        double phi_mean = phi_sum / grid.Nd();
        for (int i = 0; i < Nd; i++) h_phi[i] -= phi_mean;

        for (int i = 0; i < Nd; i++)
            h_Veff[i] = h_Vxc[i] + h_phi[i];

        CUDA_CHECK(cudaMemcpy(d_Veff, h_Veff.data(), Nd * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_phi, h_phi.data(), Nd * sizeof(double), cudaMemcpyHostToDevice));
    }

    // ============================================================
    // Phase 5: Lanczos spectrum bounds (CPU — runs once)
    // ============================================================
    EigenSolver eigsolver;
    eigsolver.setup(ctx, hamiltonian);

    double eigval_min, eigval_max;
    {
        std::vector<double> h_Veff(Nd);
        CUDA_CHECK(cudaMemcpy(h_Veff.data(), d_Veff, Nd * sizeof(double), cudaMemcpyDeviceToHost));
        eigsolver.lanczos_bounds(h_Veff.data(), Nd, eigval_min, eigval_max);
    }
    double lambda_cutoff = 0.5 * (eigval_min + eigval_max);
    printf("Lanczos: eigmin=%.6e, eigmax=%.6e, lambda_cutoff=%.6e\n",
           eigval_min, eigval_max, lambda_cutoff);

    // Auto Chebyshev degree
    double p3 = -700.0/3, p2 = 1240.0/3, p1 = -773.0/3, p0 = 1078.0/15;
    int cheb_degree = (h_eff > 0.7) ? 14 : (int)std::round(((p3*h_eff + p2)*h_eff + p1)*h_eff + p0);
    printf("Chebyshev degree: %d (h_eff=%.6f)\n", cheb_degree, h_eff);

    // Smearing parameters
    double smearing_eV = 0.2;
    double beta_smearing = constants::EH / smearing_eV;

    // ============================================================
    // Phase 6: GPU SCF Loop
    // ============================================================
    printf("\n--- GPU SCF Loop (fully GPU-resident) ---\n");

    int max_iter = 100;
    double scf_tol = 1e-6;
    bool converged = false;
    int rho_trigger = 4;

    std::vector<double> h_eigvals(Nband), h_occ(Nband);
    std::vector<double> kpt_weights = {1.0};
    double Ef = 0.0;

    for (int scf_iter = 0; scf_iter < max_iter; scf_iter++) {
        int nchefsi = (scf_iter == 0) ? rho_trigger : 1;

        // Step 1: GPU Eigensolver (nchefsi passes of CheFSI)
        for (int pass = 0; pass < nchefsi; pass++) {
            lynx::gpu::eigensolver_solve_gpu(
                d_psi, d_eigvals, d_Veff,
                d_Y, d_Xold, d_Xnew,
                d_Hpsi, d_x_ex,
                d_Hs, d_Ms,
                Nd, Nband,
                lambda_cutoff, eigval_min, eigval_max,
                cheb_degree, dV,
                gpu_hamiltonian_apply);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download eigenvalues for occupation computation (tiny: 29 doubles)
        CUDA_CHECK(cudaMemcpy(h_eigvals.data(), d_eigvals, Nband * sizeof(double),
                               cudaMemcpyDeviceToHost));

        // Update spectral bounds
        if (scf_iter > 0) eigval_min = h_eigvals[0];
        lambda_cutoff = h_eigvals[Nband - 1] + 0.1;

        // Compute occupations on CPU (lightweight)
        for (int n = 0; n < Nband; n++)
            wfn.eigenvalues(0, 0)(n) = h_eigvals[n];

        MPIComm bandcomm_self(MPI_COMM_SELF);
        Ef = Occupation::compute(wfn, Nelectron, beta_smearing,
                                SmearingType::GaussianSmearing,
                                kpt_weights, bandcomm_self, bandcomm_self, 0);

        for (int n = 0; n < Nband; n++)
            h_occ[n] = wfn.occupations(0, 0)(n);

        CUDA_CHECK(cudaMemcpy(d_occ, h_occ.data(), Nband * sizeof(double),
                               cudaMemcpyHostToDevice));

        // Step 2: Compute new density on GPU
        CUDA_CHECK(cudaMemset(d_rho_new, 0, Nd * sizeof(double)));
        lynx::gpu::compute_density_gpu(d_psi, d_occ, d_rho_new, Nd, Nband, 2.0);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 3: Compute energy (D2H for energy components — diagnostic only)
        {
            std::vector<double> h_rho_in(Nd), h_Veff_e(Nd), h_phi_e(Nd), h_exc_e(Nd), h_Vxc_e(Nd);
            CUDA_CHECK(cudaMemcpy(h_rho_in.data(), d_rho, Nd * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_Veff_e.data(), d_Veff, Nd * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_phi_e.data(), d_phi, Nd * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_exc_e.data(), d_exc, Nd * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_Vxc_e.data(), d_Vxc, Nd * sizeof(double), cudaMemcpyDeviceToHost));

            // Download psi for Eband computation
            {
                std::vector<double> h_psi_tmp(Nd * Nband);
                CUDA_CHECK(cudaMemcpy(h_psi_tmp.data(), d_psi, (size_t)Nd * Nband * sizeof(double),
                                       cudaMemcpyDeviceToHost));
                for (int j = 0; j < Nband; j++)
                    std::memcpy(wfn.psi(0, 0).col(j), h_psi_tmp.data() + j * Nd, Nd * sizeof(double));
            }

            ElectronDensity dens_in;
            dens_in.allocate(Nd, 1);
            std::memcpy(dens_in.rho(0).data(), h_rho_in.data(), Nd * sizeof(double));
            std::memcpy(dens_in.rho_total().data(), h_rho_in.data(), Nd * sizeof(double));

            EnergyComponents energy = Energy::compute_all(
                wfn, dens_in, h_Veff_e.data(), h_phi_e.data(),
                h_exc_e.data(), h_Vxc_e.data(), elec.pseudocharge().data(),
                elec.Eself(), elec.Ec(), beta_smearing,
                SmearingType::GaussianSmearing,
                kpt_weights, Nd, dV,
                has_nlcc ? rho_core.data() : nullptr, Ef_, 0,
                nullptr, nullptr, 1);

            // SCF error from GPU density
            std::vector<double> h_rho_new(Nd);
            CUDA_CHECK(cudaMemcpy(h_rho_new.data(), d_rho_new, Nd * sizeof(double), cudaMemcpyDeviceToHost));

            double sum_sq_diff = 0, sum_sq_out = 0;
            for (int i = 0; i < Nd; i++) {
                double diff = h_rho_new[i] - h_rho_in[i];
                sum_sq_diff += diff * diff;
                sum_sq_out += h_rho_new[i] * h_rho_new[i];
            }
            double scf_error = (sum_sq_out > 0) ? std::sqrt(sum_sq_diff / sum_sq_out) : 0;

            printf("SCF iter %3d: Etot = %18.10f Ha, SCF error = %10.3e\n",
                   scf_iter + 1, energy.Etotal, scf_error);

            if (scf_iter >= 2 && scf_error < scf_tol) {
                converged = true;
                printf("\nGPU SCF converged after %d iterations.\n", scf_iter + 1);
                printf("Final energy: %.10f Ha\n", energy.Etotal);

                double ref_energy = -136.9227982;
                double diff = std::abs(energy.Etotal - ref_energy);
                printf("CPU reference: %.7f Ha\n", ref_energy);
                printf("Difference: %.2e Ha (%.4f meV/atom)\n",
                       diff, diff * 27211.386 / Natom);

                if (diff < 1e-4) {
                    printf("\n*** GPU SCF MATCHES CPU REFERENCE ***\n");
                } else {
                    printf("\n*** WARNING: GPU SCF differs from CPU ***\n");
                }
                break;
            }

            Ef_ = Ef;
        }

        // Step 4: GPU Pulay + Kerker mixer
        gpu_pulay_mix(d_rho, d_rho_new, Nd, 7, 0.3);

        // Clamp + normalize density on GPU
        {
            int bs = 256;
            int gs = lynx::gpu::ceildiv(Nd, bs);
            clamp_min_kernel<<<gs, bs>>>(d_rho, 0.0, Nd);
            CUDA_CHECK(cudaDeviceSynchronize());

            double rho_sum = gpu_sum(d_rho, Nd);
            double Ne_current = rho_sum * dV;
            if (Ne_current > 1e-10) {
                double sc = (double)Nelectron / Ne_current;
                scale_kernel<<<gs, bs>>>(d_rho, sc, Nd);
            }
        }

        // Step 5: GPU XC (GGA_PBE with NLCC)
        gpu_xc_evaluate(d_rho, d_exc, d_Vxc, Nd, has_nlcc);

        // Step 6: GPU Poisson solver
        gpu_poisson_solve(d_rho, d_phi, gpu_ctx.buf.b, Nd, scf_tol * 0.01);

        // Step 7: Veff = Vxc + phi (GPU)
        {
            int bs = 256;
            int gs = lynx::gpu::ceildiv(Nd, bs);
            veff_combine_kernel<<<gs, bs>>>(d_Vxc, d_phi, d_Veff, Nd);
        }
    }

    if (!converged) {
        printf("WARNING: GPU SCF did not converge within %d iterations.\n", max_iter);
    }

    // ============================================================
    // GPU Force/Stress Test
    // ============================================================
    if (converged) {
        printf("\n--- GPU Force/Stress Test ---\n");

        int n_phys = crystal.n_atom_total();
        int ntypes = crystal.n_types();

        // Build physical atom IP_displ
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

        // GPU force/stress computation
        std::vector<double> gpu_f_nloc(3 * n_phys, 0.0);
        std::array<double, 6> gpu_stress_k = {}, gpu_stress_nl = {};
        double gpu_energy_nl = 0.0;

        lynx::gpu::compute_force_stress_gpu(
            d_psi, d_occ,
            gpu_vnl_data.d_Chi_flat, gpu_vnl_data.d_gpos_flat,
            gpu_vnl_data.d_gpos_offsets, gpu_vnl_data.d_chi_offsets,
            gpu_vnl_data.d_ndc_arr, gpu_vnl_data.d_nproj_arr,
            gpu_vnl_data.d_IP_displ, gpu_vnl_data.d_Gamma,
            gpu_vnl_data.n_influence, gpu_vnl_data.total_phys_nproj,
            gpu_vnl_data.max_ndc, gpu_vnl_data.max_nproj,
            n_phys,
            IP_displ_phys.data(),
            atom_pos.data(),
            nx, ny, nz, FDn, Nd, Nband,
            dV, grid.dx(), grid.dy(), grid.dz(),
            domain.vertices().xs, domain.vertices().ys, domain.vertices().zs,
            2.0,  // occfac for Nspin=1
            gpu_f_nloc.data(), gpu_stress_k.data(), gpu_stress_nl.data(), &gpu_energy_nl);

        // Normalize stress by cell volume
        {
            Vec3 L = grid.lattice().lengths();
            double Jacbdet = grid.lattice().jacobian() / (L.x * L.y * L.z);
            double cell_measure = Jacbdet * L.x * L.y * L.z;  // all periodic
            for (int i = 0; i < 6; i++) {
                gpu_stress_k[i] /= cell_measure;
                gpu_stress_nl[i] /= cell_measure;
            }
        }

        // Download psi for CPU reference computation
        {
            std::vector<double> h_psi_tmp(Nd * Nband);
            CUDA_CHECK(cudaMemcpy(h_psi_tmp.data(), d_psi, (size_t)Nd * Nband * sizeof(double),
                                   cudaMemcpyDeviceToHost));
            for (int j = 0; j < Nband; j++)
                std::memcpy(wfn.psi(0, 0).col(j), h_psi_tmp.data() + j * Nd, Nd * sizeof(double));
        }
        // Download occupations
        {
            std::vector<double> h_occ_tmp(Nband);
            CUDA_CHECK(cudaMemcpy(h_occ_tmp.data(), d_occ, Nband * sizeof(double), cudaMemcpyDeviceToHost));
            for (int n = 0; n < Nband; n++)
                wfn.occupations(0, 0)(n) = h_occ_tmp[n];
        }

        // TODO: CPU force/stress comparison needs updating after API changes.
        // The old low-level Forces::compute / Stress::compute signatures
        // were replaced by high-level ones taking (LynxContext, SystemConfig, ...).
        // GPU force/stress values are printed below for manual verification.
        printf("GPU nonlocal forces (first 3 atoms):\n");
        for (int ia = 0; ia < std::min(3, n_phys); ia++)
            printf("  atom %d: [%.6e, %.6e, %.6e]\n", ia,
                   gpu_f_nloc[ia*3], gpu_f_nloc[ia*3+1], gpu_f_nloc[ia*3+2]);
        printf("Kinetic stress (GPU): [%.6e, %.6e, %.6e, %.6e, %.6e, %.6e]\n",
               gpu_stress_k[0], gpu_stress_k[1], gpu_stress_k[2],
               gpu_stress_k[3], gpu_stress_k[4], gpu_stress_k[5]);
        printf("Nonlocal stress (GPU): [%.6e, %.6e, %.6e, %.6e, %.6e, %.6e]\n",
               gpu_stress_nl[0], gpu_stress_nl[1], gpu_stress_nl[2],
               gpu_stress_nl[3], gpu_stress_nl[4], gpu_stress_nl[5]);
    }

    gpu_vnl_data.free();
    cudaFree(d_Y);
    if (g_d_rho_core) cudaFree(g_d_rho_core);
    if (g_d_pseudocharge) cudaFree(g_d_pseudocharge);
    if (g_d_mix_fkm1) cudaFree(g_d_mix_fkm1);
    LynxContext::instance().reset();
    MPI_Finalize();
    return converged ? 0 : 1;
}
