#ifdef USE_CUDA
#include <cmath>
#include <vector>
#include "core/gpu_common.cuh"
#include "core/GPUContext.cuh"
#include "core/KPoints.hpp"
#include "core/LynxContext.hpp"
#include "atoms/Crystal.hpp"
#include "operators/Hamiltonian.hpp"
#include "operators/Hamiltonian.cuh"
#include "operators/NonlocalProjector.hpp"
#include "operators/NonlocalProjector.cuh"
#include "operators/ComplexOperators.cuh"
#include "operators/SOCOperators.cuh"
#include "operators/Gradient.cuh"
#include "parallel/HaloExchange.cuh"
#include "operators/Laplacian.cuh"
#include "xc/GPUExactExchange.cuh"
#include <cublas_v2.h>

namespace lynx {
namespace gpu {

// Hamiltonian local part: H_local*psi = -0.5*Lap*psi + Veff*psi + c*psi
// Uses batched halo exchange (8 launches total instead of 8*ncol)
// and V2 Laplacian (single launch with multi-column batching)
void hamiltonian_apply_local_gpu(
    const double* d_psi, const double* d_Veff, double* d_Hpsi,
    double* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol, double c,
    bool is_orthogonal,
    bool periodic_x, bool periodic_y, bool periodic_z,
    double diag_coeff,
    bool has_xy, bool has_xz, bool has_yz,
    cudaStream_t stream)
{
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;

    // Step 1: Batched halo exchange (all columns in ~8 launches)
    halo_exchange_batched_gpu(d_psi, d_x_ex, nx, ny, nz, FDn, ncol,
                               periodic_x, periodic_y, periodic_z, stream);

    // Step 2: Apply (-0.5*Lap + Veff + c*I) using V2 kernel
    if (is_orthogonal) {
        laplacian_orth_v7_gpu(d_x_ex, d_Veff, d_Hpsi,
                               nx, ny, nz, FDn, nx_ex, ny_ex,
                               -0.5, 1.0, c, diag_coeff, ncol, stream);
    } else {
        laplacian_nonorth_gpu(d_x_ex, d_Veff, d_Hpsi,
                              nx, ny, nz, FDn, nx_ex, ny_ex,
                              -0.5, 1.0, c, diag_coeff,
                              has_xy, has_xz, has_yz, ncol, stream);
    }
}

} // namespace gpu

// ============================================================
// mGGA and spinor kernels (duplicated from GPUSCF.cu so that
// Hamiltonian's Device-dispatching methods can call them
// without depending on GPUSCF).
// These are file-static to avoid ODR violations.
// ============================================================

static __global__ void vtau_multiply_batched_kernel(
    const double* __restrict__ dpsi,
    const double* __restrict__ vtau,
    double* __restrict__ out, int Nd, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) out[idx] = vtau[idx % Nd] * dpsi[idx];
}

static __global__ void vtau_lapcT_multiply_kernel(
    const double* __restrict__ dpsi_x,
    const double* __restrict__ dpsi_y,
    const double* __restrict__ dpsi_z,
    const double* __restrict__ vtau,
    double* __restrict__ out, int N,
    double L0, double L1, double L2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = vtau[i] * (L0*dpsi_x[i] + L1*dpsi_y[i] + L2*dpsi_z[i]);
}

static __global__ void mgga_ham_sub_kernel(
    double* __restrict__ Hpsi,
    const double* __restrict__ div, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) Hpsi[i] -= 0.5 * div[i];
}

static __global__ void vtau_multiply_z_kernel(
    const cuDoubleComplex* __restrict__ dpsi,
    const double* __restrict__ vtau,
    cuDoubleComplex* __restrict__ out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i].x = vtau[i] * dpsi[i].x;
        out[i].y = vtau[i] * dpsi[i].y;
    }
}

static __global__ void vtau_lapcT_multiply_z_kernel(
    const cuDoubleComplex* __restrict__ dpsi_x,
    const cuDoubleComplex* __restrict__ dpsi_y,
    const cuDoubleComplex* __restrict__ dpsi_z,
    const double* __restrict__ vtau,
    cuDoubleComplex* __restrict__ out, int N,
    double L0, double L1, double L2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double re = L0*dpsi_x[i].x + L1*dpsi_y[i].x + L2*dpsi_z[i].x;
        double im = L0*dpsi_x[i].y + L1*dpsi_y[i].y + L2*dpsi_z[i].y;
        out[i].x = vtau[i] * re;
        out[i].y = vtau[i] * im;
    }
}

static __global__ void mgga_ham_sub_z_kernel(
    cuDoubleComplex* __restrict__ Hpsi,
    const cuDoubleComplex* __restrict__ div, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Hpsi[i].x -= 0.5 * div[i].x;
        Hpsi[i].y -= 0.5 * div[i].y;
    }
}

static __global__ void spinor_extract_kernel(
    const cuDoubleComplex* __restrict__ spinor,
    cuDoubleComplex* __restrict__ out,
    int Nd, int ncol, int component)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nd * ncol) return;
    int ig = i % Nd;
    int n = i / Nd;
    out[n * Nd + ig] = spinor[n * 2 * Nd + component * Nd + ig];
}

static __global__ void spinor_scatter_kernel(
    const cuDoubleComplex* __restrict__ in,
    cuDoubleComplex* __restrict__ spinor,
    int Nd, int ncol, int component)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nd * ncol) return;
    int ig = i % Nd;
    int n = i / Nd;
    spinor[n * 2 * Nd + component * Nd + ig] = in[n * Nd + ig];
}

// ============================================================
// GPUHamiltonianState — holds all GPU-side data needed by the
// Device-dispatching apply() methods.
// ============================================================

struct GPUHamiltonianState {
    // Grid parameters
    int nx = 0, ny = 0, nz = 0, FDn = 0, Nd = 0;
    double dV = 0.0;

    // Stencil parameters
    double diag_coeff_ham = 0.0;
    bool is_orth = true;
    bool has_mixed_deriv = false;

    // Workspace for halo-exchanged psi (real and complex)
    double* d_x_ex = nullptr;
    cuDoubleComplex* d_x_ex_z = nullptr;

    // mGGA work buffers (real)
    double* d_mgga_dpsi = nullptr;
    double* d_mgga_dpsi_y = nullptr;
    double* d_mgga_dpsi_z_r = nullptr;
    double* d_mgga_vtdpsi = nullptr;
    double* d_mgga_div = nullptr;
    double* d_mgga_vt_ex = nullptr;
    // mGGA work buffers (complex)
    cuDoubleComplex* d_mgga_dpsi_z = nullptr;
    cuDoubleComplex* d_mgga_dpsi_yz = nullptr;
    cuDoubleComplex* d_mgga_dpsi_zz = nullptr;
    cuDoubleComplex* d_mgga_vtdpsi_z = nullptr;
    cuDoubleComplex* d_mgga_div_z = nullptr;
    cuDoubleComplex* d_mgga_vt_ex_z = nullptr;

    // Current vtau pointer (device)
    const double* d_vtau_active = nullptr;

    // Metric tensor for non-orthogonal cells (row-major 3x3)
    double lapcT[9] = {};

    // EXX state (gamma)
    double* d_Xi = nullptr;
    double* d_Y_exx = nullptr;
    double exx_frac = 0.0;
    int exx_Nocc = 0;
    bool exx_active = false;
    int exx_spin = 0;
    int exx_kpt = 0;

    // EXX state (k-point)
    std::vector<cuDoubleComplex*> d_Xi_kpt;
    cuDoubleComplex* d_Y_exx_z = nullptr;

    // Bloch factors
    double kxLx = 0, kyLy = 0, kzLz = 0;
    double* d_bloch_fac = nullptr;
    std::vector<Vec3> h_image_shifts;  // per-atom image shifts for Bloch phase computation

    // GPU nonlocal data (mirroring GPUSCF's GPUNonlocalData)
    struct {
        double* d_Chi_flat = nullptr;
        int*    d_gpos_flat = nullptr;
        int*    d_gpos_offsets = nullptr;
        int*    d_chi_offsets = nullptr;
        int*    d_ndc_arr = nullptr;
        int*    d_nproj_arr = nullptr;
        int*    d_IP_displ = nullptr;
        double* d_Gamma = nullptr;
        double* d_alpha = nullptr;
        int n_influence = 0;
        int total_phys_nproj = 0;
        int max_ndc = 0;
        int max_nproj = 0;
    } gpu_vnl;

    // Complex nonlocal alpha
    cuDoubleComplex* d_alpha_z = nullptr;

    // SOC data pointers
    struct {
        cuDoubleComplex* d_Chi_soc_flat = nullptr;
        int*    d_gpos_offsets_soc = nullptr;
        int*    d_chi_soc_offsets = nullptr;
        int*    d_ndc_arr_soc = nullptr;
        int*    d_nproj_soc_arr = nullptr;
        int*    d_IP_displ_soc = nullptr;
        double* d_Gamma_soc = nullptr;
        int*    d_proj_l = nullptr;
        int*    d_proj_m = nullptr;
        cuDoubleComplex* d_alpha_soc_up = nullptr;
        cuDoubleComplex* d_alpha_soc_dn = nullptr;
        int n_influence_soc = 0;
        int total_soc_nproj = 0;
        int max_ndc_soc = 0;
        int max_nproj_soc = 0;
    } gpu_soc;

    bool has_soc = false;

    // K-point info
    const class KPoints* kpoints = nullptr;
};

// ── setup_gpu / cleanup_gpu ──────────────────────────────────

void Hamiltonian::setup_gpu(const LynxContext& ctx,
                            const NonlocalProjector* vnl,
                            const Crystal& crystal,
                            const std::vector<AtomNlocInfluence>& nloc_influence,
                            int Nband)
{
    if (!gpu_state_raw_)
        gpu_state_raw_ = new GPUHamiltonianState();
    auto* gs = static_cast<GPUHamiltonianState*>(gpu_state_raw_);

    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;

    // ── Grid parameters ─────────────────────────────────────────
    const auto& grid    = ctx.grid();
    const auto& domain  = ctx.domain();
    const auto& stencil = ctx.stencil();

    gs->nx  = grid.Nx();
    gs->ny  = grid.Ny();
    gs->nz  = grid.Nz();
    gs->FDn = stencil.FDn();
    gs->Nd  = domain.Nd_d();
    gs->dV  = grid.dV();

    gs->is_orth         = grid.lattice().is_orthogonal();
    gs->has_mixed_deriv = !gs->is_orth;

    // ── Stencil coefficients ────────────────────────────────────
    const double* D2x = stencil.D2_coeff_x();
    const double* D2y = stencil.D2_coeff_y();
    const double* D2z = stencil.D2_coeff_z();
    gs->diag_coeff_ham = -0.5 * (D2x[0] + D2y[0] + D2z[0]);

    gpu::upload_stencil_coefficients(
        D2x, D2y, D2z,
        stencil.D1_coeff_x(), stencil.D1_coeff_y(), stencil.D1_coeff_z(),
        stencil.D2_coeff_xy(), stencil.D2_coeff_xz(), stencil.D2_coeff_yz(),
        gs->FDn);

    // ── Metric tensor for non-orthogonal cells ──────────────────
    {
        const auto& L = grid.lattice().lapc_T();
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                gs->lapcT[i * 3 + j] = L(i, j);
    }

    // ── x_ex workspace (halo-extended psi) ──────────────────────
    int nx_ex = gs->nx + 2 * gs->FDn;
    int ny_ex = gs->ny + 2 * gs->FDn;
    int nz_ex = gs->nz + 2 * gs->FDn;
    size_t nd_ex = (size_t)nx_ex * ny_ex * nz_ex;

    if (gs->d_x_ex) cudaFreeAsync(gs->d_x_ex, stream);
    CUDA_CHECK(cudaMallocAsync(&gs->d_x_ex, nd_ex * Nband * sizeof(double), stream));

    bool is_kpt = ctx.is_kpt();
    if (is_kpt) {
        if (gs->d_x_ex_z) cudaFreeAsync(gs->d_x_ex_z, stream);
        CUDA_CHECK(cudaMallocAsync(&gs->d_x_ex_z, nd_ex * Nband * sizeof(cuDoubleComplex), stream));
    }

    // ── Nonlocal projector data upload ──────────────────────────
    if (vnl && vnl->is_setup() && vnl->total_nproj() > 0) {
        auto& gv = gs->gpu_vnl;

        int ntypes = crystal.n_types();
        int n_phys = crystal.n_atom_total();

        // Build IP_displ_global for physical atoms
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
        gv.total_phys_nproj = IP_displ_global[n_phys];

        // Count influence atoms
        gv.n_influence = 0;
        for (int it = 0; it < ntypes; it++)
            gv.n_influence += nloc_influence[it].n_atom;

        // Flatten per-atom data
        std::vector<int> h_ndc_arr, h_nproj_arr, h_IP_displ_arr;
        std::vector<int> h_gpos_offsets(1, 0), h_chi_offsets(1, 0);
        std::vector<int> h_gpos_flat;
        std::vector<double> h_Chi_flat;
        gv.max_ndc = 0;
        gv.max_nproj = 0;

        const auto& Chi = vnl->Chi();

        gs->h_image_shifts.clear();
        gs->h_image_shifts.reserve(gv.n_influence);

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

                // Store image shift for Bloch factor computation
                if (iat < static_cast<int>(inf.image_shift.size()))
                    gs->h_image_shifts.push_back(inf.image_shift[iat]);
                else
                    gs->h_image_shifts.push_back({0.0, 0.0, 0.0});

                for (int ig = 0; ig < ndc; ig++)
                    h_gpos_flat.push_back(inf.grid_pos[iat][ig]);

                if (ndc > 0 && nproj > 0) {
                    const double* chi_data = Chi[it][iat].data();
                    int chi_ld = Chi[it][iat].ld();
                    for (int jp = 0; jp < nproj; jp++)
                        for (int ig = 0; ig < ndc; ig++)
                            h_Chi_flat.push_back(chi_data[ig + jp * chi_ld]);
                }

                gv.max_ndc = std::max(gv.max_ndc, ndc);
                gv.max_nproj = std::max(gv.max_nproj, nproj);
            }
        }

        // Build Gamma indexed by physical atom projectors
        std::vector<double> h_Gamma(gv.total_phys_nproj, 0.0);
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

        // Upload to GPU
        int total_gpos = h_gpos_offsets.back();
        int total_chi  = h_chi_offsets.back();

        CUDA_CHECK(cudaMallocAsync(&gv.d_Chi_flat,    std::max(1, total_chi) * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&gv.d_gpos_flat,   std::max(1, total_gpos) * sizeof(int), stream));
        CUDA_CHECK(cudaMallocAsync(&gv.d_gpos_offsets, (gv.n_influence + 1) * sizeof(int), stream));
        CUDA_CHECK(cudaMallocAsync(&gv.d_chi_offsets,  (gv.n_influence + 1) * sizeof(int), stream));
        CUDA_CHECK(cudaMallocAsync(&gv.d_ndc_arr,      gv.n_influence * sizeof(int), stream));
        CUDA_CHECK(cudaMallocAsync(&gv.d_nproj_arr,    gv.n_influence * sizeof(int), stream));
        CUDA_CHECK(cudaMallocAsync(&gv.d_IP_displ,     gv.n_influence * sizeof(int), stream));
        CUDA_CHECK(cudaMallocAsync(&gv.d_Gamma,        gv.total_phys_nproj * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&gv.d_alpha,        (size_t)gv.total_phys_nproj * Nband * sizeof(double), stream));

        if (total_chi > 0)
            CUDA_CHECK(cudaMemcpyAsync(gv.d_Chi_flat, h_Chi_flat.data(), total_chi * sizeof(double), cudaMemcpyHostToDevice, stream));
        if (total_gpos > 0)
            CUDA_CHECK(cudaMemcpyAsync(gv.d_gpos_flat, h_gpos_flat.data(), total_gpos * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(gv.d_gpos_offsets, h_gpos_offsets.data(), (gv.n_influence + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(gv.d_chi_offsets, h_chi_offsets.data(), (gv.n_influence + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(gv.d_ndc_arr, h_ndc_arr.data(), gv.n_influence * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(gv.d_nproj_arr, h_nproj_arr.data(), gv.n_influence * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(gv.d_IP_displ, h_IP_displ_arr.data(), gv.n_influence * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(gv.d_Gamma, h_Gamma.data(), gv.total_phys_nproj * sizeof(double), cudaMemcpyHostToDevice, stream));

        // Complex alpha for k-point
        if (is_kpt) {
            if (gs->d_alpha_z) cudaFreeAsync(gs->d_alpha_z, stream);
            CUDA_CHECK(cudaMallocAsync(&gs->d_alpha_z,
                (size_t)gv.total_phys_nproj * Nband * sizeof(cuDoubleComplex), stream));
        }

        printf("Hamiltonian::setup_gpu: %d influence atoms, %d phys projectors, "
               "max_ndc=%d, max_nproj=%d\n",
               gv.n_influence, gv.total_phys_nproj, gv.max_ndc, gv.max_nproj);
    }

    // ── mGGA work buffers ───────────────────────────────────────
    bool is_mgga = (vtau_ != nullptr);
    if (is_mgga) {
        size_t Nd_Nb = (size_t)gs->Nd * Nband;

        if (gs->d_mgga_dpsi) cudaFreeAsync(gs->d_mgga_dpsi, stream);
        CUDA_CHECK(cudaMallocAsync(&gs->d_mgga_dpsi, Nd_Nb * sizeof(double), stream));

        if (gs->d_mgga_vtdpsi) cudaFreeAsync(gs->d_mgga_vtdpsi, stream);
        CUDA_CHECK(cudaMallocAsync(&gs->d_mgga_vtdpsi, Nd_Nb * sizeof(double), stream));

        if (gs->d_mgga_div) cudaFreeAsync(gs->d_mgga_div, stream);
        CUDA_CHECK(cudaMallocAsync(&gs->d_mgga_div, Nd_Nb * sizeof(double), stream));

        if (gs->d_mgga_vt_ex) cudaFreeAsync(gs->d_mgga_vt_ex, stream);
        CUDA_CHECK(cudaMallocAsync(&gs->d_mgga_vt_ex, nd_ex * Nband * sizeof(double), stream));

        // Extra gradient buffers for non-orthogonal cells
        if (!gs->is_orth) {
            if (gs->d_mgga_dpsi_y) cudaFreeAsync(gs->d_mgga_dpsi_y, stream);
            CUDA_CHECK(cudaMallocAsync(&gs->d_mgga_dpsi_y, Nd_Nb * sizeof(double), stream));

            if (gs->d_mgga_dpsi_z_r) cudaFreeAsync(gs->d_mgga_dpsi_z_r, stream);
            CUDA_CHECK(cudaMallocAsync(&gs->d_mgga_dpsi_z_r, Nd_Nb * sizeof(double), stream));
        }

        // Complex mGGA buffers for k-point
        if (is_kpt) {
            if (gs->d_mgga_dpsi_z) cudaFreeAsync(gs->d_mgga_dpsi_z, stream);
            CUDA_CHECK(cudaMallocAsync(&gs->d_mgga_dpsi_z, gs->Nd * sizeof(cuDoubleComplex), stream));

            if (gs->d_mgga_vtdpsi_z) cudaFreeAsync(gs->d_mgga_vtdpsi_z, stream);
            CUDA_CHECK(cudaMallocAsync(&gs->d_mgga_vtdpsi_z, gs->Nd * sizeof(cuDoubleComplex), stream));

            if (gs->d_mgga_div_z) cudaFreeAsync(gs->d_mgga_div_z, stream);
            CUDA_CHECK(cudaMallocAsync(&gs->d_mgga_div_z, gs->Nd * sizeof(cuDoubleComplex), stream));

            if (gs->d_mgga_vt_ex_z) cudaFreeAsync(gs->d_mgga_vt_ex_z, stream);
            CUDA_CHECK(cudaMallocAsync(&gs->d_mgga_vt_ex_z, nd_ex * sizeof(cuDoubleComplex), stream));

            if (!gs->is_orth) {
                if (gs->d_mgga_dpsi_yz) cudaFreeAsync(gs->d_mgga_dpsi_yz, stream);
                CUDA_CHECK(cudaMallocAsync(&gs->d_mgga_dpsi_yz, gs->Nd * sizeof(cuDoubleComplex), stream));

                if (gs->d_mgga_dpsi_zz) cudaFreeAsync(gs->d_mgga_dpsi_zz, stream);
                CUDA_CHECK(cudaMallocAsync(&gs->d_mgga_dpsi_zz, gs->Nd * sizeof(cuDoubleComplex), stream));
            }
        }
    }

    // ── K-point info ────────────────────────────────────────────
    if (is_kpt) {
        gs->kpoints = &ctx.kpoints();
    }

    // ── EXX state initially inactive ────────────────────────────
    gs->exx_active = false;
    gs->exx_Nocc = 0;
    gs->exx_frac = 0.0;
}

void Hamiltonian::cleanup_gpu() {
    if (!gpu_state_raw_) return;
    auto* gs = static_cast<GPUHamiltonianState*>(gpu_state_raw_);

    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    auto safe_free = [stream](auto*& p) { if (p) { cudaFreeAsync(p, stream); p = nullptr; } };

    // x_ex workspace
    safe_free(gs->d_x_ex);
    safe_free(gs->d_x_ex_z);

    // Nonlocal data
    {
        auto& gv = gs->gpu_vnl;
        safe_free(gv.d_Chi_flat);
        safe_free(gv.d_gpos_flat);
        safe_free(gv.d_gpos_offsets);
        safe_free(gv.d_chi_offsets);
        safe_free(gv.d_ndc_arr);
        safe_free(gv.d_nproj_arr);
        safe_free(gv.d_IP_displ);
        safe_free(gv.d_Gamma);
        safe_free(gv.d_alpha);
    }
    safe_free(gs->d_alpha_z);

    // mGGA buffers
    safe_free(gs->d_mgga_dpsi);
    safe_free(gs->d_mgga_dpsi_y);
    safe_free(gs->d_mgga_dpsi_z_r);
    safe_free(gs->d_mgga_vtdpsi);
    safe_free(gs->d_mgga_div);
    safe_free(gs->d_mgga_vt_ex);
    safe_free(gs->d_mgga_dpsi_z);
    safe_free(gs->d_mgga_dpsi_yz);
    safe_free(gs->d_mgga_dpsi_zz);
    safe_free(gs->d_mgga_vtdpsi_z);
    safe_free(gs->d_mgga_div_z);
    safe_free(gs->d_mgga_vt_ex_z);

    // EXX
    safe_free(gs->d_Xi);
    safe_free(gs->d_Y_exx);
    safe_free(gs->d_Y_exx_z);
    for (auto*& p : gs->d_Xi_kpt) safe_free(p);
    gs->d_Xi_kpt.clear();

    // Bloch factors
    safe_free(gs->d_bloch_fac);

    // SOC
    {
        auto& sc = gs->gpu_soc;
        safe_free(sc.d_Chi_soc_flat);
        safe_free(sc.d_gpos_offsets_soc);
        safe_free(sc.d_chi_soc_offsets);
        safe_free(sc.d_ndc_arr_soc);
        safe_free(sc.d_nproj_soc_arr);
        safe_free(sc.d_IP_displ_soc);
        safe_free(sc.d_Gamma_soc);
        safe_free(sc.d_proj_l);
        safe_free(sc.d_proj_m);
        safe_free(sc.d_alpha_soc_up);
        safe_free(sc.d_alpha_soc_dn);
    }

    delete gs;
    gpu_state_raw_ = nullptr;
}

Hamiltonian::~Hamiltonian() {
    cleanup_gpu();
}

void Hamiltonian::set_kpoint_gpu(const Vec3& kpt_cart, const Vec3& cell_lengths) {
    auto* gs = static_cast<GPUHamiltonianState*>(gpu_state_raw_);
    if (!gs) return;
    gs->kxLx = kpt_cart.x * cell_lengths.x;
    gs->kyLy = kpt_cart.y * cell_lengths.y;
    gs->kzLz = kpt_cart.z * cell_lengths.z;

    // Compute and upload Bloch factors for nonlocal projector k-point path.
    // bloch_fac[2*iat+0] = cos(theta), bloch_fac[2*iat+1] = sin(theta)
    // where theta = -k . image_shift[iat]
    int n_inf = gs->gpu_vnl.n_influence;
    if (n_inf > 0 && !gs->h_image_shifts.empty()) {
        cudaStream_t stream = gpu::GPUContext::instance().compute_stream;

        // Allocate d_bloch_fac if not yet allocated
        if (!gs->d_bloch_fac) {
            CUDA_CHECK(cudaMallocAsync(&gs->d_bloch_fac, 2 * n_inf * sizeof(double), stream));
        }

        // Compute on CPU (n_inf is small, ~tens of atoms)
        std::vector<double> h_bf(2 * n_inf);
        for (int i = 0; i < n_inf; ++i) {
            const Vec3& R = gs->h_image_shifts[i];
            double theta = -(kpt_cart.x * R.x + kpt_cart.y * R.y + kpt_cart.z * R.z);
            h_bf[2*i]     = std::cos(theta);
            h_bf[2*i + 1] = std::sin(theta);
        }

        CUDA_CHECK(cudaMemcpyAsync(gs->d_bloch_fac, h_bf.data(),
                                   2 * n_inf * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    }
}

// ============================================================
// Device-dispatching apply() — real (Gamma-point)
// ============================================================

void Hamiltonian::apply(const double* psi, const double* Veff, double* y,
                        int ncol, Device dev, double c) const {
    if (dev == Device::CPU) {
        apply(psi, Veff, y, ncol, c);
        return;
    }

    // GPU path — mirrors GPUSCF::hamiltonian_apply_cb
    auto* gs = static_cast<GPUHamiltonianState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;

    // Local part: -0.5*Lap + Veff
    gpu::hamiltonian_apply_local_gpu(
        psi, Veff, y, gs->d_x_ex,
        gs->nx, gs->ny, gs->nz, gs->FDn, ncol, c,
        gs->is_orth, true, true, true,
        gs->diag_coeff_ham,
        gs->has_mixed_deriv, gs->has_mixed_deriv, gs->has_mixed_deriv,
        stream);

    // Nonlocal part: Vnl*psi
    if (gs->gpu_vnl.total_phys_nproj > 0) {
        gpu::nonlocal_projector_apply_gpu(
            psi, y,
            gs->gpu_vnl.d_Chi_flat, gs->gpu_vnl.d_gpos_flat,
            gs->gpu_vnl.d_gpos_offsets, gs->gpu_vnl.d_chi_offsets,
            gs->gpu_vnl.d_ndc_arr, gs->gpu_vnl.d_nproj_arr,
            gs->gpu_vnl.d_IP_displ, gs->gpu_vnl.d_Gamma,
            gs->gpu_vnl.d_alpha,
            gs->Nd, ncol, gs->dV,
            gs->gpu_vnl.n_influence, gs->gpu_vnl.total_phys_nproj,
            gs->gpu_vnl.max_ndc, gs->gpu_vnl.max_nproj, stream);
    }

    // mGGA Hamiltonian term
    if (gs->d_vtau_active) {
        auto& ctx = gpu::GPUContext::instance();
        int Nd = gs->Nd;
        int nx_ex = gs->nx + 2 * gs->FDn;
        int ny_ex = gs->ny + 2 * gs->FDn;
        int bs = 256;
        int total = Nd * ncol;
        int gs_total = gpu::ceildiv(total, bs);

        double* d_dpsi   = gs->d_mgga_dpsi;
        double* d_vtdpsi = gs->d_mgga_vtdpsi;
        double* d_div    = gs->d_mgga_div;
        double* d_vt_ex  = gs->d_mgga_vt_ex;

        CUDA_CHECK(cudaMemsetAsync(d_div, 0, (size_t)total * sizeof(double), stream));

        if (gs->is_orth) {
            gpu::halo_exchange_batched_nomemset_gpu(psi, gs->d_x_ex,
                gs->nx, gs->ny, gs->nz, gs->FDn, ncol, true, true, true, stream);
            for (int dir = 0; dir < 3; dir++) {
                gpu::gradient_v3_gpu(gs->d_x_ex, d_dpsi,
                    gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex, dir, ncol, stream);
                vtau_multiply_batched_kernel<<<gs_total, bs, 0, stream>>>(
                    d_dpsi, gs->d_vtau_active, d_vtdpsi, Nd, total);
                gpu::halo_exchange_batched_nomemset_gpu(d_vtdpsi, d_vt_ex,
                    gs->nx, gs->ny, gs->nz, gs->FDn, ncol, true, true, true, stream);
                gpu::gradient_v3_gpu(d_vt_ex, d_dpsi,
                    gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex, dir, ncol, stream);
                double one = 1.0;
                cublasDaxpy(ctx.cublas, total, &one, d_dpsi, 1, d_div, 1);
            }
        } else {
            double* d_dpsi_x = d_dpsi;
            double* d_dpsi_y = gs->d_mgga_dpsi_y;
            double* d_dpsi_zr = gs->d_mgga_dpsi_z_r;
            int gs_nd = gpu::ceildiv(Nd, bs);
            gpu::halo_exchange_batched_nomemset_gpu(psi, gs->d_x_ex,
                gs->nx, gs->ny, gs->nz, gs->FDn, ncol, true, true, true, stream);
            gpu::gradient_v3_gpu(gs->d_x_ex, d_dpsi_x,
                gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex, 0, ncol, stream);
            gpu::gradient_v3_gpu(gs->d_x_ex, d_dpsi_y,
                gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex, 1, ncol, stream);
            gpu::gradient_v3_gpu(gs->d_x_ex, d_dpsi_zr,
                gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex, 2, ncol, stream);
            for (int dir = 0; dir < 3; dir++) {
                for (int n = 0; n < ncol; n++) {
                    vtau_lapcT_multiply_kernel<<<gs_nd, bs, 0, stream>>>(
                        d_dpsi_x + (size_t)n * Nd, d_dpsi_y + (size_t)n * Nd,
                        d_dpsi_zr + (size_t)n * Nd,
                        gs->d_vtau_active, d_vtdpsi + (size_t)n * Nd, Nd,
                        gs->lapcT[dir*3+0], gs->lapcT[dir*3+1], gs->lapcT[dir*3+2]);
                }
                gpu::halo_exchange_batched_nomemset_gpu(d_vtdpsi, d_vt_ex,
                    gs->nx, gs->ny, gs->nz, gs->FDn, ncol, true, true, true, stream);
                gpu::gradient_v3_gpu(d_vt_ex, d_vtdpsi,
                    gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex, dir, ncol, stream);
                double one = 1.0;
                cublasDaxpy(ctx.cublas, total, &one, d_vtdpsi, 1, d_div, 1);
            }
        }

        mgga_ham_sub_kernel<<<gs_total, bs, 0, stream>>>(y, d_div, total);
    }

    // Exact exchange
    if (gs->exx_active && gs->d_Xi && gs->exx_Nocc > 0) {
        auto& ctx = gpu::GPUContext::instance();
        gpu::apply_Vx_gpu(ctx.cublas,
                          gs->d_Xi, gs->Nd, gs->exx_Nocc,
                          psi, gs->Nd, ncol,
                          y, gs->Nd,
                          gs->d_Y_exx,
                          gs->exx_frac);
    }
}

// ============================================================
// Device-dispatching apply_kpt() — complex (k-point)
// ============================================================

void Hamiltonian::apply_kpt(const Complex* psi, const double* Veff, Complex* y,
                            int ncol, const Vec3& kpt_cart, const Vec3& cell_lengths,
                            Device dev, double c) const {
    if (dev == Device::CPU) {
        apply_kpt(psi, Veff, y, ncol, kpt_cart, cell_lengths, c);
        return;
    }

    // GPU path — mirrors GPUSCF::hamiltonian_apply_z_cb
    auto* gs = static_cast<GPUHamiltonianState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;

    auto* d_psi = reinterpret_cast<const cuDoubleComplex*>(psi);
    auto* d_Hpsi = reinterpret_cast<cuDoubleComplex*>(y);
    auto* d_x_ex = gs->d_x_ex_z;

    // Local part with Bloch phases
    gpu::hamiltonian_apply_local_z_gpu(
        d_psi, Veff, d_Hpsi, d_x_ex,
        gs->nx, gs->ny, gs->nz, gs->FDn, ncol, c,
        gs->is_orth, true, true, true,
        gs->diag_coeff_ham,
        gs->has_mixed_deriv, gs->has_mixed_deriv, gs->has_mixed_deriv,
        gs->kxLx, gs->kyLy, gs->kzLz, stream);

    // Nonlocal part with Bloch phases
    if (gs->gpu_vnl.total_phys_nproj > 0 && gs->d_bloch_fac) {
        gpu::nonlocal_projector_apply_z_gpu(
            d_psi, d_Hpsi,
            gs->gpu_vnl.d_Chi_flat, gs->gpu_vnl.d_gpos_flat,
            gs->gpu_vnl.d_gpos_offsets, gs->gpu_vnl.d_chi_offsets,
            gs->gpu_vnl.d_ndc_arr, gs->gpu_vnl.d_nproj_arr,
            gs->gpu_vnl.d_IP_displ, gs->gpu_vnl.d_Gamma,
            gs->d_alpha_z,
            gs->d_bloch_fac,
            gs->Nd, ncol, gs->dV,
            gs->gpu_vnl.n_influence, gs->gpu_vnl.total_phys_nproj,
            gs->gpu_vnl.max_ndc, gs->gpu_vnl.max_nproj, stream);
    }

    // mGGA Hamiltonian term (complex)
    if (gs->d_vtau_active) {
        auto& ctx = gpu::GPUContext::instance();
        int Nd = gs->Nd;
        int nx_ex = gs->nx + 2 * gs->FDn;
        int ny_ex = gs->ny + 2 * gs->FDn;
        int bs = 256;
        int gs_nd = gpu::ceildiv(Nd, bs);

        cuDoubleComplex* d_dpsi_z   = gs->d_mgga_dpsi_z;
        cuDoubleComplex* d_vtdpsi_z = gs->d_mgga_vtdpsi_z;
        cuDoubleComplex* d_div_z    = gs->d_mgga_div_z;
        cuDoubleComplex* d_vt_ex_z  = gs->d_mgga_vt_ex_z;

        for (int n = 0; n < ncol; n++) {
            const cuDoubleComplex* d_psi_n = d_psi + (size_t)n * Nd;
            cuDoubleComplex* d_Hpsi_n = d_Hpsi + (size_t)n * Nd;

            CUDA_CHECK(cudaMemsetAsync(d_div_z, 0, Nd * sizeof(cuDoubleComplex), stream));

            if (gs->is_orth) {
                for (int dir = 0; dir < 3; dir++) {
                    gpu::halo_exchange_z_gpu(d_psi_n, d_x_ex,
                        gs->nx, gs->ny, gs->nz, gs->FDn, 1,
                        true, true, true, gs->kxLx, gs->kyLy, gs->kzLz, stream);
                    gpu::gradient_z_gpu(d_x_ex, d_dpsi_z,
                        gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex, dir, 1, stream);
                    vtau_multiply_z_kernel<<<gs_nd, bs, 0, stream>>>(
                        d_dpsi_z, gs->d_vtau_active, d_vtdpsi_z, Nd);
                    gpu::halo_exchange_z_gpu(d_vtdpsi_z, d_vt_ex_z,
                        gs->nx, gs->ny, gs->nz, gs->FDn, 1,
                        true, true, true, gs->kxLx, gs->kyLy, gs->kzLz, stream);
                    gpu::gradient_z_gpu(d_vt_ex_z, d_dpsi_z,
                        gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex, dir, 1, stream);
                    cuDoubleComplex one = {1.0, 0.0};
                    cublasZaxpy(ctx.cublas, Nd, &one, d_dpsi_z, 1, d_div_z, 1);
                }
            } else {
                cuDoubleComplex* d_dpsi_xz = gs->d_mgga_dpsi_z;
                cuDoubleComplex* d_dpsi_yz = gs->d_mgga_dpsi_yz;
                cuDoubleComplex* d_dpsi_zz = gs->d_mgga_dpsi_zz;
                gpu::halo_exchange_z_gpu(d_psi_n, d_x_ex,
                    gs->nx, gs->ny, gs->nz, gs->FDn, 1,
                    true, true, true, gs->kxLx, gs->kyLy, gs->kzLz, stream);
                gpu::gradient_z_gpu(d_x_ex, d_dpsi_xz,
                    gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex, 0, 1, stream);
                gpu::gradient_z_gpu(d_x_ex, d_dpsi_yz,
                    gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex, 1, 1, stream);
                gpu::gradient_z_gpu(d_x_ex, d_dpsi_zz,
                    gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex, 2, 1, stream);
                for (int dir = 0; dir < 3; dir++) {
                    vtau_lapcT_multiply_z_kernel<<<gs_nd, bs, 0, stream>>>(
                        d_dpsi_xz, d_dpsi_yz, d_dpsi_zz,
                        gs->d_vtau_active, d_vtdpsi_z, Nd,
                        gs->lapcT[dir*3+0], gs->lapcT[dir*3+1], gs->lapcT[dir*3+2]);
                    gpu::halo_exchange_z_gpu(d_vtdpsi_z, d_vt_ex_z,
                        gs->nx, gs->ny, gs->nz, gs->FDn, 1,
                        true, true, true, gs->kxLx, gs->kyLy, gs->kzLz, stream);
                    gpu::gradient_z_gpu(d_vt_ex_z, d_vtdpsi_z,
                        gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex, dir, 1, stream);
                    cuDoubleComplex one = {1.0, 0.0};
                    cublasZaxpy(ctx.cublas, Nd, &one, d_vtdpsi_z, 1, d_div_z, 1);
                }
            }

            mgga_ham_sub_z_kernel<<<gs_nd, bs, 0, stream>>>(d_Hpsi_n, d_div_z, Nd);
        }
    }

    // Exact exchange (k-point)
    if (gs->exx_active && gs->exx_Nocc > 0 && !gs->d_Xi_kpt.empty()) {
        int kpt_loc = gs->exx_kpt;
        int spin = gs->exx_spin;
        int Nkpts = gs->kpoints ? gs->kpoints->Nkpts() : 1;
        int idx = spin * Nkpts + kpt_loc;
        if (idx >= 0 && idx < (int)gs->d_Xi_kpt.size() && gs->d_Xi_kpt[idx]) {
            auto& ctx = gpu::GPUContext::instance();
            gpu::apply_Vx_kpt_gpu(ctx.cublas,
                                   gs->d_Xi_kpt[idx], gs->Nd, gs->exx_Nocc,
                                   d_psi, gs->Nd, ncol,
                                   d_Hpsi, gs->Nd,
                                   gs->d_Y_exx_z,
                                   gs->exx_frac);
        }
    }
}

// ============================================================
// Device-dispatching apply_spinor_kpt() — SOC spinor
// ============================================================

void Hamiltonian::apply_spinor_kpt(const Complex* psi, const double* Veff_spinor, Complex* y,
                                    int ncol, int Nd_d, const Vec3& kpt_cart, const Vec3& cell_lengths,
                                    Device dev, double c) const {
    if (dev == Device::CPU) {
        apply_spinor_kpt(psi, Veff_spinor, y, ncol, Nd_d, kpt_cart, cell_lengths, c);
        return;
    }

    // GPU path — mirrors GPUSCF::hamiltonian_apply_spinor_z_cb
    auto* gs = static_cast<GPUHamiltonianState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    auto& ctx = gpu::GPUContext::instance();

    auto* d_psi = reinterpret_cast<const cuDoubleComplex*>(psi);
    auto* d_Hpsi = reinterpret_cast<cuDoubleComplex*>(y);
    auto* d_x_ex = gs->d_x_ex_z;
    int Nd_d_spinor = 2 * Nd_d;

    const double* V_uu = Veff_spinor;
    const double* V_dd = Veff_spinor + Nd_d;
    const double* V_ud_re = Veff_spinor + 2 * Nd_d;
    const double* V_ud_im = Veff_spinor + 3 * Nd_d;

    // Process all bands at once (matching GPUSCF logic)
    int batch_size = ncol;
    {
        int nx_ex = gs->nx + 2*gs->FDn, ny_ex = gs->ny + 2*gs->FDn, nz_ex = gs->nz + 2*gs->FDn;
        size_t Nd_ex = (size_t)nx_ex * ny_ex * nz_ex;
        size_t per_band = (4 * Nd_d + Nd_ex) * sizeof(cuDoubleComplex);
        size_t max_scratch = 200ULL * 1024 * 1024;
        if (per_band * ncol > max_scratch)
            batch_size = std::max(1, (int)(max_scratch / per_band));
    }

    for (int col_start = 0; col_start < ncol; col_start += batch_size) {
        int cols = std::min(batch_size, ncol - col_start);
        int total = Nd_d * cols;
        int bs = 256;
        int gs_k = gpu::ceildiv(total, bs);

        size_t sp_cp = ctx.scratch_pool.checkpoint();
        cuDoubleComplex* d_psi_up = ctx.scratch_pool.alloc<cuDoubleComplex>((size_t)Nd_d * cols);
        cuDoubleComplex* d_psi_dn = ctx.scratch_pool.alloc<cuDoubleComplex>((size_t)Nd_d * cols);
        cuDoubleComplex* d_Hp_up = ctx.scratch_pool.alloc<cuDoubleComplex>((size_t)Nd_d * cols);
        cuDoubleComplex* d_Hp_dn = ctx.scratch_pool.alloc<cuDoubleComplex>((size_t)Nd_d * cols);
        int nx_ex = gs->nx + 2*gs->FDn, ny_ex = gs->ny + 2*gs->FDn, nz_ex = gs->nz + 2*gs->FDn;
        size_t Nd_ex = (size_t)nx_ex * ny_ex * nz_ex;
        cuDoubleComplex* d_xex_batch = ctx.scratch_pool.alloc<cuDoubleComplex>(Nd_ex * cols);

        const cuDoubleComplex* d_psi_batch = d_psi + col_start * Nd_d_spinor;
        cuDoubleComplex* d_Hpsi_batch = d_Hpsi + col_start * Nd_d_spinor;
        spinor_extract_kernel<<<gs_k, bs, 0, stream>>>(d_psi_batch, d_psi_up, Nd_d, cols, 0);
        spinor_extract_kernel<<<gs_k, bs, 0, stream>>>(d_psi_batch, d_psi_dn, Nd_d, cols, 1);

        // Local H for each spin component
        gpu::hamiltonian_apply_local_z_gpu(
            d_psi_up, V_uu, d_Hp_up, d_xex_batch,
            gs->nx, gs->ny, gs->nz, gs->FDn, cols, c,
            gs->is_orth, true, true, true,
            gs->diag_coeff_ham,
            gs->has_mixed_deriv, gs->has_mixed_deriv, gs->has_mixed_deriv,
            gs->kxLx, gs->kyLy, gs->kzLz, stream);

        gpu::hamiltonian_apply_local_z_gpu(
            d_psi_dn, V_dd, d_Hp_dn, d_xex_batch,
            gs->nx, gs->ny, gs->nz, gs->FDn, cols, c,
            gs->is_orth, true, true, true,
            gs->diag_coeff_ham,
            gs->has_mixed_deriv, gs->has_mixed_deriv, gs->has_mixed_deriv,
            gs->kxLx, gs->kyLy, gs->kzLz, stream);

        // Nonlocal per spin component
        if (gs->gpu_vnl.total_phys_nproj > 0 && gs->d_bloch_fac) {
            gpu::nonlocal_projector_apply_z_gpu(
                d_psi_up, d_Hp_up,
                gs->gpu_vnl.d_Chi_flat, gs->gpu_vnl.d_gpos_flat,
                gs->gpu_vnl.d_gpos_offsets, gs->gpu_vnl.d_chi_offsets,
                gs->gpu_vnl.d_ndc_arr, gs->gpu_vnl.d_nproj_arr,
                gs->gpu_vnl.d_IP_displ, gs->gpu_vnl.d_Gamma,
                gs->d_alpha_z,
                gs->d_bloch_fac,
                Nd_d, cols, gs->dV,
                gs->gpu_vnl.n_influence, gs->gpu_vnl.total_phys_nproj,
                gs->gpu_vnl.max_ndc, gs->gpu_vnl.max_nproj, stream);

            gpu::nonlocal_projector_apply_z_gpu(
                d_psi_dn, d_Hp_dn,
                gs->gpu_vnl.d_Chi_flat, gs->gpu_vnl.d_gpos_flat,
                gs->gpu_vnl.d_gpos_offsets, gs->gpu_vnl.d_chi_offsets,
                gs->gpu_vnl.d_ndc_arr, gs->gpu_vnl.d_nproj_arr,
                gs->gpu_vnl.d_IP_displ, gs->gpu_vnl.d_Gamma,
                gs->d_alpha_z,
                gs->d_bloch_fac,
                Nd_d, cols, gs->dV,
                gs->gpu_vnl.n_influence, gs->gpu_vnl.total_phys_nproj,
                gs->gpu_vnl.max_ndc, gs->gpu_vnl.max_nproj, stream);
        }

        // Scatter back to spinor layout
        spinor_scatter_kernel<<<gs_k, bs, 0, stream>>>(d_Hp_up, d_Hpsi_batch, Nd_d, cols, 0);
        spinor_scatter_kernel<<<gs_k, bs, 0, stream>>>(d_Hp_dn, d_Hpsi_batch, Nd_d, cols, 1);

        ctx.scratch_pool.restore(sp_cp);
    }

    // Off-diagonal Veff
    gpu::spinor_offdiag_veff_gpu(d_Hpsi, d_psi, V_ud_re, V_ud_im, Nd_d, ncol, stream);

    // SOC terms
    if (gs->has_soc && gs->gpu_soc.total_soc_nproj > 0) {
        gpu::soc_apply_z_gpu(
            d_psi, d_Hpsi,
            gs->gpu_soc.d_Chi_soc_flat, gs->gpu_vnl.d_gpos_flat,
            gs->gpu_soc.d_gpos_offsets_soc, gs->gpu_soc.d_chi_soc_offsets,
            gs->gpu_soc.d_ndc_arr_soc, gs->gpu_soc.d_nproj_soc_arr,
            gs->gpu_soc.d_IP_displ_soc,
            gs->gpu_soc.d_Gamma_soc,
            gs->gpu_soc.d_proj_l, gs->gpu_soc.d_proj_m,
            gs->d_bloch_fac,
            gs->gpu_soc.d_alpha_soc_up,
            gs->gpu_soc.d_alpha_soc_dn,
            Nd_d, ncol, gs->dV,
            gs->gpu_soc.n_influence_soc, gs->gpu_soc.total_soc_nproj,
            gs->gpu_soc.max_ndc_soc, gs->gpu_soc.max_nproj_soc, stream);
    }
}

} // namespace lynx
#endif // USE_CUDA
