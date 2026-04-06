#ifdef USE_CUDA

#include "electronic/KineticEnergyDensity.hpp"
#include "core/LynxContext.hpp"
#include "core/DeviceTag.hpp"
#include "electronic/Wavefunction.hpp"
#include <mpi.h>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "core/gpu_common.cuh"
#include "core/GPUContext.cuh"
#include "parallel/HaloExchange.cuh"
#include "operators/Gradient.cuh"

namespace lynx {

// ============================================================
// File-static kernels
// ============================================================

namespace {

// tau[i] += weight * (dpsi_x[i]^2 + dpsi_y[i]^2 + dpsi_z[i]^2)
__global__ void tau_accumulate_kernel(
    const double* __restrict__ dpsi_x,
    const double* __restrict__ dpsi_y,
    const double* __restrict__ dpsi_z,
    double* __restrict__ tau,
    double weight, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    tau[i] += weight * (dpsi_x[i]*dpsi_x[i] + dpsi_y[i]*dpsi_y[i] + dpsi_z[i]*dpsi_z[i]);
}

// Non-orthogonal: applies lapcT metric tensor
__global__ void tau_accumulate_nonorth_kernel(
    const double* __restrict__ dpsi_x,
    const double* __restrict__ dpsi_y,
    const double* __restrict__ dpsi_z,
    double* __restrict__ tau,
    double weight, int N,
    double L00, double L11, double L22, double L01, double L02, double L12)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double dx = dpsi_x[i], dy = dpsi_y[i], dz = dpsi_z[i];
    double val = L00*dx*dx + L11*dy*dy + L22*dz*dz
               + 2.0*L01*dx*dy + 2.0*L02*dx*dz + 2.0*L12*dy*dz;
    tau[i] += weight * val;
}

} // anonymous namespace

// ============================================================
// GPUTauState
// ============================================================
struct GPUTauState {
    int Nd = 0;
    int Nspin = 1;
    int nx = 0, ny = 0, nz = 0, FDn = 0;
    int Nd_ex = 0;
    bool is_orth = true;
    double lapcT[9] = {};

    double* d_tau  = nullptr;
    double* d_vtau = nullptr;
};

// ============================================================
// Setup / Cleanup / Accessors
// ============================================================

void KineticEnergyDensity::setup_gpu(const LynxContext& ctx, int Nspin) {
    if (!gpu_state_)
        gpu_state_.reset(new GPUTauState());
    auto* gs = gpu_state_.as<GPUTauState>();

    const auto& grid = ctx.grid();
    const auto& domain = ctx.domain();
    const auto& stencil = ctx.stencil();

    gs->Nd    = domain.Nd_d();
    gs->Nspin = Nspin;
    gs->nx = grid.Nx();
    gs->ny = grid.Ny();
    gs->nz = grid.Nz();
    gs->FDn = stencil.FDn();
    gs->Nd_ex = (gs->nx + 2*gs->FDn) * (gs->ny + 2*gs->FDn) * (gs->nz + 2*gs->FDn);
    gs->is_orth = grid.lattice().is_orthogonal();

    const auto& L = grid.lattice().lapc_T();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            gs->lapcT[i*3+j] = L(i,j);

    int tau_size = (Nspin >= 2) ? 3 * gs->Nd : gs->Nd;

    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    CUDA_CHECK(cudaMallocAsync(&gs->d_tau,  tau_size * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&gs->d_vtau, tau_size * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(gs->d_tau,  0, tau_size * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(gs->d_vtau, 0, tau_size * sizeof(double), stream));
}

void KineticEnergyDensity::cleanup_gpu() {
    if (!gpu_state_) return;
    auto* gs = gpu_state_.as<GPUTauState>();

    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    auto safe_free = [stream](auto*& p) { if (p) { cudaFreeAsync(p, stream); p = nullptr; } };

    safe_free(gs->d_tau);
    safe_free(gs->d_vtau);

    gpu_state_.reset();
}

KineticEnergyDensity::~KineticEnergyDensity() {
    cleanup_gpu();
}

double* KineticEnergyDensity::d_tau() {
    auto* gs = gpu_state_.as<GPUTauState>();
    return gs ? gs->d_tau : nullptr;
}

double* KineticEnergyDensity::d_vtau() {
    auto* gs = gpu_state_.as<GPUTauState>();
    return gs ? gs->d_vtau : nullptr;
}

// ============================================================
// GPU kernel wrapper: halo + gradient + tau accumulation for one band
// ============================================================
void KineticEnergyDensity::gradient_accumulate_tau_gpu(
    const double* d_psi_col, double* d_tau_s, double weight, int Nd)
{
    auto* gs = gpu_state_.as<GPUTauState>();
    auto& gctx = gpu::GPUContext::instance();
    cudaStream_t stream = gctx.compute_stream;
    int bs = 256;
    int grid_sz = gpu::ceildiv(Nd, bs);
    int nx_ex = gs->nx + 2*gs->FDn;
    int ny_ex = gs->ny + 2*gs->FDn;

    auto& sp = gctx.scratch_pool;
    size_t sp_cp = sp.checkpoint();

    double* d_psi_ex = sp.alloc<double>(gs->Nd_ex);
    double* d_dpsi_x = sp.alloc<double>(Nd);
    double* d_dpsi_y = sp.alloc<double>(Nd);
    double* d_dpsi_z = sp.alloc<double>(Nd);

    // Halo exchange
    gpu::halo_exchange_gpu(d_psi_col, d_psi_ex, gs->nx, gs->ny, gs->nz,
                           gs->FDn, 1, true, true, true, stream);

    // Gradient in x, y, z
    gpu::gradient_gpu(d_psi_ex, d_dpsi_x, gs->nx, gs->ny, gs->nz,
                      gs->FDn, nx_ex, ny_ex, 0, 1, stream);
    gpu::gradient_gpu(d_psi_ex, d_dpsi_y, gs->nx, gs->ny, gs->nz,
                      gs->FDn, nx_ex, ny_ex, 1, 1, stream);
    gpu::gradient_gpu(d_psi_ex, d_dpsi_z, gs->nx, gs->ny, gs->nz,
                      gs->FDn, nx_ex, ny_ex, 2, 1, stream);

    // Accumulate tau
    if (gs->is_orth) {
        tau_accumulate_kernel<<<grid_sz, bs, 0, stream>>>(
            d_dpsi_x, d_dpsi_y, d_dpsi_z, d_tau_s, weight, Nd);
    } else {
        tau_accumulate_nonorth_kernel<<<grid_sz, bs, 0, stream>>>(
            d_dpsi_x, d_dpsi_y, d_dpsi_z, d_tau_s, weight, Nd,
            gs->lapcT[0], gs->lapcT[4], gs->lapcT[8],
            gs->lapcT[1], gs->lapcT[2], gs->lapcT[5]);
    }

    sp.restore(sp_cp);
}

// ============================================================
// GPU compute — full tau computation on device
// ============================================================
void KineticEnergyDensity::compute_gpu(const LynxContext& ctx,
                                        const Wavefunction& wfn,
                                        const std::vector<double>& kpt_weights)
{
    if (!gpu_state_) {
        // No GPU state — fall back to CPU
        dev_ = Device::CPU;
        compute(ctx, wfn, kpt_weights);
        dev_ = Device::GPU;
        return;
    }

    int Nd = ctx.domain().Nd_d();
    int Nspin_global = ctx.Nspin();
    int Nspin_local = ctx.Nspin_local();
    int spin_start = ctx.spin_start();
    int kpt_start = ctx.kpt_start();
    int band_start = ctx.band_start();
    int Nband = wfn.Nband();
    int Nkpts = wfn.Nkpts();
    bool is_kpt = !ctx.kpoints().is_gamma_only();

    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;

    // Zero tau on device
    int tau_size = (Nspin_global >= 2) ? 3 * Nd : Nd;
    allocate(Nd, Nspin_global);

    double* d_tau_base = d_tau();
    CUDA_CHECK(cudaMemsetAsync(d_tau_base, 0, tau_size * sizeof(double), stream));

    for (int s = 0; s < Nspin_local; ++s) {
        int s_glob = spin_start + s;
        double* d_tau_s = d_tau_base + s_glob * Nd;

        for (int k = 0; k < Nkpts; ++k) {
            double wk = kpt_weights[kpt_start + k];
            const auto& occ = wfn.occupations(s, k);

            if (is_kpt) {
                // Complex k-point: fall back to CPU
                dev_ = Device::CPU;
                compute(ctx, wfn, kpt_weights);
                dev_ = Device::GPU;
                return;
            }

            const auto& psi = wfn.psi(s, k);

            for (int n = 0; n < Nband; ++n) {
                double fn = occ(band_start + n);
                if (fn < 1e-16) continue;
                double g_nk = wk * fn;

                auto& sp = gpu::GPUContext::instance().scratch_pool;
                size_t sp_cp = sp.checkpoint();
                double* d_psi_col = sp.alloc<double>(Nd);
                const double* col = psi.col(n);
                CUDA_CHECK(cudaMemcpyAsync(d_psi_col, col, Nd * sizeof(double),
                                           cudaMemcpyHostToDevice, stream));

                gradient_accumulate_tau_gpu(d_psi_col, d_tau_s, g_nk, Nd);

                sp.restore(sp_cp);
            }
        }
    }

    // Download tau to host
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(tau_.data(), d_tau_base, tau_size * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // MPI reductions on host
    const auto& bandcomm = ctx.scf_bandcomm();
    const auto& kptcomm = ctx.kpt_bridge();
    const auto& spincomm = ctx.spin_bridge();

    if (!bandcomm.is_null() && bandcomm.size() > 1) {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start + s;
            bandcomm.allreduce_sum(tau_.data() + s_glob * Nd, Nd);
        }
    }

    if (!kptcomm.is_null() && kptcomm.size() > 1) {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start + s;
            kptcomm.allreduce_sum(tau_.data() + s_glob * Nd, Nd);
        }
    }

    if (!spincomm.is_null() && spincomm.size() > 1 && Nspin_global == 2) {
        int my_spin = spin_start;
        int other_spin = 1 - my_spin;
        int partner = (spincomm.rank() == 0) ? 1 : 0;
        MPI_Sendrecv(tau_.data() + my_spin * Nd, Nd, MPI_DOUBLE, partner, 0,
                     tau_.data() + other_spin * Nd, Nd, MPI_DOUBLE, partner, 0,
                     spincomm.comm(), MPI_STATUS_IGNORE);
    }

    // Spin finalize
    if (Nspin_global == 2) {
        double* tau_up = tau_.data();
        double* tau_dn = tau_.data() + Nd;
        double* tau_tot = tau_.data() + 2 * Nd;
        for (int i = 0; i < Nd; ++i) {
            tau_up[i] *= 0.5;
            tau_dn[i] *= 0.5;
            tau_tot[i] = tau_up[i] + tau_dn[i];
        }
    }

    // Re-upload final tau to device
    CUDA_CHECK(cudaMemcpyAsync(d_tau_base, tau_.data(), tau_size * sizeof(double),
                               cudaMemcpyHostToDevice, stream));

    valid_ = true;
}

// ============================================================
// GPU compute from per-(spin,kpt) device-resident psi — zero psi H2D transfers
// ============================================================
void KineticEnergyDensity::compute_gpu_from_device(
    const LynxContext& ctx,
    const Wavefunction& wfn,
    const std::vector<double>& kpt_weights,
    const std::vector<const double*>& d_psi_real_ptrs,
    const std::vector<const void*>& d_psi_z_ptrs)
{
    if (!gpu_state_) {
        dev_ = Device::CPU;
        compute(ctx, wfn, kpt_weights);
        dev_ = Device::GPU;
        return;
    }

    int Nd = ctx.domain().Nd_d();
    int Nspin_global = ctx.Nspin();
    int Nspin_local = ctx.Nspin_local();
    int spin_start = ctx.spin_start();
    int kpt_start = ctx.kpt_start();
    int band_start = ctx.band_start();
    int Nband = wfn.Nband();
    int Nkpts = wfn.Nkpts();
    bool is_kpt = !ctx.kpoints().is_gamma_only();

    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;

    int tau_size = (Nspin_global >= 2) ? 3 * Nd : Nd;
    allocate(Nd, Nspin_global);

    double* d_tau_base = d_tau();
    CUDA_CHECK(cudaMemsetAsync(d_tau_base, 0, tau_size * sizeof(double), stream));

    for (int s = 0; s < Nspin_local; ++s) {
        int s_glob = spin_start + s;
        double* d_tau_s = d_tau_base + s_glob * Nd;

        for (int k = 0; k < Nkpts; ++k) {
            double wk = kpt_weights[kpt_start + k];
            const auto& occ = wfn.occupations(s, k);

            if (is_kpt) {
                // Complex k-point tau: fall back to CPU (no complex gradient_accumulate_tau_gpu yet)
                dev_ = Device::CPU;
                compute(ctx, wfn, kpt_weights);
                dev_ = Device::GPU;
                return;
            }

            int idx = s * Nkpts + k;
            const double* d_psi = d_psi_real_ptrs[idx];

            for (int n = 0; n < Nband; ++n) {
                double fn = occ(band_start + n);
                if (fn < 1e-16) continue;
                double g_nk = wk * fn;

                // psi column is already on device — just offset the pointer
                const double* d_psi_col = d_psi + (size_t)n * Nd;
                gradient_accumulate_tau_gpu(d_psi_col, d_tau_s, g_nk, Nd);
            }
        }
    }

    // Download tau to host for MPI reductions
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(tau_.data(), d_tau_base, tau_size * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // MPI reductions on host
    const auto& bandcomm = ctx.scf_bandcomm();
    const auto& kptcomm = ctx.kpt_bridge();
    const auto& spincomm = ctx.spin_bridge();

    if (!bandcomm.is_null() && bandcomm.size() > 1) {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start + s;
            bandcomm.allreduce_sum(tau_.data() + s_glob * Nd, Nd);
        }
    }

    if (!kptcomm.is_null() && kptcomm.size() > 1) {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start + s;
            kptcomm.allreduce_sum(tau_.data() + s_glob * Nd, Nd);
        }
    }

    if (!spincomm.is_null() && spincomm.size() > 1 && Nspin_global == 2) {
        int my_spin = spin_start;
        int other_spin = 1 - my_spin;
        int partner = (spincomm.rank() == 0) ? 1 : 0;
        MPI_Sendrecv(tau_.data() + my_spin * Nd, Nd, MPI_DOUBLE, partner, 0,
                     tau_.data() + other_spin * Nd, Nd, MPI_DOUBLE, partner, 0,
                     spincomm.comm(), MPI_STATUS_IGNORE);
    }

    if (Nspin_global == 2) {
        double* tau_up = tau_.data();
        double* tau_dn = tau_.data() + Nd;
        double* tau_tot = tau_.data() + 2 * Nd;
        for (int i = 0; i < Nd; ++i) {
            tau_up[i] *= 0.5;
            tau_dn[i] *= 0.5;
            tau_tot[i] = tau_up[i] + tau_dn[i];
        }
    }

    // Re-upload final tau to device
    CUDA_CHECK(cudaMemcpyAsync(d_tau_base, tau_.data(), tau_size * sizeof(double),
                               cudaMemcpyHostToDevice, stream));

    valid_ = true;
}

} // namespace lynx

#endif // USE_CUDA
