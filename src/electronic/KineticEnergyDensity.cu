#ifdef USE_CUDA

#include "electronic/KineticEnergyDensity.hpp"
#include "core/LynxContext.hpp"
#include "core/DeviceTag.hpp"

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

// tau[i] += weight * (dpsi_x[i]^2 + dpsi_y[i]^2 + dpsi_z[i]^2)
static __global__ void tau_accumulate_kernel(
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
static __global__ void tau_accumulate_nonorth_kernel(
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

// Complex k-point tau accumulation
static __global__ void tau_accumulate_z_kernel(
    const cuDoubleComplex* __restrict__ dpsi_x,
    const cuDoubleComplex* __restrict__ dpsi_y,
    const cuDoubleComplex* __restrict__ dpsi_z,
    double* __restrict__ tau,
    double weight, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double dx2 = dpsi_x[i].x*dpsi_x[i].x + dpsi_x[i].y*dpsi_x[i].y;
    double dy2 = dpsi_y[i].x*dpsi_y[i].x + dpsi_y[i].y*dpsi_y[i].y;
    double dz2 = dpsi_z[i].x*dpsi_z[i].x + dpsi_z[i].y*dpsi_z[i].y;
    tau[i] += weight * (dx2 + dy2 + dz2);
}

// Complex non-orthogonal tau: tau += weight * Re(conj(grad psi) . lapcT . grad psi)
static __global__ void tau_accumulate_z_nonorth_kernel(
    const cuDoubleComplex* __restrict__ dpsi_x,
    const cuDoubleComplex* __restrict__ dpsi_y,
    const cuDoubleComplex* __restrict__ dpsi_z,
    double* __restrict__ tau,
    double weight, int N,
    double L00, double L11, double L22, double L01, double L02, double L12)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double ndx = dpsi_x[i].x*dpsi_x[i].x + dpsi_x[i].y*dpsi_x[i].y;
    double ndy = dpsi_y[i].x*dpsi_y[i].x + dpsi_y[i].y*dpsi_y[i].y;
    double ndz = dpsi_z[i].x*dpsi_z[i].x + dpsi_z[i].y*dpsi_z[i].y;
    double cdxy = dpsi_x[i].x*dpsi_y[i].x + dpsi_x[i].y*dpsi_y[i].y;
    double cdxz = dpsi_x[i].x*dpsi_z[i].x + dpsi_x[i].y*dpsi_z[i].y;
    double cdyz = dpsi_y[i].x*dpsi_z[i].x + dpsi_y[i].y*dpsi_z[i].y;
    double val = L00*ndx + L11*ndy + L22*ndz + 2.0*L01*cdxy + 2.0*L02*cdxz + 2.0*L12*cdyz;
    tau[i] += weight * val;
}

// Spin 0.5 factor + total: tau_up *= 0.5, tau_dn *= 0.5, tau_tot = tau_up + tau_dn
static __global__ void tau_spin_finalize_kernel(
    double* __restrict__ tau_up,
    double* __restrict__ tau_dn,
    double* __restrict__ tau_tot,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    tau_up[i] *= 0.5;
    tau_dn[i] *= 0.5;
    tau_tot[i] = tau_up[i] + tau_dn[i];
}

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

    // Device tau buffer (accumulated on GPU, downloaded to host)
    double* d_tau  = nullptr;  // Nd * Nspin (or 3*Nd for spin: [up|dn|total])
    double* d_vtau = nullptr;  // Nd * Nspin
};

void KineticEnergyDensity::setup_gpu(const LynxContext& ctx, int Nspin) {
    if (!gpu_state_raw_)
        gpu_state_raw_ = new GPUTauState();
    auto* gs = static_cast<GPUTauState*>(gpu_state_raw_);

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
    if (!gpu_state_raw_) return;
    auto* gs = static_cast<GPUTauState*>(gpu_state_raw_);

    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    auto safe_free = [stream](auto*& p) { if (p) { cudaFreeAsync(p, stream); p = nullptr; } };

    safe_free(gs->d_tau);
    safe_free(gs->d_vtau);

    delete gs;
    gpu_state_raw_ = nullptr;
}

KineticEnergyDensity::~KineticEnergyDensity() {
    cleanup_gpu();
}

// ============================================================
// Device-dispatching compute() — GPU path
// ============================================================
void KineticEnergyDensity::compute(const LynxContext& ctx,
                                    const Wavefunction& wfn,
                                    const std::vector<double>& kpt_weights,
                                    Device dev)
{
    if (dev == Device::CPU) {
        compute(ctx, wfn, kpt_weights);
        return;
    }

    // GPU path: per-band gradient + tau accumulation on device
    auto* gs = static_cast<GPUTauState*>(gpu_state_raw_);
    if (!gs) {
        compute(ctx, wfn, kpt_weights);
        return;
    }

    int Nd = gs->Nd;
    int Nspin_global = ctx.Nspin();
    int Nspin_local = ctx.Nspin_local();
    int spin_start = ctx.spin_start();
    int kpt_start = ctx.kpt_start();
    int band_start = ctx.band_start();
    int Nband = wfn.Nband();
    int Nkpts = wfn.Nkpts();
    bool is_kpt = !ctx.kpoints().is_gamma_only();

    auto& gctx = gpu::GPUContext::instance();
    cudaStream_t stream = gctx.compute_stream;
    int bs = 256;
    int grid_sz = gpu::ceildiv(Nd, bs);

    int nx_ex = gs->nx + 2*gs->FDn;
    int ny_ex = gs->ny + 2*gs->FDn;

    // Allocate scratch from pool
    auto& sp = gctx.scratch_pool;
    size_t sp_cp = sp.checkpoint();

    double* d_psi_ex = sp.alloc<double>(gs->Nd_ex);
    double* d_dpsi_x = sp.alloc<double>(Nd);
    double* d_dpsi_y = sp.alloc<double>(Nd);
    double* d_dpsi_z = sp.alloc<double>(Nd);

    // Zero tau on device
    int tau_size = (Nspin_global >= 2) ? 3 * Nd : Nd;
    allocate(Nd, Nspin_global);
    CUDA_CHECK(cudaMemsetAsync(gs->d_tau, 0, tau_size * sizeof(double), stream));

    for (int s = 0; s < Nspin_local; ++s) {
        int s_glob = spin_start + s;
        double* d_tau_s = gs->d_tau + s_glob * Nd;

        for (int k = 0; k < Nkpts; ++k) {
            double wk = kpt_weights[kpt_start + k];
            const auto& occ = wfn.occupations(s, k);

            if (is_kpt) {
                // Complex k-point path: upload psi columns from host
                const auto& psi_c = wfn.psi_kpt(s, k);
                Vec3 kpt = ctx.kpoints().kpts_cart()[kpt_start + k];

                // Allocate complex scratch
                size_t sp_cp2 = sp.checkpoint();
                cuDoubleComplex* d_psi_z_col = sp.alloc<cuDoubleComplex>(Nd);
                cuDoubleComplex* d_psi_z_ex = sp.alloc<cuDoubleComplex>(gs->Nd_ex);
                cuDoubleComplex* d_dpsi_z_x = sp.alloc<cuDoubleComplex>(Nd);
                cuDoubleComplex* d_dpsi_z_y = sp.alloc<cuDoubleComplex>(Nd);
                cuDoubleComplex* d_dpsi_z_z = sp.alloc<cuDoubleComplex>(Nd);

                for (int n = 0; n < Nband; ++n) {
                    double fn = occ(band_start + n);
                    if (fn < 1e-16) continue;
                    double g_nk = wk * fn;

                    // Upload psi column to device
                    const Complex* col = psi_c.col(n);
                    CUDA_CHECK(cudaMemcpyAsync(d_psi_z_col, col, Nd * sizeof(cuDoubleComplex),
                                               cudaMemcpyHostToDevice, stream));

                    // TODO: complex halo exchange + gradient on GPU
                    // For now, fall back to CPU for k-point tau
                    // (complex gradient GPU kernels not yet available)
                }

                sp.restore(sp_cp2);

                // Fall back to CPU for k-point tau (complex halo+gradient not wired)
                if (Nband > 0) {
                    sp.restore(sp_cp);
                    compute(ctx, wfn, kpt_weights);
                    return;
                }
            } else {
                // Real gamma-point path
                const auto& psi = wfn.psi(s, k);

                for (int n = 0; n < Nband; ++n) {
                    double fn = occ(band_start + n);
                    if (fn < 1e-16) continue;
                    double g_nk = wk * fn;

                    // Upload psi column to device
                    const double* col = psi.col(n);
                    CUDA_CHECK(cudaMemcpyAsync(d_dpsi_x, col, Nd * sizeof(double),
                                               cudaMemcpyHostToDevice, stream));

                    // Halo exchange
                    gpu::halo_exchange_gpu(d_dpsi_x, d_psi_ex, gs->nx, gs->ny, gs->nz,
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
                            d_dpsi_x, d_dpsi_y, d_dpsi_z, d_tau_s, g_nk, Nd);
                    } else {
                        tau_accumulate_nonorth_kernel<<<grid_sz, bs, 0, stream>>>(
                            d_dpsi_x, d_dpsi_y, d_dpsi_z, d_tau_s, g_nk, Nd,
                            gs->lapcT[0], gs->lapcT[4], gs->lapcT[8],
                            gs->lapcT[1], gs->lapcT[2], gs->lapcT[5]);
                    }
                }
            }
        }
    }

    sp.restore(sp_cp);

    // Download tau to host
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(tau_.data(), gs->d_tau, tau_size * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // MPI reductions on host (same as CPU path)
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

    // Spin finalize: 0.5 factor + total
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

    valid_ = true;
}

} // namespace lynx

#endif // USE_CUDA
