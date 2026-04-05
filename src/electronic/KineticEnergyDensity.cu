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

double* KineticEnergyDensity::d_tau() {
    auto* gs = static_cast<GPUTauState*>(gpu_state_raw_);
    return gs ? gs->d_tau : nullptr;
}

double* KineticEnergyDensity::d_vtau() {
    auto* gs = static_cast<GPUTauState*>(gpu_state_raw_);
    return gs ? gs->d_vtau : nullptr;
}

// ============================================================
// GPU kernel wrapper: halo + gradient + tau accumulation for one band
// ============================================================
void KineticEnergyDensity::gradient_accumulate_tau_gpu(
    const double* d_psi_col, double* d_tau_s, double weight, int Nd)
{
    auto* gs = static_cast<GPUTauState*>(gpu_state_raw_);
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

} // namespace lynx

#endif // USE_CUDA
