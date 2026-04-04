#ifdef USE_CUDA

#include "electronic/KineticEnergyDensity.hpp"
#include "core/LynxContext.hpp"
#include "core/DeviceTag.hpp"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "core/gpu_common.cuh"
#include "core/GPUContext.cuh"

namespace lynx {

// ============================================================
// File-static kernels (duplicated from GPUSCF.cu)
// ============================================================

// tau[i] += weight * (dpsi_x[i]^2 + dpsi_y[i]^2 + dpsi_z[i]^2)
static __global__ void tau_accumulate_kernel_local(
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
static __global__ void tau_accumulate_nonorth_kernel_local(
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
static __global__ void tau_accumulate_z_kernel_local(
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
static __global__ void tau_accumulate_z_nonorth_kernel_local(
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

// ============================================================
// GPUTauState — stub for future device-resident tau buffers
// ============================================================
struct GPUTauState {
    int Nd = 0;
    int Nspin = 1;

    // tau and vtau arrays (owned)
    double* d_tau  = nullptr;  // Nd * Nspin (or 2*Nd for spin)
    double* d_vtau = nullptr;  // Nd * Nspin
};

void KineticEnergyDensity::setup_gpu(const LynxContext& ctx, int Nspin) {
    if (!gpu_state_raw_)
        gpu_state_raw_ = new GPUTauState();
    auto* gs = static_cast<GPUTauState*>(gpu_state_raw_);

    gs->Nd    = ctx.domain().Nd_d();
    gs->Nspin = Nspin;

    int tau_size = (Nspin >= 2) ? 2 * gs->Nd : gs->Nd;

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
// Device-dispatching compute()
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

    // GPU path: tau computation uses gradient operations that don't yet have
    // Device dispatch. CPU fallback is correct since tau data is host-resident
    // and only consumed by XC evaluation (also on CPU).
    compute(ctx, wfn, kpt_weights);
}

} // namespace lynx

#endif // USE_CUDA
