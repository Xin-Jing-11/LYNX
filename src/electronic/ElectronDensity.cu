#ifdef USE_CUDA

#include "electronic/ElectronDensity.hpp"
#include "core/LynxContext.hpp"
#include "electronic/Wavefunction.hpp"
#include "core/DeviceTag.hpp"

// GPU function declaration for complex density accumulation
#include "solvers/EigenSolver.cuh"    // gpu::compute_density_z_gpu

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "core/gpu_common.cuh"
#include "core/GPUContext.cuh"

namespace lynx {

// ============================================================
// Density accumulation kernel (real, gamma-point)
// rho[i] += weight * occ[n] * |psi[i + n*Nd]|^2
// ============================================================
__global__ static void compute_density_kernel(
    const double* __restrict__ psi,
    const double* __restrict__ occ,
    double* __restrict__ rho,
    int Nd, int Ns, double weight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Nd) {
        double sum = 0.0;
        for (int n = 0; n < Ns; ++n) {
            double val = psi[idx + n * Nd];
            sum += occ[n] * val * val;
        }
        rho[idx] += weight * sum;
    }
}

// ============================================================
// GPUDensityState — stub for future device-resident rho arrays
// ============================================================
struct GPUDensityState {
    int Nd = 0;
    int Nspin = 1;
};

void ElectronDensity::setup_gpu(const LynxContext& ctx, int Nspin) {
    if (!gpu_state_raw_)
        gpu_state_raw_ = new GPUDensityState();
    auto* gs = static_cast<GPUDensityState*>(gpu_state_raw_);

    gs->Nd    = ctx.domain().Nd_d();
    gs->Nspin = Nspin;
}

void ElectronDensity::cleanup_gpu() {
    delete static_cast<GPUDensityState*>(gpu_state_raw_);
    gpu_state_raw_ = nullptr;
}

ElectronDensity::~ElectronDensity() {
    cleanup_gpu();
}

// ============================================================
// GPU kernel wrappers — thin wrappers, no algorithm logic
// ============================================================

void ElectronDensity::accumulate_band_gpu(const double* d_psi, const double* d_occ,
                                           double* d_rho, int Nd, int Nband, double weight) {
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int bs = 256;
    int grid = gpu::ceildiv(Nd, bs);
    compute_density_kernel<<<grid, bs, 0, stream>>>(d_psi, d_occ, d_rho, Nd, Nband, weight);
}

// gpu:: namespace wrapper — retained for test compatibility
namespace gpu {
void compute_density_gpu(
    const double* d_psi, const double* d_occ, double* d_rho,
    int Nd, int Ns, double weight, cudaStream_t stream)
{
    int bs = 256;
    int grid = ceildiv(Nd, bs);
    compute_density_kernel<<<grid, bs, 0, stream>>>(d_psi, d_occ, d_rho, Nd, Ns, weight);
}
} // namespace gpu

void ElectronDensity::accumulate_band_kpt_gpu(const void* d_psi_z, const double* d_occ,
                                               double* d_rho, int Nd, int Nband, double weight) {
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    gpu::compute_density_z_gpu(
        static_cast<const cuDoubleComplex*>(d_psi_z),
        d_occ, d_rho, Nd, Nband, weight, stream);
}

} // namespace lynx

#endif // USE_CUDA
