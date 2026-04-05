#ifdef USE_CUDA

#include "electronic/ElectronDensity.hpp"
#include "core/LynxContext.hpp"
#include "electronic/Wavefunction.hpp"
#include "core/DeviceTag.hpp"

// GPU function declarations from existing .cuh headers
#include "solvers/LinearSolver.cuh"   // gpu::compute_density_gpu
#include "solvers/EigenSolver.cuh"    // gpu::compute_density_z_gpu

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "core/gpu_common.cuh"
#include "core/GPUContext.cuh"

namespace lynx {

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
    gpu::compute_density_gpu(d_psi, d_occ, d_rho, Nd, Nband, weight, stream);
}

void ElectronDensity::accumulate_band_kpt_gpu(const void* d_psi_z, const double* d_occ,
                                               double* d_rho, int Nd, int Nband, double weight) {
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    gpu::compute_density_z_gpu(
        static_cast<const cuDoubleComplex*>(d_psi_z),
        d_occ, d_rho, Nd, Nband, weight, stream);
}

} // namespace lynx

#endif // USE_CUDA
