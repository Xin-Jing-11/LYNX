#ifdef USE_CUDA

#include "physics/EffectivePotential.hpp"
#include "core/LynxContext.hpp"
#include "core/DeviceTag.hpp"
#include "core/constants.hpp"

#include <cuda_runtime.h>
#include "core/gpu_common.cuh"
#include "core/GPUContext.cuh"

namespace lynx {

// ============================================================
// File-static kernels
// ============================================================

// Veff = Vxc + phi (per spin channel)
static __global__ void veff_combine_kernel(
    const double* __restrict__ vxc,
    const double* __restrict__ phi,
    double* __restrict__ veff,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) veff[i] = vxc[i] + phi[i];
}

// rhs = fourpi * (rho + pseudocharge)
static __global__ void poisson_rhs_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ pseudocharge,
    double* __restrict__ rhs,
    double fourpi, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) rhs[i] = fourpi * (rho[i] + pseudocharge[i]);
}

// rhs = fourpi * rho (no pseudocharge)
static __global__ void poisson_rhs_nob_kernel(
    const double* __restrict__ rho,
    double* __restrict__ rhs,
    double fourpi, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) rhs[i] = fourpi * rho[i];
}

// Construct 4-component Veff_spinor from XC output and magnetization
static __global__ void veff_spinor_from_xc_kernel(
    const double* __restrict__ Vxc_up,
    const double* __restrict__ Vxc_dn,
    const double* __restrict__ phi,
    const double* __restrict__ mag_x,
    const double* __restrict__ mag_y,
    const double* __restrict__ mag_z,
    double* __restrict__ Veff_spinor,
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
            Veff_spinor[i]          = vavg + vdiff * mz * inv_m + p;
            Veff_spinor[Nd + i]     = vavg - vdiff * mz * inv_m + p;
            Veff_spinor[2*Nd + i]   = vdiff * mx * inv_m;
            Veff_spinor[3*Nd + i]   = -vdiff * my * inv_m;
        } else {
            Veff_spinor[i]          = vavg + p;
            Veff_spinor[Nd + i]     = vavg + p;
            Veff_spinor[2*Nd + i]   = 0.0;
            Veff_spinor[3*Nd + i]   = 0.0;
        }
    }
}

static bool is_mgga_type(XCType t) {
    return t == XCType::MGGA_SCAN || t == XCType::MGGA_RSCAN || t == XCType::MGGA_R2SCAN;
}

// ============================================================
// GPUVeffState — device-resident potential and density arrays
// ============================================================
struct GPUVeffState {
    int Nd = 0;
    int Nspin = 1;

    // Persistent uploads (uploaded once from CPU, used every SCF iteration)
    double* d_pseudocharge = nullptr; // Nd
    double* d_rho_core   = nullptr;  // Nd (if NLCC)

    // Persistent device arrays for GPU-resident Veff computation.
    double* d_Veff = nullptr;        // (Nd * Nspin) effective potential
    double* d_Vxc  = nullptr;        // (Nd * Nspin) XC potential
    double* d_exc  = nullptr;        // (Nd) XC energy density
    double* d_phi  = nullptr;        // (Nd) electrostatic potential
    double* d_rho  = nullptr;        // (Nd * Nspin) electron density (working copy)
    double* d_rho_total = nullptr;   // (Nd) total density
    double* d_rhs  = nullptr;        // (Nd) Poisson RHS

    bool buffers_allocated = false;

    // Persistent operator instances for GPU dispatch
    XCFunctional xc;
    PoissonSolver poisson;

    void allocate_buffers() {
        if (buffers_allocated) return;
        cudaStream_t stream = gpu::GPUContext::instance().compute_stream;

        CUDA_CHECK(cudaMallocAsync(&d_Veff, Nd * Nspin * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_Vxc,  Nd * Nspin * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_exc,  Nd * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_phi,  Nd * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_rho,  Nd * Nspin * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_rho_total, Nd * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_rhs,  Nd * sizeof(double), stream));

        buffers_allocated = true;
    }

    void free_buffers() {
        if (!buffers_allocated) return;
        cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
        auto safe_free = [stream](auto*& p) { if (p) { cudaFreeAsync(p, stream); p = nullptr; } };

        safe_free(d_Veff); safe_free(d_Vxc); safe_free(d_exc);
        safe_free(d_phi); safe_free(d_rho); safe_free(d_rho_total);
        safe_free(d_rhs);

        buffers_allocated = false;
    }
};

void EffectivePotential::setup_gpu(const LynxContext& ctx, int Nspin,
                                         XCType xc_type, const double* rho_b,
                                         const double* rho_core) {
    if (!gpu_state_raw_)
        gpu_state_raw_ = new GPUVeffState();
    auto* gs = static_cast<GPUVeffState*>(gpu_state_raw_);

    gs->Nd    = ctx.domain().Nd_d();
    gs->Nspin = Nspin;
    int Nd = gs->Nd;

    // Upload persistent data (pseudocharge and NLCC core density)
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    CUDA_CHECK(cudaMallocAsync(&gs->d_pseudocharge, Nd * sizeof(double), stream));
    if (rho_b) {
        CUDA_CHECK(cudaMemcpyAsync(gs->d_pseudocharge, rho_b, Nd * sizeof(double), cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemsetAsync(gs->d_pseudocharge, 0, Nd * sizeof(double), stream));
    }

    if (rho_core) {
        CUDA_CHECK(cudaMallocAsync(&gs->d_rho_core, Nd * sizeof(double), stream));
        CUDA_CHECK(cudaMemcpyAsync(gs->d_rho_core, rho_core, Nd * sizeof(double), cudaMemcpyHostToDevice, stream));
    }

    // Allocate persistent potential/density device arrays
    gs->allocate_buffers();

    // Setup XCFunctional for GPU dispatch
    gs->xc.setup(xc_type, ctx.domain(), ctx.grid(), &ctx.gradient(), &ctx.halo());
    gs->xc.setup_gpu(ctx, Nspin);
    if (rho_core) {
        gs->xc.set_gpu_nlcc(rho_core, Nd);
    }

    // Setup PoissonSolver for GPU dispatch
    gs->poisson.setup(ctx.laplacian(), ctx.stencil(), ctx.domain(), ctx.grid(), ctx.halo());
    gs->poisson.setup_gpu(ctx);
}

void EffectivePotential::cleanup_gpu() {
    if (!gpu_state_raw_) return;
    auto* gs = static_cast<GPUVeffState*>(gpu_state_raw_);

    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    auto safe_free = [stream](auto*& p) { if (p) { cudaFreeAsync(p, stream); p = nullptr; } };

    safe_free(gs->d_pseudocharge);
    safe_free(gs->d_rho_core);

    gs->free_buffers();

    // XCFunctional and PoissonSolver clean up their own GPU state in destructors
    gs->xc.cleanup_gpu();
    gs->poisson.cleanup_gpu();

    delete gs;
    gpu_state_raw_ = nullptr;
}

EffectivePotential::~EffectivePotential() {
    cleanup_gpu();
}

// ============================================================
// Device-dispatching compute() — GPU path
// ============================================================
void EffectivePotential::compute(const ElectronDensity& density,
                                  const double* rho_b,
                                  const double* rho_core,
                                  XCType xc_type,
                                  double exx_frac_scale,
                                  double poisson_tol,
                                  VeffArrays& arrays,
                                  Device dev,
                                  const double* tau,
                                  bool tau_valid)
{
    if (dev == Device::CPU) {
        compute(density, rho_b, rho_core, xc_type, exx_frac_scale,
                poisson_tol, arrays, tau, tau_valid);
        return;
    }

    // GPU path: XC + Poisson + combine all on GPU
    auto* gs = static_cast<GPUVeffState*>(gpu_state_raw_);
    if (!gs || !gs->buffers_allocated) {
        // Fallback to CPU if GPU state not initialized
        compute(density, rho_b, rho_core, xc_type, exx_frac_scale,
                poisson_tol, arrays, tau, tau_valid);
        return;
    }

    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int Nd = gs->Nd;
    int Nspin = gs->Nspin;
    int bs = 256;
    int grid_sz = gpu::ceildiv(Nd, bs);

    // For hybrid functionals: scale exchange
    if (exx_frac_scale > 0.0) {
        gs->xc.set_exchange_scale(1.0 - exx_frac_scale);
    }

    // For mGGA: set tau_valid so GPU XC path uses SCAN kernels when tau is available
    if (is_mgga_type(xc_type)) {
        gs->xc.set_gpu_tau_valid(tau_valid);
    }

    // 1. Upload density from host to device
    upload_density(density);

    // 2. XC evaluation on GPU
    // XC handles NLCC internally (rho_core was uploaded in setup_gpu via set_gpu_nlcc).
    // Pass device pointers for rho, Vxc, exc.
    // Note: tau/vtau for mGGA must be device pointers. When tau is host-only
    // (KineticEnergyDensity GPU path not yet wired), pass nullptr — XC GPU falls back to PBE.
    // TODO: pass device tau/vtau when KineticEnergyDensity GPU path is complete.
    if (Nspin == 2) {
        // Spin: GPU evaluate_spin expects [up(Nd)|dn(Nd)] — matches d_rho layout
        gs->xc.evaluate_spin(gs->d_rho, gs->d_Vxc, gs->d_exc, Nd,
                              Device::GPU);
    } else {
        // Non-spin: pass total density
        gs->xc.evaluate(gs->d_rho_total, gs->d_Vxc, gs->d_exc, Nd,
                         Device::GPU);
    }

    // 3. Prepare Poisson RHS on device: 4*pi*(rho_total + pseudocharge)
    if (rho_b) {
        poisson_rhs_kernel<<<grid_sz, bs, 0, stream>>>(
            gs->d_rho_total, gs->d_pseudocharge, gs->d_rhs,
            4.0 * constants::PI, Nd);
    } else {
        poisson_rhs_nob_kernel<<<grid_sz, bs, 0, stream>>>(
            gs->d_rho_total, gs->d_rhs,
            4.0 * constants::PI, Nd);
    }

    // 4. Solve Poisson equation on GPU (handles mean-subtraction internally)
    gs->poisson.solve(gs->d_rhs, gs->d_phi, poisson_tol, Device::GPU);

    // 5. Veff = Vxc + phi (per spin channel)
    for (int s = 0; s < Nspin; ++s) {
        veff_combine_kernel<<<grid_sz, bs, 0, stream>>>(
            gs->d_Vxc + s * Nd, gs->d_phi, gs->d_Veff + s * Nd, Nd);
    }

    // 6. Download results to host arrays (for Energy::compute_all and convergence checks)
    download_to_host(arrays);

    // 7. Set vtau on Hamiltonian for mGGA
    if (is_mgga_type(xc_type) && tau_valid) {
        const_cast<Hamiltonian*>(hamiltonian_)->set_vtau(arrays.vtau.data());
    } else if (is_mgga_type(xc_type)) {
        const_cast<Hamiltonian*>(hamiltonian_)->set_vtau(nullptr);
    }
}

// ============================================================
// Device-dispatching compute_spinor() — GPU path
// ============================================================
void EffectivePotential::compute_spinor(const ElectronDensity& density,
                                         const double* rho_b,
                                         const double* rho_core,
                                         XCType xc_type,
                                         double poisson_tol,
                                         VeffArrays& arrays,
                                         Device dev)
{
    if (dev == Device::CPU) {
        compute_spinor(density, rho_b, rho_core, xc_type, poisson_tol, arrays);
        return;
    }

    // SOC GPU path: not yet implemented (requires noncollinear density on device)
    // Fall back to CPU for correctness
    compute_spinor(density, rho_b, rho_core, xc_type, poisson_tol, arrays);
}

// ============================================================
// GPU-resident device pointer accessors
// ============================================================

double* EffectivePotential::gpu_Veff() {
    auto* gs = static_cast<GPUVeffState*>(gpu_state_raw_);
    return gs ? gs->d_Veff : nullptr;
}

const double* EffectivePotential::gpu_Veff() const {
    auto* gs = static_cast<GPUVeffState*>(gpu_state_raw_);
    return gs ? gs->d_Veff : nullptr;
}

double* EffectivePotential::gpu_phi() {
    auto* gs = static_cast<GPUVeffState*>(gpu_state_raw_);
    return gs ? gs->d_phi : nullptr;
}

double* EffectivePotential::gpu_exc() {
    auto* gs = static_cast<GPUVeffState*>(gpu_state_raw_);
    return gs ? gs->d_exc : nullptr;
}

double* EffectivePotential::gpu_Vxc() {
    auto* gs = static_cast<GPUVeffState*>(gpu_state_raw_);
    return gs ? gs->d_Vxc : nullptr;
}

double* EffectivePotential::gpu_rho() {
    auto* gs = static_cast<GPUVeffState*>(gpu_state_raw_);
    return gs ? gs->d_rho : nullptr;
}

double* EffectivePotential::gpu_rho_total() {
    auto* gs = static_cast<GPUVeffState*>(gpu_state_raw_);
    return gs ? gs->d_rho_total : nullptr;
}

// ============================================================
// Upload density from host to device buffers
// ============================================================
void EffectivePotential::upload_density(const ElectronDensity& density) {
    auto* gs = static_cast<GPUVeffState*>(gpu_state_raw_);
    if (!gs || !gs->buffers_allocated) return;

    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int Nd = gs->Nd;
    int Nspin = gs->Nspin;

    // Upload total density
    CUDA_CHECK(cudaMemcpyAsync(gs->d_rho_total, density.rho_total().data(),
                               Nd * sizeof(double), cudaMemcpyHostToDevice, stream));

    // Upload per-spin density
    for (int s = 0; s < Nspin; ++s) {
        CUDA_CHECK(cudaMemcpyAsync(gs->d_rho + s * Nd, density.rho(s).data(),
                                   Nd * sizeof(double), cudaMemcpyHostToDevice, stream));
    }
}

// ============================================================
// Download potential arrays from device to host VeffArrays
// ============================================================
void EffectivePotential::download_to_host(VeffArrays& arrays) {
    auto* gs = static_cast<GPUVeffState*>(gpu_state_raw_);
    if (!gs || !gs->buffers_allocated) return;

    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int Nd = gs->Nd;
    int Nspin = gs->Nspin;

    CUDA_CHECK(cudaMemcpyAsync(arrays.Veff.data(), gs->d_Veff,
                               Nd * Nspin * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(arrays.Vxc.data(), gs->d_Vxc,
                               Nd * Nspin * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(arrays.exc.data(), gs->d_exc,
                               Nd * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(arrays.phi.data(), gs->d_phi,
                               Nd * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace lynx

#endif // USE_CUDA
