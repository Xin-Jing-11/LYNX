#ifdef USE_CUDA

#include "electronic/ElectronDensity.hpp"
#include "core/LynxContext.hpp"
#include "electronic/Wavefunction.hpp"
#include "core/DeviceTag.hpp"
#include <mpi.h>

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
    if (!gpu_state_)
        gpu_state_.reset(new GPUDensityState());
    auto* gs = gpu_state_.as<GPUDensityState>();

    gs->Nd    = ctx.domain().Nd_d();
    gs->Nspin = Nspin;
}

void ElectronDensity::cleanup_gpu() {
    gpu_state_.reset();
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

// ============================================================
// Spinor density accumulation kernel (SOC/noncollinear)
// psi layout: [psi_up(Nd) | psi_dn(Nd)] per band, Nband bands
// rho[i] += w * (|up|^2 + |dn|^2)
// mx[i]  += w * 2*Re(conj(up)*dn)
// my[i]  -= w * 2*Im(conj(up)*dn)
// mz[i]  += w * (|up|^2 - |dn|^2)
// ============================================================
__global__ static void compute_spinor_density_kernel(
    const cuDoubleComplex* __restrict__ psi,
    const double* __restrict__ occ,
    double* __restrict__ rho,
    double* __restrict__ mx,
    double* __restrict__ my,
    double* __restrict__ mz,
    int Nd, int Nband, double weight)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Nd) {
        double sum_rho = 0.0, sum_mx = 0.0, sum_my = 0.0, sum_mz = 0.0;
        int Nd_spinor = 2 * Nd;
        for (int n = 0; n < Nband; ++n) {
            double fn = occ[n];
            if (fn < 1e-16) continue;
            const cuDoubleComplex* col = psi + n * Nd_spinor;
            cuDoubleComplex up = col[i];
            cuDoubleComplex dn = col[i + Nd];
            double up2 = up.x * up.x + up.y * up.y;
            double dn2 = dn.x * dn.x + dn.y * dn.y;
            // conj(up)*dn = (up.x - i*up.y)*(dn.x + i*dn.y)
            //             = (up.x*dn.x + up.y*dn.y) + i*(up.x*dn.y - up.y*dn.x)
            double cross_re = up.x * dn.x + up.y * dn.y;
            double cross_im = up.x * dn.y - up.y * dn.x;
            double w = fn;
            sum_rho += w * (up2 + dn2);
            sum_mx  += w * 2.0 * cross_re;
            sum_my  -= w * 2.0 * cross_im;
            sum_mz  += w * (up2 - dn2);
        }
        rho[i] += weight * sum_rho;
        mx[i]  += weight * sum_mx;
        my[i]  += weight * sum_my;
        mz[i]  += weight * sum_mz;
    }
}

void ElectronDensity::accumulate_spinor_band_gpu(const void* d_psi_z, const double* d_occ,
                                                   double* d_rho, double* d_mx, double* d_my, double* d_mz,
                                                   int Nd, int Nband, double weight) {
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int bs = 256;
    int grid = gpu::ceildiv(Nd, bs);
    compute_spinor_density_kernel<<<grid, bs, 0, stream>>>(
        static_cast<const cuDoubleComplex*>(d_psi_z), d_occ,
        d_rho, d_mx, d_my, d_mz, Nd, Nband, weight);
}

// ============================================================
// Spinor density from device-resident psi — zero psi H2D transfers
// ============================================================
void ElectronDensity::compute_spinor_from_device_ptrs(
    const LynxContext& ctx,
    const Wavefunction& wfn,
    const std::vector<double>& kpt_weights,
    const std::vector<const void*>& d_psi_z_ptrs)
{
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int Nd = ctx.domain().Nd_d();
    int Nband = wfn.Nband();
    int kpt_start = ctx.kpt_start();
    int Nkpts = wfn.Nkpts();

    allocate_noncollinear(Nd);

    // Allocate device rho/mag arrays
    double *d_rho = nullptr, *d_mx = nullptr, *d_my = nullptr, *d_mz = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_rho, Nd * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_mx, Nd * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_my, Nd * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_mz, Nd * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(d_rho, 0, Nd * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(d_mx, 0, Nd * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(d_my, 0, Nd * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(d_mz, 0, Nd * sizeof(double), stream));

    for (int k = 0; k < Nkpts; ++k) {
        int k_glob = kpt_start + k;
        double wk = kpt_weights[k_glob];

        // Upload occupations (tiny: Nband doubles)
        const double* occ_h = wfn.occupations(0, k).data();
        double* d_occ = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_occ, Nband * sizeof(double), stream));
        CUDA_CHECK(cudaMemcpyAsync(d_occ, occ_h, Nband * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));

        accumulate_spinor_band_gpu(d_psi_z_ptrs[k], d_occ,
                                    d_rho, d_mx, d_my, d_mz,
                                    Nd, Nband, wk);
        cudaFreeAsync(d_occ, stream);
    }

    // Download results to host
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(rho_total_.data(), d_rho, Nd * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(mag_x_.data(), d_mx, Nd * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(mag_y_.data(), d_my, Nd * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(mag_z_.data(), d_mz, Nd * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Keep rho_[0] in sync
    std::memcpy(rho_[0].data(), rho_total_.data(), Nd * sizeof(double));

    // Free device arrays
    cudaFreeAsync(d_rho, stream);
    cudaFreeAsync(d_mx, stream);
    cudaFreeAsync(d_my, stream);
    cudaFreeAsync(d_mz, stream);

    // MPI reductions
    const auto& bandcomm = ctx.scf_bandcomm();
    const auto& kptcomm = ctx.kpt_bridge();

    if (!bandcomm.is_null() && bandcomm.size() > 1) {
        bandcomm.allreduce_sum(rho_total_.data(), Nd);
        bandcomm.allreduce_sum(mag_x_.data(), Nd);
        bandcomm.allreduce_sum(mag_y_.data(), Nd);
        bandcomm.allreduce_sum(mag_z_.data(), Nd);
    }

    if (!kptcomm.is_null() && kptcomm.size() > 1) {
        kptcomm.allreduce_sum(rho_total_.data(), Nd);
        kptcomm.allreduce_sum(mag_x_.data(), Nd);
        kptcomm.allreduce_sum(mag_y_.data(), Nd);
        kptcomm.allreduce_sum(mag_z_.data(), Nd);
    }

    std::memcpy(rho_[0].data(), rho_total_.data(), Nd * sizeof(double));
}

// NOTE: Legacy compute_gpu() that uploaded psi from host has been removed.
// Production code uses compute_from_device_ptrs() with device-resident psi pointers.

void ElectronDensity::compute_spinor_gpu(const LynxContext& ctx,
                                          const Wavefunction& wfn,
                                          const std::vector<double>& kpt_weights)
{
    // Spinor GPU path not yet wired — fall back to CPU.
    static bool warned = false;
    if (!warned) { fprintf(stderr, "INFO: Spinor density GPU path not yet wired, using CPU\n"); warned = true; }
    dev_ = Device::CPU;
    compute_spinor(ctx, wfn, kpt_weights);
    dev_ = Device::GPU;
}

// NOTE: Legacy compute_from_device() that uploaded psi from host for kpt path has been removed.
// Production code uses compute_from_device_ptrs() with per-(spin,kpt) device-resident psi pointers.

// ============================================================
// Compute density from per-(spin,kpt) device-resident psi — zero psi H2D transfers
// ============================================================
void ElectronDensity::compute_from_device_ptrs(
    const LynxContext& ctx,
    const Wavefunction& wfn,
    const std::vector<double>& kpt_weights,
    const std::vector<const double*>& d_psi_real_ptrs,
    const std::vector<const void*>& d_psi_z_ptrs)
{
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int Nd = ctx.domain().Nd_d();
    int Nband = wfn.Nband();
    int Nspin_local = ctx.Nspin_local();
    int spin_start = ctx.spin_start();
    int Nspin_global = ctx.Nspin();
    int kpt_start = ctx.kpt_start();
    int Nkpts = wfn.Nkpts();
    double spin_fac = (Nspin_global == 1) ? 2.0 : 1.0;
    bool is_kpt = !ctx.kpoints().is_gamma_only();

    allocate(Nd, Nspin_global);

    for (int s = 0; s < Nspin_local; ++s) {
        int s_glob = spin_start + s;

        double* d_rho_s = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_rho_s, Nd * sizeof(double), stream));
        CUDA_CHECK(cudaMemsetAsync(d_rho_s, 0, Nd * sizeof(double), stream));

        for (int k = 0; k < Nkpts; ++k) {
            int k_glob = kpt_start + k;
            double wk = kpt_weights[k_glob] * spin_fac;

            // Upload occupations (tiny: Nband doubles)
            const double* occ_h = wfn.occupations(s, k).data();
            double* d_occ_tmp = nullptr;
            CUDA_CHECK(cudaMallocAsync(&d_occ_tmp, Nband * sizeof(double), stream));
            CUDA_CHECK(cudaMemcpyAsync(d_occ_tmp, occ_h, Nband * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));

            if (is_kpt) {
                // psi already on device — no transfer
                int idx = s * Nkpts + k;
                const void* d_psi_z = d_psi_z_ptrs[idx];
                accumulate_band_kpt_gpu(d_psi_z, d_occ_tmp, d_rho_s, Nd, Nband, wk);
            } else {
                // psi already on device — no transfer
                int idx = s * Nkpts + k;
                const double* d_psi = d_psi_real_ptrs[idx];
                accumulate_band_gpu(d_psi, d_occ_tmp, d_rho_s, Nd, Nband, wk);
            }

            cudaFreeAsync(d_occ_tmp, stream);
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpyAsync(rho(s_glob).data(), d_rho_s, Nd * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        cudaFreeAsync(d_rho_s, stream);
    }

    // MPI reductions — same as other GPU paths
    const auto& bandcomm = ctx.scf_bandcomm();
    const auto& kptcomm = ctx.kpt_bridge();

    if (!bandcomm.is_null() && bandcomm.size() > 1) {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start + s;
            bandcomm.allreduce_sum(rho(s_glob).data(), Nd);
        }
    }

    if (!kptcomm.is_null() && kptcomm.size() > 1) {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start + s;
            kptcomm.allreduce_sum(rho(s_glob).data(), Nd);
        }
    }

    if (Nspin_global == 2) {
        const auto& spincomm = ctx.spin_bridge();
        if (!spincomm.is_null() && spincomm.size() > 1) {
            int my_spin = spin_start;
            int other_spin = 1 - my_spin;
            int partner = (spincomm.rank() == 0) ? 1 : 0;
            MPI_Sendrecv(rho(my_spin).data(), Nd, MPI_DOUBLE, partner, 0,
                         rho(other_spin).data(), Nd, MPI_DOUBLE, partner, 0,
                         spincomm.comm(), MPI_STATUS_IGNORE);
        }
    }

    rho_total_.zero();
    for (int s = 0; s < Nspin_global; ++s) {
        double* rt = rho_total_.data();
        const double* rs = rho(s).data();
        for (int i = 0; i < Nd; ++i) rt[i] += rs[i];
    }
}

} // namespace lynx

#endif // USE_CUDA
