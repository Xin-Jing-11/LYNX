#ifdef USE_CUDA

#include "electronic/ElectronDensity.hpp"
#include "core/LynxContext.hpp"
#include "electronic/Wavefunction.hpp"
#include "core/DeviceTag.hpp"

// GPU function declarations from existing .cuh headers
#include "solvers/LinearSolver.cuh"   // gpu::compute_density_gpu
#include "solvers/EigenSolver.cuh"    // gpu::compute_density_z_gpu
#include "operators/SOCOperators.cuh" // gpu::spinor_density_gpu

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <mpi.h>
#include <cstring>
#include "core/gpu_common.cuh"
#include "core/GPUContext.cuh"

namespace lynx {

// ============================================================
// File-static kernels (duplicated from GPUSCF.cu to avoid coupling)
// ============================================================

// out[i] = a[i] + b[i]
static __global__ void density_add_kernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i] + b[i];
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
    // Density arrays are outputs written by compute() — no separate allocation
    // needed here. The rho data lives in EffectivePotential or is passed as pointers.
}

void ElectronDensity::cleanup_gpu() {
    delete static_cast<GPUDensityState*>(gpu_state_raw_);
    gpu_state_raw_ = nullptr;
}

ElectronDensity::~ElectronDensity() {
    cleanup_gpu();
}

// ============================================================
// Device-dispatching compute()
// ============================================================
void ElectronDensity::compute(const LynxContext& ctx,
                               const Wavefunction& wfn,
                               const std::vector<double>& kpt_weights,
                               Device dev)
{
    if (dev == Device::CPU) {
        compute(ctx, wfn, kpt_weights);
        return;
    }

    // GPU path: upload psi/occ per spin/kpt, accumulate density on GPU, download.
    // Then do the same MPI reductions as the CPU path.
    // This is correct but not optimal (upload/download per kpt).
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
    int band_start = ctx.band_start();

    // Allocate output on host (same as CPU)
    allocate(Nd, Nspin_global);

    // Per-spin accumulation on GPU, then download to host rho_[s_glob]
    for (int s = 0; s < Nspin_local; ++s) {
        int s_glob = spin_start + s;

        // Allocate device rho for this spin channel
        double* d_rho_s = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_rho_s, Nd * sizeof(double), stream));
        CUDA_CHECK(cudaMemsetAsync(d_rho_s, 0, Nd * sizeof(double), stream));

        for (int k = 0; k < Nkpts; ++k) {
            int k_glob = kpt_start + k;
            double wk = kpt_weights[k_glob] * spin_fac;

            if (is_kpt) {
                const auto& psi_k = wfn.psi_kpt(s, k);
                const double* occ = wfn.occupations(s, k).data();

                cuDoubleComplex* d_psi_z = nullptr;
                double* d_occ = nullptr;
                size_t psi_bytes = (size_t)Nd * Nband * sizeof(cuDoubleComplex);
                CUDA_CHECK(cudaMallocAsync(&d_psi_z, psi_bytes, stream));
                CUDA_CHECK(cudaMallocAsync(&d_occ, Nband * sizeof(double), stream));
                CUDA_CHECK(cudaMemcpyAsync(d_psi_z, psi_k.data(), psi_bytes, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(d_occ, occ, Nband * sizeof(double), cudaMemcpyHostToDevice, stream));

                gpu::compute_density_z_gpu(d_psi_z, d_occ, d_rho_s, Nd, Nband, wk);

                cudaFreeAsync(d_psi_z, stream);
                cudaFreeAsync(d_occ, stream);
            } else {
                const double* psi_data = wfn.psi(s, k).data();
                const double* occ = wfn.occupations(s, k).data();

                double* d_psi = nullptr;
                double* d_occ = nullptr;
                size_t psi_bytes = (size_t)Nd * Nband * sizeof(double);
                CUDA_CHECK(cudaMallocAsync(&d_psi, psi_bytes, stream));
                CUDA_CHECK(cudaMallocAsync(&d_occ, Nband * sizeof(double), stream));
                CUDA_CHECK(cudaMemcpyAsync(d_psi, psi_data, psi_bytes, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(d_occ, occ, Nband * sizeof(double), cudaMemcpyHostToDevice, stream));

                gpu::compute_density_gpu(d_psi, d_occ, d_rho_s, Nd, Nband, wk);

                cudaFreeAsync(d_psi, stream);
                cudaFreeAsync(d_occ, stream);
            }
        }

        // Download per-spin density to host
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpyAsync(rho(s_glob).data(), d_rho_s, Nd * sizeof(double), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);  // CPU needs this data now
        cudaFreeAsync(d_rho_s, stream);
    }

    // MPI reductions — same logic as CPU path
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

    // Exchange spin densities across spin communicator
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

    // Compute rho_total from spin channels
    rho_total_.zero();
    for (int s = 0; s < Nspin_global; ++s) {
        double* rt = rho_total_.data();
        const double* rs = rho(s).data();
        for (int i = 0; i < Nd; ++i) {
            rt[i] += rs[i];
        }
    }
}

// ============================================================
// Device-dispatching compute_spinor()
// ============================================================
void ElectronDensity::compute_spinor(const LynxContext& ctx,
                                      const Wavefunction& wfn,
                                      const std::vector<double>& kpt_weights,
                                      Device dev)
{
    if (dev == Device::CPU) {
        // Delegate to existing CPU implementation
        compute_spinor(ctx, wfn, kpt_weights);
        return;
    }

    // Spinor GPU path not yet wired — fall back to CPU.
    // TODO: Wire spinor density GPU computation.
    static bool warned = false;
    if (!warned) { fprintf(stderr, "INFO: Spinor density GPU path not yet wired, using CPU\n"); warned = true; }
    compute_spinor(ctx, wfn, kpt_weights);
}

// ============================================================
// GPU-resident compute: reads psi and occ from device pointers directly
// ============================================================
void ElectronDensity::compute_from_device(const LynxContext& ctx,
                                           const Wavefunction& wfn,
                                           const std::vector<double>& kpt_weights,
                                           const double* d_psi_real,
                                           const void* d_psi_z_ptr,
                                           const double* d_occ)
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

    // Allocate output on host (same as CPU path)
    allocate(Nd, Nspin_global);

    // Per-spin accumulation on GPU, then download to host
    for (int s = 0; s < Nspin_local; ++s) {
        int s_glob = spin_start + s;

        // Allocate device rho for this spin channel
        double* d_rho_s = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_rho_s, Nd * sizeof(double), stream));
        CUDA_CHECK(cudaMemsetAsync(d_rho_s, 0, Nd * sizeof(double), stream));

        for (int k = 0; k < Nkpts; ++k) {
            int k_glob = kpt_start + k;
            double wk = kpt_weights[k_glob] * spin_fac;

            if (is_kpt) {
                // For k-point: psi is complex, still upload from host since each k-point
                // has a separate psi array and eigsolver only keeps current k-point on device.
                // TODO: When multi-kpt GPU buffers are added, read from device directly.
                const auto& psi_k = wfn.psi_kpt(s, k);
                const double* occ = wfn.occupations(s, k).data();

                cuDoubleComplex* d_psi_z_tmp = nullptr;
                double* d_occ_tmp = nullptr;
                size_t psi_bytes = (size_t)Nd * Nband * sizeof(cuDoubleComplex);
                CUDA_CHECK(cudaMallocAsync(&d_psi_z_tmp, psi_bytes, stream));
                CUDA_CHECK(cudaMallocAsync(&d_occ_tmp, Nband * sizeof(double), stream));
                CUDA_CHECK(cudaMemcpyAsync(d_psi_z_tmp, psi_k.data(), psi_bytes, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(d_occ_tmp, occ, Nband * sizeof(double), cudaMemcpyHostToDevice, stream));

                gpu::compute_density_z_gpu(d_psi_z_tmp, d_occ_tmp, d_rho_s, Nd, Nband, wk);

                cudaFreeAsync(d_psi_z_tmp, stream);
                cudaFreeAsync(d_occ_tmp, stream);
            } else {
                // Gamma-point: use device psi directly (no upload needed)
                // Occupations are on host (computed by CPU Occupation::compute),
                // upload them to a temp device buffer.
                const double* occ = wfn.occupations(s, k).data();
                double* d_occ_tmp = nullptr;
                CUDA_CHECK(cudaMallocAsync(&d_occ_tmp, Nband * sizeof(double), stream));
                CUDA_CHECK(cudaMemcpyAsync(d_occ_tmp, occ, Nband * sizeof(double), cudaMemcpyHostToDevice, stream));

                gpu::compute_density_gpu(d_psi_real, d_occ_tmp, d_rho_s, Nd, Nband, wk);

                cudaFreeAsync(d_occ_tmp, stream);
            }
        }

        // Download per-spin density to host
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpyAsync(rho(s_glob).data(), d_rho_s, Nd * sizeof(double), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        cudaFreeAsync(d_rho_s, stream);
    }

    // MPI reductions — same logic as CPU path
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

    // Exchange spin densities across spin communicator
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

    // Compute rho_total from spin channels
    rho_total_.zero();
    for (int s = 0; s < Nspin_global; ++s) {
        double* rt = rho_total_.data();
        const double* rs = rho(s).data();
        for (int i = 0; i < Nd; ++i) {
            rt[i] += rs[i];
        }
    }
}

} // namespace lynx

#endif // USE_CUDA
