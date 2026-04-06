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
// GPU compute — full density computation on device
// ============================================================
void ElectronDensity::compute_gpu(const LynxContext& ctx,
                                   const Wavefunction& wfn,
                                   const std::vector<double>& kpt_weights)
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

            if (is_kpt) {
                const auto& psi_k = wfn.psi_kpt(s, k);
                const double* occ = wfn.occupations(s, k).data();

                void* d_psi_z = nullptr;
                double* d_occ_tmp = nullptr;
                size_t psi_bytes = (size_t)Nd * Nband * 2 * sizeof(double);
                CUDA_CHECK(cudaMallocAsync(&d_psi_z, psi_bytes, stream));
                CUDA_CHECK(cudaMallocAsync(&d_occ_tmp, Nband * sizeof(double), stream));
                CUDA_CHECK(cudaMemcpyAsync(d_psi_z, psi_k.data(), psi_bytes, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(d_occ_tmp, occ, Nband * sizeof(double), cudaMemcpyHostToDevice, stream));

                accumulate_band_kpt_gpu(d_psi_z, d_occ_tmp, d_rho_s, Nd, Nband, wk);

                cudaFreeAsync(d_psi_z, stream);
                cudaFreeAsync(d_occ_tmp, stream);
            } else {
                const double* psi_data = wfn.psi(s, k).data();
                const double* occ = wfn.occupations(s, k).data();

                double* d_psi = nullptr;
                double* d_occ_tmp = nullptr;
                size_t psi_bytes = (size_t)Nd * Nband * sizeof(double);
                CUDA_CHECK(cudaMallocAsync(&d_psi, psi_bytes, stream));
                CUDA_CHECK(cudaMallocAsync(&d_occ_tmp, Nband * sizeof(double), stream));
                CUDA_CHECK(cudaMemcpyAsync(d_psi, psi_data, psi_bytes, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(d_occ_tmp, occ, Nband * sizeof(double), cudaMemcpyHostToDevice, stream));

                accumulate_band_gpu(d_psi, d_occ_tmp, d_rho_s, Nd, Nband, wk);

                cudaFreeAsync(d_psi, stream);
                cudaFreeAsync(d_occ_tmp, stream);
            }
        }

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

// ============================================================
// GPU-resident compute — reads psi and occ from device pointers directly
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

    allocate(Nd, Nspin_global);

    for (int s = 0; s < Nspin_local; ++s) {
        int s_glob = spin_start + s;

        double* d_rho_s = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_rho_s, Nd * sizeof(double), stream));
        CUDA_CHECK(cudaMemsetAsync(d_rho_s, 0, Nd * sizeof(double), stream));

        for (int k = 0; k < Nkpts; ++k) {
            int k_glob = kpt_start + k;
            double wk = kpt_weights[k_glob] * spin_fac;

            if (is_kpt) {
                const auto& psi_k = wfn.psi_kpt(s, k);
                const double* occ_h = wfn.occupations(s, k).data();

                void* d_psi_z_tmp = nullptr;
                double* d_occ_tmp = nullptr;
                size_t psi_bytes = (size_t)Nd * Nband * 2 * sizeof(double);
                CUDA_CHECK(cudaMallocAsync(&d_psi_z_tmp, psi_bytes, stream));
                CUDA_CHECK(cudaMallocAsync(&d_occ_tmp, Nband * sizeof(double), stream));
                CUDA_CHECK(cudaMemcpyAsync(d_psi_z_tmp, psi_k.data(), psi_bytes, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(d_occ_tmp, occ_h, Nband * sizeof(double), cudaMemcpyHostToDevice, stream));

                accumulate_band_kpt_gpu(d_psi_z_tmp, d_occ_tmp, d_rho_s, Nd, Nband, wk);

                cudaFreeAsync(d_psi_z_tmp, stream);
                cudaFreeAsync(d_occ_tmp, stream);
            } else {
                const double* occ_h = wfn.occupations(s, k).data();
                double* d_occ_tmp = nullptr;
                CUDA_CHECK(cudaMallocAsync(&d_occ_tmp, Nband * sizeof(double), stream));
                CUDA_CHECK(cudaMemcpyAsync(d_occ_tmp, occ_h, Nband * sizeof(double), cudaMemcpyHostToDevice, stream));

                accumulate_band_gpu(d_psi_real, d_occ_tmp, d_rho_s, Nd, Nband, wk);

                cudaFreeAsync(d_occ_tmp, stream);
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpyAsync(rho(s_glob).data(), d_rho_s, Nd * sizeof(double), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        cudaFreeAsync(d_rho_s, stream);
    }

    // MPI reductions
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
