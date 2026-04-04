#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include "core/gpu_common.cuh"
#include "core/GPUContext.cuh"
#include "solvers/Mixer.hpp"
#include "solvers/LinearSolver.cuh"
#include "parallel/HaloExchange.cuh"
#include "operators/Laplacian.cuh"

namespace lynx {

// ============================================================
// Kernels duplicated from GPUSCF.cu (file-static)
// ============================================================

namespace {

// f[i] = g[i] - x[i]
__global__ void mixer_residual_kernel(
    const double* __restrict__ g,
    const double* __restrict__ x,
    double* __restrict__ f, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) f[i] = g[i] - x[i];
}

// X(:,col) = x - x_old, F(:,col) = f - f_old
__global__ void mixer_store_history_kernel(
    const double* __restrict__ x,
    const double* __restrict__ x_old,
    const double* __restrict__ f,
    const double* __restrict__ f_old,
    double* __restrict__ X_hist,
    double* __restrict__ F_hist,
    int col, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        X_hist[col * N + i] = x[i] - x_old[i];
        F_hist[col * N + i] = f[i] - f_old[i];
    }
}

// f[i] = scale * r[i]
__global__ void mixer_jacobi_scale_kernel(
    const double* __restrict__ r,
    double* __restrict__ f,
    double scale, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) f[i] = scale * r[i];
}

} // anonymous namespace

// ============================================================
// GPUMixerState
// ============================================================

struct GPUMixerState {
    // Grid parameters (needed for Kerker preconditioner)
    int nx = 0, ny = 0, nz = 0, FDn = 0, Nd = 0;
    bool is_orth = true;
    bool has_mixed_deriv = false;

    // Mixer parameters
    int m_depth = 7;
    int ncol = 1;
    double beta = 0.3;
    int mix_iter = 0;

    // Kerker preconditioner coefficients
    double kerker_diag = 0.0;
    double kerker_rhs_diag = 0.0;
    double kerker_m_inv = 0.0;
    double precond_tol = 1e-3;

    // NOTE: Pulay history buffers (R, F, fk, xkm1) are NOT owned here —
    // they come from GPUContext::buf at call time in mix().
    // Only d_fkm1 is owned (previous residual, persistent across iterations).
    double* d_fkm1  = nullptr;  // Nd * ncol — previous residual
};

// Thread-local instance pointer for Kerker callback trampolines
static thread_local GPUMixerState* s_mixer_state_ = nullptr;

// ============================================================
// Static Kerker callbacks
// ============================================================

static void mixer_kerker_op_cb(const double* d_x, double* d_Ax) {
    auto* s = s_mixer_state_;
    auto& ctx = gpu::GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;

    gpu::halo_exchange_gpu(d_x, ctx.buf.aar_x_ex,
        s->nx, s->ny, s->nz, s->FDn, 1, true, true, true, stream);
    int nx_ex = s->nx + 2 * s->FDn, ny_ex = s->ny + 2 * s->FDn;
    constexpr double kTF2 = 1.0;
    if (s->is_orth) {
        gpu::laplacian_orth_v7_gpu(ctx.buf.aar_x_ex, nullptr, d_Ax,
            s->nx, s->ny, s->nz, s->FDn, nx_ex, ny_ex,
            -1.0, 0.0, kTF2, s->kerker_diag, 1, stream);
    } else {
        gpu::laplacian_nonorth_gpu(ctx.buf.aar_x_ex, nullptr, d_Ax,
            s->nx, s->ny, s->nz, s->FDn, nx_ex, ny_ex,
            -1.0, 0.0, kTF2, s->kerker_diag,
            s->has_mixed_deriv, s->has_mixed_deriv, s->has_mixed_deriv, 1, stream);
    }
}

static void mixer_kerker_precond_cb(const double* d_r, double* d_f) {
    auto* s = s_mixer_state_;
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int bs = 256;
    mixer_jacobi_scale_kernel<<<gpu::ceildiv(s->Nd, bs), bs, 0, stream>>>(
        d_r, d_f, s->kerker_m_inv, s->Nd);
}

// ============================================================
// Stubs
// ============================================================

void Mixer::setup_gpu(int Nd_d, int ncol, int m_depth, double beta_mix) {
    if (!gpu_state_raw_)
        gpu_state_raw_ = new GPUMixerState();
    auto* gs = static_cast<GPUMixerState*>(gpu_state_raw_);

    gs->m_depth = m_depth;
    gs->ncol    = ncol;
    gs->beta    = beta_mix;
    gs->mix_iter = 0;
    gs->Nd      = Nd_d;

    // Pulay history buffers (R, F, fk, xkm1) come from GPUContext::buf.
    // Only d_fkm1 is allocated lazily on first use in mix().
}

void Mixer::cleanup_gpu() {
    if (!gpu_state_raw_) return;
    auto* gs = static_cast<GPUMixerState*>(gpu_state_raw_);

    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    auto safe_free = [stream](auto*& p) { if (p) { cudaFreeAsync(p, stream); p = nullptr; } };

    safe_free(gs->d_fkm1);

    delete gs;
    gpu_state_raw_ = nullptr;
}

Mixer::~Mixer() {
    cleanup_gpu();
}

// ============================================================
// Device-dispatching mix()
// ============================================================

void Mixer::mix(double* x_k_inout, const double* g_k, int Nd_d, int ncol, Device dev) {
    if (dev == Device::CPU || !gpu_state_raw_) {
        // Fall back to CPU if GPU state not initialized
        mix(x_k_inout, g_k, Nd_d, ncol);
        return;
    }

    // GPU path — mirrors GPUSCF::gpu_pulay_mix()
    auto* gs = static_cast<GPUMixerState*>(gpu_state_raw_);
    auto& ctx = gpu::GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;

    int Nd = Nd_d * ncol;
    int m_depth = gs->m_depth;
    double beta_mix = gs->beta;
    int Nd_kerker = Nd_d;  // Kerker on first Nd_d, simple mix on rest
    double beta_mag = beta_mix;  // same for magnetization by default

    int bs = 256;
    int grid_sz = gpu::ceildiv(Nd, bs);

    double* d_fk   = ctx.buf.mix_fk;
    double* d_xkm1 = ctx.buf.mix_xkm1;
    double* d_R    = ctx.buf.mix_R;
    double* d_F    = ctx.buf.mix_F;

    // Ensure fkm1 buffer exists
    if (!gs->d_fkm1) {
        CUDA_CHECK(cudaMallocAsync(&gs->d_fkm1, Nd * sizeof(double), stream));
        CUDA_CHECK(cudaMemsetAsync(gs->d_fkm1, 0, Nd * sizeof(double), stream));
    }

    // Save old f_k -> f_km1
    if (gs->mix_iter > 0) {
        CUDA_CHECK(cudaMemcpyAsync(gs->d_fkm1, d_fk, Nd * sizeof(double),
                                   cudaMemcpyDeviceToDevice, stream));
    }

    // f_k = g - x
    mixer_residual_kernel<<<grid_sz, bs, 0, stream>>>(g_k, x_k_inout, d_fk, Nd);

    // Store history
    if (gs->mix_iter > 0) {
        int i_hist = (gs->mix_iter - 1) % m_depth;
        mixer_store_history_kernel<<<grid_sz, bs, 0, stream>>>(
            x_k_inout, d_xkm1, d_fk, gs->d_fkm1, d_R, d_F, i_hist, Nd);
    }

    // Workspace
    auto& sp = ctx.scratch_pool;
    size_t sp_cp = sp.checkpoint();
    double* d_x_wavg = sp.alloc<double>(Nd);
    double* d_f_wavg = sp.alloc<double>(Nd);

    if (gs->mix_iter > 0) {
        int cols = std::min(gs->mix_iter, m_depth);

        // Build F^T*F and F^T*f_k
        std::vector<double> h_FtF(cols * cols);
        std::vector<double> h_Ftf(cols);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int ii = 0; ii < cols; ii++) {
            double* Fi = d_F + ii * Nd;
            cublasDdot(ctx.cublas, Nd, Fi, 1, d_fk, 1, &h_Ftf[ii]);
            for (int jj = 0; jj <= ii; jj++) {
                double* Fj = d_F + jj * Nd;
                cublasDdot(ctx.cublas, Nd, Fi, 1, Fj, 1, &h_FtF[ii * cols + jj]);
                h_FtF[jj * cols + ii] = h_FtF[ii * cols + jj];
            }
        }

        // Solve Gamma on CPU (tiny matrix)
        std::vector<double> Gamma(cols, 0.0);
        {
            std::vector<double> A(h_FtF);
            std::vector<double> b(h_Ftf);
            for (int k = 0; k < cols; k++) {
                int pivot = k;
                for (int i = k+1; i < cols; i++)
                    if (std::abs(A[i*cols+k]) > std::abs(A[pivot*cols+k])) pivot = i;
                if (pivot != k) {
                    for (int j = 0; j < cols; j++) std::swap(A[k*cols+j], A[pivot*cols+j]);
                    std::swap(b[k], b[pivot]);
                }
                double d = A[k*cols+k];
                if (std::abs(d) < 1e-14) continue;
                for (int i = k+1; i < cols; i++) {
                    double fac = A[i*cols+k] / d;
                    for (int j = k+1; j < cols; j++) A[i*cols+j] -= fac * A[k*cols+j];
                    b[i] -= fac * b[k];
                }
            }
            for (int k = cols-1; k >= 0; k--) {
                if (std::abs(A[k*cols+k]) < 1e-14) continue;
                Gamma[k] = b[k];
                for (int j = k+1; j < cols; j++) Gamma[k] -= A[k*cols+j] * Gamma[j];
                Gamma[k] /= A[k*cols+k];
            }
        }

        // x_wavg = x - R * Gamma, f_wavg = f_k - F * Gamma
        CUDA_CHECK(cudaMemcpyAsync(d_x_wavg, x_k_inout, Nd * sizeof(double),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_f_wavg, d_fk, Nd * sizeof(double),
                                   cudaMemcpyDeviceToDevice, stream));

        for (int j = 0; j < cols; j++) {
            double neg_gj = -Gamma[j];
            cublasDaxpy(ctx.cublas, Nd, &neg_gj, d_R + j * Nd, 1, d_x_wavg, 1);
            cublasDaxpy(ctx.cublas, Nd, &neg_gj, d_F + j * Nd, 1, d_f_wavg, 1);
        }
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_x_wavg, x_k_inout, Nd * sizeof(double),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_f_wavg, d_fk, Nd * sizeof(double),
                                   cudaMemcpyDeviceToDevice, stream));
    }

    // Kerker preconditioner
    double* d_Pf = sp.alloc<double>(Nd);
    CUDA_CHECK(cudaMemsetAsync(d_Pf, 0, Nd * sizeof(double), stream));

    // Set callback pointer
    s_mixer_state_ = gs;

    // Apply Kerker to first Nd_kerker elements
    {
        double* d_f_col = d_f_wavg;
        double* d_Pf_col = d_Pf;

        // Step 1: Lf = (Lap - idiemac*kTF^2) * f_wavg
        double* d_Lf = sp.alloc<double>(Nd_kerker);
        {
            constexpr double idiemac_kTF2 = 0.1;
            int nx_ex = gs->nx + 2 * gs->FDn, ny_ex = gs->ny + 2 * gs->FDn;
            gpu::halo_exchange_gpu(d_f_col, ctx.buf.aar_x_ex,
                gs->nx, gs->ny, gs->nz, gs->FDn, 1, true, true, true, stream);
            if (gs->is_orth) {
                gpu::laplacian_orth_v7_gpu(ctx.buf.aar_x_ex, nullptr, d_Lf,
                    gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex,
                    1.0, 0.0, -idiemac_kTF2, gs->kerker_rhs_diag, 1, stream);
            } else {
                gpu::laplacian_nonorth_gpu(ctx.buf.aar_x_ex, nullptr, d_Lf,
                    gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex,
                    1.0, 0.0, -idiemac_kTF2, gs->kerker_rhs_diag,
                    gs->has_mixed_deriv, gs->has_mixed_deriv, gs->has_mixed_deriv, 1, stream);
            }
        }

        // Step 2: Solve (-Lap + kTF^2)*Pf = Lf via AAR
        {
            double* d_kr    = sp.alloc<double>(Nd_kerker);
            double* d_kf    = sp.alloc<double>(Nd_kerker);
            double* d_kAx   = sp.alloc<double>(Nd_kerker);
            double* d_kX    = sp.alloc<double>(Nd_kerker * 7);
            double* d_kF    = sp.alloc<double>(Nd_kerker * 7);
            double* d_kxold = sp.alloc<double>(Nd_kerker);
            double* d_kfold = sp.alloc<double>(Nd_kerker);

            gpu::aar_gpu(
                mixer_kerker_op_cb, mixer_kerker_precond_cb,
                d_Lf, d_Pf_col, Nd_kerker,
                0.6, 0.6, 7, 6, gs->precond_tol, 1000,
                d_kr, d_kf, d_kAx, d_kX, d_kF, d_kxold, d_kfold, stream);
        }

        // Step 3: Pf *= -beta_mix
        {
            double neg_beta = -beta_mix;
            cublasDscal(ctx.cublas, Nd_kerker, &neg_beta, d_Pf_col, 1);
        }
    }

    // Simple mixing for magnetization part (if ncol > 1)
    if (Nd_kerker < Nd) {
        int Nd_mag = Nd - Nd_kerker;
        CUDA_CHECK(cudaMemcpyAsync(d_Pf + Nd_kerker, d_f_wavg + Nd_kerker,
                              Nd_mag * sizeof(double), cudaMemcpyDeviceToDevice, stream));
        cublasDscal(ctx.cublas, Nd_mag, &beta_mag, d_Pf + Nd_kerker, 1);
    }

    // Save x_km1
    CUDA_CHECK(cudaMemcpyAsync(d_xkm1, x_k_inout, Nd * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream));

    // x_{k+1} = x_wavg + Pf
    CUDA_CHECK(cudaMemcpyAsync(x_k_inout, d_x_wavg, Nd * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream));
    {
        double one = 1.0;
        cublasDaxpy(ctx.cublas, Nd, &one, d_Pf, 1, x_k_inout, 1);
    }

    sp.restore(sp_cp);
    gs->mix_iter++;
}

} // namespace lynx

#endif // USE_CUDA
