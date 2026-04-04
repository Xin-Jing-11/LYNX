#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include "core/gpu_common.cuh"
#include "core/GPUContext.cuh"
#include "solvers/Mixer.hpp"
#include "parallel/HaloExchange.cuh"
#include "operators/Laplacian.cuh"

namespace lynx {

// ============================================================
// GPU kernels (file-static)
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

// AAR kernels for Kerker inner solve

// r[i] = b[i] - Ax[i]
__global__ void aar_residual_kernel(
    const double* __restrict__ b,
    const double* __restrict__ Ax,
    double* __restrict__ r, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) r[i] = b[i] - Ax[i];
}

// x[i] = x_old[i] + omega * f[i]
__global__ void aar_richardson_kernel(
    const double* __restrict__ x_old,
    const double* __restrict__ f,
    double* __restrict__ x,
    double omega, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = x_old[i] + omega * f[i];
}

// X(:,col) = x - x_old, F(:,col) = f - f_old
__global__ void aar_store_history_kernel(
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

// Anderson extrapolation
__global__ void aar_anderson_kernel(
    const double* __restrict__ x_old,
    const double* __restrict__ f,
    const double* __restrict__ X_hist,
    const double* __restrict__ F_hist,
    const double* __restrict__ gamma,
    double* __restrict__ x,
    double beta, int cols, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double val = x_old[i] + beta * f[i];
        for (int j = 0; j < cols; ++j)
            val -= gamma[j] * (X_hist[j * N + i] + beta * F_hist[j * N + i]);
        x[i] = val;
    }
}

__global__ void norm2_kernel(
    const double* __restrict__ r,
    double* __restrict__ d_norm2, int N)
{
    extern __shared__ double sdata[];
    double sum = 0.0;
    for (int idx = threadIdx.x; idx < N; idx += blockDim.x)
        sum += r[idx] * r[idx];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) *d_norm2 = sdata[0];
}

__global__ void fused_gram_kernel(
    const double* __restrict__ F_hist,
    const double* __restrict__ f,
    double* __restrict__ d_out,
    const int* __restrict__ d_pair_i,
    const int* __restrict__ d_pair_j,
    int N, int cols, int n_jobs)
{
    int job = blockIdx.x;
    if (job >= n_jobs) return;
    int ci = d_pair_i[job];
    int cj = d_pair_j[job];
    const double* a = F_hist + ci * N;
    const double* b = (cj >= 0) ? (F_hist + cj * N) : f;
    extern __shared__ double sdata[];
    double sum = 0.0;
    for (int idx = threadIdx.x; idx < N; idx += blockDim.x)
        sum += a[idx] * b[idx];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) d_out[job] = sdata[0];
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

    // Persistent buffer for previous residual
    double* d_fkm1  = nullptr;
};

// ============================================================
// Kerker inner AAR solve (no callbacks, direct kernel calls)
// Solves (-Lap + kTF^2)*Pf = Lf
// ============================================================
static void kerker_aar_solve_gpu(
    GPUMixerState* gs,
    const double* d_Lf,   // RHS
    double* d_Pf,          // solution (in/out)
    int Nd_kerker,
    cudaStream_t stream)
{
    auto& ctx = gpu::GPUContext::instance();
    auto& sp = ctx.scratch_pool;
    size_t sp_cp = sp.checkpoint();

    double* d_kr    = sp.alloc<double>(Nd_kerker);
    double* d_kf    = sp.alloc<double>(Nd_kerker);
    double* d_kAx   = sp.alloc<double>(Nd_kerker);
    double* d_kX    = sp.alloc<double>(Nd_kerker * 7);
    double* d_kF    = sp.alloc<double>(Nd_kerker * 7);
    double* d_kxold = sp.alloc<double>(Nd_kerker);
    double* d_kfold = sp.alloc<double>(Nd_kerker);

    int bs = 256;
    int grid_sz = gpu::ceildiv(Nd_kerker, bs);
    int m = 7, p = 6;
    double omega = 0.6, beta_aar = 0.6;
    double tol = gs->precond_tol;
    int max_iter = 1000;

    // x_old = x (Pf)
    CUDA_CHECK(cudaMemcpyAsync(d_kxold, d_Pf, Nd_kerker * sizeof(double), cudaMemcpyDeviceToDevice, stream));

    // ||b||
    double* d_norm2 = sp.alloc<double>(1);
    norm2_kernel<<<1, 256, 256 * sizeof(double), stream>>>(d_Lf, d_norm2, Nd_kerker);
    double b_norm2;
    CUDA_CHECK(cudaMemcpyAsync(&b_norm2, d_norm2, sizeof(double), cudaMemcpyDeviceToHost, stream));
    double abs_tol = tol * std::sqrt(b_norm2);

    // Kerker operator: (-Lap + kTF^2)*x
    auto apply_kerker_op = [&](const double* d_x, double* d_Ax) {
        gpu::halo_exchange_gpu(d_x, ctx.buf.aar_x_ex,
            gs->nx, gs->ny, gs->nz, gs->FDn, 1, true, true, true, stream);
        int nx_ex = gs->nx + 2 * gs->FDn, ny_ex = gs->ny + 2 * gs->FDn;
        constexpr double kTF2 = 1.0;
        if (gs->is_orth) {
            gpu::laplacian_orth_v7_gpu(ctx.buf.aar_x_ex, nullptr, d_Ax,
                gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex,
                -1.0, 0.0, kTF2, gs->kerker_diag, 1, stream);
        } else {
            gpu::laplacian_nonorth_gpu(ctx.buf.aar_x_ex, nullptr, d_Ax,
                gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex,
                -1.0, 0.0, kTF2, gs->kerker_diag,
                gs->has_mixed_deriv, gs->has_mixed_deriv, gs->has_mixed_deriv, 1, stream);
        }
    };

    // Initial residual
    apply_kerker_op(d_Pf, d_kAx);
    aar_residual_kernel<<<grid_sz, bs, 0, stream>>>(d_Lf, d_kAx, d_kr, Nd_kerker);

    double r_2norm = abs_tol + 1.0;
    int iter_count = 0;

    // Host buffers for Gram solve
    int max_jobs = m * (m + 1) / 2 + m;
    std::vector<double> h_FTF(m * m);
    std::vector<double> h_gamma(m);
    std::vector<int> h_pair_i(max_jobs), h_pair_j(max_jobs);
    std::vector<double> h_gram_out(max_jobs);

    while (r_2norm > abs_tol && iter_count < max_iter) {
        // Precondition: f = M^{-1} * r
        mixer_jacobi_scale_kernel<<<grid_sz, bs, 0, stream>>>(d_kr, d_kf, gs->kerker_m_inv, Nd_kerker);

        // Store history
        if (iter_count > 0) {
            int i_hist = (iter_count - 1) % m;
            aar_store_history_kernel<<<grid_sz, bs, 0, stream>>>(
                d_Pf, d_kxold, d_kf, d_kfold, d_kX, d_kF, i_hist, Nd_kerker);
        }

        CUDA_CHECK(cudaMemcpyAsync(d_kxold, d_Pf, Nd_kerker * sizeof(double), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_kfold, d_kf, Nd_kerker * sizeof(double), cudaMemcpyDeviceToDevice, stream));

        if ((iter_count + 1) % p == 0 && iter_count > 0) {
            int cols = std::min(iter_count, m);

            // Fused Gram kernel
            int* d_pi = reinterpret_cast<int*>(d_kAx);
            int* d_pj = d_pi + max_jobs;
            double* d_go = reinterpret_cast<double*>(d_pj + max_jobs);

            int n_jobs = 0;
            for (int ii = 0; ii < cols; ++ii)
                for (int jj = 0; jj <= ii; ++jj) {
                    h_pair_i[n_jobs] = ii; h_pair_j[n_jobs] = jj; n_jobs++;
                }
            int ftf_pairs = n_jobs;
            for (int ii = 0; ii < cols; ++ii) {
                h_pair_i[n_jobs] = ii; h_pair_j[n_jobs] = -1; n_jobs++;
            }

            CUDA_CHECK(cudaMemcpyAsync(d_pi, h_pair_i.data(), n_jobs * sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_pj, h_pair_j.data(), n_jobs * sizeof(int), cudaMemcpyHostToDevice, stream));
            int gram_bs = std::min(256, Nd_kerker);
            fused_gram_kernel<<<n_jobs, gram_bs, gram_bs * sizeof(double), stream>>>(
                d_kF, d_kf, d_go, d_pi, d_pj, Nd_kerker, cols, n_jobs);
            CUDA_CHECK(cudaMemcpyAsync(h_gram_out.data(), d_go, n_jobs * sizeof(double), cudaMemcpyDeviceToHost, stream));

            int k = 0;
            for (int ii = 0; ii < cols; ++ii)
                for (int jj = 0; jj <= ii; ++jj) {
                    h_FTF[ii * cols + jj] = h_gram_out[k];
                    h_FTF[jj * cols + ii] = h_gram_out[k];
                    k++;
                }
            for (int ii = 0; ii < cols; ++ii)
                h_gamma[ii] = h_gram_out[ftf_pairs + ii];

            // Gaussian elimination
            {
                std::vector<double> A(h_FTF.begin(), h_FTF.begin() + cols * cols);
                for (int kk = 0; kk < cols; ++kk) {
                    int pivot = kk;
                    for (int ii = kk + 1; ii < cols; ++ii)
                        if (std::abs(A[ii * cols + kk]) > std::abs(A[pivot * cols + kk])) pivot = ii;
                    if (pivot != kk) {
                        for (int j = 0; j < cols; ++j) std::swap(A[kk * cols + j], A[pivot * cols + j]);
                        std::swap(h_gamma[kk], h_gamma[pivot]);
                    }
                    double d = A[kk * cols + kk];
                    if (std::abs(d) < 1e-14) continue;
                    for (int ii = kk + 1; ii < cols; ++ii) {
                        double factor = A[ii * cols + kk] / d;
                        for (int j = kk + 1; j < cols; ++j) A[ii * cols + j] -= factor * A[kk * cols + j];
                        h_gamma[ii] -= factor * h_gamma[kk];
                    }
                }
                for (int kk = cols - 1; kk >= 0; --kk) {
                    if (std::abs(A[kk * cols + kk]) < 1e-14) continue;
                    for (int j = kk + 1; j < cols; ++j) h_gamma[kk] -= A[kk * cols + j] * h_gamma[j];
                    h_gamma[kk] /= A[kk * cols + kk];
                }
            }

            double* d_gamma = d_kAx;
            CUDA_CHECK(cudaMemcpyAsync(d_gamma, h_gamma.data(), cols * sizeof(double), cudaMemcpyHostToDevice, stream));
            aar_anderson_kernel<<<grid_sz, bs, 0, stream>>>(
                d_kxold, d_kf, d_kX, d_kF, d_gamma, d_Pf, beta_aar, cols, Nd_kerker);

            // Recompute residual + convergence check
            apply_kerker_op(d_Pf, d_kAx);
            aar_residual_kernel<<<grid_sz, bs, 0, stream>>>(d_Lf, d_kAx, d_kr, Nd_kerker);
            norm2_kernel<<<1, 256, 256 * sizeof(double), stream>>>(d_kr, d_norm2, Nd_kerker);
            double r_norm2;
            CUDA_CHECK(cudaMemcpyAsync(&r_norm2, d_norm2, sizeof(double), cudaMemcpyDeviceToHost, stream));
            r_2norm = std::sqrt(r_norm2);
        } else {
            aar_richardson_kernel<<<grid_sz, bs, 0, stream>>>(d_kxold, d_kf, d_Pf, omega, Nd_kerker);
            apply_kerker_op(d_Pf, d_kAx);
            aar_residual_kernel<<<grid_sz, bs, 0, stream>>>(d_Lf, d_kAx, d_kr, Nd_kerker);
        }
        iter_count++;
    }

    sp.restore(sp_cp);
}

// ============================================================
// Setup / Cleanup
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
// GPU Pulay mixing — dispatched from mix() via mix_gpu()
// ============================================================

void Mixer::mix_gpu(double* x_k_inout, const double* g_k, int Nd_d, int ncol) {
    auto* gs = static_cast<GPUMixerState*>(gpu_state_raw_);
    auto& ctx = gpu::GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;

    int Nd = Nd_d * ncol;
    int m_depth = gs->m_depth;
    double beta_mix = gs->beta;
    int Nd_kerker = Nd_d;
    double beta_mag = beta_mix;

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

        // Build F^T*F and F^T*f_k via cuBLAS
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

        // Solve Gamma on CPU
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

        // Step 2: Solve (-Lap + kTF^2)*Pf = Lf via AAR (no callbacks)
        kerker_aar_solve_gpu(gs, d_Lf, d_Pf_col, Nd_kerker, stream);

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
