#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <vector>
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
__global__ void jacobi_scale_kernel(
    const double* __restrict__ r,
    double* __restrict__ f,
    double scale, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) f[i] = scale * r[i];
}

// r[i] = b[i] - Ax[i]
__global__ void residual_kernel(
    const double* __restrict__ b,
    const double* __restrict__ Ax,
    double* __restrict__ r, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) r[i] = b[i] - Ax[i];
}

// x[i] = x_old[i] + omega * f[i]
__global__ void richardson_kernel(
    const double* __restrict__ x_old,
    const double* __restrict__ f,
    double* __restrict__ x,
    double omega, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = x_old[i] + omega * f[i];
}

// Store history: X(:,col) = x - x_old, F(:,col) = f - f_old
__global__ void store_history_kernel(
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
__global__ void anderson_kernel(
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
// GPUMixerState — grid/Kerker parameters for kernel wrappers
// ============================================================

struct GPUMixerState {
    // Grid parameters (needed for Kerker preconditioner)
    int nx = 0, ny = 0, nz = 0, FDn = 0, Nd = 0;
    bool is_orth = true;
    bool has_mixed_deriv = false;

    // Kerker preconditioner coefficients
    double kerker_diag = 0.0;
    double kerker_rhs_diag = 0.0;
    double kerker_m_inv = 0.0;
};

// ============================================================
// Setup / Cleanup
// ============================================================

void Mixer::setup_gpu(int Nd_d, int ncol, int m_depth, double beta_mix) {
    if (!gpu_state_raw_)
        gpu_state_raw_ = new GPUMixerState();
    auto* gs = static_cast<GPUMixerState*>(gpu_state_raw_);

    gs->Nd = Nd_d;
    // m_depth, beta, ncol are stored in class members (m_, beta_)
    gpu_mix_iter_ = 0;
}

void Mixer::cleanup_gpu() {
    if (!gpu_state_raw_) return;
    auto* gs = static_cast<GPUMixerState*>(gpu_state_raw_);

    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    if (d_fkm1_) { cudaFreeAsync(d_fkm1_, stream); d_fkm1_ = nullptr; }

    delete gs;
    gpu_state_raw_ = nullptr;
}

Mixer::~Mixer() {
    cleanup_gpu();
}

// ============================================================
// GPU kernel wrapper methods (thin wrappers, no algorithm logic)
// ============================================================

void Mixer::mixer_residual_gpu(const double* d_g, const double* d_x, double* d_f, int N) {
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int bs = 256;
    mixer_residual_kernel<<<gpu::ceildiv(N, bs), bs, 0, stream>>>(d_g, d_x, d_f, N);
}

void Mixer::mixer_store_history_gpu(const double* d_x, const double* d_x_old,
                                     const double* d_f, const double* d_f_old,
                                     double* d_X_hist, double* d_F_hist, int col, int N) {
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int bs = 256;
    mixer_store_history_kernel<<<gpu::ceildiv(N, bs), bs, 0, stream>>>(
        d_x, d_x_old, d_f, d_f_old, d_X_hist, d_F_hist, col, N);
}

void Mixer::kerker_apply_op_gpu(const double* d_x, double* d_Ax, int Nd) {
    auto* gs = static_cast<GPUMixerState*>(gpu_state_raw_);
    auto& ctx = gpu::GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;

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
}

void Mixer::kerker_apply_rhs_gpu(const double* d_f, double* d_Lf, int Nd) {
    auto* gs = static_cast<GPUMixerState*>(gpu_state_raw_);
    auto& ctx = gpu::GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;

    constexpr double idiemac_kTF2 = 0.1;
    int nx_ex = gs->nx + 2 * gs->FDn, ny_ex = gs->ny + 2 * gs->FDn;
    gpu::halo_exchange_gpu(d_f, ctx.buf.aar_x_ex,
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

void Mixer::kerker_precondition_gpu(const double* d_r, double* d_f, int N) {
    auto* gs = static_cast<GPUMixerState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int bs = 256;
    jacobi_scale_kernel<<<gpu::ceildiv(N, bs), bs, 0, stream>>>(d_r, d_f, gs->kerker_m_inv, N);
}

void Mixer::aar_residual_gpu(const double* d_b, const double* d_Ax, double* d_r, int N) {
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int bs = 256;
    residual_kernel<<<gpu::ceildiv(N, bs), bs, 0, stream>>>(d_b, d_Ax, d_r, N);
}

void Mixer::aar_richardson_gpu(const double* d_x_old, const double* d_f, double* d_x, double omega, int N) {
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int bs = 256;
    richardson_kernel<<<gpu::ceildiv(N, bs), bs, 0, stream>>>(d_x_old, d_f, d_x, omega, N);
}

void Mixer::aar_store_history_gpu(const double* d_x, const double* d_x_old,
                                   const double* d_f, const double* d_f_old,
                                   double* d_X_hist, double* d_F_hist, int col, int N) {
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int bs = 256;
    store_history_kernel<<<gpu::ceildiv(N, bs), bs, 0, stream>>>(
        d_x, d_x_old, d_f, d_f_old, d_X_hist, d_F_hist, col, N);
}

void Mixer::aar_anderson_gpu(const double* d_x_old, const double* d_f,
                              const double* d_X_hist, const double* d_F_hist,
                              const double* d_gamma, double* d_x,
                              double beta, int cols, int N) {
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int bs = 256;
    anderson_kernel<<<gpu::ceildiv(N, bs), bs, 0, stream>>>(
        d_x_old, d_f, d_X_hist, d_F_hist, d_gamma, d_x, beta, cols, N);
}

void Mixer::aar_fused_gram_gpu(const double* d_F_hist, const double* d_f,
                                double* d_gram_out, const int* d_pair_i,
                                const int* d_pair_j, int N, int cols, int n_jobs) {
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int gram_bs = std::min(256, N);
    fused_gram_kernel<<<n_jobs, gram_bs, gram_bs * sizeof(double), stream>>>(
        d_F_hist, d_f, d_gram_out, d_pair_i, d_pair_j, N, cols, n_jobs);
}

double Mixer::aar_norm2_gpu(const double* d_r, int N) {
    auto& ctx = gpu::GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;
    auto& sp = ctx.scratch_pool;
    size_t cp = sp.checkpoint();
    double* d_norm2 = sp.alloc<double>(1);
    int bs = 256;
    norm2_kernel<<<1, bs, bs * sizeof(double), stream>>>(d_r, d_norm2, N);
    double result;
    CUDA_CHECK(cudaMemcpyAsync(&result, d_norm2, sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    sp.restore(cp);
    return result;
}

void Mixer::aar_copy_gpu(double* dst, const double* src, int N) {
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    CUDA_CHECK(cudaMemcpyAsync(dst, src, N * sizeof(double), cudaMemcpyDeviceToDevice, stream));
}

} // namespace lynx

#endif // USE_CUDA
