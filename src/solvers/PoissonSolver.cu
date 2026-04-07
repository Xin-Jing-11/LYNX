#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include "core/gpu_common.cuh"
#include "core/GPUContext.cuh"
#include "core/LynxContext.hpp"
#include "solvers/PoissonSolver.hpp"
#include "core/NumericalMethods.hpp"
#include "parallel/HaloExchange.cuh"
#include "operators/Laplacian.cuh"

namespace lynx {

// ============================================================
// GPU kernels (file-static)
// ============================================================

namespace {

// r[i] = b[i] - Ax[i]
__global__ void residual_kernel(
    const double* __restrict__ b,
    const double* __restrict__ Ax,
    double* __restrict__ r,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) r[idx] = b[idx] - Ax[idx];
}

// x[i] = x_old[i] + omega * f[i]
__global__ void richardson_kernel(
    const double* __restrict__ x_old,
    const double* __restrict__ f,
    double* __restrict__ x,
    double omega, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) x[idx] = x_old[idx] + omega * f[idx];
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        X_hist[col * N + idx] = x[idx] - x_old[idx];
        F_hist[col * N + idx] = f[idx] - f_old[idx];
    }
}

// Anderson extrapolation:
// x[i] = x_old[i] + beta*f[i] - sum_j gamma[j]*(X(i,j) + beta*F(i,j))
__global__ void anderson_kernel(
    const double* __restrict__ x_old,
    const double* __restrict__ f,
    const double* __restrict__ X_hist,
    const double* __restrict__ F_hist,
    const double* __restrict__ gamma,
    double* __restrict__ x,
    double beta, int cols, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double val = x_old[idx] + beta * f[idx];
        for (int j = 0; j < cols; ++j) {
            val -= gamma[j] * (X_hist[j * N + idx] + beta * F_hist[j * N + idx]);
        }
        x[idx] = val;
    }
}

// Fused norm^2 kernel
__global__ void norm2_kernel(
    const double* __restrict__ r,
    double* __restrict__ d_norm2,
    int N)
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

// Fused Gram matrix kernel
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

// f[i] = scale * r[i]
__global__ void jacobi_scale_kernel(
    const double* __restrict__ r,
    double* __restrict__ f,
    double scale, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) f[i] = scale * r[i];
}

// x[i] -= mean
__global__ void mean_subtract_kernel(double* __restrict__ x, double mean, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] -= mean;
}

} // anonymous namespace

// ============================================================
// GPUPoissonState — grid parameters for laplacian dispatch
// ============================================================

struct GPUPoissonState {
    int nx = 0, ny = 0, nz = 0, FDn = 0, Nd = 0;
    bool is_orth = true;
    bool has_mixed_deriv = false;
    double poisson_diag = 0.0;
    double jacobi_m_inv = 0.0;
    // Scratch pool checkpoint for per-solve allocations
    mutable size_t solve_sp_cp = 0;
};

// ============================================================
// Setup / Cleanup
// ============================================================

void PoissonSolver::setup_gpu(const LynxContext& ctx) {
    if (!gpu_state_)
        gpu_state_.reset(new GPUPoissonState());
    auto* gs = gpu_state_.as<GPUPoissonState>();

    const auto& grid    = ctx.grid();
    const auto& domain  = ctx.domain();
    const auto& stencil = ctx.stencil();

    gs->nx  = grid.Nx();
    gs->ny  = grid.Ny();
    gs->nz  = grid.Nz();
    gs->FDn = stencil.FDn();
    gs->Nd  = domain.Nd_d();
    gs->is_orth         = grid.lattice().is_orthogonal();
    gs->has_mixed_deriv = !gs->is_orth;

    const double* D2x = stencil.D2_coeff_x();
    const double* D2y = stencil.D2_coeff_y();
    const double* D2z = stencil.D2_coeff_z();
    double D2sum = D2x[0] + D2y[0] + D2z[0];

    gs->poisson_diag = -1.0 * D2sum;
    gs->jacobi_m_inv = -1.0 / D2sum;
}

void PoissonSolver::cleanup_gpu() {
    gpu_state_.reset();
}

PoissonSolver::~PoissonSolver() {
    cleanup_gpu();
}

// ============================================================
// GPU workspace allocation for a single solve call
// ============================================================

void PoissonSolver::alloc_gpu_scratch(AARWorkspace& ws, int Nd) const {
    auto& ctx = gpu::GPUContext::instance();
    auto& sp = ctx.scratch_pool;
    auto* gs = gpu_state_.as<GPUPoissonState>();
    gs->solve_sp_cp = sp.checkpoint();

    // Use pre-allocated AAR buffers from GPUContext for main arrays
    ws.r      = ctx.buf.aar_r;
    ws.f      = ctx.buf.aar_f;
    ws.Ax     = ctx.buf.aar_Ax;
    ws.X_hist = ctx.buf.aar_X;
    ws.F_hist = ctx.buf.aar_F;

    // Allocate per-solve temporaries from scratch pool
    ws.x_old  = sp.alloc<double>(Nd);
    ws.f_old  = sp.alloc<double>(Nd);
    ws.rhs_ms = sp.alloc<double>(Nd);
}

void PoissonSolver::free_gpu_scratch() const {
    auto& sp = gpu::GPUContext::instance().scratch_pool;
    auto* gs = gpu_state_.as<GPUPoissonState>();
    sp.restore(gs->solve_sp_cp);
}

// ============================================================
// GPU method implementations — thin kernel wrappers only
// ============================================================

void PoissonSolver::apply_laplacian_gpu(const double* d_x, double* d_Ax) const {
    auto* gs = gpu_state_.as<GPUPoissonState>();
    auto& ctx = gpu::GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;

    gpu::halo_exchange_gpu(d_x, ctx.buf.aar_x_ex,
        gs->nx, gs->ny, gs->nz, gs->FDn, 1, true, true, true, stream);
    int nx_ex = gs->nx + 2 * gs->FDn, ny_ex = gs->ny + 2 * gs->FDn;
    if (gs->is_orth) {
        gpu::laplacian_orth_v7_gpu(ctx.buf.aar_x_ex, nullptr, d_Ax,
            gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex,
            -1.0, 0.0, 0.0, gs->poisson_diag, 1, stream);
    } else {
        gpu::laplacian_nonorth_gpu(ctx.buf.aar_x_ex, nullptr, d_Ax,
            gs->nx, gs->ny, gs->nz, gs->FDn, nx_ex, ny_ex,
            -1.0, 0.0, 0.0, gs->poisson_diag,
            gs->has_mixed_deriv, gs->has_mixed_deriv, gs->has_mixed_deriv, 1, stream);
    }
}

void PoissonSolver::apply_preconditioner_gpu(const double* d_r, double* d_f, int N) const {
    auto* gs = gpu_state_.as<GPUPoissonState>();
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int bs = 256;
    jacobi_scale_kernel<<<gpu::ceildiv(N, bs), bs, 0, stream>>>(
        d_r, d_f, gs->jacobi_m_inv, N);
}

void PoissonSolver::compute_residual_gpu(const double* d_b, const double* d_Ax, double* d_r, int N) const {
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int bs = 256;
    residual_kernel<<<gpu::ceildiv(N, bs), bs, 0, stream>>>(d_b, d_Ax, d_r, N);
}

void PoissonSolver::richardson_update_gpu(const double* d_x_old, const double* d_f, double* d_x, double omega, int N) const {
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int bs = 256;
    richardson_kernel<<<gpu::ceildiv(N, bs), bs, 0, stream>>>(d_x_old, d_f, d_x, omega, N);
}

void PoissonSolver::store_history_gpu(const double* d_x, const double* d_x_old,
                                       const double* d_f, const double* d_f_old,
                                       double* d_X_hist, double* d_F_hist, int col, int N) const {
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int bs = 256;
    store_history_kernel<<<gpu::ceildiv(N, bs), bs, 0, stream>>>(
        d_x, d_x_old, d_f, d_f_old, d_X_hist, d_F_hist, col, N);
}

double PoissonSolver::compute_norm2_gpu(const double* d_r, int N) const {
    auto& ctx = gpu::GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;
    auto& sp = ctx.scratch_pool;
    size_t cp = sp.checkpoint();
    double* d_norm2 = sp.alloc<double>(1);
    int bs = 256;
    norm2_kernel<<<1, bs, bs * sizeof(double), stream>>>(d_r, d_norm2, N);
    double result;
    cudaMemcpyAsync(&result, d_norm2, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    sp.restore(cp);
    return result;
}

void PoissonSolver::vec_copy_gpu(double* dst, const double* src, int N) const {
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    cudaMemcpyAsync(dst, src, N * sizeof(double), cudaMemcpyDeviceToDevice, stream);
}

void PoissonSolver::subtract_mean_gpu(double* d_x, double mean, int N) const {
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int bs = 256;
    mean_subtract_kernel<<<gpu::ceildiv(N, bs), bs, 0, stream>>>(d_x, mean, N);
}

double PoissonSolver::compute_mean_gpu(const double* d_x, int N) const {
    auto& ctx = gpu::GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;
    std::vector<double> h(N);
    cudaMemcpyAsync(h.data(), d_x, N * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    double sum = 0.0;
    for (int i = 0; i < N; i++) sum += h[i];
    return sum / N;
}

// Full Anderson step on GPU: fused Gram matrix computation, CPU solve, extrapolation.
// d_Ax_scratch is reused for Gram pair indices and gamma upload (it will be
// recomputed after the Anderson step in the main loop).
void PoissonSolver::anderson_step_gpu(const double* d_x_old, const double* d_f,
                                       const double* d_X_hist, const double* d_F_hist,
                                       double* d_Ax_scratch, double* d_x,
                                       double beta, int cols, int N) const {
    auto& ctx = gpu::GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;
    int m = aar_params_.m;

    // Reuse d_Ax_scratch for pair indices and gram output
    int max_jobs = m * (m + 1) / 2 + m;
    int* d_pair_i     = reinterpret_cast<int*>(d_Ax_scratch);
    int* d_pair_j     = d_pair_i + max_jobs;
    double* d_gram_out = reinterpret_cast<double*>(d_pair_j + max_jobs);

    std::vector<int> h_pair_i(max_jobs), h_pair_j(max_jobs);
    std::vector<double> h_gram_out(max_jobs);
    std::vector<double> h_FTF(m * m);
    std::vector<double> h_gamma(m);

    // Build pair list for fused Gram kernel
    int n_jobs = 0;
    for (int ii = 0; ii < cols; ++ii)
        for (int jj = 0; jj <= ii; ++jj) {
            h_pair_i[n_jobs] = ii; h_pair_j[n_jobs] = jj; n_jobs++;
        }
    int ftf_pairs = n_jobs;
    for (int ii = 0; ii < cols; ++ii) {
        h_pair_i[n_jobs] = ii; h_pair_j[n_jobs] = -1; n_jobs++;
    }

    // Upload pair indices
    cudaMemcpyAsync(d_pair_i, h_pair_i.data(), n_jobs * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_pair_j, h_pair_j.data(), n_jobs * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Launch fused Gram kernel
    int gram_bs = std::min(256, N);
    fused_gram_kernel<<<n_jobs, gram_bs, gram_bs * sizeof(double), stream>>>(
        d_F_hist, d_f, d_gram_out, d_pair_i, d_pair_j, N, cols, n_jobs);

    // Download Gram results
    cudaMemcpyAsync(h_gram_out.data(), d_gram_out, n_jobs * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Build symmetric FTF and RHS on CPU
    int k = 0;
    for (int ii = 0; ii < cols; ++ii)
        for (int jj = 0; jj <= ii; ++jj) {
            h_FTF[ii * cols + jj] = h_gram_out[k];
            h_FTF[jj * cols + ii] = h_gram_out[k];
            k++;
        }
    for (int ii = 0; ii < cols; ++ii)
        h_gamma[ii] = h_gram_out[ftf_pairs + ii];

    // Solve (F^T F) gamma = F^T f on CPU
    gauss_solve(h_FTF.data(), h_gamma.data(), h_gamma.data(), cols);

    // Upload gamma to device (reuse d_Ax_scratch)
    double* d_gamma = d_Ax_scratch;
    cudaMemcpyAsync(d_gamma, h_gamma.data(), cols * sizeof(double), cudaMemcpyHostToDevice, stream);

    // Launch Anderson extrapolation kernel
    int bs = 256;
    anderson_kernel<<<gpu::ceildiv(N, bs), bs, 0, stream>>>(
        d_x_old, d_f, d_X_hist, d_F_hist, d_gamma, d_x, beta, cols, N);
}

} // namespace lynx

#endif // USE_CUDA
