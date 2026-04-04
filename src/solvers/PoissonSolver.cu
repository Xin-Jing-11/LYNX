#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include "core/gpu_common.cuh"
#include "core/GPUContext.cuh"
#include "core/LynxContext.hpp"
#include "solvers/PoissonSolver.hpp"
#include "solvers/LinearSolver.cuh"
#include "parallel/HaloExchange.cuh"
#include "operators/Laplacian.cuh"

namespace lynx {

// ============================================================
// Kernels duplicated from GPUSCF.cu (file-static)
// ============================================================

namespace {

// rhs = fourpi * (rho + pseudocharge)
__global__ void poisson_rhs_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ pseudocharge,
    double* __restrict__ rhs,
    double fourpi, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) rhs[i] = fourpi * (rho[i] + pseudocharge[i]);
}

// x[i] -= mean
__global__ void poisson_mean_subtract_kernel(double* __restrict__ x, double mean, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] -= mean;
}

// f[i] = scale * r[i]
__global__ void poisson_jacobi_scale_kernel(
    const double* __restrict__ r,
    double* __restrict__ f,
    double scale, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) f[i] = scale * r[i];
}

} // anonymous namespace

// ============================================================
// GPUPoissonState
// ============================================================

struct GPUPoissonState {
    // Grid parameters
    int nx = 0, ny = 0, nz = 0, FDn = 0, Nd = 0;
    bool is_orth = true;
    bool has_mixed_deriv = false;

    // Preconditioner coefficients
    double poisson_diag = 0.0;
    double jacobi_m_inv = 0.0;

    // AAR workspace (owned by this operator)
    double* d_r       = nullptr;  // Nd — residual
    double* d_f       = nullptr;  // Nd — preconditioned residual
    double* d_Ax      = nullptr;  // Nd — operator applied
    double* d_X_hist  = nullptr;  // Nd * m — iterate history
    double* d_F_hist  = nullptr;  // Nd * m — residual history
    double* d_x_old   = nullptr;  // Nd — previous iterate
    double* d_f_old   = nullptr;  // Nd — previous residual
    double* d_aar_x_ex= nullptr;  // nd_ex — halo workspace
    double* d_rhs_buf = nullptr;  // Nd — RHS construction buffer
    int m = 7;                    // AAR history depth
};

// Static instance pointer for callback trampolines
static GPUPoissonState* s_poisson_state_ = nullptr;

// ============================================================
// Static callbacks for AAR solver
// ============================================================

static void poisson_op_cb(const double* d_x, double* d_Ax) {
    auto* s = s_poisson_state_;
    auto& ctx = gpu::GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;

    gpu::halo_exchange_gpu(d_x, ctx.buf.aar_x_ex,
        s->nx, s->ny, s->nz, s->FDn, 1, true, true, true, stream);
    int nx_ex = s->nx + 2 * s->FDn, ny_ex = s->ny + 2 * s->FDn;
    if (s->is_orth) {
        gpu::laplacian_orth_v7_gpu(ctx.buf.aar_x_ex, nullptr, d_Ax,
            s->nx, s->ny, s->nz, s->FDn, nx_ex, ny_ex,
            -1.0, 0.0, 0.0, s->poisson_diag, 1, stream);
    } else {
        gpu::laplacian_nonorth_gpu(ctx.buf.aar_x_ex, nullptr, d_Ax,
            s->nx, s->ny, s->nz, s->FDn, nx_ex, ny_ex,
            -1.0, 0.0, 0.0, s->poisson_diag,
            s->has_mixed_deriv, s->has_mixed_deriv, s->has_mixed_deriv, 1, stream);
    }
}

static void poisson_precond_cb(const double* d_r, double* d_f) {
    auto* s = s_poisson_state_;
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    int bs = 256;
    poisson_jacobi_scale_kernel<<<gpu::ceildiv(s->Nd, bs), bs, 0, stream>>>(
        d_r, d_f, s->jacobi_m_inv, s->Nd);
}

// Helper: sum on CPU
static double poisson_gpu_sum(const double* d_x, int N) {
    std::vector<double> h(N);
    CUDA_CHECK(cudaMemcpy(h.data(), d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
    double s = 0;
    for (int i = 0; i < N; i++) s += h[i];
    return s;
}

// ============================================================
// Stubs
// ============================================================

void PoissonSolver::setup_gpu(const LynxContext& ctx) {
    if (!gpu_state_raw_)
        gpu_state_raw_ = new GPUPoissonState();
    auto* gs = static_cast<GPUPoissonState*>(gpu_state_raw_);

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

    int Nd = gs->Nd;
    int nx_ex = gs->nx + 2 * gs->FDn;
    int ny_ex = gs->ny + 2 * gs->FDn;
    int nz_ex = gs->nz + 2 * gs->FDn;
    size_t nd_ex = (size_t)nx_ex * ny_ex * nz_ex;
    int m = gs->m;

    CUDA_CHECK(cudaMalloc(&gs->d_r,        Nd * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gs->d_f,        Nd * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gs->d_Ax,       Nd * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gs->d_X_hist,   (size_t)Nd * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gs->d_F_hist,   (size_t)Nd * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gs->d_x_old,    Nd * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gs->d_f_old,    Nd * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gs->d_aar_x_ex, nd_ex * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gs->d_rhs_buf,  Nd * sizeof(double)));
}

void PoissonSolver::cleanup_gpu() {
    if (!gpu_state_raw_) return;
    auto* gs = static_cast<GPUPoissonState*>(gpu_state_raw_);

    auto safe_free = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };

    safe_free(gs->d_r);
    safe_free(gs->d_f);
    safe_free(gs->d_Ax);
    safe_free(gs->d_X_hist);
    safe_free(gs->d_F_hist);
    safe_free(gs->d_x_old);
    safe_free(gs->d_f_old);
    safe_free(gs->d_aar_x_ex);
    safe_free(gs->d_rhs_buf);

    delete gs;
    gpu_state_raw_ = nullptr;
}

PoissonSolver::~PoissonSolver() {
    cleanup_gpu();
}

// ============================================================
// Device-dispatching solve()
// ============================================================

int PoissonSolver::solve(const double* rhs, double* phi, double tol, Device dev) const {
    if (dev == Device::CPU) {
        return solve(rhs, phi, tol);
    }

    // GPU path — mirrors GPUSCF::gpu_poisson_solve()
    auto* gs = static_cast<GPUPoissonState*>(gpu_state_raw_);
    auto& ctx = gpu::GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;
    int Nd = gs->Nd;
    int bs = 256;
    int grid_sz = gpu::ceildiv(Nd, bs);

    // The rhs is already prepared (4*pi*(rho+b)) by the caller.
    // Mean-subtract rhs (we work on a copy via scratch pool)
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double rhs_mean = poisson_gpu_sum(rhs, Nd) / Nd;

    auto& sp = ctx.scratch_pool;
    size_t sp_cp = sp.checkpoint();

    double* d_rhs_ms = sp.alloc<double>(Nd);
    CUDA_CHECK(cudaMemcpyAsync(d_rhs_ms, rhs, Nd * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    poisson_mean_subtract_kernel<<<grid_sz, bs, 0, stream>>>(d_rhs_ms, rhs_mean, Nd);

    double* d_xold = sp.alloc<double>(Nd);
    double* d_fold = sp.alloc<double>(Nd);

    // Set up static callback pointer
    s_poisson_state_ = gs;

    int iters = gpu::aar_gpu(
        poisson_op_cb, poisson_precond_cb,
        d_rhs_ms, phi, Nd,
        0.6, 0.6, 7, 6, tol, 3000,
        ctx.buf.aar_r, ctx.buf.aar_f, ctx.buf.aar_Ax,
        ctx.buf.aar_X, ctx.buf.aar_F,
        d_xold, d_fold, stream);

    sp.restore(sp_cp);

    // Mean-subtract phi
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double phi_mean = poisson_gpu_sum(phi, Nd) / Nd;
    poisson_mean_subtract_kernel<<<grid_sz, bs, 0, stream>>>(phi, phi_mean, Nd);

    return iters;
}

} // namespace lynx

#endif // USE_CUDA
