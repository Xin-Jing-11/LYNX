#include "solvers/PoissonSolver.hpp"
#include "core/NumericalMethods.hpp"
#include "parallel/MPIComm.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdio>

#ifdef USE_CUDA
#include "core/GPUContext.cuh"
#endif

namespace lynx {

#ifndef USE_CUDA
PoissonSolver::~PoissonSolver() = default;
#endif

void PoissonSolver::setup(const Laplacian& laplacian,
                            const FDStencil& stencil,
                            const Domain& domain,
                            const FDGrid& grid,
                            const HaloExchange& halo) {
    laplacian_ = &laplacian;
    stencil_ = &stencil;
    domain_ = &domain;
    grid_ = &grid;
    halo_ = &halo;

    // Jacobi preconditioner weight: m_inv = -1 / (D2x[0] + D2y[0] + D2z[0])
    double diag = stencil.D2_coeff_x()[0] + stencil.D2_coeff_y()[0] + stencil.D2_coeff_z()[0];
    jacobi_weight_ = (std::abs(diag) < 1e-14) ? 1.0 : (-1.0 / diag);

    // AAR params: reference uses omega=0.6, beta=0.6
    aar_params_.omega = 0.6;
    aar_params_.beta = 0.6;
    aar_params_.m = 7;
    aar_params_.p = 6;
    aar_params_.max_iter = 3000;
}

// ===================================================================
// Unified solve: ONE AAR loop, dispatches sub-ops to CPU or GPU
// ===================================================================

int PoissonSolver::solve(const double* rhs, double* phi, double tol) const {
    int Nd = domain_->Nd_d();
    int m = aar_params_.m;
    int p = aar_params_.p;
    double omega = aar_params_.omega;
    double beta = aar_params_.beta;
    int max_iter = aar_params_.max_iter;

    if (dev_ == Device::CPU) {
        // ---- CPU path: use existing LinearSolver::aar ----
        AARParams params = aar_params_;
        params.tol = tol;

        auto op = [this, Nd](const double* x, double* Ax) {
            apply_laplacian_cpu(x, Ax);
        };

        double m_inv = jacobi_weight_;
        LinearSolver::PrecondFunc jacobi = [m_inv, Nd](const double* r, double* f) {
            for (int i = 0; i < Nd; ++i)
                f[i] = m_inv * r[i];
        };

        MPIComm self_comm(MPI_COMM_SELF);
        return LinearSolver::aar(op, rhs, phi, Nd, params, self_comm, &jacobi);
    }

#ifdef USE_CUDA
    // ---- GPU path: AAR loop in .cpp, kernel launches via _gpu() methods ----
    auto& ctx = gpu::GPUContext::instance();
    auto& sp = ctx.scratch_pool;
    size_t sp_cp = sp.checkpoint();

    // Mean-subtract rhs (work on copy)
    double* d_rhs_ms = sp.alloc<double>(Nd);
    aar_copy_gpu(d_rhs_ms, rhs, Nd);
    {
        // Compute mean on CPU (small sync)
        double rhs_mean = 0.0;
        {
            std::vector<double> h(Nd);
            cudaStream_t stream = ctx.compute_stream;
            cudaMemcpyAsync(h.data(), rhs, Nd * sizeof(double), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            for (int i = 0; i < Nd; i++) rhs_mean += h[i];
            rhs_mean /= Nd;
        }
        mean_subtract_gpu(d_rhs_ms, rhs_mean, Nd);
    }

    // Workspace from pre-allocated SCF buffers
    double* d_r      = ctx.buf.aar_r;
    double* d_f      = ctx.buf.aar_f;
    double* d_Ax     = ctx.buf.aar_Ax;
    double* d_X_hist = ctx.buf.aar_X;
    double* d_F_hist = ctx.buf.aar_F;

    double* d_x_old  = sp.alloc<double>(Nd);
    double* d_f_old  = sp.alloc<double>(Nd);

    // x_old = x (phi)
    aar_copy_gpu(d_x_old, phi, Nd);

    // ||b||
    double b_norm2 = aar_norm2_gpu(d_rhs_ms, Nd);
    double abs_tol = tol * std::sqrt(b_norm2);

    // Initial residual: r = b - A*x
    apply_laplacian_gpu(phi, d_Ax);
    aar_residual_gpu(d_rhs_ms, d_Ax, d_r, Nd);

    double r_2norm = abs_tol + 1.0;
    int iter_count = 0;

    // Small host buffers for Gram solve
    std::vector<double> h_FTF(m * m);
    std::vector<double> h_gamma(m);

    while (r_2norm > abs_tol && iter_count < max_iter) {
        // Precondition: f = M^{-1} * r
        apply_preconditioner_gpu(d_r, d_f, Nd);

        // Store history
        if (iter_count > 0) {
            int i_hist = (iter_count - 1) % m;
            aar_store_history_gpu(phi, d_x_old, d_f, d_f_old,
                                  d_X_hist, d_F_hist, i_hist, Nd);
        }

        // Save current state
        aar_copy_gpu(d_x_old, phi, Nd);
        aar_copy_gpu(d_f_old, d_f, Nd);

        if ((iter_count + 1) % p == 0 && iter_count > 0) {
            // Anderson extrapolation
            int cols = std::min(iter_count, m);

            // Build Gram matrix using fused kernel approach:
            // Reuse d_Ax memory for pair indices and output (safe: we recompute Ax after)
            int max_jobs = m * (m + 1) / 2 + m;
            int* d_pair_i     = reinterpret_cast<int*>(d_Ax);
            int* d_pair_j     = d_pair_i + max_jobs;
            double* d_gram_out = reinterpret_cast<double*>(d_pair_j + max_jobs);

            std::vector<int> h_pair_i(max_jobs), h_pair_j(max_jobs);
            std::vector<double> h_gram_out(max_jobs);

            int n_jobs = 0;
            // Upper triangle of F^T*F
            for (int ii = 0; ii < cols; ++ii)
                for (int jj = 0; jj <= ii; ++jj) {
                    h_pair_i[n_jobs] = ii;
                    h_pair_j[n_jobs] = jj;
                    n_jobs++;
                }
            int ftf_pairs = n_jobs;
            // F^T*f entries
            for (int ii = 0; ii < cols; ++ii) {
                h_pair_i[n_jobs] = ii;
                h_pair_j[n_jobs] = -1;
                n_jobs++;
            }

            // Upload pair indices, launch fused Gram kernel, download results
            cudaStream_t stream = ctx.compute_stream;
            cudaMemcpyAsync(d_pair_i, h_pair_i.data(), n_jobs * sizeof(int), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_pair_j, h_pair_j.data(), n_jobs * sizeof(int), cudaMemcpyHostToDevice, stream);

            aar_fused_gram_gpu(d_F_hist, d_f, d_gram_out, d_pair_i, d_pair_j, Nd, cols, n_jobs);

            cudaMemcpyAsync(h_gram_out.data(), d_gram_out, n_jobs * sizeof(double), cudaMemcpyDeviceToHost, stream);

            // Unpack into h_FTF and h_gamma
            int k = 0;
            for (int ii = 0; ii < cols; ++ii)
                for (int jj = 0; jj <= ii; ++jj) {
                    h_FTF[ii * cols + jj] = h_gram_out[k];
                    h_FTF[jj * cols + ii] = h_gram_out[k];
                    k++;
                }
            for (int ii = 0; ii < cols; ++ii)
                h_gamma[ii] = h_gram_out[ftf_pairs + ii];

            // Solve (F^T*F) * gamma = F^T*f via Gaussian elimination (CPU, tiny matrix)
            gauss_solve(h_FTF.data(), h_gamma.data(), h_gamma.data(), cols);

            // Upload gamma and run Anderson kernel
            double* d_gamma = d_Ax;
            cudaMemcpyAsync(d_gamma, h_gamma.data(), cols * sizeof(double), cudaMemcpyHostToDevice, stream);

            aar_anderson_gpu(d_x_old, d_f, d_X_hist, d_F_hist,
                             d_gamma, phi, beta, cols, Nd);

            // Recompute residual + check convergence
            apply_laplacian_gpu(phi, d_Ax);
            aar_residual_gpu(d_rhs_ms, d_Ax, d_r, Nd);
            r_2norm = std::sqrt(aar_norm2_gpu(d_r, Nd));
        } else {
            // Richardson update
            aar_richardson_gpu(d_x_old, d_f, phi, omega, Nd);

            // Recompute residual
            apply_laplacian_gpu(phi, d_Ax);
            aar_residual_gpu(d_rhs_ms, d_Ax, d_r, Nd);
        }

        iter_count++;
    }

    // Mean-subtract phi
    {
        std::vector<double> h(Nd);
        cudaStream_t stream = ctx.compute_stream;
        cudaStreamSynchronize(stream);
        cudaMemcpyAsync(h.data(), phi, Nd * sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        double phi_mean = 0.0;
        for (int i = 0; i < Nd; i++) phi_mean += h[i];
        phi_mean /= Nd;
        mean_subtract_gpu(phi, phi_mean, Nd);
    }

    sp.restore(sp_cp);
    return iter_count;
#else
    (void)tol;
    throw std::runtime_error("PoissonSolver: GPU requested but USE_CUDA is off");
#endif
}

// ===================================================================
// Dispatchers
// ===================================================================

void PoissonSolver::apply_laplacian(const double* x, double* Ax) const {
#ifdef USE_CUDA
    if (dev_ == Device::GPU) { apply_laplacian_gpu(x, Ax); return; }
#endif
    apply_laplacian_cpu(x, Ax);
}

void PoissonSolver::apply_preconditioner(const double* r, double* f, int N) const {
#ifdef USE_CUDA
    if (dev_ == Device::GPU) { apply_preconditioner_gpu(r, f, N); return; }
#endif
    apply_preconditioner_cpu(r, f, N);
}

// ===================================================================
// CPU implementations
// ===================================================================

void PoissonSolver::apply_laplacian_cpu(const double* x, double* Ax) const {
    int nd_ex = halo_->nd_ex();
    std::vector<double> x_ex(nd_ex, 0.0);
    halo_->execute(x, x_ex.data(), 1);
    laplacian_->apply(x_ex.data(), Ax, -1.0, 0.0, 1);
}

void PoissonSolver::apply_preconditioner_cpu(const double* r, double* f, int N) const {
    double m_inv = jacobi_weight_;
    for (int i = 0; i < N; ++i)
        f[i] = m_inv * r[i];
}

} // namespace lynx
