#include "solvers/PoissonSolver.hpp"
#include "core/NumericalMethods.hpp"
#include "parallel/MPIComm.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>

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

    // --- Workspace allocation ---
    AARWorkspace ws;

    // CPU storage (unused on GPU path)
    std::vector<double> cpu_r, cpu_f, cpu_f_old, cpu_x_old, cpu_Ax;
    std::vector<double> cpu_X_hist, cpu_F_hist;

    if (dev_ == Device::CPU) {
        cpu_r.resize(Nd, 0.0);
        cpu_f.resize(Nd, 0.0);
        cpu_f_old.resize(Nd, 0.0);
        cpu_x_old.resize(Nd, 0.0);
        cpu_Ax.resize(Nd, 0.0);
        cpu_X_hist.resize(static_cast<size_t>(Nd) * m, 0.0);
        cpu_F_hist.resize(static_cast<size_t>(Nd) * m, 0.0);

        ws.r      = cpu_r.data();
        ws.f      = cpu_f.data();
        ws.f_old  = cpu_f_old.data();
        ws.x_old  = cpu_x_old.data();
        ws.Ax     = cpu_Ax.data();
        ws.X_hist = cpu_X_hist.data();
        ws.F_hist = cpu_F_hist.data();
        ws.rhs_ms = nullptr;  // CPU does not mean-subtract
    }
#ifdef USE_CUDA
    else {
        alloc_gpu_scratch(ws, Nd);
    }
#else
    else {
        throw std::runtime_error("PoissonSolver: GPU requested but USE_CUDA is off");
    }
#endif

    // --- Mean-subtract RHS on GPU path ---
    const double* b = rhs;  // effective RHS pointer
#ifdef USE_CUDA
    if (dev_ == Device::GPU) {
        vec_copy(ws.rhs_ms, rhs, Nd);
        double rhs_mean = compute_mean(rhs, Nd);
        subtract_mean(ws.rhs_ms, rhs_mean, Nd);
        b = ws.rhs_ms;
    }
#endif

    // --- Initialize ---
    vec_copy(ws.x_old, phi, Nd);

    double b_norm2 = compute_norm2(b, Nd);
    double abs_tol = tol * std::sqrt(b_norm2);

    // Initial residual: r = b - A*phi
    apply_laplacian(phi, ws.Ax);
    compute_residual(b, ws.Ax, ws.r, Nd);

    double r_2norm = abs_tol + 1.0;  // skip initial norm check (matches reference)
    int iter_count = 0;

    // --- AAR iteration loop ---
    while (r_2norm > abs_tol && iter_count < max_iter) {
        // Apply preconditioner: f = M^{-1} * r
        apply_preconditioner(ws.r, ws.f, Nd);

        // Store history: X(:,i_hist) = x - x_old, F(:,i_hist) = f - f_old
        if (iter_count > 0) {
            int i_hist = (iter_count - 1) % m;
            store_history(phi, ws.x_old, ws.f, ws.f_old,
                          ws.X_hist, ws.F_hist, i_hist, Nd);
        }

        // Save current state
        vec_copy(ws.x_old, phi, Nd);
        vec_copy(ws.f_old, ws.f, Nd);

        if ((iter_count + 1) % p == 0 && iter_count > 0) {
            // --- Anderson extrapolation step ---
            int cols = std::min(iter_count, m);

            // Compute gamma and apply Anderson extrapolation
            anderson_step(ws.x_old, ws.f, ws.X_hist, ws.F_hist,
                          ws.Ax, phi, beta, cols, Nd);

            // Recompute residual after Anderson step
            apply_laplacian(phi, ws.Ax);
            compute_residual(b, ws.Ax, ws.r, Nd);
            r_2norm = std::sqrt(compute_norm2(ws.r, Nd));
        } else {
            // --- Richardson update step ---
            richardson_update(ws.x_old, ws.f, phi, omega, Nd);

            // Recompute residual after Richardson step
            apply_laplacian(phi, ws.Ax);
            compute_residual(b, ws.Ax, ws.r, Nd);
        }

        iter_count++;
    }

    // --- Mean-subtract phi on GPU path ---
#ifdef USE_CUDA
    if (dev_ == Device::GPU) {
        double phi_mean = compute_mean(phi, Nd);
        subtract_mean(phi, phi_mean, Nd);
        free_gpu_scratch();
    }
#endif

    return iter_count;
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

void PoissonSolver::compute_residual(const double* b, const double* Ax, double* r, int N) const {
#ifdef USE_CUDA
    if (dev_ == Device::GPU) { compute_residual_gpu(b, Ax, r, N); return; }
#endif
    compute_residual_cpu(b, Ax, r, N);
}

void PoissonSolver::richardson_update(const double* x_old, const double* f, double* x, double omega, int N) const {
#ifdef USE_CUDA
    if (dev_ == Device::GPU) { richardson_update_gpu(x_old, f, x, omega, N); return; }
#endif
    richardson_update_cpu(x_old, f, x, omega, N);
}

void PoissonSolver::store_history(const double* x, const double* x_old,
                                   const double* f, const double* f_old,
                                   double* X_hist, double* F_hist, int col, int N) const {
#ifdef USE_CUDA
    if (dev_ == Device::GPU) { store_history_gpu(x, x_old, f, f_old, X_hist, F_hist, col, N); return; }
#endif
    store_history_cpu(x, x_old, f, f_old, X_hist, F_hist, col, N);
}

void PoissonSolver::anderson_step(const double* x_old, const double* f,
                                   const double* X_hist, const double* F_hist,
                                   double* Ax_scratch, double* x,
                                   double beta, int cols, int N) const {
#ifdef USE_CUDA
    if (dev_ == Device::GPU) {
        anderson_step_gpu(x_old, f, X_hist, F_hist, Ax_scratch, x, beta, cols, N);
        return;
    }
#endif
    anderson_step_cpu(x_old, f, X_hist, F_hist, x, beta, cols, N);
}

void PoissonSolver::vec_copy(double* dst, const double* src, int N) const {
#ifdef USE_CUDA
    if (dev_ == Device::GPU) { vec_copy_gpu(dst, src, N); return; }
#endif
    vec_copy_cpu(dst, src, N);
}

double PoissonSolver::compute_norm2(const double* r, int N) const {
#ifdef USE_CUDA
    if (dev_ == Device::GPU) { return compute_norm2_gpu(r, N); }
#endif
    return compute_norm2_cpu(r, N);
}

void PoissonSolver::subtract_mean(double* x, double mean, int N) const {
#ifdef USE_CUDA
    if (dev_ == Device::GPU) { subtract_mean_gpu(x, mean, N); return; }
#endif
    subtract_mean_cpu(x, mean, N);
}

double PoissonSolver::compute_mean(const double* x, int N) const {
#ifdef USE_CUDA
    if (dev_ == Device::GPU) { return compute_mean_gpu(x, N); }
#endif
    return compute_mean_cpu(x, N);
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

void PoissonSolver::compute_residual_cpu(const double* b, const double* Ax, double* r, int N) const {
    for (int i = 0; i < N; ++i)
        r[i] = b[i] - Ax[i];
}

void PoissonSolver::richardson_update_cpu(const double* x_old, const double* f, double* x, double omega, int N) const {
    for (int i = 0; i < N; ++i)
        x[i] = x_old[i] + omega * f[i];
}

void PoissonSolver::store_history_cpu(const double* x, const double* x_old,
                                       const double* f, const double* f_old,
                                       double* X_hist, double* F_hist, int col, int N) const {
    for (int i = 0; i < N; ++i) {
        X_hist[col * N + i] = x[i] - x_old[i];
        F_hist[col * N + i] = f[i] - f_old[i];
    }
}

void PoissonSolver::anderson_step_cpu(const double* x_old, const double* f,
                                       const double* X_hist, const double* F_hist,
                                       double* x, double beta, int cols, int N) const {
    // Compute Gram matrix F^T F and right-hand side F^T f
    std::vector<double> FTF(cols * cols, 0.0);
    std::vector<double> gamma(cols, 0.0);

    for (int ii = 0; ii < cols; ++ii) {
        const double* Fi = F_hist + ii * N;
        double dot_fi = 0.0;
        for (int k = 0; k < N; ++k) dot_fi += Fi[k] * f[k];
        gamma[ii] = dot_fi;

        for (int jj = 0; jj <= ii; ++jj) {
            const double* Fj = F_hist + jj * N;
            double dot_ij = 0.0;
            for (int k = 0; k < N; ++k) dot_ij += Fi[k] * Fj[k];
            FTF[ii * cols + jj] = dot_ij;
            FTF[jj * cols + ii] = dot_ij;
        }
    }

    // Solve (F^T F) gamma = F^T f
    std::vector<double> rhs_vec(gamma);
    gauss_solve(FTF.data(), rhs_vec.data(), gamma.data(), cols);

    // Anderson extrapolation: x = x_old + beta*f - sum gamma_j*(X_j + beta*F_j)
    for (int i = 0; i < N; ++i)
        x[i] = x_old[i] + beta * f[i];
    for (int j = 0; j < cols; ++j) {
        double gj = gamma[j];
        for (int i = 0; i < N; ++i)
            x[i] -= gj * (X_hist[j * N + i] + beta * F_hist[j * N + i]);
    }
}

void PoissonSolver::vec_copy_cpu(double* dst, const double* src, int N) const {
    std::memcpy(dst, src, static_cast<size_t>(N) * sizeof(double));
}

double PoissonSolver::compute_norm2_cpu(const double* r, int N) const {
    double sum = 0.0;
    for (int i = 0; i < N; ++i)
        sum += r[i] * r[i];
    return sum;
}

void PoissonSolver::subtract_mean_cpu(double* x, double mean, int N) const {
    for (int i = 0; i < N; ++i)
        x[i] -= mean;
}

double PoissonSolver::compute_mean_cpu(const double* x, int N) const {
    double sum = 0.0;
    for (int i = 0; i < N; ++i)
        sum += x[i];
    return sum / N;
}

} // namespace lynx
