#include "solvers/PoissonSolver.hpp"
#include "core/NumericalMethods.hpp"
#include "parallel/MPIComm.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdio>

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
    return solve_gpu(rhs, phi, tol);
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
