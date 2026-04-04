#include "solvers/PoissonSolver.hpp"
#include "parallel/MPIComm.hpp"
#include <vector>
#include <cmath>
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

    // Jacobi preconditioner weight: m_inv = -1 / (D2x[0] + D2y[0] + D2z[0] + c)
    // For Poisson (c=0): m_inv = -1 / (D2x[0] + D2y[0] + D2z[0])
    // Reference: Jacobi_preconditioner in electrostatics.c
    double diag = stencil.D2_coeff_x()[0] + stencil.D2_coeff_y()[0] + stencil.D2_coeff_z()[0];
    jacobi_weight_ = (std::abs(diag) < 1e-14) ? 1.0 : (-1.0 / diag);

    // AAR params: reference uses omega=0.6, beta=0.6 (NOT scaled by jacobi_weight)
    // Jacobi preconditioning is applied inside AAR to the residual
    aar_params_.omega = 0.6;
    aar_params_.beta = 0.6;
    aar_params_.m = 7;
    aar_params_.p = 6;
    aar_params_.max_iter = 3000;
}

int PoissonSolver::solve(const double* rhs, double* phi, double tol) const {
    int Nd_d = domain_->Nd_d();
    AARParams params = aar_params_;
    params.tol = tol;

    // Operator: applies -Laplacian (i.e., the operator A in A*x = b)
    // Reference: poisson_residual computes r = b + Lap*x, so A = -Lap
    auto op = [this, Nd_d](const double* x, double* Ax) {
        int nd_ex = halo_->nd_ex();
        std::vector<double> x_ex(nd_ex, 0.0);
        halo_->execute(x, x_ex.data(), 1);
        laplacian_->apply(x_ex.data(), Ax, -1.0, 0.0, 1);
    };

    // Jacobi preconditioner: f[i] = m_inv * r[i]
    // Reference: Jacobi_preconditioner with c=0
    double m_inv = jacobi_weight_;
    LinearSolver::PrecondFunc jacobi = [m_inv, Nd_d](const double* r, double* f) {
        for (int i = 0; i < Nd_d; ++i)
            f[i] = m_inv * r[i];
    };

    MPIComm self_comm(MPI_COMM_SELF);
    int iters = LinearSolver::aar(op, rhs, phi, Nd_d, params, self_comm, &jacobi);
    return iters;
}

// ---------------------------------------------------------------------------
// Device-dispatching overload (CPU-only fallback when USE_CUDA is off)
// GPU implementation lives in PoissonSolver.cu.
// ---------------------------------------------------------------------------
#ifndef USE_CUDA

int PoissonSolver::solve(const double* rhs, double* phi, double tol, Device dev) const {
    if (dev == Device::GPU)
        throw std::runtime_error("PoissonSolver::solve(GPU) called but USE_CUDA is off");
    return solve(rhs, phi, tol);
}

#endif // !USE_CUDA

} // namespace lynx
