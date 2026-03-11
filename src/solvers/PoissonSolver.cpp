#include "solvers/PoissonSolver.hpp"
#include <vector>
#include <cmath>
#include <cstdio>

namespace sparc {

void PoissonSolver::setup(const Laplacian& laplacian,
                            const FDStencil& stencil,
                            const Domain& domain,
                            const FDGrid& grid,
                            const HaloExchange& halo,
                            const MPIComm& dmcomm) {
    laplacian_ = &laplacian;
    stencil_ = &stencil;
    domain_ = &domain;
    grid_ = &grid;
    halo_ = &halo;
    dmcomm_ = &dmcomm;

    // Jacobi preconditioner weight: 1 / diagonal of -Laplacian
    // For orthogonal cells: diag = -2*(D2x[0] + D2y[0] + D2z[0])
    // Note: D2_coeff already includes 1/h^2 scaling, and the Laplacian has coefficient -1
    // So the diagonal of -Lap is -(D2x[0] + D2y[0] + D2z[0])
    double diag = -(stencil.D2_coeff_x()[0] + stencil.D2_coeff_y()[0] + stencil.D2_coeff_z()[0]);
    jacobi_weight_ = 1.0 / diag;

    // Set default AAR params tuned for Poisson
    // omega is the Jacobi relaxation parameter, scaled by jacobi_weight
    aar_params_.omega = 0.6 * jacobi_weight_;
    aar_params_.beta = 0.6 * jacobi_weight_;
    aar_params_.m = 7;
    aar_params_.p = 6;
    aar_params_.max_iter = 3000;
}

int PoissonSolver::solve(const double* rhs, double* phi, double tol) const {
    int Nd_d = domain_->Nd_d();
    AARParams params = aar_params_;
    params.tol = tol;

    // Operator: applies -Laplacian
    auto op = [this, Nd_d](const double* x, double* Ax) {
        int nd_ex = halo_->nd_ex();
        std::vector<double> x_ex(nd_ex, 0.0);
        halo_->execute(x, x_ex.data(), 1);
        laplacian_->apply(x_ex.data(), Ax, -1.0, 0.0, 1);
    };

    int iters = LinearSolver::aar(op, rhs, phi, Nd_d, params, *dmcomm_);
    return iters;
}

} // namespace sparc
