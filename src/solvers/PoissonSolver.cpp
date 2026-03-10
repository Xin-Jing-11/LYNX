#include "solvers/PoissonSolver.hpp"
#include <vector>
#include <cmath>

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
    double diag = -2.0 * (stencil.D2_coeff_x()[0] + stencil.D2_coeff_y()[0] + stencil.D2_coeff_z()[0]);
    jacobi_weight_ = 1.0 / diag;

    // Set default AAR params tuned for Poisson
    aar_params_.omega = 0.6;
    aar_params_.beta = 0.6;
    aar_params_.m = 7;
    aar_params_.p = 6;
    aar_params_.max_iter = 1000;
}

int PoissonSolver::solve(const double* rhs, double* phi, double tol) const {
    int Nd_d = domain_->Nd_d();
    AARParams params = aar_params_;
    params.tol = tol;

    // Operator: applies -Laplacian
    auto op = [this, Nd_d](const double* x, double* Ax) {
        // We need to fill halos, then apply -Lap
        int nd_ex = halo_->nd_ex();
        std::vector<double> x_ex(nd_ex, 0.0);
        halo_->execute(x, x_ex.data(), 1);
        laplacian_->apply(x_ex.data(), Ax, -1.0, 0.0, 1);
    };

    return LinearSolver::aar(op, rhs, phi, Nd_d, params, *dmcomm_);
}

} // namespace sparc
