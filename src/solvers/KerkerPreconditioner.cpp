#include "solvers/KerkerPreconditioner.hpp"
#include "solvers/LinearSolver.hpp"
#include "parallel/MPIComm.hpp"
#include <vector>
#include <cstring>
#include <cmath>

namespace lynx {

KerkerPreconditioner::KerkerPreconditioner(const Laplacian* lap, const HaloExchange* halo,
                                           const FDGrid* grid, double kTF, double idiemac,
                                           double precond_tol)
    : laplacian_(lap), halo_(halo), grid_(grid), kTF_(kTF), idiemac_(idiemac)
{
    if (precond_tol > 0.0) {
        // Use caller-provided tolerance (from ParameterDefaults)
        precond_tol_ = precond_tol;
    } else {
        // Compute TOL_PRECOND = h_eff^2 * 1e-3 (reference: initialization.c:2655-2664)
        precond_tol_ = 1e-4;  // default
        if (grid_) {
            double dx = grid_->dx(), dy = grid_->dy(), dz = grid_->dz();
            double h_eff;
            if (std::abs(dx - dy) < 1e-12 && std::abs(dy - dz) < 1e-12) {
                h_eff = dx;
            } else {
                double dx2_inv = 1.0 / (dx * dx);
                double dy2_inv = 1.0 / (dy * dy);
                double dz2_inv = 1.0 / (dz * dz);
                h_eff = std::sqrt(3.0 / (dx2_inv + dy2_inv + dz2_inv));
            }
            precond_tol_ = h_eff * h_eff * 1e-3;
        }
    }
}

void KerkerPreconditioner::apply(const double* r, double* z, int N, double mixing_param) {
    if (!laplacian_ || !halo_) {
        // Fallback: just scale
        for (int i = 0; i < N; ++i)
            z[i] = mixing_param * r[i];
        return;
    }

    int Nd_d = N;
    int nd_ex = halo_->nd_ex();
    double kTF = kTF_;
    double idiemac = idiemac_;

    // Step 1: Compute Lf = (Lap - idiemac*kTF^2) * r
    std::vector<double> r_ex(nd_ex, 0.0);
    halo_->execute(r, r_ex.data(), 1);

    std::vector<double> Lf(Nd_d);
    laplacian_->apply(r_ex.data(), Lf.data(), 1.0, -idiemac * kTF * kTF, 1);

    // Step 2: Solve (-Lap + kTF^2) * z = Lf
    auto op = [this, Nd_d](const double* x, double* Ax) {
        int nd_ex = halo_->nd_ex();
        std::vector<double> x_ex(nd_ex, 0.0);
        halo_->execute(x, x_ex.data(), 1);
        laplacian_->apply(x_ex.data(), Ax, -1.0, kTF_ * kTF_, 1);
    };

    // Jacobi preconditioner for -(Lap + c) where c = -kTF^2
    const auto& stencil = laplacian_->stencil();
    double m_diag = stencil.D2_coeff_x()[0] + stencil.D2_coeff_y()[0]
                  + stencil.D2_coeff_z()[0] + (-kTF * kTF);
    double m_inv = (std::abs(m_diag) < 1e-14) ? 1.0 : (-1.0 / m_diag);

    auto jacobi_precond = [m_inv, Nd_d](const double* rv, double* zv) {
        for (int i = 0; i < Nd_d; ++i)
            zv[i] = m_inv * rv[i];
    };

    // Initial guess: z = 0
    std::memset(z, 0, Nd_d * sizeof(double));

    AARParams params;
    params.omega = 0.6;
    params.beta = 0.6;
    params.m = 7;
    params.p = 6;
    params.tol = precond_tol_;
    params.max_iter = 1000;

    LinearSolver::PrecondFunc precond_fn = jacobi_precond;
    MPIComm self_comm(MPI_COMM_SELF);
    LinearSolver::aar(op, Lf.data(), z, Nd_d, params, self_comm, &precond_fn);

    // Step 3: Scale by -mixing_param
    for (int i = 0; i < Nd_d; ++i) {
        z[i] *= -mixing_param;
    }
}

} // namespace lynx
