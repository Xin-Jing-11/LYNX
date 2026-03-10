#include "operators/Hamiltonian.hpp"
#include <vector>
#include <cstring>

namespace sparc {

void Hamiltonian::setup(const FDStencil& stencil,
                        const Domain& domain,
                        const FDGrid& grid,
                        const HaloExchange& halo,
                        const NonlocalProjector* vnl,
                        const MPIComm& dmcomm) {
    stencil_ = &stencil;
    domain_ = &domain;
    grid_ = &grid;
    halo_ = &halo;
    vnl_ = vnl;
    dmcomm_ = &dmcomm;
}

void Hamiltonian::apply(const double* psi, const double* Veff, double* y,
                        int ncol, double c) const {
    // Step 1: Apply local part: -0.5*Lap + Veff + c
    apply_local(psi, Veff, y, ncol, c);

    // Step 2: Apply nonlocal part: y += Vnl * psi
    if (vnl_ && vnl_->is_setup()) {
        vnl_->apply(psi, y, ncol, grid_->dV(), *dmcomm_);
    }
}

void Hamiltonian::apply_local(const double* psi, const double* Veff, double* y,
                              int ncol, double c) const {
    int nx = domain_->Nx_d();
    int ny = domain_->Ny_d();
    int nz = domain_->Nz_d();
    int nd = nx * ny * nz;
    int FDn = stencil_->FDn();
    int nx_ex = halo_->nx_ex();
    int ny_ex = halo_->ny_ex();
    int nz_ex = halo_->nz_ex();
    int nd_ex = nx_ex * ny_ex * nz_ex;

    // Create extended array with ghost zones
    std::vector<double> x_ex(ncol * nd_ex, 0.0);
    halo_->execute(psi, x_ex.data(), ncol);

    if (domain_->global_grid().lattice().is_orthogonal()) {
        lap_plus_diag_orth(x_ex.data(), Veff, y, ncol, c);
    } else {
        // Non-orthogonal: for now use same kernel but with correct coefficients
        // Full non-orth implementation with mixed derivatives is deferred
        lap_plus_diag_orth(x_ex.data(), Veff, y, ncol, c);
    }
}

void Hamiltonian::lap_plus_diag_orth(const double* x_ex, const double* Veff,
                                      double* y, int ncol, double c) const {
    int nx = domain_->Nx_d();
    int ny = domain_->Ny_d();
    int nz = domain_->Nz_d();
    int nd = nx * ny * nz;
    int FDn = stencil_->FDn();
    int nx_ex = halo_->nx_ex();
    int ny_ex = halo_->ny_ex();
    int nxny_ex = nx_ex * ny_ex;
    int nd_ex = nxny_ex * halo_->nz_ex();

    const double* cx = stencil_->D2_coeff_x();
    const double* cy = stencil_->D2_coeff_y();
    const double* cz = stencil_->D2_coeff_z();

    // Diagonal coefficient: -0.5 * (cx[0] + cy[0] + cz[0]) + c
    double diag_lap = -0.5 * (cx[0] + cy[0] + cz[0]);

    for (int n = 0; n < ncol; ++n) {
        const double* xn = x_ex + n * nd_ex;
        double* yn = y + n * nd;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx_ex = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
                    int idx_loc = i + j * nx + k * nx * ny;

                    // -0.5 * Lap * psi
                    double lap = cx[0] * xn[idx_ex];
                    lap += cy[0] * xn[idx_ex];
                    lap += cz[0] * xn[idx_ex];

                    for (int p = 1; p <= FDn; ++p) {
                        lap += cx[p] * (xn[idx_ex + p] + xn[idx_ex - p]);
                        lap += cy[p] * (xn[idx_ex + p * nx_ex] + xn[idx_ex - p * nx_ex]);
                        lap += cz[p] * (xn[idx_ex + p * nxny_ex] + xn[idx_ex - p * nxny_ex]);
                    }

                    // H*psi = -0.5*Lap*psi + Veff*psi + c*psi
                    double veff_val = Veff ? Veff[idx_loc] : 0.0;
                    yn[idx_loc] = -0.5 * lap + (veff_val + c) * xn[idx_ex];
                }
            }
        }
    }
}

} // namespace sparc
