#include "operators/Hamiltonian.hpp"
#include <vector>
#include <cstring>
#include <cmath>

namespace sparc {

void Hamiltonian::setup(const FDStencil& stencil,
                        const Domain& domain,
                        const FDGrid& grid,
                        const HaloExchange& halo,
                        const NonlocalProjector* vnl) {
    stencil_ = &stencil;
    domain_ = &domain;
    grid_ = &grid;
    halo_ = &halo;
    vnl_ = vnl;
}

// ---------------------------------------------------------------------------
// Real (Gamma-point) interface
// ---------------------------------------------------------------------------

void Hamiltonian::apply(const double* psi, const double* Veff, double* y,
                        int ncol, double c) const {
    apply_local(psi, Veff, y, ncol, c);
    if (vnl_ && vnl_->is_setup()) {
        vnl_->apply(psi, y, ncol, grid_->dV());
    }
}

void Hamiltonian::apply_local(const double* psi, const double* Veff, double* y,
                              int ncol, double c) const {
    int nd_ex = halo_->nx_ex() * halo_->ny_ex() * halo_->nz_ex();
    std::vector<double> x_ex(ncol * nd_ex, 0.0);
    halo_->execute(psi, x_ex.data(), ncol);

    if (domain_->global_grid().lattice().is_orthogonal()) {
        lap_plus_diag_orth_impl(x_ex.data(), Veff, y, ncol, c);
    } else {
        lap_plus_diag_nonorth_impl(x_ex.data(), Veff, y, ncol, c);
    }
}

// ---------------------------------------------------------------------------
// Complex (k-point) interface
// ---------------------------------------------------------------------------

void Hamiltonian::apply_kpt(const Complex* psi, const double* Veff, Complex* y,
                            int ncol, const Vec3& kpt_cart, const Vec3& cell_lengths,
                            double c) const {
    apply_local_kpt(psi, Veff, y, ncol, kpt_cart, cell_lengths, c);
    if (vnl_kpt_ && vnl_kpt_->is_setup()) {
        vnl_kpt_->apply_kpt(psi, y, ncol, grid_->dV());
    }
}

void Hamiltonian::apply_local_kpt(const Complex* psi, const double* Veff, Complex* y,
                                   int ncol, const Vec3& kpt_cart, const Vec3& cell_lengths,
                                   double c) const {
    int nd_ex = halo_->nx_ex() * halo_->ny_ex() * halo_->nz_ex();
    std::vector<Complex> x_ex(ncol * nd_ex, Complex(0.0, 0.0));
    halo_->execute_kpt(psi, x_ex.data(), ncol, kpt_cart, cell_lengths);

    if (domain_->global_grid().lattice().is_orthogonal()) {
        lap_plus_diag_orth_impl(x_ex.data(), Veff, y, ncol, c);
    } else {
        lap_plus_diag_nonorth_impl(x_ex.data(), Veff, y, ncol, c);
    }
}

// ---------------------------------------------------------------------------
// Templated stencil implementations
// ---------------------------------------------------------------------------

template<typename T>
void Hamiltonian::lap_plus_diag_orth_impl(const T* x_ex, const double* Veff,
                                           T* y, int ncol, double c) const {
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

    for (int n = 0; n < ncol; ++n) {
        const T* xn = x_ex + n * nd_ex;
        T* yn = y + n * nd;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx_ex = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
                    int idx_loc = i + j * nx + k * nx * ny;

                    T lap = (cx[0] + cy[0] + cz[0]) * xn[idx_ex];

                    for (int p = 1; p <= FDn; ++p) {
                        lap += cx[p] * (xn[idx_ex + p] + xn[idx_ex - p]);
                        lap += cy[p] * (xn[idx_ex + p * nx_ex] + xn[idx_ex - p * nx_ex]);
                        lap += cz[p] * (xn[idx_ex + p * nxny_ex] + xn[idx_ex - p * nxny_ex]);
                    }

                    double veff_val = Veff ? Veff[idx_loc] : 0.0;
                    yn[idx_loc] = -0.5 * lap + (veff_val + c) * xn[idx_ex];
                }
            }
        }
    }
}

template<typename T>
void Hamiltonian::lap_plus_diag_nonorth_impl(const T* x_ex, const double* Veff,
                                              T* y, int ncol, double c) const {
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
    const double* cxy = stencil_->D2_coeff_xy();
    const double* cxz = stencil_->D2_coeff_xz();
    const double* cyz = stencil_->D2_coeff_yz();

    const double* d1y = stencil_->D1_coeff_y();
    const double* d1z = stencil_->D1_coeff_z();

    bool has_xy = cxy && std::abs(cxy[1]) > 1e-30;
    bool has_xz = cxz && std::abs(cxz[1]) > 1e-30;
    bool has_yz = cyz && std::abs(cyz[1]) > 1e-30;

    for (int n = 0; n < ncol; ++n) {
        const T* xn = x_ex + n * nd_ex;
        T* yn = y + n * nd;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
                    int loc = i + j * nx + k * nx * ny;

                    T lap = (cx[0] + cy[0] + cz[0]) * xn[idx];

                    for (int p = 1; p <= FDn; ++p) {
                        lap += cx[p] * (xn[idx + p] + xn[idx - p]);
                        lap += cy[p] * (xn[idx + p * nx_ex] + xn[idx - p * nx_ex]);
                        lap += cz[p] * (xn[idx + p * nxny_ex] + xn[idx - p * nxny_ex]);
                    }

                    if (has_xy) {
                        for (int p = 1; p <= FDn; ++p) {
                            for (int q = 1; q <= FDn; ++q) {
                                T mixed = xn[idx + p + q * nx_ex]
                                        - xn[idx + p - q * nx_ex]
                                        - xn[idx - p + q * nx_ex]
                                        + xn[idx - p - q * nx_ex];
                                lap += cxy[p] * d1y[q] * mixed;
                            }
                        }
                    }
                    if (has_xz) {
                        for (int p = 1; p <= FDn; ++p) {
                            for (int q = 1; q <= FDn; ++q) {
                                T mixed = xn[idx + p + q * nxny_ex]
                                        - xn[idx + p - q * nxny_ex]
                                        - xn[idx - p + q * nxny_ex]
                                        + xn[idx - p - q * nxny_ex];
                                lap += cxz[p] * d1z[q] * mixed;
                            }
                        }
                    }
                    if (has_yz) {
                        for (int p = 1; p <= FDn; ++p) {
                            for (int q = 1; q <= FDn; ++q) {
                                T mixed = xn[idx + p * nx_ex + q * nxny_ex]
                                        - xn[idx + p * nx_ex - q * nxny_ex]
                                        - xn[idx - p * nx_ex + q * nxny_ex]
                                        + xn[idx - p * nx_ex - q * nxny_ex];
                                lap += cyz[p] * d1z[q] * mixed;
                            }
                        }
                    }

                    double veff_val = Veff ? Veff[loc] : 0.0;
                    yn[loc] = -0.5 * lap + (veff_val + c) * xn[idx];
                }
            }
        }
    }
}

// Legacy real wrappers
void Hamiltonian::lap_plus_diag_orth(const double* x_ex, const double* Veff,
                                      double* y, int ncol, double c) const {
    lap_plus_diag_orth_impl(x_ex, Veff, y, ncol, c);
}

void Hamiltonian::lap_plus_diag_nonorth(const double* x_ex, const double* Veff,
                                         double* y, int ncol, double c) const {
    lap_plus_diag_nonorth_impl(x_ex, Veff, y, ncol, c);
}

// Explicit template instantiations
template void Hamiltonian::lap_plus_diag_orth_impl<double>(const double*, const double*,
                                                            double*, int, double) const;
template void Hamiltonian::lap_plus_diag_orth_impl<Complex>(const Complex*, const double*,
                                                             Complex*, int, double) const;
template void Hamiltonian::lap_plus_diag_nonorth_impl<double>(const double*, const double*,
                                                               double*, int, double) const;
template void Hamiltonian::lap_plus_diag_nonorth_impl<Complex>(const Complex*, const double*,
                                                                Complex*, int, double) const;

} // namespace sparc
