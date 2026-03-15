#include "operators/Laplacian.hpp"
#include <cmath>

namespace lynx {

Laplacian::Laplacian(const FDStencil& stencil, const Domain& domain)
    : stencil_(&stencil), domain_(&domain) {}

// Real versions
void Laplacian::apply(const double* x_ex, double* y, double a, double c, int ncol) const {
    if (domain_->global_grid().lattice().is_orthogonal()) {
        apply_orth_impl(x_ex, nullptr, y, a, 0.0, c, ncol);
    } else {
        apply_nonorth_impl(x_ex, nullptr, y, a, 0.0, c, ncol);
    }
}

void Laplacian::apply_with_diag(const double* x_ex, const double* V, double* y,
                                double a, double b, double c, int ncol) const {
    if (domain_->global_grid().lattice().is_orthogonal()) {
        apply_orth_impl(x_ex, V, y, a, b, c, ncol);
    } else {
        apply_nonorth_impl(x_ex, V, y, a, b, c, ncol);
    }
}

// Complex versions
void Laplacian::apply(const Complex* x_ex, Complex* y, double a, double c, int ncol) const {
    if (domain_->global_grid().lattice().is_orthogonal()) {
        apply_orth_impl(x_ex, nullptr, y, a, 0.0, c, ncol);
    } else {
        apply_nonorth_impl(x_ex, nullptr, y, a, 0.0, c, ncol);
    }
}

void Laplacian::apply_with_diag(const Complex* x_ex, const double* V, Complex* y,
                                double a, double b, double c, int ncol) const {
    if (domain_->global_grid().lattice().is_orthogonal()) {
        apply_orth_impl(x_ex, V, y, a, b, c, ncol);
    } else {
        apply_nonorth_impl(x_ex, V, y, a, b, c, ncol);
    }
}

// Backward-compat wrappers
void Laplacian::apply_orth(const double* x_ex, const double* V, double* y,
                           double a, double b, double c, int ncol) const {
    apply_orth_impl(x_ex, V, y, a, b, c, ncol);
}

void Laplacian::apply_nonorth(const double* x_ex, const double* V, double* y,
                              double a, double b, double c, int ncol) const {
    apply_nonorth_impl(x_ex, V, y, a, b, c, ncol);
}

template<typename T>
void Laplacian::apply_orth_impl(const T* x_ex, const double* V, T* y,
                                double a, double b, double c, int ncol) const {
    int nx = domain_->Nx_d();
    int ny = domain_->Ny_d();
    int nz = domain_->Nz_d();
    int nd = nx * ny * nz;
    int FDn = stencil_->FDn();
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nxny_ex = nx_ex * ny_ex;
    int nd_ex = nxny_ex * (nz + 2 * FDn);

    const double* cx = stencil_->D2_coeff_x();
    const double* cy = stencil_->D2_coeff_y();
    const double* cz = stencil_->D2_coeff_z();

    double diag_coeff = a * (cx[0] + cy[0] + cz[0]) + c;

    for (int n = 0; n < ncol; ++n) {
        const T* xn = x_ex + n * nd_ex;
        T* yn = y + n * nd;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
                    int loc = i + j * nx + k * nx * ny;

                    T val = diag_coeff * xn[idx];

                    for (int p = 1; p <= FDn; ++p) {
                        val += a * cx[p] * (xn[idx + p] + xn[idx - p]);
                        val += a * cy[p] * (xn[idx + p * nx_ex] + xn[idx - p * nx_ex]);
                        val += a * cz[p] * (xn[idx + p * nxny_ex] + xn[idx - p * nxny_ex]);
                    }

                    if (V) val += b * V[loc] * xn[idx];

                    yn[loc] = val;
                }
            }
        }
    }
}

template<typename T>
void Laplacian::apply_nonorth_impl(const T* x_ex, const double* V, T* y,
                                   double a, double b, double c, int ncol) const {
    int nx = domain_->Nx_d();
    int ny = domain_->Ny_d();
    int nz = domain_->Nz_d();
    int nd = nx * ny * nz;
    int FDn = stencil_->FDn();
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nxny_ex = nx_ex * ny_ex;
    int nd_ex = nxny_ex * (nz + 2 * FDn);

    const double* cx = stencil_->D2_coeff_x();
    const double* cy = stencil_->D2_coeff_y();
    const double* cz = stencil_->D2_coeff_z();
    const double* cxy = stencil_->D2_coeff_xy();
    const double* cxz = stencil_->D2_coeff_xz();
    const double* cyz = stencil_->D2_coeff_yz();

    double diag_coeff = a * (cx[0] + cy[0] + cz[0]) + c;

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

                    T val = diag_coeff * xn[idx];

                    for (int p = 1; p <= FDn; ++p) {
                        val += a * cx[p] * (xn[idx + p] + xn[idx - p]);
                        val += a * cy[p] * (xn[idx + p * nx_ex] + xn[idx - p * nx_ex]);
                        val += a * cz[p] * (xn[idx + p * nxny_ex] + xn[idx - p * nxny_ex]);
                    }

                    if (has_xy) {
                        for (int p = 1; p <= FDn; ++p) {
                            for (int q = 1; q <= FDn; ++q) {
                                T mixed = xn[idx + p + q * nx_ex]
                                        - xn[idx + p - q * nx_ex]
                                        - xn[idx - p + q * nx_ex]
                                        + xn[idx - p - q * nx_ex];
                                val += a * cxy[p] * d1y[q] * mixed;
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
                                val += a * cxz[p] * d1z[q] * mixed;
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
                                val += a * cyz[p] * d1z[q] * mixed;
                            }
                        }
                    }

                    if (V) val += b * V[loc] * xn[idx];

                    yn[loc] = val;
                }
            }
        }
    }
}

// Explicit template instantiations
template void Laplacian::apply_orth_impl<double>(const double*, const double*, double*,
                                                  double, double, double, int) const;
template void Laplacian::apply_orth_impl<Complex>(const Complex*, const double*, Complex*,
                                                   double, double, double, int) const;
template void Laplacian::apply_nonorth_impl<double>(const double*, const double*, double*,
                                                     double, double, double, int) const;
template void Laplacian::apply_nonorth_impl<Complex>(const Complex*, const double*, Complex*,
                                                      double, double, double, int) const;

} // namespace lynx
