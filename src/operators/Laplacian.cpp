#include "operators/Laplacian.hpp"

namespace sparc {

Laplacian::Laplacian(const FDStencil& stencil, const Domain& domain)
    : stencil_(&stencil), domain_(&domain) {}

void Laplacian::apply(const double* x_ex, double* y, double a, double c, int ncol) const {
    if (domain_->global_grid().lattice().is_orthogonal()) {
        apply_orth(x_ex, nullptr, y, a, 0.0, c, ncol);
    } else {
        apply_nonorth(x_ex, nullptr, y, a, 0.0, c, ncol);
    }
}

void Laplacian::apply_with_diag(const double* x_ex, const double* V, double* y,
                                double a, double b, double c, int ncol) const {
    if (domain_->global_grid().lattice().is_orthogonal()) {
        apply_orth(x_ex, V, y, a, b, c, ncol);
    } else {
        apply_nonorth(x_ex, V, y, a, b, c, ncol);
    }
}

void Laplacian::apply_orth(const double* x_ex, const double* V, double* y,
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
        const double* xn = x_ex + n * nd_ex;
        double* yn = y + n * nd;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
                    int loc = i + j * nx + k * nx * ny;

                    double val = diag_coeff * xn[idx];

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

void Laplacian::apply_nonorth(const double* x_ex, const double* V, double* y,
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

    // Non-orthogonal Laplacian:
    // Lap = d^2/dx^2 * lapcT(0,0) + d^2/dy^2 * lapcT(1,1) + d^2/dz^2 * lapcT(2,2)
    //     + 2 * lapcT(0,1) * d^2/dxdy + 2 * lapcT(0,2) * d^2/dxdz + 2 * lapcT(1,2) * d^2/dydz
    //
    // The mixed derivatives are computed as:
    //   d^2f/dxdy ≈ sum_p sum_q w1[p]*w1[q]/(dx*dy) * (f(i+p,j+q) - f(i+p,j-q) - f(i-p,j+q) + f(i-p,j-q))
    //
    // In the compact form used by SPARC, this is done via intermediate first derivatives.

    for (int n = 0; n < ncol; ++n) {
        const double* xn = x_ex + n * nd_ex;
        double* yn = y + n * nd;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
                    int loc = i + j * nx + k * nx * ny;

                    // Diagonal second derivatives
                    double val = diag_coeff * xn[idx];

                    for (int p = 1; p <= FDn; ++p) {
                        val += a * cx[p] * (xn[idx + p] + xn[idx - p]);
                        val += a * cy[p] * (xn[idx + p * nx_ex] + xn[idx - p * nx_ex]);
                        val += a * cz[p] * (xn[idx + p * nxny_ex] + xn[idx - p * nxny_ex]);
                    }

                    // Mixed derivatives: d^2f/dxdy via cross stencil
                    // cxy[p] already contains 2*lapcT(0,1)*w1[p]/dx
                    // Need to apply w1[q]/dy to get the full mixed derivative
                    const double* d1y = stencil_->D1_coeff_y();
                    const double* d1z = stencil_->D1_coeff_z();
                    const double* d1x = stencil_->D1_coeff_x();

                    // xy mixed: sum_p cxy[p] * sum_q d1y[q] * [f(i+p,j+q) - f(i+p,j-q) - f(i-p,j+q) + f(i-p,j-q)]
                    // Simplified: sum_p,q cxy[p]*d1y[q] * [...]
                    // But more efficient: first compute df/dy, then apply df/dx to that.
                    // For now, direct double-sum approach (accurate, can optimize later):
                    if (cxy && std::abs(cxy[1]) > 1e-30) {
                        for (int p = 1; p <= FDn; ++p) {
                            for (int q = 1; q <= FDn; ++q) {
                                double mixed = xn[idx + p + q * nx_ex]
                                             - xn[idx + p - q * nx_ex]
                                             - xn[idx - p + q * nx_ex]
                                             + xn[idx - p - q * nx_ex];
                                val += a * cxy[p] * d1y[q] * mixed;
                            }
                        }
                    }
                    if (cxz && std::abs(cxz[1]) > 1e-30) {
                        for (int p = 1; p <= FDn; ++p) {
                            for (int q = 1; q <= FDn; ++q) {
                                double mixed = xn[idx + p + q * nxny_ex]
                                             - xn[idx + p - q * nxny_ex]
                                             - xn[idx - p + q * nxny_ex]
                                             + xn[idx - p - q * nxny_ex];
                                val += a * cxz[p] * d1z[q] * mixed;
                            }
                        }
                    }
                    if (cyz && std::abs(cyz[1]) > 1e-30) {
                        for (int p = 1; p <= FDn; ++p) {
                            for (int q = 1; q <= FDn; ++q) {
                                double mixed = xn[idx + p * nx_ex + q * nxny_ex]
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

} // namespace sparc
