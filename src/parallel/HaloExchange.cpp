#include "parallel/HaloExchange.hpp"
#include <cstring>
#include <cmath>

namespace lynx {

HaloExchange::HaloExchange(const Domain& domain, int FDn)
    : nx_(domain.Nx_d()), ny_(domain.Ny_d()), nz_(domain.Nz_d()),
      FDn_(FDn) {
    nx_ex_ = nx_ + 2 * FDn;
    ny_ex_ = ny_ + 2 * FDn;
    nz_ex_ = nz_ + 2 * FDn;

    const auto& grid = domain.global_grid();
    periods_[0] = (grid.bcx() == BCType::Periodic);
    periods_[1] = (grid.bcy() == BCType::Periodic);
    periods_[2] = (grid.bcz() == BCType::Periodic);
}

void HaloExchange::copy_to_interior(const double* x, double* x_ex, int ncol) const {
    for (int n = 0; n < ncol; ++n) {
        const double* xn = x + n * static_cast<std::ptrdiff_t>(nx_ * ny_ * nz_);
        double* xn_ex = x_ex + n * static_cast<std::ptrdiff_t>(nx_ex_ * ny_ex_ * nz_ex_);
        for (int k = 0; k < nz_; ++k) {
            for (int j = 0; j < ny_; ++j) {
                const double* src = xn + j * nx_ + k * nx_ * ny_;
                double* dst = xn_ex + (j + FDn_) * nx_ex_ + (k + FDn_) * nx_ex_ * ny_ex_ + FDn_;
                std::memcpy(dst, src, nx_ * sizeof(double));
            }
        }
    }
}

void HaloExchange::execute(const double* x, double* x_ex, int ncol) const {
    int nd_ex_total = nx_ex_ * ny_ex_ * nz_ex_;
    std::memset(x_ex, 0, ncol * nd_ex_total * sizeof(double));
    copy_to_interior(x, x_ex, ncol);
    apply_periodic_bc(x_ex, ncol);
}

void HaloExchange::apply_periodic_bc(double* x_ex, int ncol) const {
    int nxny_ex = nx_ex_ * ny_ex_;

    for (int n = 0; n < ncol; ++n) {
        double* xn = x_ex + n * static_cast<std::ptrdiff_t>(nx_ex_ * ny_ex_ * nz_ex_);

        // Z-direction wrapping
        if (periods_[2]) {
            for (int k = 0; k < FDn_; ++k) {
                std::memcpy(xn + k * nxny_ex,
                           xn + (nz_ + k) * nxny_ex, nxny_ex * sizeof(double));
                std::memcpy(xn + (nz_ + FDn_ + k) * nxny_ex,
                           xn + (FDn_ + k) * nxny_ex, nxny_ex * sizeof(double));
            }
        }

        // Y-direction wrapping
        if (periods_[1]) {
            for (int k = 0; k < nz_ex_; ++k) {
                for (int j = 0; j < FDn_; ++j) {
                    std::memcpy(xn + j * nx_ex_ + k * nxny_ex,
                               xn + (ny_ + j) * nx_ex_ + k * nxny_ex, nx_ex_ * sizeof(double));
                    std::memcpy(xn + (ny_ + FDn_ + j) * nx_ex_ + k * nxny_ex,
                               xn + (FDn_ + j) * nx_ex_ + k * nxny_ex, nx_ex_ * sizeof(double));
                }
            }
        }

        // X-direction wrapping
        if (periods_[0]) {
            for (int k = 0; k < nz_ex_; ++k) {
                for (int j = 0; j < ny_ex_; ++j) {
                    int base = j * nx_ex_ + k * nxny_ex;
                    for (int i = 0; i < FDn_; ++i) {
                        xn[base + i] = xn[base + nx_ + i];
                        xn[base + nx_ + FDn_ + i] = xn[base + FDn_ + i];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Complex (k-point) versions
// ---------------------------------------------------------------------------

void HaloExchange::copy_to_interior(const Complex* x, Complex* x_ex, int ncol) const {
    std::ptrdiff_t nd = static_cast<std::ptrdiff_t>(nx_) * ny_ * nz_;
    std::ptrdiff_t nd_ex = static_cast<std::ptrdiff_t>(nx_ex_) * ny_ex_ * nz_ex_;
    for (int n = 0; n < ncol; ++n) {
        const Complex* xn = x + n * nd;
        Complex* xn_ex = x_ex + n * nd_ex;
        for (int k = 0; k < nz_; ++k) {
            for (int j = 0; j < ny_; ++j) {
                const Complex* src = xn + j * nx_ + k * nx_ * ny_;
                Complex* dst = xn_ex + (j + FDn_) * nx_ex_ + (k + FDn_) * nx_ex_ * ny_ex_ + FDn_;
                std::memcpy(dst, src, nx_ * sizeof(Complex));
            }
        }
    }
}

void HaloExchange::execute_kpt(const Complex* x, Complex* x_ex, int ncol,
                                const Vec3& kpt_cart, const Vec3& cell_lengths) const {
    int nd_ex_total = nx_ex_ * ny_ex_ * nz_ex_;
    std::memset(x_ex, 0, ncol * nd_ex_total * sizeof(Complex));
    copy_to_interior(x, x_ex, ncol);

    // Compute Bloch phase factors: e^{-ik·L} for left, e^{+ik·L} for right
    // Matching reference: phase_fac_l = cos(k*L) - i*sin(k*L)
    double theta_x = kpt_cart.x * cell_lengths.x;
    double theta_y = kpt_cart.y * cell_lengths.y;
    double theta_z = kpt_cart.z * cell_lengths.z;

    Complex phase_l_x(std::cos(theta_x), -std::sin(theta_x));
    Complex phase_r_x(std::cos(theta_x),  std::sin(theta_x));
    Complex phase_l_y(std::cos(theta_y), -std::sin(theta_y));
    Complex phase_r_y(std::cos(theta_y),  std::sin(theta_y));
    Complex phase_l_z(std::cos(theta_z), -std::sin(theta_z));
    Complex phase_r_z(std::cos(theta_z),  std::sin(theta_z));

    apply_periodic_bc_kpt(x_ex, ncol,
                          phase_l_x, phase_r_x,
                          phase_l_y, phase_r_y,
                          phase_l_z, phase_r_z);
}

void HaloExchange::apply_periodic_bc_kpt(Complex* x_ex, int ncol,
                                          Complex phase_l_x, Complex phase_r_x,
                                          Complex phase_l_y, Complex phase_r_y,
                                          Complex phase_l_z, Complex phase_r_z) const {
    int nxny_ex = nx_ex_ * ny_ex_;

    for (int n = 0; n < ncol; ++n) {
        Complex* xn = x_ex + n * static_cast<std::ptrdiff_t>(nx_ex_ * ny_ex_ * nz_ex_);

        // Z-direction wrapping
        if (periods_[2]) {
            for (int k = 0; k < FDn_; ++k) {
                // Left ghost (k < FDn): data from right interior, phase_l_z
                Complex* dst_l = xn + k * nxny_ex;
                const Complex* src_l = xn + (nz_ + k) * nxny_ex;
                for (int idx = 0; idx < nxny_ex; ++idx)
                    dst_l[idx] = src_l[idx] * phase_l_z;

                // Right ghost (k >= nz+FDn): data from left interior, phase_r_z
                Complex* dst_r = xn + (nz_ + FDn_ + k) * nxny_ex;
                const Complex* src_r = xn + (FDn_ + k) * nxny_ex;
                for (int idx = 0; idx < nxny_ex; ++idx)
                    dst_r[idx] = src_r[idx] * phase_r_z;
            }
        }

        // Y-direction wrapping
        if (periods_[1]) {
            for (int k = 0; k < nz_ex_; ++k) {
                for (int j = 0; j < FDn_; ++j) {
                    // Left ghost
                    Complex* dst_l = xn + j * nx_ex_ + k * nxny_ex;
                    const Complex* src_l = xn + (ny_ + j) * nx_ex_ + k * nxny_ex;
                    for (int i = 0; i < nx_ex_; ++i)
                        dst_l[i] = src_l[i] * phase_l_y;

                    // Right ghost
                    Complex* dst_r = xn + (ny_ + FDn_ + j) * nx_ex_ + k * nxny_ex;
                    const Complex* src_r = xn + (FDn_ + j) * nx_ex_ + k * nxny_ex;
                    for (int i = 0; i < nx_ex_; ++i)
                        dst_r[i] = src_r[i] * phase_r_y;
                }
            }
        }

        // X-direction wrapping
        if (periods_[0]) {
            for (int k = 0; k < nz_ex_; ++k) {
                for (int j = 0; j < ny_ex_; ++j) {
                    int base = j * nx_ex_ + k * nxny_ex;
                    for (int i = 0; i < FDn_; ++i) {
                        // Left ghost
                        xn[base + i] = xn[base + nx_ + i] * phase_l_x;
                        // Right ghost
                        xn[base + nx_ + FDn_ + i] = xn[base + FDn_ + i] * phase_r_x;
                    }
                }
            }
        }
    }
}

} // namespace lynx
