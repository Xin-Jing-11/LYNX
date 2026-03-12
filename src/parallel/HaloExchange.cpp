#include "parallel/HaloExchange.hpp"
#include <cstring>

namespace sparc {

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

} // namespace sparc
