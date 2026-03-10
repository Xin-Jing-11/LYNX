#include "operators/FDStencil.hpp"
#include "core/constants.hpp"
#include <cmath>

namespace sparc {

double FDStencil::fract(int n, int k) {
    double Nr = 1.0, Dr = 1.0;
    for (int i = n - k + 1; i <= n; ++i)
        Nr *= i;
    for (int i = n + 1; i <= n + k; ++i)
        Dr *= i;
    return Nr / Dr;
}

FDStencil::FDStencil(int order, const FDGrid& grid, const Lattice& lattice)
    : order_(order) {
    compute_weights();
    scale_for_grid(grid, lattice);
    compute_max_eigval(grid, lattice);
}

void FDStencil::compute_weights() {
    int FDn = order_ / 2;
    int n = FDn + 1;

    w_D1_.resize(n, 0.0);
    w_D2_.resize(n, 0.0);

    // First derivative weights: w1[p] = (-1)^(p+1) * fract(FDn, p) / p
    w_D1_[0] = 0.0;
    for (int p = 1; p < n; ++p) {
        double sign = (p % 2 == 1) ? 1.0 : -1.0;
        w_D1_[p] = sign * fract(FDn, p) / p;
    }

    // Second derivative weights: w2[0] = -sum 2/p^2, w2[p] = (-1)^(p+1) * 2*fract(FDn,p)/p^2
    w_D2_[0] = 0.0;
    for (int p = 1; p < n; ++p) {
        w_D2_[0] -= 2.0 / (static_cast<double>(p) * p);
        double sign = (p % 2 == 1) ? 1.0 : -1.0;
        w_D2_[p] = sign * 2.0 * fract(FDn, p) / (static_cast<double>(p) * p);
    }
}

void FDStencil::scale_for_grid(const FDGrid& grid, const Lattice& lattice) {
    int n = order_ / 2 + 1;

    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    double dx_inv = 1.0 / dx, dy_inv = 1.0 / dy, dz_inv = 1.0 / dz;
    double dx2_inv = dx_inv * dx_inv;
    double dy2_inv = dy_inv * dy_inv;
    double dz2_inv = dz_inv * dz_inv;

    D1_x_.resize(n);
    D1_y_.resize(n);
    D1_z_.resize(n);
    D2_x_.resize(n);
    D2_y_.resize(n);
    D2_z_.resize(n);

    if (lattice.is_orthogonal()) {
        // Orthogonal: simple scaling
        for (int p = 0; p < n; ++p) {
            D1_x_[p] = w_D1_[p] * dx_inv;
            D1_y_[p] = w_D1_[p] * dy_inv;
            D1_z_[p] = w_D1_[p] * dz_inv;
            D2_x_[p] = w_D2_[p] * dx2_inv;
            D2_y_[p] = w_D2_[p] * dy2_inv;
            D2_z_[p] = w_D2_[p] * dz2_inv;
        }

        D2_xy_.assign(n, 0.0);
        D2_xz_.assign(n, 0.0);
        D2_yz_.assign(n, 0.0);
    } else {
        // Non-orthogonal: use Laplacian transformation matrix
        const Mat3& lapcT = lattice.lapc_T();

        for (int p = 0; p < n; ++p) {
            // Diagonal second derivatives
            D2_x_[p] = lapcT(0, 0) * w_D2_[p] * dx2_inv;
            D2_y_[p] = lapcT(1, 1) * w_D2_[p] * dy2_inv;
            D2_z_[p] = lapcT(2, 2) * w_D2_[p] * dz2_inv;

            // First derivatives (for gradient)
            D1_x_[p] = w_D1_[p] * dx_inv;
            D1_y_[p] = w_D1_[p] * dy_inv;
            D1_z_[p] = w_D1_[p] * dz_inv;
        }

        // Mixed second derivative coefficients (for non-orthogonal Laplacian)
        D2_xy_.resize(n);
        D2_xz_.resize(n);
        D2_yz_.resize(n);
        for (int p = 0; p < n; ++p) {
            D2_xy_[p] = 2.0 * lapcT(0, 1) * w_D1_[p] * dx_inv;
            D2_xz_[p] = 2.0 * lapcT(0, 2) * w_D1_[p] * dx_inv;
            D2_yz_[p] = 2.0 * lapcT(1, 2) * w_D1_[p] * dy_inv;
        }
    }
}

void FDStencil::compute_max_eigval(const FDGrid& grid, const Lattice& lattice) {
    int FDn = order_ / 2;

    if (lattice.is_orthogonal()) {
        // Max eigenvalue of -0.5*Laplacian using Nyquist-like frequency
        double scal_x = (grid.Nx() - grid.Nx() % 2) / static_cast<double>(grid.Nx());
        double scal_y = (grid.Ny() - grid.Ny() % 2) / static_cast<double>(grid.Ny());
        double scal_z = (grid.Nz() - grid.Nz() % 2) / static_cast<double>(grid.Nz());

        max_eigval_ = D2_x_[0] + D2_y_[0] + D2_z_[0];
        for (int p = 1; p <= FDn; ++p) {
            max_eigval_ += 2.0 * (D2_x_[p] * std::cos(constants::PI * p * scal_x)
                                + D2_y_[p] * std::cos(constants::PI * p * scal_y)
                                + D2_z_[p] * std::cos(constants::PI * p * scal_z));
        }
        max_eigval_ *= -0.5;
    } else {
        // For non-orthogonal: conservative upper bound
        // Sum absolute values of all coefficients
        max_eigval_ = std::abs(D2_x_[0]) + std::abs(D2_y_[0]) + std::abs(D2_z_[0]);
        for (int p = 1; p <= FDn; ++p) {
            max_eigval_ += 2.0 * (std::abs(D2_x_[p]) + std::abs(D2_y_[p]) + std::abs(D2_z_[p]));
            max_eigval_ += 2.0 * (std::abs(D2_xy_[p]) + std::abs(D2_xz_[p]) + std::abs(D2_yz_[p]));
        }
        max_eigval_ *= 0.5;
    }
}

} // namespace sparc
