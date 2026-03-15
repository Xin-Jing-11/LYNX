#pragma once

#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "core/types.hpp"
#include <complex>
#include <vector>

namespace lynx {

using Complex = std::complex<double>;

// Manages ghost/halo zone exchange for finite-difference stencil operations.
// For a domain of size (nx, ny, nz) with stencil half-width FDn,
// the extended domain has size (nx+2*FDn, ny+2*FDn, nz+2*FDn).
// Periodic boundaries are handled by wrapping.
class HaloExchange {
public:
    HaloExchange() = default;

    // Setup for a given domain and stencil half-width
    HaloExchange(const Domain& domain, int FDn);

    // Fill ghost zones of x_ex from local data x using periodic wrapping (real, Gamma-only).
    void execute(const double* x, double* x_ex, int ncol) const;

    // Fill ghost zones with Bloch phase factors for k-point calculations (complex).
    // kpt_cart: k-point in Cartesian reciprocal coords (k1, k2, k3 in 2π/L units)
    // cell_lengths: (Lx, Ly, Lz) cell lengths
    // Phase convention: left ghost *= e^{-ik·L}, right ghost *= e^{+ik·L}
    void execute_kpt(const Complex* x, Complex* x_ex, int ncol,
                     const Vec3& kpt_cart, const Vec3& cell_lengths) const;

    // Dimensions of extended domain
    int nx_ex() const { return nx_ex_; }
    int ny_ex() const { return ny_ex_; }
    int nz_ex() const { return nz_ex_; }
    int nd_ex() const { return nx_ex_ * ny_ex_ * nz_ex_; }

    int FDn() const { return FDn_; }

private:
    int nx_ = 0, ny_ = 0, nz_ = 0;
    int nx_ex_ = 0, ny_ex_ = 0, nz_ex_ = 0;
    int FDn_ = 0;
    bool periods_[3] = {true, true, true};

    // Copy local data into extended array interior
    void copy_to_interior(const double* x, double* x_ex, int ncol) const;
    void copy_to_interior(const Complex* x, Complex* x_ex, int ncol) const;

    // Wrap periodic boundaries
    void apply_periodic_bc(double* x_ex, int ncol) const;

    // Wrap periodic boundaries with Bloch phase factors
    void apply_periodic_bc_kpt(Complex* x_ex, int ncol,
                               Complex phase_l_x, Complex phase_r_x,
                               Complex phase_l_y, Complex phase_r_y,
                               Complex phase_l_z, Complex phase_r_z) const;
};

} // namespace lynx
