#pragma once

#include <vector>
#include "core/FDGrid.hpp"
#include "core/Lattice.hpp"

namespace sparc {

class FDStencil {
public:
    FDStencil() = default;
    FDStencil(int order, const FDGrid& grid, const Lattice& lattice);

    int order() const { return order_; }
    int FDn() const { return order_ / 2; }

    const double* weights_D1() const { return w_D1_.data(); }
    const double* weights_D2() const { return w_D2_.data(); }

    const double* D2_coeff_x() const { return D2_x_.data(); }
    const double* D2_coeff_y() const { return D2_y_.data(); }
    const double* D2_coeff_z() const { return D2_z_.data(); }

    const double* D2_coeff_xy() const { return D2_xy_.data(); }
    const double* D2_coeff_xz() const { return D2_xz_.data(); }
    const double* D2_coeff_yz() const { return D2_yz_.data(); }

    const double* D1_coeff_x() const { return D1_x_.data(); }
    const double* D1_coeff_y() const { return D1_y_.data(); }
    const double* D1_coeff_z() const { return D1_z_.data(); }

    double max_eigval_half_lap() const { return max_eigval_; }

private:
    int order_ = 12;
    std::vector<double> w_D1_, w_D2_;
    std::vector<double> D2_x_, D2_y_, D2_z_;
    std::vector<double> D2_xy_, D2_xz_, D2_yz_;
    std::vector<double> D1_x_, D1_y_, D1_z_;
    double max_eigval_ = 0.0;

    void compute_weights();
    void scale_for_grid(const FDGrid& grid, const Lattice& lattice);
    void compute_max_eigval(const FDGrid& grid, const Lattice& lattice);

    // fract(n,k) = n! / ((n-k)! * (n+k)!)
    static double fract(int n, int k);
};

} // namespace sparc
