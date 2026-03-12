#pragma once

#include "types.hpp"

namespace sparc {

class Lattice {
public:
    Lattice() = default;
    Lattice(const Mat3& latvec, CellType cell_type);

    const Mat3& latvec() const { return latvec_; }
    const Mat3& metric_tensor() const { return metric_T_; }
    const Mat3& grad_T() const { return grad_T_; }
    const Mat3& lapc_T() const { return lapc_T_; }
    const Mat3& lat_uvec() const { return lat_uvec_; }
    const Mat3& lat_uvec_inv() const { return lat_uvec_inv_; }
    double jacobian() const { return jacobian_; }
    CellType cell_type() const { return cell_type_; }
    bool is_orthogonal() const { return cell_type_ == CellType::Orthogonal; }
    Vec3 lengths() const;

    Vec3 frac_to_cart(const Vec3& frac) const;
    Vec3 cart_to_frac(const Vec3& cart) const;

    // Non-Cartesian coords: scaled fractional (x_nc = frac * |a_i|)
    // Grid point (i,j,k) has non-Cart coords (i*dx, j*dy, k*dz)
    Vec3 cart_to_nonCart(const Vec3& cart) const;
    Vec3 nonCart_to_cart(const Vec3& nonCart) const;

    // Distance using metric tensor (for non-Cart coordinate differences)
    double metric_distance(double dx, double dy, double dz) const;

    Mat3 reciprocal_latvec() const;

    // For a Cartesian sphere of radius R, compute the maximum extent in
    // non-Cartesian coordinates for each direction. Returns factors f_i such
    // that the bounding box in non-Cart coords is [-R*f_i, +R*f_i].
    // Since nc = (U^{-1})^T * cart, f_i = ||column_i(U^{-1})|| = sqrt(sum_j (U^{-1}_{ji})^2)
    Vec3 nonCart_sphere_extent() const { return nc_sphere_extent_; }

private:
    Mat3 latvec_;
    Mat3 lat_uvec_;
    Mat3 lat_uvec_inv_;
    Mat3 metric_T_;
    Mat3 grad_T_;
    Mat3 lapc_T_;
    Vec3 nc_sphere_extent_ = {1.0, 1.0, 1.0};
    double jacobian_ = 1.0;
    CellType cell_type_ = CellType::Orthogonal;

    void compute_derived();
};

} // namespace sparc
