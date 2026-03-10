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
    double jacobian() const { return jacobian_; }
    CellType cell_type() const { return cell_type_; }
    bool is_orthogonal() const { return cell_type_ == CellType::Orthogonal; }
    Vec3 lengths() const;

    Vec3 frac_to_cart(const Vec3& frac) const;
    Vec3 cart_to_frac(const Vec3& cart) const;

    Mat3 reciprocal_latvec() const;

private:
    Mat3 latvec_;
    Mat3 lat_uvec_;
    Mat3 metric_T_;
    Mat3 grad_T_;
    Mat3 lapc_T_;
    double jacobian_ = 1.0;
    CellType cell_type_ = CellType::Orthogonal;

    void compute_derived();
};

} // namespace sparc
