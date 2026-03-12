#include "core/Lattice.hpp"
#include "core/constants.hpp"
#include <cmath>
#include <stdexcept>

namespace sparc {

Lattice::Lattice(const Mat3& latvec, CellType cell_type)
    : latvec_(latvec), cell_type_(cell_type) {
    compute_derived();
}

void Lattice::compute_derived() {
    // Compute lattice vector lengths and unit vectors
    for (int i = 0; i < 3; ++i) {
        double len = 0.0;
        for (int j = 0; j < 3; ++j)
            len += latvec_(i, j) * latvec_(i, j);
        len = std::sqrt(len);
        if (len < 1e-14)
            throw std::runtime_error("Lattice vector has zero length");
        for (int j = 0; j < 3; ++j)
            lat_uvec_(i, j) = latvec_(i, j) / len;
    }

    // Jacobian = |det(latvec)|
    jacobian_ = std::abs(latvec_.determinant());
    if (jacobian_ < 1e-14)
        throw std::runtime_error("Lattice vectors are coplanar (zero volume)");

    // gradT = inverse(latvec)^T  (transforms Cartesian gradient to lattice coords)
    Mat3 inv = latvec_.inverse();
    grad_T_ = inv.transpose();

    // Metric tensor: metric_T = latvec^T * latvec
    Mat3 ltrans = latvec_.transpose();
    metric_T_ = ltrans * latvec_;

    // Laplacian transformation: lapc_T = (LatUVec^{-1})^T * LatUVec^{-1}
    // Must use unit lattice vectors (not full latvec), matching reference SPARC
    lat_uvec_inv_ = lat_uvec_.inverse();
    lapc_T_ = lat_uvec_inv_.transpose() * lat_uvec_inv_;

    // Non-Cart sphere extent factors: ||column_i(U^{-1})||
    // Since nc = (U^{-1})^T * cart, by Cauchy-Schwarz:
    //   |nc_i| = |column_i(U^{-1}) · cart| ≤ ||column_i(U^{-1})|| * ||cart||
    // So f_i = ||column_i(U^{-1})|| = sqrt(sum_j (U^{-1}_{ji})^2)
    for (int i = 0; i < 3; ++i) {
        double sum = 0.0;
        for (int j = 0; j < 3; ++j)
            sum += lat_uvec_inv_(j, i) * lat_uvec_inv_(j, i);
        (&nc_sphere_extent_.x)[i] = std::sqrt(sum);
    }
}

Vec3 Lattice::lengths() const {
    Vec3 L;
    L.x = std::sqrt(latvec_(0, 0) * latvec_(0, 0) +
                    latvec_(0, 1) * latvec_(0, 1) +
                    latvec_(0, 2) * latvec_(0, 2));
    L.y = std::sqrt(latvec_(1, 0) * latvec_(1, 0) +
                    latvec_(1, 1) * latvec_(1, 1) +
                    latvec_(1, 2) * latvec_(1, 2));
    L.z = std::sqrt(latvec_(2, 0) * latvec_(2, 0) +
                    latvec_(2, 1) * latvec_(2, 1) +
                    latvec_(2, 2) * latvec_(2, 2));
    return L;
}

Vec3 Lattice::frac_to_cart(const Vec3& frac) const {
    // cart = frac^T * latvec (row vectors)
    // cart_j = sum_i frac_i * latvec(i,j)
    Vec3 cart;
    cart.x = frac.x * latvec_(0, 0) + frac.y * latvec_(1, 0) + frac.z * latvec_(2, 0);
    cart.y = frac.x * latvec_(0, 1) + frac.y * latvec_(1, 1) + frac.z * latvec_(2, 1);
    cart.z = frac.x * latvec_(0, 2) + frac.y * latvec_(1, 2) + frac.z * latvec_(2, 2);
    return cart;
}

Vec3 Lattice::cart_to_frac(const Vec3& cart) const {
    Mat3 inv = latvec_.inverse();
    Vec3 frac;
    frac.x = cart.x * inv(0, 0) + cart.y * inv(1, 0) + cart.z * inv(2, 0);
    frac.y = cart.x * inv(0, 1) + cart.y * inv(1, 1) + cart.z * inv(2, 1);
    frac.z = cart.x * inv(0, 2) + cart.y * inv(1, 2) + cart.z * inv(2, 2);
    return frac;
}

Vec3 Lattice::cart_to_nonCart(const Vec3& cart) const {
    // Non-Cart coords = fractional * |a_i|
    Vec3 frac = cart_to_frac(cart);
    Vec3 L = lengths();
    return {frac.x * L.x, frac.y * L.y, frac.z * L.z};
}

Vec3 Lattice::nonCart_to_cart(const Vec3& nc) const {
    // Non-Cart to Cartesian: nc_i / |a_i| = frac_i, then frac_to_cart
    // Equivalent to: cart_j = sum_i (nc_i / |a_i|) * latvec(i,j)
    //              = sum_i nc_i * LatUVec(i,j)
    Vec3 cart;
    cart.x = nc.x * lat_uvec_(0, 0) + nc.y * lat_uvec_(1, 0) + nc.z * lat_uvec_(2, 0);
    cart.y = nc.x * lat_uvec_(0, 1) + nc.y * lat_uvec_(1, 1) + nc.z * lat_uvec_(2, 1);
    cart.z = nc.x * lat_uvec_(0, 2) + nc.y * lat_uvec_(1, 2) + nc.z * lat_uvec_(2, 2);
    return cart;
}

double Lattice::metric_distance(double dx, double dy, double dz) const {
    // Convert non-Cart coordinate difference to Cartesian, return Euclidean distance
    Vec3 cart = nonCart_to_cart({dx, dy, dz});
    return std::sqrt(cart.x*cart.x + cart.y*cart.y + cart.z*cart.z);
}

Mat3 Lattice::reciprocal_latvec() const {
    // Reciprocal lattice: b_i = 2*pi * (a_j x a_k) / V
    // Equivalent to 2*pi * inverse(latvec)^T
    double factor = 2.0 * constants::PI;
    Mat3 inv = latvec_.inverse();
    Mat3 recip;
    for (int i = 0; i < 9; ++i)
        recip.data[i] = factor * inv.data[i];
    // We want rows of recip to be reciprocal vectors, so transpose
    return recip.transpose();
}

} // namespace sparc
