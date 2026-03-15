#include <gtest/gtest.h>
#include "core/Lattice.hpp"
#include "core/constants.hpp"
#include <cmath>

using namespace lynx;

TEST(Lattice, OrthogonalBasic) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);

    EXPECT_TRUE(lat.is_orthogonal());
    EXPECT_DOUBLE_EQ(lat.jacobian(), 1000.0);

    Vec3 L = lat.lengths();
    EXPECT_DOUBLE_EQ(L.x, 10.0);
    EXPECT_DOUBLE_EQ(L.y, 10.0);
    EXPECT_DOUBLE_EQ(L.z, 10.0);
}

TEST(Lattice, OrthogonalMetricTensor) {
    Mat3 lv;
    lv(0, 0) = 5.0; lv(1, 1) = 7.0; lv(2, 2) = 9.0;
    Lattice lat(lv, CellType::Orthogonal);

    // metric_T = L^T * L → diagonal with squares of lengths
    auto& mt = lat.metric_tensor();
    EXPECT_NEAR(mt(0, 0), 25.0, 1e-12);
    EXPECT_NEAR(mt(1, 1), 49.0, 1e-12);
    EXPECT_NEAR(mt(2, 2), 81.0, 1e-12);
    EXPECT_NEAR(mt(0, 1), 0.0, 1e-12);
}

TEST(Lattice, FracToCart) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);

    Vec3 frac(0.5, 0.5, 0.5);
    Vec3 cart = lat.frac_to_cart(frac);
    EXPECT_NEAR(cart.x, 5.0, 1e-12);
    EXPECT_NEAR(cart.y, 5.0, 1e-12);
    EXPECT_NEAR(cart.z, 5.0, 1e-12);
}

TEST(Lattice, CartToFrac) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);

    Vec3 cart(5.0, 5.0, 5.0);
    Vec3 frac = lat.cart_to_frac(cart);
    EXPECT_NEAR(frac.x, 0.5, 1e-12);
    EXPECT_NEAR(frac.y, 0.5, 1e-12);
    EXPECT_NEAR(frac.z, 0.5, 1e-12);
}

TEST(Lattice, FracCartRoundTrip) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);

    Vec3 orig(0.3, 0.7, 0.1);
    Vec3 cart = lat.frac_to_cart(orig);
    Vec3 back = lat.cart_to_frac(cart);
    EXPECT_NEAR(back.x, orig.x, 1e-12);
    EXPECT_NEAR(back.y, orig.y, 1e-12);
    EXPECT_NEAR(back.z, orig.z, 1e-12);
}

TEST(Lattice, ReciprocalLattice) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);

    Mat3 recip = lat.reciprocal_latvec();
    // For cubic: b_i = 2*pi/a_i along diagonal
    EXPECT_NEAR(recip(0, 0), 2.0 * constants::PI / 10.0, 1e-12);
    EXPECT_NEAR(recip(1, 1), 2.0 * constants::PI / 10.0, 1e-12);
    EXPECT_NEAR(recip(2, 2), 2.0 * constants::PI / 10.0, 1e-12);
    EXPECT_NEAR(recip(0, 1), 0.0, 1e-12);
}

TEST(Lattice, NonOrthogonal) {
    // FCC-like lattice
    Mat3 lv;
    lv(0, 0) = 0.0; lv(0, 1) = 5.0; lv(0, 2) = 5.0;
    lv(1, 0) = 5.0; lv(1, 1) = 0.0; lv(1, 2) = 5.0;
    lv(2, 0) = 5.0; lv(2, 1) = 5.0; lv(2, 2) = 0.0;
    Lattice lat(lv, CellType::NonOrthogonal);

    EXPECT_FALSE(lat.is_orthogonal());
    EXPECT_NEAR(lat.jacobian(), 250.0, 1e-10);  // det of FCC

    // Round trip
    Vec3 frac(0.25, 0.75, 0.5);
    Vec3 cart = lat.frac_to_cart(frac);
    Vec3 back = lat.cart_to_frac(cart);
    EXPECT_NEAR(back.x, frac.x, 1e-10);
    EXPECT_NEAR(back.y, frac.y, 1e-10);
    EXPECT_NEAR(back.z, frac.z, 1e-10);
}

TEST(Lattice, ZeroVolumeThrows) {
    Mat3 lv;
    lv(0, 0) = 1.0; lv(0, 1) = 0.0; lv(0, 2) = 0.0;
    lv(1, 0) = 2.0; lv(1, 1) = 0.0; lv(1, 2) = 0.0;  // parallel to row 0
    lv(2, 0) = 0.0; lv(2, 1) = 0.0; lv(2, 2) = 1.0;
    EXPECT_THROW(Lattice(lv, CellType::Orthogonal), std::runtime_error);
}

TEST(Lattice, LapcT_Orthogonal) {
    // lapcT = (LatUVec^{-1})^T * LatUVec^{-1}
    // For orthogonal cells, LatUVec = I (unit vectors), so lapcT = I.
    // The 1/dx^2 scaling is applied separately in FDStencil, not in lapcT.
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);

    auto& lt = lat.lapc_T();
    EXPECT_NEAR(lt(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(lt(1, 1), 1.0, 1e-12);
    EXPECT_NEAR(lt(2, 2), 1.0, 1e-12);
    EXPECT_NEAR(lt(0, 1), 0.0, 1e-12);
}
