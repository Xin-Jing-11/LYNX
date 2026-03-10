#include <gtest/gtest.h>
#include "operators/FDStencil.hpp"
#include <cmath>

using namespace sparc;

TEST(FDStencil, Order6Weights) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);
    FDGrid grid(20, 20, 20, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);

    FDStencil stencil(6, grid, lat);

    EXPECT_EQ(stencil.order(), 6);
    EXPECT_EQ(stencil.FDn(), 3);

    // First derivative weights for order 6 (FDn=3):
    // w1[0] = 0
    // w1[1] = fract(3,1)/1 = (3!/(2!*4!)) / 1 = 3/24 = 1/8 = 0.125  ... wait
    // fract(n,k) = product(n-k+1..n) / product(n+1..n+k)
    // fract(3,1) = 3 / 4 = 0.75
    // w1[1] = +0.75/1 = 0.75
    // fract(3,2) = (2*3)/(4*5) = 6/20 = 0.3
    // w1[2] = -0.3/2 = -0.15
    // fract(3,3) = (1*2*3)/(4*5*6) = 6/120 = 0.05
    // w1[3] = +0.05/3 ≈ 0.01667

    const double* w1 = stencil.weights_D1();
    EXPECT_DOUBLE_EQ(w1[0], 0.0);
    EXPECT_NEAR(w1[1], 0.75, 1e-12);
    EXPECT_NEAR(w1[2], -0.15, 1e-12);
    EXPECT_NEAR(w1[3], 1.0 / 60.0, 1e-12);
}

TEST(FDStencil, Order6SecondDerivWeights) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);
    FDGrid grid(20, 20, 20, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);

    FDStencil stencil(6, grid, lat);

    const double* w2 = stencil.weights_D2();

    // w2[0] = -(2/1 + 2/4 + 2/9) = -(2 + 0.5 + 0.2222..) = -2.7222..
    EXPECT_NEAR(w2[0], -(2.0 + 0.5 + 2.0 / 9.0), 1e-12);

    // w2[1] = 2*fract(3,1)/1 = 2*0.75 = 1.5
    EXPECT_NEAR(w2[1], 1.5, 1e-12);

    // w2[2] = -2*fract(3,2)/4 = -2*0.3/4 = -0.15
    EXPECT_NEAR(w2[2], -0.15, 1e-12);

    // w2[3] = 2*fract(3,3)/9 = 2*0.05/9 = 0.01111..
    EXPECT_NEAR(w2[3], 2.0 * 0.05 / 9.0, 1e-12);
}

TEST(FDStencil, ScaledCoefficients) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);
    FDGrid grid(20, 20, 20, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);

    FDStencil stencil(12, grid, lat);

    double dx = grid.dx();  // 0.5
    const double* w2 = stencil.weights_D2();
    const double* cx = stencil.D2_coeff_x();

    // D2_coeff_x[p] = w2[p] / dx^2
    for (int p = 0; p <= stencil.FDn(); ++p) {
        EXPECT_NEAR(cx[p], w2[p] / (dx * dx), 1e-10);
    }
}

TEST(FDStencil, MaxEigvalPositive) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);
    FDGrid grid(20, 20, 20, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);

    FDStencil stencil(12, grid, lat);

    // Max eigenvalue of -0.5*Laplacian should be positive
    EXPECT_GT(stencil.max_eigval_half_lap(), 0.0);
}

TEST(FDStencil, SumOfSecondDerivWeightsConsistency) {
    // For second derivative FD: sum of all weights (center + 2*off-center) should be 0
    // This is because applying to constant function gives 0
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);
    FDGrid grid(20, 20, 20, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);

    FDStencil stencil(12, grid, lat);
    const double* w2 = stencil.weights_D2();

    double sum = w2[0];
    for (int p = 1; p <= stencil.FDn(); ++p)
        sum += 2.0 * w2[p];

    EXPECT_NEAR(sum, 0.0, 1e-12);
}

TEST(FDStencil, Order12) {
    Mat3 lv;
    lv(0, 0) = 7.63; lv(1, 1) = 7.63; lv(2, 2) = 7.63;
    Lattice lat(lv, CellType::Orthogonal);
    FDGrid grid(30, 30, 30, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);

    FDStencil stencil(12, grid, lat);
    EXPECT_EQ(stencil.FDn(), 6);

    // Verify positive max eigval
    EXPECT_GT(stencil.max_eigval_half_lap(), 0.0);
}
