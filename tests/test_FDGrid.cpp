#include <gtest/gtest.h>
#include "core/FDGrid.hpp"

using namespace sparc;

TEST(FDGrid, BasicPeriodic) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);

    FDGrid grid(20, 20, 20, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);

    EXPECT_EQ(grid.Nx(), 20);
    EXPECT_EQ(grid.Ny(), 20);
    EXPECT_EQ(grid.Nz(), 20);
    EXPECT_EQ(grid.Nd(), 8000);

    // dx = L / N for periodic
    EXPECT_NEAR(grid.dx(), 0.5, 1e-12);
    EXPECT_NEAR(grid.dy(), 0.5, 1e-12);
    EXPECT_NEAR(grid.dz(), 0.5, 1e-12);
}

TEST(FDGrid, DirichletSpacing) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);

    FDGrid grid(11, 11, 11, lat, BCType::Dirichlet, BCType::Dirichlet, BCType::Dirichlet);

    // dx = L / (N-1) for Dirichlet
    EXPECT_NEAR(grid.dx(), 1.0, 1e-12);
}

TEST(FDGrid, DV) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);

    FDGrid grid(20, 20, 20, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);

    // dV = Volume / (Nx * Ny * Nz)
    EXPECT_NEAR(grid.dV(), 1000.0 / 8000.0, 1e-12);
}

TEST(FDGrid, BoundaryConditions) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);

    FDGrid grid(20, 20, 20, lat, BCType::Periodic, BCType::Dirichlet, BCType::Periodic);
    EXPECT_EQ(grid.bcx(), BCType::Periodic);
    EXPECT_EQ(grid.bcy(), BCType::Dirichlet);
    EXPECT_EQ(grid.bcz(), BCType::Periodic);
}

TEST(FDGrid, InvalidDimensions) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);

    EXPECT_THROW(FDGrid(0, 10, 10, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic),
                 std::invalid_argument);
}

TEST(FDGrid, NonUniform) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 20.0; lv(2, 2) = 30.0;
    Lattice lat(lv, CellType::Orthogonal);

    FDGrid grid(20, 40, 60, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);
    EXPECT_NEAR(grid.dx(), 0.5, 1e-12);
    EXPECT_NEAR(grid.dy(), 0.5, 1e-12);
    EXPECT_NEAR(grid.dz(), 0.5, 1e-12);
}
