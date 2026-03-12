#include <gtest/gtest.h>
#include "parallel/HaloExchange.hpp"
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include "core/Domain.hpp"
#include <vector>
#include <cmath>

using namespace sparc;

TEST(HaloExchange, PeriodicSingleProcess) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);
    FDGrid grid(10, 10, 10, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);
    DomainVertices v{0, 9, 0, 9, 0, 9};
    Domain domain(grid, v);

    int FDn = 3;
    HaloExchange halo(domain, FDn);

    EXPECT_EQ(halo.nx_ex(), 16);
    EXPECT_EQ(halo.ny_ex(), 16);
    EXPECT_EQ(halo.nz_ex(), 16);

    int nd = 10 * 10 * 10;
    std::vector<double> x(nd);
    for (int i = 0; i < nd; ++i) x[i] = i + 1.0;

    int nd_ex = halo.nd_ex();
    std::vector<double> x_ex(nd_ex, -1.0);
    halo.execute(x.data(), x_ex.data(), 1);

    // Check interior is preserved
    int nx = 10, ny = 10;
    int nx_ex = 16, ny_ex = 16;
    for (int k = 0; k < 10; ++k)
        for (int j = 0; j < 10; ++j)
            for (int i = 0; i < 10; ++i) {
                int loc = i + j * nx + k * nx * ny;
                int ext = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nx_ex * ny_ex;
                EXPECT_DOUBLE_EQ(x_ex[ext], x[loc])
                    << "Interior mismatch at (" << i << "," << j << "," << k << ")";
            }

    // Check periodic ghost in x direction:
    // x_ex[FDn-1, FDn, FDn] should equal x[nx-1, 0, 0]
    {
        int ext = (FDn - 1) + FDn * nx_ex + FDn * nx_ex * ny_ex;
        int loc = (nx - 1) + 0 * nx + 0 * nx * ny;
        EXPECT_DOUBLE_EQ(x_ex[ext], x[loc]) << "Periodic ghost x-low failed";
    }

    // x_ex[FDn+nx, FDn, FDn] should equal x[0, 0, 0]
    {
        int ext = (FDn + nx) + FDn * nx_ex + FDn * nx_ex * ny_ex;
        int loc = 0 + 0 * nx + 0 * nx * ny;
        EXPECT_DOUBLE_EQ(x_ex[ext], x[loc]) << "Periodic ghost x-high failed";
    }
}

TEST(HaloExchange, ExtendedDimensions) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);
    FDGrid grid(20, 30, 40, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);
    DomainVertices v{0, 19, 0, 29, 0, 39};
    Domain domain(grid, v);

    HaloExchange halo(domain, 6);
    EXPECT_EQ(halo.nx_ex(), 32);
    EXPECT_EQ(halo.ny_ex(), 42);
    EXPECT_EQ(halo.nz_ex(), 52);
}
