#include <gtest/gtest.h>
#include "core/Domain.hpp"

using namespace lynx;

TEST(Domain, BasicConstruction) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);
    FDGrid grid(20, 20, 20, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);

    DomainVertices v{0, 9, 0, 9, 0, 9};
    Domain dom(grid, v);

    EXPECT_EQ(dom.Nx_d(), 10);
    EXPECT_EQ(dom.Ny_d(), 10);
    EXPECT_EQ(dom.Nz_d(), 10);
    EXPECT_EQ(dom.Nd_d(), 1000);
}

TEST(Domain, FlatIndex) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);
    FDGrid grid(20, 20, 20, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);

    DomainVertices v{0, 4, 0, 4, 0, 4};
    Domain dom(grid, v);

    EXPECT_EQ(dom.flat_index(0, 0, 0), 0);
    EXPECT_EQ(dom.flat_index(1, 0, 0), 1);
    EXPECT_EQ(dom.flat_index(0, 1, 0), 5);
    EXPECT_EQ(dom.flat_index(0, 0, 1), 25);
}

TEST(Domain, LocalToGlobal) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);
    FDGrid grid(20, 20, 20, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);

    DomainVertices v{10, 19, 10, 19, 0, 9};
    Domain dom(grid, v);

    // local (0,0,0) → global (10, 10, 0)
    int g = dom.local_to_global(0, 0, 0);
    int expected = 10 + 10 * 20 + 0 * 20 * 20;
    EXPECT_EQ(g, expected);
}

TEST(Domain, VerticesAccessor) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);
    FDGrid grid(20, 20, 20, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);

    DomainVertices v{5, 14, 3, 12, 0, 19};
    Domain dom(grid, v);

    auto& vv = dom.vertices();
    EXPECT_EQ(vv.xs, 5);
    EXPECT_EQ(vv.xe, 14);
    EXPECT_EQ(vv.ys, 3);
    EXPECT_EQ(vv.ye, 12);
}
