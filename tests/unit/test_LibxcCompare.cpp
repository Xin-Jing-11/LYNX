#include <gtest/gtest.h>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <cstdio>

#include "xc/XCFunctional.hpp"
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include "core/Domain.hpp"
#include "operators/FDStencil.hpp"
#include "operators/Gradient.hpp"
#include "parallel/HaloExchange.hpp"

using namespace lynx;

struct XCTestFixture : public ::testing::Test {
    Lattice lattice;
    FDGrid grid;
    Domain domain;
    FDStencil stencil;
    HaloExchange halo;
    Gradient gradient;
    int Nd_d;

    void SetUp() override {
        Mat3 lv{};
        lv(0,0) = 10.0; lv(1,1) = 10.0; lv(2,2) = 10.0;
        lattice = Lattice(lv, CellType::Orthogonal);
        grid = FDGrid(12, 12, 12, lattice, BCType::Periodic, BCType::Periodic, BCType::Periodic);
        stencil = FDStencil(12, grid, lattice);

        DomainVertices v{0, grid.Nx()-1, 0, grid.Ny()-1, 0, grid.Nz()-1};
        domain = Domain(grid, v);
        Nd_d = domain.Nd_d();

        halo = HaloExchange(domain, stencil.FDn());
        gradient = Gradient(stencil, domain);
    }

    std::vector<double> make_density(double rho0 = 0.05, double amp = 0.01) {
        std::vector<double> rho(Nd_d);
        int nx = domain.Nx_d(), ny = domain.Ny_d();
        for (int i = 0; i < Nd_d; i++) {
            int ix = i % nx, iy = (i / nx) % ny;
            rho[i] = rho0 + amp * std::sin(2*M_PI*ix/nx) * std::cos(2*M_PI*iy/ny);
            if (rho[i] < 1e-10) rho[i] = 1e-10;
        }
        return rho;
    }

    std::vector<double> make_spin_density(double rho0 = 0.05, double mag = 0.005) {
        auto rho_total = make_density(rho0);
        std::vector<double> rho(3 * Nd_d);
        for (int i = 0; i < Nd_d; i++) {
            double m = mag * std::sin(2*M_PI*i / Nd_d);
            double up = 0.5 * (rho_total[i] + m);
            double dn = 0.5 * (rho_total[i] - m);
            if (up < 1e-10) up = 1e-10;
            if (dn < 1e-10) dn = 1e-10;
            rho[i] = up + dn;
            rho[Nd_d + i] = up;
            rho[2*Nd_d + i] = dn;
        }
        return rho;
    }
};

TEST_F(XCTestFixture, LDA_PW_Nonspin) {
    auto rho = make_density();
    XCFunctional xc;
    xc.setup(XCType::LDA_PW, domain, grid);

    std::vector<double> Vxc(Nd_d), exc(Nd_d);
    xc.evaluate(rho.data(), Vxc.data(), exc.data(), Nd_d);

    // LDA PW: exc should be negative, Vxc should be negative
    for (int i = 0; i < Nd_d; i++) {
        EXPECT_TRUE(std::isfinite(exc[i]));
        EXPECT_TRUE(std::isfinite(Vxc[i]));
        EXPECT_LT(exc[i], 0.0);
        EXPECT_LT(Vxc[i], 0.0);
    }
}

TEST_F(XCTestFixture, LDA_PZ_Nonspin) {
    auto rho = make_density();
    XCFunctional xc;
    xc.setup(XCType::LDA_PZ, domain, grid);

    std::vector<double> Vxc(Nd_d), exc(Nd_d);
    xc.evaluate(rho.data(), Vxc.data(), exc.data(), Nd_d);

    for (int i = 0; i < Nd_d; i++) {
        EXPECT_TRUE(std::isfinite(exc[i]));
        EXPECT_LT(exc[i], 0.0);
    }
}

TEST_F(XCTestFixture, GGA_PBE_Nonspin) {
    auto rho = make_density();
    XCFunctional xc;
    xc.setup(XCType::GGA_PBE, domain, grid, &gradient, &halo);

    std::vector<double> Vxc(Nd_d), exc(Nd_d), dxc(Nd_d);
    xc.evaluate(rho.data(), Vxc.data(), exc.data(), Nd_d, dxc.data());

    for (int i = 0; i < Nd_d; i++) {
        EXPECT_TRUE(std::isfinite(exc[i]));
        EXPECT_TRUE(std::isfinite(Vxc[i]));
        EXPECT_TRUE(std::isfinite(dxc[i]));
        EXPECT_LT(exc[i], 0.0);
    }

    // Dxcdgrho should be non-trivial (not all zero)
    double dxc_norm = 0;
    for (int i = 0; i < Nd_d; i++) dxc_norm += dxc[i] * dxc[i];
    EXPECT_GT(dxc_norm, 0.0);
}

TEST_F(XCTestFixture, LDA_PW_Spin) {
    auto rho = make_spin_density();
    XCFunctional xc;
    xc.setup(XCType::LDA_PW, domain, grid);

    std::vector<double> Vxc(2*Nd_d), exc(Nd_d);
    xc.evaluate_spin(rho.data(), Vxc.data(), exc.data(), Nd_d);

    for (int i = 0; i < Nd_d; i++) {
        EXPECT_TRUE(std::isfinite(exc[i]));
        EXPECT_TRUE(std::isfinite(Vxc[i]));
        EXPECT_TRUE(std::isfinite(Vxc[Nd_d + i]));
        EXPECT_LT(exc[i], 0.0);
    }
}

TEST_F(XCTestFixture, GGA_PBE_Spin) {
    auto rho = make_spin_density();
    XCFunctional xc;
    xc.setup(XCType::GGA_PBE, domain, grid, &gradient, &halo);

    std::vector<double> Vxc(2*Nd_d), exc(Nd_d), dxc(3*Nd_d);
    xc.evaluate_spin(rho.data(), Vxc.data(), exc.data(), Nd_d, dxc.data());

    for (int i = 0; i < Nd_d; i++) {
        EXPECT_TRUE(std::isfinite(exc[i]));
        EXPECT_TRUE(std::isfinite(Vxc[i]));
        EXPECT_TRUE(std::isfinite(Vxc[Nd_d + i]));
        EXPECT_LT(exc[i], 0.0);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
