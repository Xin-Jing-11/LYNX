#include <gtest/gtest.h>
#include "operators/Laplacian.hpp"
#include "operators/FDStencil.hpp"
#include "parallel/HaloExchange.hpp"
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include "core/Domain.hpp"
#include "core/constants.hpp"
#include <cmath>
#include <vector>

using namespace sparc;

// Helper: create a simple single-process test setup
struct LapTestSetup {
    Mat3 lv;
    Lattice lat;
    FDGrid grid;
    DomainVertices verts;
    Domain domain;
    FDStencil stencil;
    HaloExchange halo;
    Laplacian lap;

    LapTestSetup(int N = 20, int order = 12) {
        lv = Mat3{};
        double L = 2.0 * constants::PI;  // convenient for trig functions
        lv(0, 0) = L; lv(1, 1) = L; lv(2, 2) = L;
        lat = Lattice(lv, CellType::Orthogonal);
        grid = FDGrid(N, N, N, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);
        verts = {0, N - 1, 0, N - 1, 0, N - 1};
        domain = Domain(grid, verts);
        stencil = FDStencil(order, grid, lat);
        halo = HaloExchange(domain, stencil.FDn(), MPI_COMM_NULL);
        lap = Laplacian(stencil, domain);
    }
};

TEST(Laplacian, ConstantFunction) {
    // Laplacian of a constant should be zero (plus c*constant)
    LapTestSetup s(20, 12);
    int nd = s.domain.Nd_d();
    std::vector<double> f(nd, 5.0);

    int FDn = s.stencil.FDn();
    int nx_ex = s.halo.nx_ex();
    int ny_ex = s.halo.ny_ex();
    int nz_ex = s.halo.nz_ex();
    int nd_ex = nx_ex * ny_ex * nz_ex;

    std::vector<double> f_ex(nd_ex, 0.0);
    s.halo.execute(f.data(), f_ex.data(), 1);

    std::vector<double> y(nd, 0.0);
    s.lap.apply(f_ex.data(), y.data(), 1.0, 0.0, 1);

    // Lap(const) = 0, so y should be c*f = 0
    for (int i = 0; i < nd; ++i) {
        EXPECT_NEAR(y[i], 0.0, 1e-10) << "at index " << i;
    }
}

TEST(Laplacian, ConstantWithShift) {
    // (Lap + c*I) applied to constant f → c*f
    LapTestSetup s(20, 12);
    int nd = s.domain.Nd_d();
    double f_val = 3.0, c_val = 2.5;
    std::vector<double> f(nd, f_val);

    int nd_ex = s.halo.nd_ex();
    std::vector<double> f_ex(nd_ex, 0.0);
    s.halo.execute(f.data(), f_ex.data(), 1);

    std::vector<double> y(nd, 0.0);
    s.lap.apply(f_ex.data(), y.data(), 1.0, c_val, 1);

    for (int i = 0; i < nd; ++i) {
        EXPECT_NEAR(y[i], c_val * f_val, 1e-10);
    }
}

TEST(Laplacian, SinFunction) {
    // Lap(sin(kx)) = -k^2 * sin(kx)
    // Use k=1 (one period in [0, 2*pi))
    int N = 40;
    LapTestSetup s(N, 12);
    int nx = s.domain.Nx_d();
    int ny = s.domain.Ny_d();
    int nz = s.domain.Nz_d();
    int nd = nx * ny * nz;
    double dx = s.grid.dx();

    std::vector<double> f(nd, 0.0);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                double x = i * dx;
                f[i + j * nx + k * nx * ny] = std::sin(x);
            }

    int nd_ex = s.halo.nd_ex();
    std::vector<double> f_ex(nd_ex, 0.0);
    s.halo.execute(f.data(), f_ex.data(), 1);

    std::vector<double> y(nd, 0.0);
    s.lap.apply(f_ex.data(), y.data(), 1.0, 0.0, 1);

    // Expected: -sin(x)  (k^2 = 1)
    double max_err = 0.0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                double x = i * dx;
                double expected = -std::sin(x);
                int idx = i + j * nx + k * nx * ny;
                double err = std::abs(y[idx] - expected);
                max_err = std::max(max_err, err);
            }

    // FD order 12 on 40-point grid should give ~1e-8 accuracy for sin
    EXPECT_LT(max_err, 1e-6) << "Max error: " << max_err;
}

TEST(Laplacian, MultiColumn) {
    LapTestSetup s(20, 12);
    int nd = s.domain.Nd_d();
    int ncol = 3;

    std::vector<double> f(nd * ncol, 2.0);
    int nd_ex = s.halo.nd_ex();
    std::vector<double> f_ex(nd_ex * ncol, 0.0);
    s.halo.execute(f.data(), f_ex.data(), ncol);

    std::vector<double> y(nd * ncol, 0.0);
    double c = 1.5;
    s.lap.apply(f_ex.data(), y.data(), 1.0, c, ncol);

    // Each column: (Lap + c)*const = c*const
    for (int n = 0; n < ncol; ++n)
        for (int i = 0; i < nd; ++i)
            EXPECT_NEAR(y[n * nd + i], c * 2.0, 1e-10);
}
