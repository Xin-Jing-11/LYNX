#include <gtest/gtest.h>
#include "operators/Hamiltonian.hpp"
#include "operators/FDStencil.hpp"
#include "parallel/HaloExchange.hpp"
#include "parallel/MPIComm.hpp"
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include "core/Domain.hpp"
#include "core/constants.hpp"
#include <cmath>
#include <vector>

using namespace lynx;

struct HamTestSetup {
    Mat3 lv;
    Lattice lat;
    FDGrid grid;
    DomainVertices verts;
    Domain domain;
    FDStencil stencil;
    HaloExchange halo;
    Hamiltonian ham;

    HamTestSetup(int N = 20) {
        double L = 2.0 * constants::PI;
        lv = Mat3{};
        lv(0, 0) = L; lv(1, 1) = L; lv(2, 2) = L;
        lat = Lattice(lv, CellType::Orthogonal);
        grid = FDGrid(N, N, N, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);
        verts = {0, N - 1, 0, N - 1, 0, N - 1};
        domain = Domain(grid, verts);
        stencil = FDStencil(12, grid, lat);
        halo = HaloExchange(domain, stencil.FDn());
        ham.setup(stencil, domain, grid, halo, nullptr);
    }
};

TEST(Hamiltonian, FreeParticle) {
    // H = -0.5*Lap, Veff = 0
    // Apply to sin(x): should give 0.5*sin(x)
    int N = 40;
    HamTestSetup s(N);
    int nd = s.domain.Nd_d();
    double dx = s.grid.dx();

    std::vector<double> psi(nd, 0.0);
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                double x = i * dx;
                psi[i + j * N + k * N * N] = std::sin(x);
            }

    std::vector<double> Veff(nd, 0.0);
    std::vector<double> Hpsi(nd, 0.0);

    s.ham.apply(psi.data(), Veff.data(), Hpsi.data(), 1);

    // Expected: 0.5 * sin(x)
    double max_err = 0.0;
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                double x = i * dx;
                double expected = 0.5 * std::sin(x);
                int idx = i + j * N + k * N * N;
                max_err = std::max(max_err, std::abs(Hpsi[idx] - expected));
            }

    EXPECT_LT(max_err, 1e-6) << "Max error: " << max_err;
}

TEST(Hamiltonian, WithPotential) {
    // H = -0.5*Lap + V, psi = constant
    // H*const = V*const (since Lap(const)=0)
    int N = 20;
    HamTestSetup s(N);
    int nd = s.domain.Nd_d();

    double psi_val = 2.0;
    std::vector<double> psi(nd, psi_val);
    std::vector<double> Veff(nd);
    for (int i = 0; i < nd; ++i)
        Veff[i] = 0.5 * (i % 10);

    std::vector<double> Hpsi(nd, 0.0);
    s.ham.apply(psi.data(), Veff.data(), Hpsi.data(), 1);

    for (int i = 0; i < nd; ++i) {
        EXPECT_NEAR(Hpsi[i], Veff[i] * psi_val, 1e-10)
            << "at index " << i;
    }
}

TEST(Hamiltonian, MultiColumn) {
    int N = 20;
    HamTestSetup s(N);
    int nd = s.domain.Nd_d();
    int ncol = 2;

    std::vector<double> psi(nd * ncol, 1.0);
    std::vector<double> Veff(nd, 3.0);
    std::vector<double> Hpsi(nd * ncol, 0.0);

    s.ham.apply(psi.data(), Veff.data(), Hpsi.data(), ncol);

    // Lap(const)=0, so H*const = V*const = 3.0
    for (int n = 0; n < ncol; ++n)
        for (int i = 0; i < nd; ++i)
            EXPECT_NEAR(Hpsi[n * nd + i], 3.0, 1e-10);
}
