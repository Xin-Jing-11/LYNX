#include <gtest/gtest.h>
#include "electronic/ElectronDensity.hpp"
#include "electronic/Wavefunction.hpp"
#include "parallel/MPIComm.hpp"
#include <cmath>
#include <vector>

using namespace sparc;

TEST(ElectronDensity, Allocate) {
    ElectronDensity rho;
    rho.allocate(100, 1);
    EXPECT_EQ(rho.Nd_d(), 100);
    EXPECT_EQ(rho.Nspin(), 1);
    EXPECT_EQ(rho.rho_total().size(), 100);
}

TEST(ElectronDensity, AllocateSpin) {
    ElectronDensity rho;
    rho.allocate(50, 2);
    EXPECT_EQ(rho.Nspin(), 2);
    EXPECT_EQ(rho.mag().size(), 50);
}

TEST(ElectronDensity, ComputeFromWavefunction) {
    // Simple test: uniform orbitals → uniform density
    int Nd_d = 20;
    int Nband = 2;

    Wavefunction wfn;
    wfn.allocate(Nd_d, Nband, 1, 1);

    // Set orbitals to normalized constant: psi = 1/sqrt(Nd_d * dV)
    double dV = 0.1;
    double norm_val = 1.0 / std::sqrt(Nd_d * dV);
    for (int n = 0; n < Nband; ++n) {
        double* col = wfn.psi(0, 0).col(n);
        for (int i = 0; i < Nd_d; ++i) {
            col[i] = norm_val;
        }
    }

    // Set eigenvalues and occupations manually
    wfn.eigenvalues(0, 0)(0) = 0.1;
    wfn.eigenvalues(0, 0)(1) = 0.2;
    wfn.occupations(0, 0)(0) = 1.0;
    wfn.occupations(0, 0)(1) = 1.0;

    ElectronDensity density;
    density.allocate(Nd_d, 1);

    std::vector<double> kpt_weights = {1.0};
    MPIComm null_comm;
    density.compute(wfn, kpt_weights, dV, null_comm, null_comm);

    // Each point should have density = 2 * sum(f_n * |psi_n|^2)
    // = 2 * (1.0 + 1.0) * (1/(Nd_d*dV)) = 2 * 2 / (Nd_d*dV)
    // Wait: density is computed without spin_fac inside; it's in occ.
    // rho = sum kw * f * |psi|^2 = 1.0 * 1.0 * norm_val^2 * 2 bands
    double expected = 2.0 * norm_val * norm_val;  // 2 bands, each f=1
    for (int i = 0; i < Nd_d; ++i) {
        EXPECT_NEAR(density.rho_total()(i), expected, 1e-10);
    }
}

TEST(ElectronDensity, Integrate) {
    int Nd_d = 50;
    double dV = 0.5;

    ElectronDensity density;
    density.allocate(Nd_d, 1);

    // Set uniform density
    double rho_val = 0.02;
    for (int i = 0; i < Nd_d; ++i) {
        density.rho_total()(i) = rho_val;
        density.rho(0)(i) = rho_val;
    }

    double Ne = density.integrate(dV);
    EXPECT_NEAR(Ne, rho_val * Nd_d * dV, 1e-10);
}
