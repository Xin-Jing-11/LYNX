#include <gtest/gtest.h>
#include "physics/Energy.hpp"
#include "electronic/Wavefunction.hpp"
#include "electronic/ElectronDensity.hpp"
#include "parallel/MPIComm.hpp"
#include <vector>

using namespace sparc;

TEST(Energy, BandEnergy) {
    Wavefunction wfn;
    wfn.allocate(10, 4, 1, 1);

    auto& eig = wfn.eigenvalues(0, 0);
    auto& occ = wfn.occupations(0, 0);
    eig(0) = 0.1; eig(1) = 0.2; eig(2) = 0.3; eig(3) = 0.4;
    occ(0) = 1.0; occ(1) = 1.0; occ(2) = 0.5; occ(3) = 0.0;

    std::vector<double> kpt_weights = {1.0};
    // spin_fac = 2 for Nspin=1
    double Eband = Energy::band_energy(wfn, kpt_weights, 1);

    // Expected: 2*(1.0*0.1 + 1.0*0.2 + 0.5*0.3 + 0.0*0.4) = 2*0.45 = 0.9
    EXPECT_NEAR(Eband, 0.9, 1e-10);
}

TEST(Energy, XCEnergy) {
    int Nd_d = 50;
    double dV = 0.1;

    std::vector<double> rho(Nd_d, 0.02);
    std::vector<double> exc(Nd_d, -0.5);

    MPIComm null_comm;
    double Exc = Energy::xc_energy(rho.data(), exc.data(), Nd_d, dV, null_comm);

    EXPECT_NEAR(Exc, 0.02 * (-0.5) * Nd_d * dV, 1e-10);
}

TEST(Energy, HartreeEnergy) {
    int Nd_d = 50;
    double dV = 0.1;

    std::vector<double> rho(Nd_d, 0.02);
    std::vector<double> phi(Nd_d, 1.0);

    MPIComm null_comm;
    double Ehart = Energy::hartree_energy(rho.data(), phi.data(), Nd_d, dV, null_comm);

    // 0.5 * int rho * phi dV = 0.5 * 0.02 * 1.0 * 50 * 0.1 = 0.05
    EXPECT_NEAR(Ehart, 0.05, 1e-10);
}
