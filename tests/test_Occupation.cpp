#include <gtest/gtest.h>
#include "electronic/Occupation.hpp"
#include "electronic/Wavefunction.hpp"
#include "parallel/MPIComm.hpp"
#include <cmath>
#include <vector>

using namespace sparc;

TEST(Occupation, FermiDirac) {
    // At T=0 (beta→inf): f(x<0)=1, f(x>0)=0
    EXPECT_NEAR(Occupation::fermi_dirac(-1.0, 1000.0), 1.0, 1e-10);
    EXPECT_NEAR(Occupation::fermi_dirac(1.0, 1000.0), 0.0, 1e-10);
    // At x=0: f(0) = 0.5
    EXPECT_NEAR(Occupation::fermi_dirac(0.0, 1.0), 0.5, 1e-10);
}

TEST(Occupation, GaussianSmearing) {
    // At x=0: g(0) = 0.5
    EXPECT_NEAR(Occupation::gaussian_smearing(0.0, 1.0), 0.5, 1e-10);
    // Large negative x: g → 1
    EXPECT_NEAR(Occupation::gaussian_smearing(-10.0, 1.0), 1.0, 1e-6);
    // Large positive x: g → 0
    EXPECT_NEAR(Occupation::gaussian_smearing(10.0, 1.0), 0.0, 1e-6);
}

TEST(Occupation, FindFermiLevel) {
    // Simple system: 4 states with eigenvalues 0.1, 0.2, 0.3, 0.4
    // 2 electrons, spin_fac=2 → need to fill 1 state
    Wavefunction wfn;
    wfn.allocate(10, 4, 1, 1);

    auto& eig = wfn.eigenvalues(0, 0);
    eig(0) = 0.1; eig(1) = 0.2; eig(2) = 0.3; eig(3) = 0.4;

    std::vector<double> kpt_weights = {1.0};
    MPIComm null_comm;

    double beta = 100.0;  // cold system
    double Ef = Occupation::compute(wfn, 2.0, beta, SmearingType::FermiDirac,
                                     kpt_weights, null_comm, null_comm);

    // Fermi level should be between 0.1 and 0.2
    EXPECT_GT(Ef, 0.05);
    EXPECT_LT(Ef, 0.25);

    // First state should be nearly fully occupied
    EXPECT_NEAR(wfn.occupations(0, 0)(0), 1.0, 0.05);
    // Last states should be nearly empty
    EXPECT_NEAR(wfn.occupations(0, 0)(3), 0.0, 0.05);
}

TEST(Occupation, TotalOccupation) {
    // 10 electrons, 10 states → first 5 doubly occupied
    Wavefunction wfn;
    wfn.allocate(10, 10, 1, 1);

    auto& eig = wfn.eigenvalues(0, 0);
    for (int n = 0; n < 10; ++n) {
        eig(n) = 0.1 * (n + 1);
    }

    std::vector<double> kpt_weights = {1.0};
    MPIComm null_comm;

    double beta = 200.0;
    double Ef = Occupation::compute(wfn, 10.0, beta, SmearingType::FermiDirac,
                                     kpt_weights, null_comm, null_comm);

    // Check total occupation sums to ~10
    double total = 0.0;
    const auto& occ = wfn.occupations(0, 0);
    for (int n = 0; n < 10; ++n) {
        total += 2.0 * occ(n);  // spin_fac=2
    }
    EXPECT_NEAR(total, 10.0, 0.01);
}

TEST(Occupation, Entropy) {
    Wavefunction wfn;
    wfn.allocate(10, 4, 1, 1);

    auto& eig = wfn.eigenvalues(0, 0);
    eig(0) = 0.1; eig(1) = 0.2; eig(2) = 0.3; eig(3) = 0.4;

    std::vector<double> kpt_weights = {1.0};
    MPIComm null_comm;

    double beta = 10.0;  // warm system for non-trivial entropy
    Occupation::compute(wfn, 2.0, beta, SmearingType::FermiDirac,
                         kpt_weights, null_comm, null_comm);

    double S = Occupation::entropy(wfn, beta, SmearingType::FermiDirac, kpt_weights);
    // Entropy should be negative (it's -T*S)
    EXPECT_LE(S, 0.0);
}
