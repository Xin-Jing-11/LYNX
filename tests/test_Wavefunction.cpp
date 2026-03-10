#include <gtest/gtest.h>
#include "electronic/Wavefunction.hpp"
#include <cmath>

using namespace sparc;

TEST(Wavefunction, Allocate) {
    Wavefunction wfn;
    wfn.allocate(100, 10, 1, 1);

    EXPECT_EQ(wfn.Nd_d(), 100);
    EXPECT_EQ(wfn.Nband(), 10);
    EXPECT_EQ(wfn.Nspin(), 1);
    EXPECT_EQ(wfn.Nkpts(), 1);

    EXPECT_EQ(wfn.psi(0, 0).rows(), 100);
    EXPECT_EQ(wfn.psi(0, 0).cols(), 10);
    EXPECT_EQ(wfn.eigenvalues(0, 0).size(), 10);
    EXPECT_EQ(wfn.occupations(0, 0).size(), 10);
}

TEST(Wavefunction, AllocateSpinKpt) {
    Wavefunction wfn;
    wfn.allocate(50, 5, 2, 3);

    EXPECT_EQ(wfn.Nspin(), 2);
    EXPECT_EQ(wfn.Nkpts(), 3);

    // Access each spin/kpt combination
    for (int s = 0; s < 2; ++s) {
        for (int k = 0; k < 3; ++k) {
            EXPECT_EQ(wfn.psi(s, k).rows(), 50);
            EXPECT_EQ(wfn.psi(s, k).cols(), 5);
        }
    }
}

TEST(Wavefunction, Randomize) {
    Wavefunction wfn;
    wfn.allocate(100, 5, 1, 1);
    wfn.randomize(0, 0, 42);

    // Check that values are non-zero
    const auto& psi = wfn.psi(0, 0);
    double sum = 0.0;
    for (int j = 0; j < 5; ++j) {
        for (int i = 0; i < 100; ++i) {
            sum += std::abs(psi(i, j));
        }
    }
    EXPECT_GT(sum, 0.0);

    // Check deterministic (same seed = same values)
    Wavefunction wfn2;
    wfn2.allocate(100, 5, 1, 1);
    wfn2.randomize(0, 0, 42);
    for (int j = 0; j < 5; ++j) {
        for (int i = 0; i < 100; ++i) {
            EXPECT_DOUBLE_EQ(wfn.psi(0, 0)(i, j), wfn2.psi(0, 0)(i, j));
        }
    }
}

TEST(Wavefunction, EigenvaluesZeroInitialized) {
    Wavefunction wfn;
    wfn.allocate(50, 5, 1, 1);
    for (int n = 0; n < 5; ++n) {
        EXPECT_DOUBLE_EQ(wfn.eigenvalues(0, 0)(n), 0.0);
        EXPECT_DOUBLE_EQ(wfn.occupations(0, 0)(n), 0.0);
    }
}
