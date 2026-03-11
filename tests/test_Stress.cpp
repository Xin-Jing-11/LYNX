#include <gtest/gtest.h>
#include "physics/Stress.hpp"
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include "core/Domain.hpp"
#include "parallel/MPIComm.hpp"
#include <cmath>
#include <vector>

using namespace sparc;

TEST(Stress, PressureFormula) {
    // P = -(1/3)(σ_xx + σ_yy + σ_zz)
    // For uniform compression σ_ii = -1, P = 1
    double sigma_xx = -1.0, sigma_yy = -1.0, sigma_zz = -1.0;
    double P = -(sigma_xx + sigma_yy + sigma_zz) / 3.0;
    EXPECT_DOUBLE_EQ(P, 1.0);
}

TEST(Stress, VoigtIndexing) {
    // Voigt: [xx, xy, xz, yy, yz, zz] = indices [0,1,2,3,4,5]
    std::array<double, 6> s = {1.0, 0.5, 0.3, 2.0, 0.4, 3.0};
    double P = -(s[0] + s[3] + s[5]) / 3.0;
    EXPECT_DOUBLE_EQ(P, -2.0);  // -(1+2+3)/3 = -2
}

TEST(Stress, ZeroPressure) {
    std::array<double, 6> zero_stress = {};
    double P = -(zero_stress[0] + zero_stress[3] + zero_stress[5]) / 3.0;
    EXPECT_DOUBLE_EQ(P, 0.0);
}

TEST(Stress, ObjectCreation) {
    Stress stress;
    EXPECT_DOUBLE_EQ(stress.pressure(), 0.0);
    for (int i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(stress.total_stress()[i], 0.0);
    }
}
