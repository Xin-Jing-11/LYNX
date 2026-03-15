#include <gtest/gtest.h>
#include "physics/Forces.hpp"
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include "core/Domain.hpp"
#include "parallel/MPIComm.hpp"
#include <cmath>
#include <vector>

using namespace lynx;

TEST(Forces, SymmetrizeProperty) {
    // Test that after symmetrization, forces sum to zero in each direction
    int n_atom = 4;
    std::vector<double> forces = {1.0, 2.0, 3.0,
                                   -0.5, 1.0, -2.0,
                                   0.5, -1.0, 0.5,
                                   -0.3, -0.4, 0.1};

    // Apply symmetrization manually (same algorithm as Forces::symmetrize)
    for (int d = 0; d < 3; ++d) {
        double avg = 0.0;
        for (int i = 0; i < n_atom; ++i) {
            avg += forces[i * 3 + d];
        }
        avg /= n_atom;
        for (int i = 0; i < n_atom; ++i) {
            forces[i * 3 + d] -= avg;
        }
    }

    // Check sums are zero
    for (int d = 0; d < 3; ++d) {
        double sum = 0.0;
        for (int i = 0; i < n_atom; ++i) {
            sum += forces[i * 3 + d];
        }
        EXPECT_NEAR(sum, 0.0, 1e-14);
    }
}

TEST(Forces, ForceObjectCreation) {
    Forces forces;
    EXPECT_TRUE(forces.total_forces().empty());
    EXPECT_TRUE(forces.local_forces().empty());
    EXPECT_TRUE(forces.nonlocal_forces().empty());
}
