#include <gtest/gtest.h>
#include "core/KPoints.hpp"
#include "core/Lattice.hpp"
#include <cmath>
#include <algorithm>
#include <cstdio>

using namespace lynx;

// Helper: create orthogonal lattice with given cell length
static Lattice make_cubic(double a) {
    Mat3 lv{};
    lv(0,0) = a; lv(1,1) = a; lv(2,2) = a;
    return Lattice(lv, CellType::Orthogonal);
}

// Helper: sort k-points for comparison (by x, then y, then z)
static std::vector<Vec3> sorted_kpts(const std::vector<Vec3>& kpts) {
    auto v = kpts;
    std::sort(v.begin(), v.end(), [](const Vec3& a, const Vec3& b) {
        if (std::fabs(a.x - b.x) > 1e-10) return a.x < b.x;
        if (std::fabs(a.y - b.y) > 1e-10) return a.y < b.y;
        return a.z < b.z;
    });
    return v;
}

// Test 1: 2x2x2 with shift [0.5, 0.5, 0.5] -> 4 k-points (from 8)
TEST(KPoints, Grid222_Shift05) {
    Lattice lat = make_cubic(10.26);
    KPoints kp;
    kp.generate(2, 2, 2, {0.5, 0.5, 0.5}, lat);

    EXPECT_EQ(kp.Nkpts_full(), 8);
    EXPECT_EQ(kp.Nkpts(), 4);
    EXPECT_FALSE(kp.is_gamma_only());

    // All weights should be 2.0 (each k paired with -k)
    for (int i = 0; i < kp.Nkpts(); ++i) {
        EXPECT_DOUBLE_EQ(kp.weights()[i], 2.0) << "k-point " << i;
    }

    // Verify weight sum = Nkpts_full
    double wsum = 0;
    for (auto w : kp.weights()) wsum += w;
    EXPECT_DOUBLE_EQ(wsum, 8.0);

    // Check normalized weights sum to 1
    auto nw = kp.normalized_weights();
    double nwsum = 0;
    for (auto w : nw) nwsum += w;
    EXPECT_NEAR(nwsum, 1.0, 1e-14);

    // Expected reduced coords: ±0.25 in each direction, 4 unique
    auto red = sorted_kpts(kp.kpts_red());
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(std::fabs(red[i].x), 0.25, 1e-10);
        EXPECT_NEAR(std::fabs(red[i].y), 0.25, 1e-10);
        EXPECT_NEAR(std::fabs(red[i].z), 0.25, 1e-10);
    }
}

// Test 2: 3x3x3 with shift [0, 0, 0] -> 14 k-points (from 27)
TEST(KPoints, Grid333_NoShift) {
    Lattice lat = make_cubic(10.26);
    KPoints kp;
    kp.generate(3, 3, 3, {0.0, 0.0, 0.0}, lat);

    EXPECT_EQ(kp.Nkpts_full(), 27);
    EXPECT_EQ(kp.Nkpts(), 14);

    // Gamma point (0,0,0) should be present with weight 1 (self-conjugate)
    bool found_gamma = false;
    int gamma_idx = -1;
    for (int i = 0; i < kp.Nkpts(); ++i) {
        const auto& k = kp.kpts_red()[i];
        if (std::fabs(k.x) < 1e-10 && std::fabs(k.y) < 1e-10 && std::fabs(k.z) < 1e-10) {
            found_gamma = true;
            gamma_idx = i;
        }
    }
    EXPECT_TRUE(found_gamma);
    if (gamma_idx >= 0) {
        EXPECT_DOUBLE_EQ(kp.weights()[gamma_idx], 1.0);
    }

    // Weight sum should be 27
    double wsum = 0;
    for (auto w : kp.weights()) wsum += w;
    EXPECT_DOUBLE_EQ(wsum, 27.0);

    // Count: 1 self-conjugate (Gamma, w=1) + 13 paired (w=2) = 1 + 26 = 27
    int w1_count = 0, w2_count = 0;
    for (auto w : kp.weights()) {
        if (std::fabs(w - 1.0) < 1e-10) w1_count++;
        else if (std::fabs(w - 2.0) < 1e-10) w2_count++;
    }
    EXPECT_EQ(w1_count, 1);
    EXPECT_EQ(w2_count, 13);
}

// Test 3: 4x4x4 with shift [0.5, 0.5, 0.5] -> 32 k-points (from 64)
TEST(KPoints, Grid444_Shift05) {
    Lattice lat = make_cubic(10.26);
    KPoints kp;
    kp.generate(4, 4, 4, {0.5, 0.5, 0.5}, lat);

    EXPECT_EQ(kp.Nkpts_full(), 64);
    EXPECT_EQ(kp.Nkpts(), 32);

    // With half-integer shift, no self-conjugate points -> all weights = 2
    for (int i = 0; i < kp.Nkpts(); ++i) {
        EXPECT_DOUBLE_EQ(kp.weights()[i], 2.0) << "k-point " << i;
    }

    double wsum = 0;
    for (auto w : kp.weights()) wsum += w;
    EXPECT_DOUBLE_EQ(wsum, 64.0);
}

// Test 4: 2x3x4 with shift [0, 0.5, 0] -> 14 k-points (from 24)
TEST(KPoints, Grid234_MixedShift) {
    Lattice lat = make_cubic(10.26);
    KPoints kp;
    kp.generate(2, 3, 4, {0.0, 0.5, 0.0}, lat);

    EXPECT_EQ(kp.Nkpts_full(), 24);
    EXPECT_EQ(kp.Nkpts(), 14);

    // 10 with weight 2.0, 4 with weight 1.0
    int w1_count = 0, w2_count = 0;
    for (auto w : kp.weights()) {
        if (std::fabs(w - 1.0) < 1e-10) w1_count++;
        else if (std::fabs(w - 2.0) < 1e-10) w2_count++;
    }
    EXPECT_EQ(w1_count, 4);
    EXPECT_EQ(w2_count, 10);

    double wsum = 0;
    for (auto w : kp.weights()) wsum += w;
    EXPECT_DOUBLE_EQ(wsum, 24.0);
}

// Test 5: Gamma-only (1x1x1 with shift [0,0,0])
TEST(KPoints, GammaOnly) {
    Lattice lat = make_cubic(10.26);
    KPoints kp;
    kp.generate(1, 1, 1, {0.0, 0.0, 0.0}, lat);

    EXPECT_EQ(kp.Nkpts_full(), 1);
    EXPECT_EQ(kp.Nkpts(), 1);
    EXPECT_TRUE(kp.is_gamma_only());
    EXPECT_DOUBLE_EQ(kp.weights()[0], 1.0);
    EXPECT_NEAR(kp.kpts_cart()[0].x, 0.0, 1e-14);
    EXPECT_NEAR(kp.kpts_cart()[0].y, 0.0, 1e-14);
    EXPECT_NEAR(kp.kpts_cart()[0].z, 0.0, 1e-14);
}

// Test 6: Verify Cartesian coordinates match 2π/L * reduced
TEST(KPoints, CartesianCoords) {
    double a = 7.63;
    Lattice lat = make_cubic(a);
    KPoints kp;
    kp.generate(4, 4, 4, {0.5, 0.5, 0.5}, lat);

    double twoPI_L = 2.0 * M_PI / a;
    for (int i = 0; i < kp.Nkpts(); ++i) {
        EXPECT_NEAR(kp.kpts_cart()[i].x, kp.kpts_red()[i].x * twoPI_L, 1e-12);
        EXPECT_NEAR(kp.kpts_cart()[i].y, kp.kpts_red()[i].y * twoPI_L, 1e-12);
        EXPECT_NEAR(kp.kpts_cart()[i].z, kp.kpts_red()[i].z * twoPI_L, 1e-12);
    }
}

// Test 7: Print k-points for manual verification against reference
TEST(KPoints, PrintForVerification) {
    Lattice lat = make_cubic(10.26);

    struct TestCase { int Kx, Ky, Kz; Vec3 shift; const char* name; };
    TestCase cases[] = {
        {2, 2, 2, {0.5, 0.5, 0.5}, "2x2x2 shift=0.5"},
        {3, 3, 3, {0.0, 0.0, 0.0}, "3x3x3 shift=0.0"},
        {4, 4, 4, {0.5, 0.5, 0.5}, "4x4x4 shift=0.5"},
        {2, 3, 4, {0.0, 0.5, 0.0}, "2x3x4 shift=(0,0.5,0)"},
    };

    for (const auto& tc : cases) {
        KPoints kp;
        kp.generate(tc.Kx, tc.Ky, tc.Kz, tc.shift, lat);
        std::printf("\n=== %s: Nkpts_full=%d, Nkpts_sym=%d ===\n",
                    tc.name, kp.Nkpts_full(), kp.Nkpts());
        for (int i = 0; i < kp.Nkpts(); ++i) {
            const auto& kr = kp.kpts_red()[i];
            std::printf("  k[%2d]: %8.4f %8.4f %8.4f  wt=%.1f\n",
                        i, kr.x, kr.y, kr.z, kp.weights()[i]);
        }
    }
}
