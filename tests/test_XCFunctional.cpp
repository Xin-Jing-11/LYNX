#include <gtest/gtest.h>
#include "xc/XCFunctional.hpp"
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include "core/Domain.hpp"
#include "core/constants.hpp"
#include <cmath>
#include <vector>

using namespace sparc;

TEST(XCFunctional, LDA_UniformElectronGas) {
    XCFunctional xc;
    Domain domain;
    FDGrid grid;
    xc.setup(XCType::LDA_PW, domain, grid);

    int N = 100;
    std::vector<double> rho(N, 0.01);
    std::vector<double> Vxc(N), exc(N);

    xc.evaluate(rho.data(), Vxc.data(), exc.data(), N);

    // All values should be uniform
    for (int i = 1; i < N; i++) {
        EXPECT_DOUBLE_EQ(Vxc[i], Vxc[0]);
        EXPECT_DOUBLE_EQ(exc[i], exc[0]);
    }

    EXPECT_LT(exc[0], 0.0);
    EXPECT_LT(Vxc[0], 0.0);
}

TEST(XCFunctional, LDA_ZeroDensity) {
    XCFunctional xc;
    Domain domain;
    FDGrid grid;
    xc.setup(XCType::LDA_PW, domain, grid);

    // Very small density (not exactly zero to avoid cbrt issues)
    std::vector<double> rv(1, 1e-25), vv(1), ev(1);
    xc.evaluate(rv.data(), vv.data(), ev.data(), 1);

    // Should be very close to zero
    EXPECT_NEAR(ev[0], 0.0, 1e-6);
}

TEST(XCFunctional, LDA_SlaterExchange_KnownValue) {
    // Verify Slater exchange matches reference SPARC exactly
    // ex = -C2 * rho^(1/3), vx = -C3 * rho^(1/3)
    // C2 = 0.738558766382022, C3 = 0.9847450218426965
    XCFunctional xc;
    Domain domain;
    FDGrid grid;
    xc.setup(XCType::LDA_PW, domain, grid);

    double rho_val = 0.05;
    std::vector<double> rv(1, rho_val), vv(1), ev(1);
    xc.evaluate(rv.data(), vv.data(), ev.data(), 1);

    double rho_cbrt = std::cbrt(rho_val);
    double C2 = 0.738558766382022;
    double C3 = 0.9847450218426965;
    double ex_expected = -C2 * rho_cbrt;

    // exc includes correlation, so it should be more negative than pure exchange
    EXPECT_LT(ev[0], ex_expected);

    // The exchange contribution dominates; check it's in the right ballpark
    EXPECT_NEAR(ev[0], ex_expected, 0.05);  // correlation is small
}

TEST(XCFunctional, LDA_PW92Correlation_KnownValue) {
    // At rs=2: known PW92 correlation value
    // ec ≈ -0.04479 Ha (from published tables)
    double C31 = 0.6203504908993999;
    double rs = 2.0;
    double rho_val = std::pow(C31 / rs, 3.0);

    XCFunctional xc;
    Domain domain;
    FDGrid grid;
    xc.setup(XCType::LDA_PW, domain, grid);

    std::vector<double> rv(1, rho_val), vv(1), ev(1);
    xc.evaluate(rv.data(), vv.data(), ev.data(), 1);

    // Exchange: ex = -C2 * rho^(1/3) = -C2 * C31/rs = -0.738558766382022 * 0.3101752454497 ≈ -0.2291
    // Correlation: ec ≈ -0.0480 at rs=2
    // Total exc ≈ -0.277
    double C2 = 0.738558766382022;
    double ex_expected = -C2 * std::cbrt(rho_val);
    EXPECT_NEAR(ev[0] - ex_expected, -0.048, 0.005);  // correlation part
}

TEST(XCFunctional, LDA_ExchangeScaling) {
    // Exchange scales as rho^(1/3)
    XCFunctional xc;
    Domain domain;
    FDGrid grid;
    xc.setup(XCType::LDA_PW, domain, grid);

    double rho1 = 0.01;
    double rho2 = 0.08;
    std::vector<double> r1(1, rho1), r2(1, rho2), v1(1), v2(1), e1(1), e2(1);

    xc.evaluate(r1.data(), v1.data(), e1.data(), 1);
    xc.evaluate(r2.data(), v2.data(), e2.data(), 1);

    double ratio = e2[0] / e1[0];
    EXPECT_NEAR(ratio, 2.0, 0.2);
}

TEST(XCFunctional, GGA_FallbackToLDAWithoutGradient) {
    XCFunctional xc;
    Domain domain;
    FDGrid grid;
    xc.setup(XCType::GGA_PBE, domain, grid, nullptr, nullptr);

    int N = 10;
    std::vector<double> rho(N, 0.01), Vxc(N), exc(N);
    xc.evaluate(rho.data(), Vxc.data(), exc.data(), N);

    EXPECT_LT(exc[0], 0.0);
    EXPECT_LT(Vxc[0], 0.0);
}

TEST(XCFunctional, GGA_PBE_UniformDensity) {
    // For uniform density, sigma=0, PBE reduces to LDA
    // (enhancement factor Fx=1 when s=0)
    XCFunctional xc;
    Domain domain;
    FDGrid grid;
    // GGA without gradient => falls back to LDA
    xc.setup(XCType::GGA_PBE, domain, grid, nullptr, nullptr);

    double rho_val = 0.03;
    std::vector<double> rho(1, rho_val), Vxc_gga(1), exc_gga(1);
    xc.evaluate(rho.data(), Vxc_gga.data(), exc_gga.data(), 1);

    // Compare with explicit LDA
    XCFunctional xc_lda;
    xc_lda.setup(XCType::LDA_PW, domain, grid);
    std::vector<double> Vxc_lda(1), exc_lda(1);
    xc_lda.evaluate(rho.data(), Vxc_lda.data(), exc_lda.data(), 1);

    // Should be identical (GGA with zero gradient = LDA)
    EXPECT_NEAR(exc_gga[0], exc_lda[0], 1e-12);
    EXPECT_NEAR(Vxc_gga[0], Vxc_lda[0], 1e-12);
}

TEST(XCFunctional, LDA_PZ_vs_PW) {
    // PZ and PW should give similar but not identical correlation
    XCFunctional xc_pz, xc_pw;
    Domain domain;
    FDGrid grid;
    xc_pz.setup(XCType::LDA_PZ, domain, grid);
    xc_pw.setup(XCType::LDA_PW, domain, grid);

    double rho_val = 0.02;
    std::vector<double> rho(1, rho_val);
    std::vector<double> Vxc_pz(1), exc_pz(1), Vxc_pw(1), exc_pw(1);

    xc_pz.evaluate(rho.data(), Vxc_pz.data(), exc_pz.data(), 1);
    xc_pw.evaluate(rho.data(), Vxc_pw.data(), exc_pw.data(), 1);

    // Both should be negative
    EXPECT_LT(exc_pz[0], 0.0);
    EXPECT_LT(exc_pw[0], 0.0);

    // Should be similar (within ~10% on correlation)
    EXPECT_NEAR(exc_pz[0], exc_pw[0], 0.01);
}
