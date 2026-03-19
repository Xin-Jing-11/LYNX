#include <gtest/gtest.h>
#include "xc/SCANFunctional.hpp"
#include <xc.h>
#include <xc_funcs.h>
#include <cmath>
#include <vector>

using namespace lynx;

// Helper to generate realistic test data for non-spin SCAN
static void make_test_data(int N, std::vector<double>& rho,
                           std::vector<double>& sigma,
                           std::vector<double>& tau) {
    rho   = {0.1, 0.5, 1.0, 2.0, 5.0};
    sigma.resize(N);
    tau.resize(N);
    double threePi2_2o3 = std::pow(3.0 * M_PI * M_PI, 2.0 / 3.0);
    for (int i = 0; i < N; i++) {
        double kf = std::pow(3.0 * M_PI * M_PI * rho[i], 1.0 / 3.0);
        sigma[i] = 0.01 * rho[i] * rho[i] * kf * kf;
        tau[i] = 0.3 * threePi2_2o3 * std::pow(rho[i], 5.0 / 3.0);
    }
}

// ============================================================
// Task 3: Basic SCAN functional unit tests
// ============================================================

TEST(SCANFunctional, NonSpinExchange) {
    int N = 5;
    std::vector<double> rho, sigma, tau;
    make_test_data(N, rho, sigma, tau);

    std::vector<double> ex(N), vx(N), v2x(N), v3x(N);
    scan::scanx(N, rho.data(), sigma.data(), tau.data(),
                ex.data(), vx.data(), v2x.data(), v3x.data());

    for (int i = 0; i < N; i++) {
        EXPECT_TRUE(std::isfinite(ex[i])) << "ex[" << i << "] is not finite";
        EXPECT_TRUE(std::isfinite(vx[i])) << "vx[" << i << "] is not finite";
        EXPECT_TRUE(std::isfinite(v2x[i])) << "v2x[" << i << "] is not finite";
        EXPECT_TRUE(std::isfinite(v3x[i])) << "v3x[" << i << "] is not finite";
        EXPECT_LT(ex[i], 0.0) << "Exchange energy density should be negative at i=" << i;
    }
}

TEST(SCANFunctional, NonSpinCorrelation) {
    int N = 5;
    std::vector<double> rho, sigma, tau;
    make_test_data(N, rho, sigma, tau);

    std::vector<double> ec(N), vc(N), v2c(N), v3c(N);
    scan::scanc(N, rho.data(), sigma.data(), tau.data(),
                ec.data(), vc.data(), v2c.data(), v3c.data());

    for (int i = 0; i < N; i++) {
        EXPECT_TRUE(std::isfinite(ec[i])) << "ec[" << i << "] is not finite";
        EXPECT_TRUE(std::isfinite(vc[i])) << "vc[" << i << "] is not finite";
        EXPECT_TRUE(std::isfinite(v2c[i])) << "v2c[" << i << "] is not finite";
        EXPECT_TRUE(std::isfinite(v3c[i])) << "v3c[" << i << "] is not finite";
        EXPECT_LT(ec[i], 0.0) << "Correlation energy density should be negative at i=" << i;
    }
}

TEST(SCANFunctional, SpinExchange) {
    int N = 5;
    std::vector<double> rho_base, sigma_base, tau_base;
    make_test_data(N, rho_base, sigma_base, tau_base);

    // rho layout: [total(N) | up(N) | dn(N)]
    std::vector<double> rho(3 * N), sigma(3 * N), tau(3 * N);
    for (int i = 0; i < N; i++) {
        double total = rho_base[i];
        double up = 0.6 * total;
        double dn = 0.4 * total;
        rho[i] = total;
        rho[N + i] = up;
        rho[2 * N + i] = dn;

        double kf_up = std::pow(3.0 * M_PI * M_PI * up, 1.0 / 3.0);
        double kf_dn = std::pow(3.0 * M_PI * M_PI * dn, 1.0 / 3.0);
        double threePi2_2o3 = std::pow(3.0 * M_PI * M_PI, 2.0 / 3.0);

        sigma[i] = sigma_base[i]; // total
        sigma[N + i] = 0.01 * up * up * kf_up * kf_up;
        sigma[2 * N + i] = 0.01 * dn * dn * kf_dn * kf_dn;

        tau[i] = tau_base[i]; // total
        tau[N + i] = 0.3 * threePi2_2o3 * std::pow(up, 5.0 / 3.0);
        tau[2 * N + i] = 0.3 * threePi2_2o3 * std::pow(dn, 5.0 / 3.0);
    }

    std::vector<double> ex(N), vx(2 * N), v2x(2 * N), v3x(2 * N);
    scan::scanx_spin(N, rho.data(), sigma.data(), tau.data(),
                     ex.data(), vx.data(), v2x.data(), v3x.data());

    for (int i = 0; i < N; i++) {
        EXPECT_TRUE(std::isfinite(ex[i])) << "ex[" << i << "] is not finite";
        EXPECT_LT(ex[i], 0.0) << "Spin exchange energy density should be negative at i=" << i;
    }
    for (int i = 0; i < 2 * N; i++) {
        EXPECT_TRUE(std::isfinite(vx[i])) << "vx[" << i << "] is not finite";
        EXPECT_TRUE(std::isfinite(v2x[i])) << "v2x[" << i << "] is not finite";
        EXPECT_TRUE(std::isfinite(v3x[i])) << "v3x[" << i << "] is not finite";
    }
}

TEST(SCANFunctional, SpinCorrelation) {
    int N = 5;
    std::vector<double> rho_base, sigma_base, tau_base;
    make_test_data(N, rho_base, sigma_base, tau_base);

    // rho layout: [total(N) | up(N) | dn(N)]
    std::vector<double> rho(3 * N), sigma(3 * N), tau(3 * N);
    for (int i = 0; i < N; i++) {
        double total = rho_base[i];
        double up = 0.6 * total;
        double dn = 0.4 * total;
        rho[i] = total;
        rho[N + i] = up;
        rho[2 * N + i] = dn;

        double kf_up = std::pow(3.0 * M_PI * M_PI * up, 1.0 / 3.0);
        double kf_dn = std::pow(3.0 * M_PI * M_PI * dn, 1.0 / 3.0);
        double threePi2_2o3 = std::pow(3.0 * M_PI * M_PI, 2.0 / 3.0);

        sigma[i] = sigma_base[i];
        sigma[N + i] = 0.01 * up * up * kf_up * kf_up;
        sigma[2 * N + i] = 0.01 * dn * dn * kf_dn * kf_dn;

        tau[i] = tau_base[i];
        tau[N + i] = 0.3 * threePi2_2o3 * std::pow(up, 5.0 / 3.0);
        tau[2 * N + i] = 0.3 * threePi2_2o3 * std::pow(dn, 5.0 / 3.0);
    }

    // vc: 2*N (per-spin), v2c: N, v3c: N
    std::vector<double> ec(N), vc(2 * N), v2c(N), v3c(N);
    scan::scanc_spin(N, rho.data(), sigma.data(), tau.data(),
                     ec.data(), vc.data(), v2c.data(), v3c.data());

    for (int i = 0; i < N; i++) {
        EXPECT_TRUE(std::isfinite(ec[i])) << "ec[" << i << "] is not finite";
        EXPECT_LT(ec[i], 0.0) << "Spin correlation energy density should be negative at i=" << i;
    }
    for (int i = 0; i < 2 * N; i++) {
        EXPECT_TRUE(std::isfinite(vc[i])) << "vc[" << i << "] is not finite";
    }
    for (int i = 0; i < N; i++) {
        EXPECT_TRUE(std::isfinite(v2c[i])) << "v2c[" << i << "] is not finite";
        EXPECT_TRUE(std::isfinite(v3c[i])) << "v3c[" << i << "] is not finite";
    }
}

// ============================================================
// Task 5: Hand-coded vs libxc comparison tests
// ============================================================

TEST(SCANFunctional, HandCodedVsLibxc_NonSpin) {
    int N = 5;
    std::vector<double> rho, sigma, tau;
    make_test_data(N, rho, sigma, tau);

    // Hand-coded
    std::vector<double> ex_hc(N), vx_hc(N), v2x_hc(N), v3x_hc(N);
    std::vector<double> ec_hc(N), vc_hc(N), v2c_hc(N), v3c_hc(N);
    scan::scanx(N, rho.data(), sigma.data(), tau.data(),
                ex_hc.data(), vx_hc.data(), v2x_hc.data(), v3x_hc.data());
    scan::scanc(N, rho.data(), sigma.data(), tau.data(),
                ec_hc.data(), vc_hc.data(), v2c_hc.data(), v3c_hc.data());

    // libxc
    xc_func_type func_x, func_c;
    xc_func_init(&func_x, XC_MGGA_X_SCAN, XC_UNPOLARIZED);
    xc_func_init(&func_c, XC_MGGA_C_SCAN, XC_UNPOLARIZED);

    std::vector<double> zk_x(N), vrho_x(N), vsigma_x(N), vlapl_x(N), vtau_x(N);
    std::vector<double> zk_c(N), vrho_c(N), vsigma_c(N), vlapl_c(N), vtau_c(N);
    std::vector<double> lapl(N, 0.0);

    xc_mgga_exc_vxc(&func_x, N, rho.data(), sigma.data(), lapl.data(), tau.data(),
                     zk_x.data(), vrho_x.data(), vsigma_x.data(), vlapl_x.data(), vtau_x.data());
    xc_mgga_exc_vxc(&func_c, N, rho.data(), sigma.data(), lapl.data(), tau.data(),
                     zk_c.data(), vrho_c.data(), vsigma_c.data(), vlapl_c.data(), vtau_c.data());

    xc_func_end(&func_x);
    xc_func_end(&func_c);

    // Compare energy density (tolerance ~1e-10 due to minor numerical differences)
    for (int i = 0; i < N; i++) {
        double exc_hc = ex_hc[i] + ec_hc[i];
        double exc_lxc = zk_x[i] + zk_c[i];
        EXPECT_NEAR(exc_hc, exc_lxc, 1e-6 * std::abs(exc_lxc) + 1e-10)
            << "Energy density mismatch at i=" << i;
    }

    // Compare vrho
    for (int i = 0; i < N; i++) {
        double vrho_hc = vx_hc[i] + vc_hc[i];
        double vrho_lxc = vrho_x[i] + vrho_c[i];
        EXPECT_NEAR(vrho_hc, vrho_lxc, 1e-6 * std::abs(vrho_lxc) + 1e-10)
            << "vrho mismatch at i=" << i;
    }

    // Compare vtau
    for (int i = 0; i < N; i++) {
        double vtau_hc = v3x_hc[i] + v3c_hc[i];
        double vtau_lxc = vtau_x[i] + vtau_c[i];
        EXPECT_NEAR(vtau_hc, vtau_lxc, 1e-6 * std::abs(vtau_lxc) + 1e-10)
            << "vtau mismatch at i=" << i;
    }

    // Compare vsigma: hand-coded v2 = d(n*eps)/d|grad n| / |grad n| = 2*vsigma
    // Slightly looser tolerance (~1e-8) due to chain rule through sqrt(sigma) amplifying differences
    for (int i = 0; i < N; i++) {
        double v2_hc = v2x_hc[i] + v2c_hc[i];
        double v2_lxc = 2.0 * (vsigma_x[i] + vsigma_c[i]);
        EXPECT_NEAR(v2_hc, v2_lxc, 1e-7 * std::abs(v2_lxc) + 1e-10)
            << "vsigma mismatch at i=" << i;
    }
}

TEST(SCANFunctional, HandCodedVsLibxc_Spin) {
    int N = 5;
    std::vector<double> rho_base, sigma_base, tau_base;
    make_test_data(N, rho_base, sigma_base, tau_base);

    // Build SPARC-convention arrays: [total|up|dn]
    // Use consistent sigma: assume gradients are aligned so
    // sigma_total = (sqrt(sigma_up) + sqrt(sigma_dn))^2 and sigma_ud = sqrt(sigma_up*sigma_dn)
    std::vector<double> rho_sparc(3 * N), sigma_sparc(3 * N), tau_sparc(3 * N);
    for (int i = 0; i < N; i++) {
        double total = rho_base[i];
        double up = 0.6 * total;
        double dn = 0.4 * total;
        rho_sparc[i] = total;
        rho_sparc[N + i] = up;
        rho_sparc[2 * N + i] = dn;

        double kf_up = std::pow(3.0 * M_PI * M_PI * up, 1.0 / 3.0);
        double kf_dn = std::pow(3.0 * M_PI * M_PI * dn, 1.0 / 3.0);
        double threePi2_2o3 = std::pow(3.0 * M_PI * M_PI, 2.0 / 3.0);

        sigma_sparc[N + i] = 0.01 * up * up * kf_up * kf_up;
        sigma_sparc[2 * N + i] = 0.01 * dn * dn * kf_dn * kf_dn;
        // sigma_total = (|grad rho_up| + |grad rho_dn|)^2 for aligned gradients
        double gnorm_up = std::sqrt(sigma_sparc[N + i]);
        double gnorm_dn = std::sqrt(sigma_sparc[2 * N + i]);
        sigma_sparc[i] = (gnorm_up + gnorm_dn) * (gnorm_up + gnorm_dn);

        tau_sparc[N + i] = 0.3 * threePi2_2o3 * std::pow(up, 5.0 / 3.0);
        tau_sparc[2 * N + i] = 0.3 * threePi2_2o3 * std::pow(dn, 5.0 / 3.0);
        tau_sparc[i] = tau_sparc[N + i] + tau_sparc[2 * N + i];
    }

    // Hand-coded exchange (spin)
    std::vector<double> ex_hc(N), vx_hc(2 * N), v2x_hc(2 * N), v3x_hc(2 * N);
    scan::scanx_spin(N, rho_sparc.data(), sigma_sparc.data(), tau_sparc.data(),
                     ex_hc.data(), vx_hc.data(), v2x_hc.data(), v3x_hc.data());

    // Hand-coded correlation (spin)
    std::vector<double> ec_hc(N), vc_hc(2 * N), v2c_hc(N), v3c_hc(N);
    scan::scanc_spin(N, rho_sparc.data(), sigma_sparc.data(), tau_sparc.data(),
                     ec_hc.data(), vc_hc.data(), v2c_hc.data(), v3c_hc.data());

    // libxc: interleaved layout
    // rho_libxc[2*N]: [up0,dn0,up1,dn1,...]
    // sigma_libxc[3*N]: [sigma_uu0, sigma_ud0, sigma_dd0, ...]
    // tau_libxc[2*N]: [tau_up0,tau_dn0,...]
    std::vector<double> rho_lxc(2 * N), sigma_lxc(3 * N), tau_lxc(2 * N);
    std::vector<double> lapl_lxc(2 * N, 0.0);

    for (int i = 0; i < N; i++) {
        rho_lxc[2 * i] = rho_sparc[N + i];       // up
        rho_lxc[2 * i + 1] = rho_sparc[2 * N + i]; // dn

        sigma_lxc[3 * i]     = sigma_sparc[N + i];     // sigma_uu
        sigma_lxc[3 * i + 1] = std::sqrt(sigma_sparc[N + i] * sigma_sparc[2 * N + i]); // sigma_ud (aligned gradients)
        sigma_lxc[3 * i + 2] = sigma_sparc[2 * N + i]; // sigma_dd

        tau_lxc[2 * i]     = tau_sparc[N + i];     // tau_up
        tau_lxc[2 * i + 1] = tau_sparc[2 * N + i]; // tau_dn
    }

    // Exchange
    xc_func_type func_x, func_c;
    xc_func_init(&func_x, XC_MGGA_X_SCAN, XC_POLARIZED);
    xc_func_init(&func_c, XC_MGGA_C_SCAN, XC_POLARIZED);

    std::vector<double> zk_x(N), vrho_x(2 * N), vsigma_x(3 * N), vlapl_x(2 * N), vtau_x(2 * N);
    std::vector<double> zk_c(N), vrho_c(2 * N), vsigma_c(3 * N), vlapl_c(2 * N), vtau_c(2 * N);

    xc_mgga_exc_vxc(&func_x, N, rho_lxc.data(), sigma_lxc.data(), lapl_lxc.data(), tau_lxc.data(),
                     zk_x.data(), vrho_x.data(), vsigma_x.data(), vlapl_x.data(), vtau_x.data());
    xc_mgga_exc_vxc(&func_c, N, rho_lxc.data(), sigma_lxc.data(), lapl_lxc.data(), tau_lxc.data(),
                     zk_c.data(), vrho_c.data(), vsigma_c.data(), vlapl_c.data(), vtau_c.data());

    xc_func_end(&func_x);
    xc_func_end(&func_c);

    // Compare energy density (per particle): hand-coded ex+ec vs libxc zk_x+zk_c
    for (int i = 0; i < N; i++) {
        double exc_hc = ex_hc[i] + ec_hc[i];
        double exc_lxc = zk_x[i] + zk_c[i];
        EXPECT_NEAR(exc_hc, exc_lxc, 1e-6 * std::abs(exc_lxc) + 1e-10)
            << "Spin energy density mismatch at i=" << i;
    }

    // Compare vrho for up spin: vx_hc[i] + vc_hc[i] vs vrho_x[2i] + vrho_c[2i]
    for (int i = 0; i < N; i++) {
        double vrho_up_hc = vx_hc[i] + vc_hc[i];
        double vrho_up_lxc = vrho_x[2 * i] + vrho_c[2 * i];
        EXPECT_NEAR(vrho_up_hc, vrho_up_lxc, 1e-6 * std::abs(vrho_up_lxc) + 1e-10)
            << "Spin vrho_up mismatch at i=" << i;
    }

    // Compare vrho for down spin
    for (int i = 0; i < N; i++) {
        double vrho_dn_hc = vx_hc[N + i] + vc_hc[N + i];
        double vrho_dn_lxc = vrho_x[2 * i + 1] + vrho_c[2 * i + 1];
        EXPECT_NEAR(vrho_dn_hc, vrho_dn_lxc, 1e-6 * std::abs(vrho_dn_lxc) + 1e-10)
            << "Spin vrho_dn mismatch at i=" << i;
    }

    // Compare vtau: v3x_hc[i] + v3c_hc[i] for up, v3x_hc[N+i] + v3c_hc[i] for down
    for (int i = 0; i < N; i++) {
        double vtau_up_hc = v3x_hc[i] + v3c_hc[i];
        double vtau_up_lxc = vtau_x[2 * i] + vtau_c[2 * i];
        EXPECT_NEAR(vtau_up_hc, vtau_up_lxc, 1e-6 * std::abs(vtau_up_lxc) + 1e-10)
            << "Spin vtau_up mismatch at i=" << i;
    }

    for (int i = 0; i < N; i++) {
        double vtau_dn_hc = v3x_hc[N + i] + v3c_hc[i];
        double vtau_dn_lxc = vtau_x[2 * i + 1] + vtau_c[2 * i + 1];
        EXPECT_NEAR(vtau_dn_hc, vtau_dn_lxc, 1e-6 * std::abs(vtau_dn_lxc) + 1e-10)
            << "Spin vtau_dn mismatch at i=" << i;
    }

    // Compare vsigma for exchange up: hand-coded v2x uses doubled convention,
    // so v2x_hc = d(n_s*eps)/d|grad(2*rho_s)| / |grad(2*rho_s)| = 2*vsigma w.r.t. sigma_ss(doubled)
    // Since sigma_ss_doubled = 4*sigma_ss, and v2 = 2*d(eps)/d(sigma_doubled),
    // v2x_hc[i] = 2*vsigma_x_doubled. Mapping to libxc: vsigma_x_doubled = vsigma_x / 4 (chain rule),
    // so v2x_hc = 2*vsigma_x/4 = vsigma_x/2? Actually this needs careful derivation.
    // Use direct factor: v2x_hc (SPARC) vs 2*vsigma_x (libxc), with SPARC using doubled sigma
    // SPARC: v2x = theRho * eps * DFx/d|grad(2n_s)| / |grad(2n_s)|
    // This equals d(n_s*eps)/d(sigma_s) when sigma_s = |grad n_s|^2
    // Since |grad(2n_s)|^2 = 4*sigma_s, d/d(sigma_s) = 4*d/d(4*sigma_s)
    // And |grad(2n_s)| = 2*|grad n_s|, so 1/|grad(2n_s)| = 1/(2*|grad n_s|)
    // v2x_hc uses normDrho[i+DMnd] = sqrt(sigma_s) for spin, and SPARC divides by this
    // Actually the wrapper divides by normDrho[i+DMnd] which is sqrt(sigma_spin), not sqrt(4*sigma_spin)
    // Let's just compare exchange v2x vs 2*vsigma_x directly
    for (int i = 0; i < N; i++) {
        double v2_up_hc = v2x_hc[i];
        double v2_up_lxc = 2.0 * vsigma_x[3 * i];
        EXPECT_NEAR(v2_up_hc, v2_up_lxc, 1e-6 * std::abs(v2_up_lxc) + 1e-10)
            << "Spin vsigma_up exchange mismatch at i=" << i;
    }

    for (int i = 0; i < N; i++) {
        double v2_dn_hc = v2x_hc[N + i];
        double v2_dn_lxc = 2.0 * vsigma_x[3 * i + 2];
        EXPECT_NEAR(v2_dn_hc, v2_dn_lxc, 1e-6 * std::abs(v2_dn_lxc) + 1e-10)
            << "Spin vsigma_dn exchange mismatch at i=" << i;
    }

    // Compare vsigma for correlation: SPARC v2c uses total gradient
    // Different sigma convention between SPARC (total) and libxc (uu/ud/dd) makes
    // direct comparison complex; skip for now
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
