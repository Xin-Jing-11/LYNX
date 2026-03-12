// XC functional implementation — ported from SPARC reference code
// (exchangeCorrelation.c, Georgia Tech, Xu/Sharma/Suryanarayana)
//
// Each function matches the reference SPARC implementation exactly
// to ensure numerical reproducibility.

#include "xc/XCFunctional.hpp"
#include "core/constants.hpp"
#include <cmath>
#include <vector>
#include <cstring>

namespace sparc {

void XCFunctional::setup(XCType type, const Domain& domain, const FDGrid& grid,
                          const Gradient* gradient, const HaloExchange* halo) {
    type_ = type;
    domain_ = &domain;
    grid_ = &grid;
    gradient_ = gradient;
    halo_ = halo;
}

int XCFunctional::get_pbe_iflag() const {
    switch (type_) {
        case XCType::GGA_PBE:    return 1;
        case XCType::GGA_PBEsol: return 2;
        case XCType::GGA_RPBE:   return 3;
        default: return 1;
    }
}

// ============================================================
// LDA Exchange: Slater
// Ref: exchangeCorrelation.c:459-469
// ============================================================
void XCFunctional::slater(int DMnd, const double* rho, double* ex, double* vx) {
    constexpr double C2 = 0.738558766382022;   // 3/4 * (3/pi)^(1/3)
    constexpr double C3 = 0.9847450218426965;  // (3/pi)^(1/3)

    for (int i = 0; i < DMnd; i++) {
        double rho_cbrt = std::cbrt(rho[i]);
        ex[i] = -C2 * rho_cbrt;
        vx[i] = -C3 * rho_cbrt;
    }
}

// ============================================================
// LDA Correlation: Perdew-Wang 92
// Ref: exchangeCorrelation.c:475-499
// J.P. Perdew and Y. Wang, PRB 45, 13244 (1992)
// ============================================================
void XCFunctional::pw(int DMnd, const double* rho, double* ec, double* vc) {
    constexpr double p = 1.0;
    constexpr double A = 0.031091;
    constexpr double alpha1 = 0.21370;
    constexpr double beta1 = 7.5957;
    constexpr double beta2 = 3.5876;
    constexpr double beta3 = 1.6382;
    constexpr double beta4 = 0.49294;
    constexpr double C31 = 0.6203504908993999; // (3/4pi)^(1/3)

    for (int i = 0; i < DMnd; i++) {
        double rho_cbrt = std::cbrt(rho[i]);
        double rs = C31 / rho_cbrt;
        double rs_sqrt = std::sqrt(rs);
        double rs_pow_1p5 = rs * rs_sqrt;
        double rs_pow_p = rs;
        double rs_pow_pplus1 = rs_pow_p * rs;
        double G2 = 2.0*A*(beta1*rs_sqrt + beta2*rs + beta3*rs_pow_1p5 + beta4*rs_pow_pplus1);
        double G1 = std::log(1.0 + 1.0 / G2);

        ec[i] = -2.0*A*(1.0 + alpha1*rs) * G1;
        vc[i] = ec[i] - (rs/3.0) * (
            -2.0*A*alpha1 * G1
            + (2.0*A*(1.0 + alpha1*rs)
               * (A*(beta1/rs_sqrt + 2.0*beta2 + 3.0*beta3*rs_sqrt + 2.0*(p+1.0)*beta4*rs_pow_p)))
              / (G2 * (G2 + 1.0))
        );
    }
}

// ============================================================
// LDA Correlation: Perdew-Zunger 81
// Ref: exchangeCorrelation.c:505-529
// ============================================================
void XCFunctional::pz(int DMnd, const double* rho, double* ec, double* vc) {
    constexpr double A_pz = 0.0311;
    constexpr double B = -0.048;
    constexpr double C = 0.002;
    constexpr double D = -0.0116;
    constexpr double gamma1 = -0.1423;
    constexpr double beta1 = 1.0529;
    constexpr double beta2 = 0.3334;
    constexpr double C31 = 0.6203504908993999;

    for (int i = 0; i < DMnd; i++) {
        double rho_cbrt = std::cbrt(rho[i]);
        double rs = C31 / rho_cbrt;
        if (rs < 1.0) {
            ec[i] = A_pz*std::log(rs) + B + C*rs*std::log(rs) + D*rs;
            vc[i] = std::log(rs)*(A_pz + (2.0/3.0)*C*rs)
                    + (B - (1.0/3.0)*A_pz)
                    + (1.0/3.0)*(2.0*D - C)*rs;
        } else {
            double sqrtrs = std::sqrt(rs);
            ec[i] = gamma1 / (1.0 + beta1*sqrtrs + beta2*rs);
            double denom = 1.0 + beta1*sqrtrs + beta2*rs;
            vc[i] = (gamma1 + (7.0/6.0)*gamma1*beta1*sqrtrs
                     + (4.0/3.0)*gamma1*beta2*rs) / (denom*denom);
        }
    }
}

// ============================================================
// GGA Exchange: PBE family
// Ref: exchangeCorrelation.c:540-586
// Matches reference exactly: uses spin-scaling (rho/2) convention
// ============================================================
void XCFunctional::pbex(int DMnd, const double* rho, const double* sigma,
                         int iflag, double* ex, double* vx, double* v2x) {
    double mu_tab[4] = {0.2195149727645171, 10.0/81.0, 0.2195149727645171, 0.2195149727645171};
    double kappa_tab[4] = {0.804, 0.804, 0.804, 1.245};

    double kappa = kappa_tab[iflag-1];
    double mu = mu_tab[iflag-1];
    double mu_divkappa = mu / kappa;
    double threefourth_divpi = (3.0/4.0) / constants::PI;
    constexpr double third = 1.0/3.0;
    double sixpi2_1_3 = std::pow(6.0*constants::PI*constants::PI, third);
    double sixpi2m1_3 = 1.0 / sixpi2_1_3;

    for (int i = 0; i < DMnd; i++) {
        double rho_updn = rho[i] / 2.0;
        double rho_updnm1_3 = std::pow(rho_updn, -third);

        double rhomot = rho_updnm1_3;
        double ex_lsd = -threefourth_divpi * sixpi2_1_3 * (rhomot * rhomot * rho_updn);

        double rho_inv = rhomot * rhomot * rhomot;
        double coeffss = (1.0/4.0) * sixpi2m1_3 * sixpi2m1_3 * (rho_inv * rho_inv * rhomot * rhomot);
        double ss = (sigma[i] / 4.0) * coeffss; // s^2

        double divss, dfxdss;
        if (iflag == 1 || iflag == 2 || iflag == 4) {
            divss = 1.0 / (1.0 + mu_divkappa * ss);
            dfxdss = mu * (divss * divss);
        } else { // iflag == 3 (RPBE)
            divss = std::exp(-mu_divkappa * ss);
            dfxdss = mu * divss;
        }

        double fx = 1.0 + kappa * (1.0 - divss);
        double dssdn = (-8.0/3.0) * (ss * rho_inv);
        double dfxdn = dfxdss * dssdn;
        double dssdg = 2.0 * coeffss;
        double dfxdg = dfxdss * dssdg;

        ex[i] = ex_lsd * fx;
        vx[i] = ex_lsd * ((4.0/3.0) * fx + rho_updn * dfxdn);
        v2x[i] = 0.5 * ex_lsd * rho_updn * dfxdg;
    }
}

// ============================================================
// GGA Correlation: PBE family
// Ref: exchangeCorrelation.c:626-754
// Chain of variable substitutions matching reference exactly:
// ec_lda → bb → cc → aa → tt → xx → pade → qq → rr → hh
// ============================================================
void XCFunctional::pbec(int DMnd, const double* rho, const double* sigma,
                         int iflag, double* ec, double* vc, double* v2c) {
    double beta_tab[3] = {0.066725, 0.046, 0.066725};

    double beta = beta_tab[iflag-1];
    double gamma = (1.0 - std::log(2.0)) / (constants::PI * constants::PI);
    double gamma_inv = 1.0 / gamma;
    double phi_zeta_inv = 1.0;
    double phi3_zeta = 1.0;
    double gamphi3inv = gamma_inv;
    constexpr double third = 1.0/3.0;
    double twom1_3 = std::pow(2.0, -third);
    constexpr double rsfac = 0.6203504908994000;
    double sq_rsfac = std::sqrt(rsfac);
    double sq_rsfac_inv = 1.0 / sq_rsfac;
    double coeff_tt = 1.0 / (16.0 / constants::PI * std::pow(3.0*constants::PI*constants::PI, third));
    constexpr double ec0_aa = 0.031091;
    constexpr double ec0_a1 = 0.21370;
    constexpr double ec0_b1 = 7.5957;
    constexpr double ec0_b2 = 3.5876;
    constexpr double ec0_b3 = 1.6382;
    constexpr double ec0_b4 = 0.49294;

    for (int i = 0; i < DMnd; i++) {
        double rho_updn = rho[i] / 2.0;
        double rho_updnm1_3 = std::pow(rho_updn, -third);
        double rhom1_3 = twom1_3 * rho_updnm1_3;
        double rhotot_inv = rhom1_3 * rhom1_3 * rhom1_3;
        double rhotmo6 = std::sqrt(rhom1_3);
        double rhoto6 = rho[i] * rhom1_3 * rhom1_3 * rhotmo6;

        // LSD correlation part (PW92 formulas A6-A8)
        double rs = rsfac * rhom1_3;
        double sqr_rs = sq_rsfac * rhotmo6;
        double rsm1_2 = sq_rsfac_inv * rhoto6;

        double ec0_q0 = -2.0 * ec0_aa * (1.0 + ec0_a1 * rs);
        double ec0_q1 = 2.0 * ec0_aa * (ec0_b1 * sqr_rs + ec0_b2 * rs + ec0_b3 * rs * sqr_rs + ec0_b4 * rs * rs);
        double ec0_q1p = ec0_aa * (ec0_b1 * rsm1_2 + 2.0 * ec0_b2 + 3.0 * ec0_b3 * sqr_rs + 4.0 * ec0_b4 * rs);
        double ec0_den = 1.0 / (ec0_q1 * ec0_q1 + ec0_q1);
        double ec0_log = -std::log(ec0_q1 * ec0_q1 * ec0_den);
        double ecrs0 = ec0_q0 * ec0_log;
        double decrs0_drs = -2.0 * ec0_aa * ec0_a1 * ec0_log - ec0_q0 * ec0_q1p * ec0_den;

        double ecrs = ecrs0;
        double decrs_drs = decrs0_drs;

        // LSD contribution
        ec[i] = ecrs;
        vc[i] = ecrs - (rs / 3.0) * decrs_drs;

        // GGA correlation: chain of variable substitutions
        // ec → bb
        double bb = ecrs * gamphi3inv;
        double dbb_drs = decrs_drs * gamphi3inv;

        // bb → cc
        double exp_pbe = std::exp(-bb);
        double cc = 1.0 / (exp_pbe - 1.0);
        double dcc_dbb = cc * cc * exp_pbe;
        double dcc_drs = dcc_dbb * dbb_drs;

        // cc → aa
        double coeff_aa = beta * gamma_inv * phi_zeta_inv * phi_zeta_inv;
        double aa = coeff_aa * cc;
        double daa_drs = coeff_aa * dcc_drs;

        // Introduce tt
        double grrho2 = sigma[i];
        double dtt_dg = 2.0 * rhotot_inv * rhotot_inv * rhom1_3 * coeff_tt;
        double tt = 0.5 * grrho2 * dtt_dg;

        // tt,aa → xx
        double xx = aa * tt;
        double dxx_drs = daa_drs * tt;
        double dxx_dtt = aa;

        // xx → pade
        double pade_den = 1.0 / (1.0 + xx * (1.0 + xx));
        double pade = (1.0 + xx) * pade_den;
        double dpade_dxx = -xx * (2.0 + xx) * (pade_den * pade_den);
        double dpade_drs = dpade_dxx * dxx_drs;
        double dpade_dtt = dpade_dxx * dxx_dtt;

        // pade → qq
        double coeff_qq = tt * phi_zeta_inv * phi_zeta_inv;
        double qq = coeff_qq * pade;
        double dqq_drs = coeff_qq * dpade_drs;
        double dqq_dtt = pade * phi_zeta_inv * phi_zeta_inv + coeff_qq * dpade_dtt;

        // qq → rr
        double arg_rr = 1.0 + beta * gamma_inv * qq;
        double div_rr = 1.0 / arg_rr;
        double rr = gamma * std::log(arg_rr);
        double drr_dqq = beta * div_rr;
        double drr_drs = drr_dqq * dqq_drs;
        double drr_dtt = drr_dqq * dqq_dtt;

        // rr → hh
        double hh = phi3_zeta * rr;
        double dhh_drs = phi3_zeta * drr_drs;
        double dhh_dtt = phi3_zeta * drr_dtt;

        // GGA correlation energy
        ec[i] += hh;

        // Derivative of energy wrt density
        double drhohh_drho = hh - third * rs * dhh_drs - (7.0/3.0) * tt * dhh_dtt;
        vc[i] += drhohh_drho;

        // Derivative wrt gradient (v2c = d(rho*ec)/d(|grad rho|^2))
        v2c[i] = rho[i] * dtt_dg * dhh_dtt;
    }
}

// ============================================================
// LDA Slater exchange, spin-polarized
// Ref: exchangeCorrelation.c:759-781
// ============================================================
void XCFunctional::slater_spin(int DMnd, const double* rho, double* ex, double* vx) {
    constexpr double third = 1.0/3.0;
    double threefourth_divpi = (3.0/4.0) / constants::PI;
    double sixpi2_1_3 = std::pow(6.0*constants::PI*constants::PI, third);

    for (int i = 0; i < DMnd; i++) {
        double rhom1_3 = std::pow(rho[i], -third);
        double rhotot_inv = rhom1_3 * rhom1_3 * rhom1_3;
        double extot = 0.0;

        for (int spn_i = 0; spn_i < 2; spn_i++) {
            double rho_updn = rho[DMnd + spn_i*DMnd + i];
            double rhomot = std::pow(rho_updn, -third);
            double ex_lsd = -threefourth_divpi * sixpi2_1_3 * (rhomot * rhomot * rho_updn);
            vx[spn_i*DMnd + i] = (4.0/3.0) * ex_lsd;
            extot += ex_lsd * rho_updn;
        }
        ex[i] = extot * rhotot_inv;
    }
}

// ============================================================
// LDA PW92 correlation, spin-polarized
// Ref: exchangeCorrelation.c:787-866
// ============================================================
void XCFunctional::pw_spin(int DMnd, const double* rho, double* ec, double* vc) {
    constexpr double third = 1.0/3.0;
    constexpr double rsfac = 0.6203504908994000;
    double sq_rsfac = std::sqrt(rsfac);
    double sq_rsfac_inv = 1.0 / sq_rsfac;
    constexpr double twom1_3 = 0.7937005259840998; // 2^(-1/3)

    // Parameters for unpolarized (ec0) and polarized (ec1) and alpha_c (mac)
    constexpr double ec0_aa = 0.031091, ec0_a1 = 0.21370;
    constexpr double ec0_b1 = 7.5957, ec0_b2 = 3.5876, ec0_b3 = 1.6382, ec0_b4 = 0.49294;
    constexpr double ec1_aa = 0.015545, ec1_a1 = 0.20548;
    constexpr double ec1_b1 = 14.1189, ec1_b2 = 6.1977, ec1_b3 = 3.3662, ec1_b4 = 0.62517;
    constexpr double mac_aa = 0.016887, mac_a1 = 0.11125;
    constexpr double mac_b1 = 10.357, mac_b2 = 3.6231, mac_b3 = 0.88026, mac_b4 = 0.49671;

    auto pw92_G = [](double rs, double sqr_rs, double rsm1_2, double aa, double a1,
                     double b1, double b2, double b3, double b4, double& ecrs, double& decrs_drs) {
        double q0 = -2.0 * aa * (1.0 + a1 * rs);
        double q1 = 2.0 * aa * (b1 * sqr_rs + b2 * rs + b3 * rs * sqr_rs + b4 * rs * rs);
        double q1p = aa * (b1 * rsm1_2 + 2.0 * b2 + 3.0 * b3 * sqr_rs + 4.0 * b4 * rs);
        double den = 1.0 / (q1 * q1 + q1);
        double logv = -std::log(q1 * q1 * den);
        ecrs = q0 * logv;
        decrs_drs = -2.0 * aa * a1 * logv - q0 * q1p * den;
    };

    constexpr double fsec_inv = 1.0/1.709920934161365; // 1/(2*(2^(1/3)-1))
    constexpr double factf_zeta = 1.709920934161365;    // 2*(2^(1/3)-1)
    constexpr double factfp_zeta = (4.0/3.0) * factf_zeta;

    for (int i = 0; i < DMnd; i++) {
        double rhom1_3 = std::pow(rho[i], -third);
        double rhotot_inv = rhom1_3 * rhom1_3 * rhom1_3;
        double rhotmo6 = std::sqrt(rhom1_3);
        double rhoto6 = rho[i] * rhom1_3 * rhom1_3 * rhotmo6;

        double rs = rsfac * rhom1_3;
        double sqr_rs = sq_rsfac * rhotmo6;
        double rsm1_2 = sq_rsfac_inv * rhoto6;

        // Unpolarized, polarized, and alpha_c
        double ecrs0, decrs0_drs;
        pw92_G(rs, sqr_rs, rsm1_2, ec0_aa, ec0_a1, ec0_b1, ec0_b2, ec0_b3, ec0_b4, ecrs0, decrs0_drs);
        double ecrs1, decrs1_drs;
        pw92_G(rs, sqr_rs, rsm1_2, ec1_aa, ec1_a1, ec1_b1, ec1_b2, ec1_b3, ec1_b4, ecrs1, decrs1_drs);
        double macrs, dmacrs_drs;
        pw92_G(rs, sqr_rs, rsm1_2, mac_aa, mac_a1, mac_b1, mac_b2, mac_b3, mac_b4, macrs, dmacrs_drs);

        // Spin polarization
        double zeta = (rho[DMnd + i] - rho[2*DMnd + i]) * rhotot_inv;
        zeta = std::max(-1.0, std::min(1.0, zeta));
        double zetp = 1.0 + zeta;
        double zetm = 1.0 - zeta;
        double zetp_1_3 = std::cbrt(zetp);
        double zetm_1_3 = std::cbrt(zetm);
        double f_zeta = (zetp * zetp_1_3 + zetm * zetm_1_3 - 2.0) * fsec_inv;
        double fp_zeta = (zetp_1_3 - zetm_1_3) * (4.0/3.0) * fsec_inv;
        double zeta4 = zeta * zeta * zeta * zeta;

        double gcrs = ecrs1 - ecrs0;
        double ecrs = ecrs0 + f_zeta * (zeta4 * gcrs - macrs) + macrs;
        double dgcrs_drs = decrs1_drs - decrs0_drs;
        double decrs_drs = decrs0_drs + f_zeta * (zeta4 * dgcrs_drs - dmacrs_drs) + dmacrs_drs;
        double dfzeta4_dzeta = 4.0 * zeta * zeta * zeta * f_zeta + fp_zeta * zeta4;
        double decrs_dzeta = dfzeta4_dzeta * gcrs + fp_zeta * (macrs - ecrs0) - f_zeta * macrs * 0.0; // simplified

        // Actually, the reference has:
        // decrs_dzeta = 4.0*zeta^3 * f_zeta * gcrs + fp_zeta * zeta4 * gcrs
        //              + fp_zeta * (macrs_correction)
        // Let me be more precise:
        decrs_dzeta = fp_zeta * (zeta4 * gcrs - macrs + macrs) // = fp_zeta * zeta4 * gcrs
                      + f_zeta * 4.0 * zeta * zeta * zeta * gcrs;
        // Simplify: fp_zeta * zeta4 * gcrs + 4*zeta^3 * f_zeta * gcrs wouldn't be right either.
        // The correct formula from the reference is:
        // d(ecrs)/dzeta = f'(zeta) * (zeta^4 * gcrs - macrs) + f(zeta) * 4*zeta^3 * gcrs + dmacrs/dzeta
        // Since macrs doesn't depend on zeta, dmacrs/dzeta = 0
        decrs_dzeta = fp_zeta * (zeta4 * gcrs - macrs) + f_zeta * 4.0 * zeta * zeta * zeta * gcrs;

        ec[i] = ecrs;
        double vxcadd = ecrs - (rs / 3.0) * decrs_drs - zeta * decrs_dzeta;
        vc[i] = vxcadd + decrs_dzeta;           // spin up
        vc[DMnd + i] = vxcadd - decrs_dzeta;    // spin down
    }
}

// ============================================================
// GGA PBE exchange, spin-polarized
// Ref: exchangeCorrelation.c:887-938
// ============================================================
void XCFunctional::pbex_spin(int DMnd, const double* rho, const double* sigma,
                              int iflag, double* ex, double* vx, double* v2x) {
    double mu_tab[4] = {0.2195149727645171, 10.0/81.0, 0.2195149727645171, 0.2195149727645171};
    double kappa_tab[4] = {0.804, 0.804, 0.804, 1.245};

    double kappa = kappa_tab[iflag-1];
    double mu = mu_tab[iflag-1];
    double mu_divkappa = mu / kappa;
    double threefourth_divpi = (3.0/4.0) / constants::PI;
    constexpr double third = 1.0/3.0;
    double sixpi2_1_3 = std::pow(6.0*constants::PI*constants::PI, third);
    double sixpi2m1_3 = 1.0 / sixpi2_1_3;

    for (int i = 0; i < DMnd; i++) {
        double rhom1_3 = std::pow(rho[i], -third);
        double rhotot_inv = rhom1_3 * rhom1_3 * rhom1_3;
        double extot = 0.0;

        for (int spn_i = 0; spn_i < 2; spn_i++) {
            double rho_updn = rho[DMnd + spn_i*DMnd + i];
            double rhomot = std::pow(rho_updn, -third);
            double ex_lsd = -threefourth_divpi * sixpi2_1_3 * (rhomot * rhomot * rho_updn);

            double rho_inv = rhomot * rhomot * rhomot;
            double coeffss = (1.0/4.0) * sixpi2m1_3 * sixpi2m1_3 * (rho_inv * rho_inv * rhomot * rhomot);
            double ss = sigma[DMnd + spn_i*DMnd + i] * coeffss; // spin sigma already divided

            double divss, dfxdss;
            if (iflag == 1 || iflag == 2 || iflag == 4) {
                divss = 1.0 / (1.0 + mu_divkappa * ss);
                dfxdss = mu * (divss * divss);
            } else {
                divss = std::exp(-mu_divkappa * ss);
                dfxdss = mu * divss;
            }

            double fx = 1.0 + kappa * (1.0 - divss);
            double dssdn = (-8.0/3.0) * (ss * rho_inv);
            double dfxdn = dfxdss * dssdn;
            double dssdg = 2.0 * coeffss;
            double dfxdg = dfxdss * dssdg;

            extot += ex_lsd * fx * rho_updn;
            vx[spn_i*DMnd + i] = ex_lsd * ((4.0/3.0) * fx + rho_updn * dfxdn);
            v2x[spn_i*DMnd + i] = ex_lsd * rho_updn * dfxdg;
        }
        ex[i] = extot * rhotot_inv;
    }
}

// ============================================================
// GGA PBE correlation, spin-polarized
// Ref: exchangeCorrelation.c:985-1156
// rho layout: [total | up | down], sigma layout: [|∇ρ_tot|²]
// vc layout: [up | down], v2c: [Nd_d] (wrt total gradient)
// ============================================================
void XCFunctional::pbec_spin(int DMnd, const double* rho, const double* sigma,
                              int iflag, double* ec, double* vc, double* v2c) {
    double beta_tab[3] = {0.066725, 0.046, 0.066725};
    double beta = beta_tab[iflag-1];
    double gamma = (1.0 - std::log(2.0)) / (constants::PI * constants::PI);
    double gamma_inv = 1.0 / gamma;
    constexpr double third = 1.0/3.0;
    constexpr double alpha_zeta2 = 1.0 - 1.0e-6;
    constexpr double alpha_zeta = 1.0 - 1.0e-6;
    constexpr double rsfac = 0.6203504908994000;
    double sq_rsfac = std::sqrt(rsfac);
    double sq_rsfac_inv = 1.0 / sq_rsfac;
    constexpr double fsec_inv = 1.0/1.709921;
    double factf_zeta = 1.0 / (std::pow(2.0, 4.0/3.0) - 2.0);
    double factfp_zeta = (4.0/3.0) * factf_zeta * alpha_zeta2;
    double coeff_tt = 1.0 / (16.0 / constants::PI * std::pow(3.0*constants::PI*constants::PI, third));

    // PW92 parameters: paramagnetic (ec0), ferromagnetic (ec1), alpha_c (mac)
    constexpr double ec0_aa = 0.031091, ec0_a1 = 0.21370;
    constexpr double ec0_b1 = 7.5957, ec0_b2 = 3.5876, ec0_b3 = 1.6382, ec0_b4 = 0.49294;
    constexpr double ec1_aa = 0.015545, ec1_a1 = 0.20548;
    constexpr double ec1_b1 = 14.1189, ec1_b2 = 6.1977, ec1_b3 = 3.3662, ec1_b4 = 0.62517;
    constexpr double mac_aa = 0.016887, mac_a1 = 0.11125;
    constexpr double mac_b1 = 10.357, mac_b2 = 3.6231, mac_b3 = 0.88026, mac_b4 = 0.49671;

    auto pw92_G = [](double rs, double sqr_rs, double rsm1_2,
                     double aa, double a1, double b1, double b2, double b3, double b4,
                     double& ecrs, double& decrs_drs) {
        double q0 = -2.0 * aa * (1.0 + a1 * rs);
        double q1 = 2.0 * aa * (b1 * sqr_rs + b2 * rs + b3 * rs * sqr_rs + b4 * rs * rs);
        double q1p = aa * (b1 * rsm1_2 + 2.0 * b2 + 3.0 * b3 * sqr_rs + 4.0 * b4 * rs);
        double den = 1.0 / (q1 * q1 + q1);
        double logv = -std::log(q1 * q1 * den);
        ecrs = q0 * logv;
        decrs_drs = -2.0 * aa * a1 * logv - q0 * q1p * den;
    };

    for (int i = 0; i < DMnd; i++) {
        double rhom1_3 = std::pow(rho[i], -third);
        double rhotot_inv = rhom1_3 * rhom1_3 * rhom1_3;
        double rhotmo6 = std::sqrt(rhom1_3);
        double rhoto6 = rho[i] * rhom1_3 * rhom1_3 * rhotmo6;

        // Spin polarization: zeta = (rho_up - rho_down) / rho_tot
        double zeta = (rho[DMnd + i] - rho[2*DMnd + i]) * rhotot_inv;
        zeta = std::max(-1.0 + 1e-12, std::min(1.0 - 1e-12, zeta));
        double zetp = 1.0 + zeta * alpha_zeta;
        double zetm = 1.0 - zeta * alpha_zeta;
        double zetpm1_3 = std::pow(zetp, -third);
        double zetmm1_3 = std::pow(zetm, -third);

        // rs = (3/(4*pi*rho))^(1/3)
        double rs = rsfac * rhom1_3;
        double sqr_rs = sq_rsfac * rhotmo6;
        double rsm1_2 = sq_rsfac_inv * rhoto6;

        // PW92 for paramagnetic, ferromagnetic, and MAC
        double ecrs0, decrs0_drs;
        pw92_G(rs, sqr_rs, rsm1_2, ec0_aa, ec0_a1, ec0_b1, ec0_b2, ec0_b3, ec0_b4, ecrs0, decrs0_drs);
        double ecrs1, decrs1_drs;
        pw92_G(rs, sqr_rs, rsm1_2, ec1_aa, ec1_a1, ec1_b1, ec1_b2, ec1_b3, ec1_b4, ecrs1, decrs1_drs);
        double macrs, dmacrs_drs;
        pw92_G(rs, sqr_rs, rsm1_2, mac_aa, mac_a1, mac_b1, mac_b2, mac_b3, mac_b4, macrs, dmacrs_drs);

        // f(zeta) and derivatives
        double zetp_1_3 = (1.0 + zeta * alpha_zeta2) * std::pow(zetpm1_3, 2.0);
        double zetm_1_3 = (1.0 - zeta * alpha_zeta2) * std::pow(zetmm1_3, 2.0);
        double f_zeta = ((1.0 + zeta * alpha_zeta2) * zetp_1_3 + (1.0 - zeta * alpha_zeta2) * zetm_1_3 - 2.0) * factf_zeta;
        double fp_zeta = (zetp_1_3 - zetm_1_3) * factfp_zeta;
        double zeta4 = zeta * zeta * zeta * zeta;

        // Interpolated LSD correlation
        double gcrs = ecrs1 - ecrs0 + macrs * fsec_inv;
        double ecrs = ecrs0 + f_zeta * (zeta4 * gcrs - macrs * fsec_inv);
        double dgcrs_drs = decrs1_drs - decrs0_drs + dmacrs_drs * fsec_inv;
        double decrs_drs = decrs0_drs + f_zeta * (zeta4 * dgcrs_drs - dmacrs_drs * fsec_inv);
        double dfzeta4_dzeta = 4.0 * zeta * zeta * zeta * f_zeta + fp_zeta * zeta4;
        double decrs_dzeta = dfzeta4_dzeta * gcrs - fp_zeta * macrs * fsec_inv;

        ec[i] = ecrs;
        double vxcadd = ecrs - rs * third * decrs_drs - zeta * decrs_dzeta;
        vc[i] = vxcadd + decrs_dzeta;
        vc[DMnd + i] = vxcadd - decrs_dzeta;

        // === GGA part ===
        // phi(zeta) = ((1+zeta)^(2/3) + (1-zeta)^(2/3)) / 2
        double phi_zeta = (zetpm1_3 * (1.0 + zeta * alpha_zeta) +
                           zetmm1_3 * (1.0 - zeta * alpha_zeta)) * 0.5;
        double phip_zeta = (zetpm1_3 - zetmm1_3) * third * alpha_zeta;
        double phi_zeta_inv = 1.0 / phi_zeta;
        double phi_logder = phip_zeta * phi_zeta_inv;
        double phi3_zeta = phi_zeta * phi_zeta * phi_zeta;
        double gamphi3inv = gamma_inv * phi_zeta_inv * phi_zeta_inv * phi_zeta_inv;

        // ec -> bb
        double bb = ecrs * gamphi3inv;
        double dbb_drs = decrs_drs * gamphi3inv;
        double dbb_dzeta = gamphi3inv * (decrs_dzeta - 3.0 * ecrs * phi_logder);

        // bb -> cc
        double exp_pbe = std::exp(-bb);
        double cc = 1.0 / (exp_pbe - 1.0);
        double dcc_dbb = cc * cc * exp_pbe;
        double dcc_drs = dcc_dbb * dbb_drs;
        double dcc_dzeta = dcc_dbb * dbb_dzeta;

        // cc -> aa
        double coeff_aa = beta * gamma_inv * phi_zeta_inv * phi_zeta_inv;
        double aa = coeff_aa * cc;
        double daa_drs = coeff_aa * dcc_drs;
        double daa_dzeta = -2.0 * aa * phi_logder + coeff_aa * dcc_dzeta;

        // Reduced gradient t
        double grrho2 = sigma[i];
        double dtt_dg = 2.0 * rhotot_inv * rhotot_inv * rhom1_3 * coeff_tt;
        double tt = 0.5 * grrho2 * dtt_dg;

        // aa, tt -> xx
        double xx = aa * tt;
        double dxx_drs = daa_drs * tt;
        double dxx_dzeta = daa_dzeta * tt;
        double dxx_dtt = aa;

        // xx -> pade
        double pade_den = 1.0 / (1.0 + xx * (1.0 + xx));
        double pade = (1.0 + xx) * pade_den;
        double dpade_dxx = -xx * (2.0 + xx) * (pade_den * pade_den);
        double dpade_drs = dpade_dxx * dxx_drs;
        double dpade_dtt = dpade_dxx * dxx_dtt;
        double dpade_dzeta = dpade_dxx * dxx_dzeta;

        // pade -> qq
        double coeff_qq = tt * phi_zeta_inv * phi_zeta_inv;
        double qq = coeff_qq * pade;
        double dqq_drs = coeff_qq * dpade_drs;
        double dqq_dtt = pade * phi_zeta_inv * phi_zeta_inv + coeff_qq * dpade_dtt;
        double dqq_dzeta = coeff_qq * (dpade_dzeta - 2.0 * pade * phi_logder);

        // qq -> rr
        double arg_rr = 1.0 + beta * gamma_inv * qq;
        double div_rr = 1.0 / arg_rr;
        double rr = gamma * std::log(arg_rr);
        double drr_dqq = beta * div_rr;
        double drr_drs = drr_dqq * dqq_drs;
        double drr_dtt = drr_dqq * dqq_dtt;
        double drr_dzeta = drr_dqq * dqq_dzeta;

        // rr -> hh
        double hh = phi3_zeta * rr;
        double dhh_drs = phi3_zeta * drr_drs;
        double dhh_dtt = phi3_zeta * drr_dtt;
        double dhh_dzeta = phi3_zeta * (drr_dzeta + 3.0 * rr * phi_logder);

        // Final GGA correlation
        ec[i] += hh;
        double drhohh_drho = hh - third * rs * dhh_drs - zeta * dhh_dzeta - (7.0/3.0) * tt * dhh_dtt;
        vc[i] += drhohh_drho + dhh_dzeta;
        vc[DMnd + i] += drhohh_drho - dhh_dzeta;

        // Derivative wrt total gradient
        v2c[i] = rho[i] * dtt_dg * dhh_dtt;
    }
}

// ============================================================
// Apply GGA divergence correction to potential
// Ref: exchangeCorrelation.c:192-210
// Vxc += -div(v2xc * grad(rho))
// For orthogonal cells: Drho *= v2xc, then take divergence
// ============================================================
void XCFunctional::apply_gga(const double* rho, double* Vxc, double* exc,
                               const double* Drho_x, const double* Drho_y, const double* Drho_z,
                               const double* v2xc, int Nd_d) const {
    if (!gradient_ || !halo_) return;

    int FDn = gradient_->stencil().FDn();
    int nx = domain_->Nx_d();
    int ny = domain_->Ny_d();
    int nz = domain_->Nz_d();
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nz_ex = nz + 2 * FDn;
    int nd_ex = nx_ex * ny_ex * nz_ex;

    // Drho_times_v2xc: for non-orth, multiply by lapcT matrix first
    // fx[i] = v2xc * (lapcT[0,0]*Dx + lapcT[0,1]*Dy + lapcT[0,2]*Dz)
    bool is_orth = grid_->lattice().is_orthogonal();
    const Mat3& lapcT = grid_->lattice().lapc_T();

    std::vector<double> fx(Nd_d), fy(Nd_d), fz(Nd_d);
    if (is_orth) {
        for (int i = 0; i < Nd_d; i++) {
            fx[i] = Drho_x[i] * v2xc[i];
            fy[i] = Drho_y[i] * v2xc[i];
            fz[i] = Drho_z[i] * v2xc[i];
        }
    } else {
        for (int i = 0; i < Nd_d; i++) {
            double dx = Drho_x[i], dy = Drho_y[i], dz = Drho_z[i];
            fx[i] = (lapcT(0,0)*dx + lapcT(0,1)*dy + lapcT(0,2)*dz) * v2xc[i];
            fy[i] = (lapcT(1,0)*dx + lapcT(1,1)*dy + lapcT(1,2)*dz) * v2xc[i];
            fz[i] = (lapcT(2,0)*dx + lapcT(2,1)*dy + lapcT(2,2)*dz) * v2xc[i];
        }
    }

    // Take divergence: grad of each component in its direction
    std::vector<double> fx_ex(nd_ex), fy_ex(nd_ex), fz_ex(nd_ex);
    halo_->execute(fx.data(), fx_ex.data(), 1);
    halo_->execute(fy.data(), fy_ex.data(), 1);
    halo_->execute(fz.data(), fz_ex.data(), 1);

    std::vector<double> DDrho_x(Nd_d), DDrho_y(Nd_d), DDrho_z(Nd_d);
    gradient_->apply(fx_ex.data(), DDrho_x.data(), 0, 1);
    gradient_->apply(fy_ex.data(), DDrho_y.data(), 1, 1);
    gradient_->apply(fz_ex.data(), DDrho_z.data(), 2, 1);

    // Vxc += -DDrho_x - DDrho_y - DDrho_z
    for (int i = 0; i < Nd_d; i++) {
        Vxc[i] += -DDrho_x[i] - DDrho_y[i] - DDrho_z[i];
    }
}

// ============================================================
// Main evaluate: non-spin-polarized
// Ref: exchangeCorrelation.c:75-227
// ============================================================
void XCFunctional::evaluate(const double* rho, double* Vxc, double* exc, int Nd_d,
                            double* Dxcdgrho_out) const {
    std::vector<double> ex(Nd_d), ec(Nd_d), vx(Nd_d), vc(Nd_d);

    if (is_gga() && gradient_ && halo_) {
        // Compute sigma = |grad(rho)|^2 and gradient components
        int FDn = gradient_->stencil().FDn();
        int nx = domain_->Nx_d();
        int ny = domain_->Ny_d();
        int nz = domain_->Nz_d();
        int nd_ex = (nx + 2*FDn) * (ny + 2*FDn) * (nz + 2*FDn);

        std::vector<double> rho_ex(nd_ex, 0.0);
        halo_->execute(rho, rho_ex.data(), 1);

        std::vector<double> Drho_x(Nd_d), Drho_y(Nd_d), Drho_z(Nd_d);
        gradient_->apply(rho_ex.data(), Drho_x.data(), 0, 1);
        gradient_->apply(rho_ex.data(), Drho_y.data(), 1, 1);
        gradient_->apply(rho_ex.data(), Drho_z.data(), 2, 1);

        // Compute sigma = |grad(rho)|^2
        // For non-orth: sigma = Drho^T * lapcT * Drho
        bool is_orth = grid_->lattice().is_orthogonal();
        const Mat3& lapcT = grid_->lattice().lapc_T();

        std::vector<double> sigma(Nd_d);
        if (is_orth) {
            for (int i = 0; i < Nd_d; i++) {
                sigma[i] = Drho_x[i]*Drho_x[i] + Drho_y[i]*Drho_y[i] + Drho_z[i]*Drho_z[i];
            }
        } else {
            for (int i = 0; i < Nd_d; i++) {
                sigma[i] = lapcT(0,0)*Drho_x[i]*Drho_x[i]
                         + lapcT(1,1)*Drho_y[i]*Drho_y[i]
                         + lapcT(2,2)*Drho_z[i]*Drho_z[i]
                         + 2.0*lapcT(0,1)*Drho_x[i]*Drho_y[i]
                         + 2.0*lapcT(0,2)*Drho_x[i]*Drho_z[i]
                         + 2.0*lapcT(1,2)*Drho_y[i]*Drho_z[i];
            }
        }

        std::vector<double> v2x(Nd_d), v2c(Nd_d);
        int iflag = get_pbe_iflag();

        // Exchange
        pbex(Nd_d, rho, sigma.data(), iflag, ex.data(), vx.data(), v2x.data());
        // Correlation
        pbec(Nd_d, rho, sigma.data(), iflag, ec.data(), vc.data(), v2c.data());

        // Combine: e_xc = ex + ec, Vxc = vx + vc, Dxcdgrho = v2x + v2c
        std::vector<double> v2xc(Nd_d);
        for (int i = 0; i < Nd_d; i++) {
            exc[i] = ex[i] + ec[i];
            Vxc[i] = vx[i] + vc[i];
            v2xc[i] = v2x[i] + v2c[i];
        }

        // Store Dxcdgrho if requested (matching reference: pSPARC->Dxcdgrho)
        if (Dxcdgrho_out) {
            for (int i = 0; i < Nd_d; i++) {
                Dxcdgrho_out[i] = v2xc[i];
            }
        }

        // Apply divergence correction
        apply_gga(rho, Vxc, exc, Drho_x.data(), Drho_y.data(), Drho_z.data(), v2xc.data(), Nd_d);

    } else {
        // LDA path
        switch (type_) {
            case XCType::LDA_PZ:
                slater(Nd_d, rho, ex.data(), vx.data());
                pz(Nd_d, rho, ec.data(), vc.data());
                break;
            case XCType::LDA_PW:
            default:
                slater(Nd_d, rho, ex.data(), vx.data());
                pw(Nd_d, rho, ec.data(), vc.data());
                break;
        }

        for (int i = 0; i < Nd_d; i++) {
            exc[i] = ex[i] + ec[i];
            Vxc[i] = vx[i] + vc[i];
        }

        // GGA without gradient available: fall back to LDA
        if (is_gga() && (!gradient_ || !halo_)) {
            slater(Nd_d, rho, ex.data(), vx.data());
            pw(Nd_d, rho, ec.data(), vc.data());
            for (int i = 0; i < Nd_d; i++) {
                exc[i] = ex[i] + ec[i];
                Vxc[i] = vx[i] + vc[i];
            }
        }
    }
}

// ============================================================
// Main evaluate: spin-polarized (collinear)
// Ref: exchangeCorrelation.c:228-389
// rho layout: [total | up | down] each of size DMnd
// Vxc layout: [up | down] each of size DMnd
// ============================================================
void XCFunctional::evaluate_spin(const double* rho, double* Vxc, double* exc, int Nd_d,
                                  double* Dxcdgrho_out) const {
    std::vector<double> ex(Nd_d), ec(Nd_d);
    std::vector<double> vx(2 * Nd_d), vc(2 * Nd_d);

    if (is_gga() && gradient_ && halo_) {
        // Spin-polarized GGA path
        // Ref: exchangeCorrelation.c:228-389
        //
        // rho layout: [total(Nd_d) | up(Nd_d) | down(Nd_d)]
        // Need gradients of all 3 density columns

        int FDn = gradient_->stencil().FDn();
        int nx = domain_->Nx_d();
        int ny = domain_->Ny_d();
        int nz = domain_->Nz_d();
        int nd_ex = (nx + 2*FDn) * (ny + 2*FDn) * (nz + 2*FDn);

        // Halo exchange and gradient for all 3 density columns (total, up, down)
        // Drho_x/y/z each have 3*Nd_d elements: [total | up | down]
        std::vector<double> Drho_x(3 * Nd_d), Drho_y(3 * Nd_d), Drho_z(3 * Nd_d);
        std::vector<double> rho_ex(nd_ex);

        for (int col = 0; col < 3; col++) {
            halo_->execute(rho + col * Nd_d, rho_ex.data(), 1);
            gradient_->apply(rho_ex.data(), Drho_x.data() + col * Nd_d, 0, 1);
            gradient_->apply(rho_ex.data(), Drho_y.data() + col * Nd_d, 1, 1);
            gradient_->apply(rho_ex.data(), Drho_z.data() + col * Nd_d, 2, 1);
        }

        // Compute sigma: [|∇ρ_total|² | |∇ρ_up|² | |∇ρ_down|²]
        // For non-orth: sigma = Drho^T * lapcT * Drho
        bool is_orth = grid_->lattice().is_orthogonal();
        const Mat3& lapcT = grid_->lattice().lapc_T();

        std::vector<double> sigma(3 * Nd_d);
        for (int col = 0; col < 3; col++) {
            for (int i = 0; i < Nd_d; i++) {
                int idx = col * Nd_d + i;
                if (is_orth) {
                    sigma[idx] = Drho_x[idx]*Drho_x[idx] + Drho_y[idx]*Drho_y[idx] + Drho_z[idx]*Drho_z[idx];
                } else {
                    double dx = Drho_x[idx], dy = Drho_y[idx], dz = Drho_z[idx];
                    sigma[idx] = lapcT(0,0)*dx*dx + lapcT(1,1)*dy*dy + lapcT(2,2)*dz*dz
                               + 2.0*lapcT(0,1)*dx*dy + 2.0*lapcT(0,2)*dx*dz + 2.0*lapcT(1,2)*dy*dz;
                }
            }
        }

        // Call spin-polarized exchange and correlation
        // v2x: 2 columns (up, down) — per-spin gradient derivatives
        // v2c: 1 column — uses total gradient
        std::vector<double> v2x(2 * Nd_d, 0.0), v2c(Nd_d, 0.0);
        int iflag = get_pbe_iflag();

        pbex_spin(Nd_d, rho, sigma.data(), iflag, ex.data(), vx.data(), v2x.data());
        pbec_spin(Nd_d, rho, sigma.data(), iflag, ec.data(), vc.data(), v2c.data());

        // Combine: exc = ex + ec, Vxc = vx + vc
        for (int i = 0; i < Nd_d; i++) {
            exc[i] = ex[i] + ec[i];
            Vxc[i] = vx[i] + vc[i];                         // spin up
            Vxc[Nd_d + i] = vx[Nd_d + i] + vc[Nd_d + i];   // spin down
        }

        // Pack Dxcdgrho: 3 columns = [v2c | v2x_up | v2x_down]
        // Ref: exchangeCorrelation.c:339-341
        std::vector<double> Dxcdgrho(3 * Nd_d);
        for (int i = 0; i < Nd_d; i++) {
            Dxcdgrho[i]            = v2c[i];            // correlation (total gradient)
            Dxcdgrho[Nd_d + i]     = v2x[i];            // exchange, spin up
            Dxcdgrho[2*Nd_d + i]   = v2x[Nd_d + i];     // exchange, spin down
        }

        if (Dxcdgrho_out) {
            std::memcpy(Dxcdgrho_out, Dxcdgrho.data(), 3 * Nd_d * sizeof(double));
        }

        // Apply 3-column divergence correction
        // Ref: exchangeCorrelation.c:357-371
        // Multiply: Drho_dir[col] *= Dxcdgrho[col] for all 3 columns
        // Then compute divergence of each column and add to Vxc
        std::vector<double> fx(3 * Nd_d), fy(3 * Nd_d), fz(3 * Nd_d);
        for (int col = 0; col < 3; col++) {
            for (int i = 0; i < Nd_d; i++) {
                int idx = col * Nd_d + i;
                if (is_orth) {
                    fx[idx] = Drho_x[idx] * Dxcdgrho[idx];
                    fy[idx] = Drho_y[idx] * Dxcdgrho[idx];
                    fz[idx] = Drho_z[idx] * Dxcdgrho[idx];
                } else {
                    double dx = Drho_x[idx], dy = Drho_y[idx], dz = Drho_z[idx];
                    double v = Dxcdgrho[idx];
                    fx[idx] = (lapcT(0,0)*dx + lapcT(0,1)*dy + lapcT(0,2)*dz) * v;
                    fy[idx] = (lapcT(1,0)*dx + lapcT(1,1)*dy + lapcT(1,2)*dz) * v;
                    fz[idx] = (lapcT(2,0)*dx + lapcT(2,1)*dy + lapcT(2,2)*dz) * v;
                }
            }
        }

        // Compute divergence for each of the 3 columns
        std::vector<double> DDrho_x(3 * Nd_d), DDrho_y(3 * Nd_d), DDrho_z(3 * Nd_d);
        std::vector<double> col_ex(nd_ex);
        for (int col = 0; col < 3; col++) {
            halo_->execute(fx.data() + col * Nd_d, col_ex.data(), 1);
            gradient_->apply(col_ex.data(), DDrho_x.data() + col * Nd_d, 0, 1);
            halo_->execute(fy.data() + col * Nd_d, col_ex.data(), 1);
            gradient_->apply(col_ex.data(), DDrho_y.data() + col * Nd_d, 1, 1);
            halo_->execute(fz.data() + col * Nd_d, col_ex.data(), 1);
            gradient_->apply(col_ex.data(), DDrho_z.data() + col * Nd_d, 2, 1);
        }

        // Add divergence correction to Vxc
        // Ref: exchangeCorrelation.c:369-370
        // Vxc_up   += -div(v2c * ∇ρ_total) - div(v2x_up * ∇ρ_up)
        // Vxc_down += -div(v2c * ∇ρ_total) - div(v2x_down * ∇ρ_down)
        for (int i = 0; i < Nd_d; i++) {
            // Column 0: correlation (total), Column 1: exchange up
            Vxc[i] += -(DDrho_x[i] + DDrho_y[i] + DDrho_z[i])
                      -(DDrho_x[Nd_d + i] + DDrho_y[Nd_d + i] + DDrho_z[Nd_d + i]);
            // Column 0: correlation (total), Column 2: exchange down
            Vxc[Nd_d + i] += -(DDrho_x[i] + DDrho_y[i] + DDrho_z[i])
                             -(DDrho_x[2*Nd_d + i] + DDrho_y[2*Nd_d + i] + DDrho_z[2*Nd_d + i]);
        }

    } else {
        // LDA spin-polarized
        slater_spin(Nd_d, rho, ex.data(), vx.data());
        pw_spin(Nd_d, rho, ec.data(), vc.data());

        for (int i = 0; i < Nd_d; i++) {
            exc[i] = ex[i] + ec[i];
            Vxc[i] = vx[i] + vc[i];                // spin up
            Vxc[Nd_d + i] = vx[Nd_d + i] + vc[Nd_d + i];  // spin down
        }
    }
}

} // namespace sparc
