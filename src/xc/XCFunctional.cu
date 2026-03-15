#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cmath>
#include "core/gpu_common.cuh"

namespace lynx {
namespace gpu {

// ============================================================
// LDA Exchange (Slater) + Correlation (PW92) — fused kernel
// Matches reference exchangeCorrelation.c exactly
// ============================================================
__global__ void lda_pw_kernel(
    const double* __restrict__ rho,
    double* __restrict__ exc,
    double* __restrict__ vxc,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double r = rho[i];
    if (r < 1e-30) {
        exc[i] = 0.0;
        vxc[i] = 0.0;
        return;
    }

    // Slater exchange
    constexpr double C2 = 0.738558766382022;   // 3/4 * (3/pi)^(1/3)
    constexpr double C3 = 0.9847450218426965;  // (3/pi)^(1/3)
    double rho_cbrt = cbrt(r);
    double ex = -C2 * rho_cbrt;
    double vx = -C3 * rho_cbrt;

    // PW92 correlation
    constexpr double A = 0.031091;
    constexpr double alpha1 = 0.21370;
    constexpr double beta1 = 7.5957;
    constexpr double beta2 = 3.5876;
    constexpr double beta3 = 1.6382;
    constexpr double beta4 = 0.49294;
    constexpr double C31 = 0.6203504908993999; // (3/4pi)^(1/3)

    double rs = C31 / rho_cbrt;
    double rs_sqrt = sqrt(rs);
    double rs_pow_1p5 = rs * rs_sqrt;
    double rs_pow_pplus1 = rs * rs;  // p=1
    double G2 = 2.0 * A * (beta1 * rs_sqrt + beta2 * rs + beta3 * rs_pow_1p5 + beta4 * rs_pow_pplus1);
    double G1 = log(1.0 + 1.0 / G2);

    double ec = -2.0 * A * (1.0 + alpha1 * rs) * G1;
    double vc = ec - (rs / 3.0) * (
        -2.0 * A * alpha1 * G1
        + (2.0 * A * (1.0 + alpha1 * rs)
           * (A * (beta1 / rs_sqrt + 2.0 * beta2 + 3.0 * beta3 * rs_sqrt + 2.0 * 2.0 * beta4 * rs)))
          / (G2 * (G2 + 1.0))
    );

    exc[i] = ex + ec;
    vxc[i] = vx + vc;
}

void lda_pw_gpu(const double* d_rho, double* d_exc, double* d_vxc, int N) {
    int bs = 256;
    int grid = (N + bs - 1) / bs;
    lda_pw_kernel<<<grid, bs>>>(d_rho, d_exc, d_vxc, N);
}

// ============================================================
// LDA Exchange (Slater) + Correlation (PZ81) — fused kernel
// ============================================================
__global__ void lda_pz_kernel(
    const double* __restrict__ rho,
    double* __restrict__ exc,
    double* __restrict__ vxc,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double r = rho[i];
    if (r < 1e-30) {
        exc[i] = 0.0;
        vxc[i] = 0.0;
        return;
    }

    // Slater exchange
    constexpr double C2 = 0.738558766382022;
    constexpr double C3 = 0.9847450218426965;
    double rho_cbrt = cbrt(r);
    double ex = -C2 * rho_cbrt;
    double vx = -C3 * rho_cbrt;

    // PZ81 correlation
    constexpr double C31 = 0.6203504908993999;
    double rs = C31 / rho_cbrt;
    double ec, vc;

    if (rs >= 1.0) {
        // Interpolation formula (Ceperley-Alder parameterization)
        constexpr double gamma_ = -0.1423;
        constexpr double beta1_ = 1.0529;
        constexpr double beta2_ = 0.3334;
        double rs_sqrt = sqrt(rs);
        double denom = 1.0 + beta1_ * rs_sqrt + beta2_ * rs;
        ec = gamma_ / denom;
        vc = ec * (1.0 + (7.0/6.0) * beta1_ * rs_sqrt + (4.0/3.0) * beta2_ * rs) / denom;
    } else {
        // High-density limit (rs < 1)
        constexpr double A_ = 0.0311;
        constexpr double B_ = -0.048;
        constexpr double C_ = 0.0020;
        constexpr double D_ = -0.0116;
        double log_rs = log(rs);
        ec = A_ * log_rs + B_ + C_ * rs * log_rs + D_ * rs;
        vc = A_ * log_rs + (B_ - A_ / 3.0) + (2.0/3.0) * C_ * rs * log_rs + (2.0 * D_ - C_) / 3.0 * rs;
    }

    exc[i] = ex + ec;
    vxc[i] = vx + vc;
}

void lda_pz_gpu(const double* d_rho, double* d_exc, double* d_vxc, int N) {
    int bs = 256;
    int grid = (N + bs - 1) / bs;
    lda_pz_kernel<<<grid, bs>>>(d_rho, d_exc, d_vxc, N);
}

// ============================================================
// Spin-polarized LDA_PW: slater_spin + pw_spin (Nspin=2)
// rho_up, rho_dn -> exc, vxc_up, vxc_dn
// ============================================================
__global__ void lda_pw_spin_kernel(
    const double* __restrict__ rho_up,
    const double* __restrict__ rho_dn,
    double* __restrict__ exc,
    double* __restrict__ vxc_up,
    double* __restrict__ vxc_dn,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double ru = rho_up[i];
    double rd = rho_dn[i];
    double rt = ru + rd;
    if (rt < 1e-30) {
        exc[i] = 0.0;
        vxc_up[i] = 0.0;
        vxc_dn[i] = 0.0;
        return;
    }

    // Spin-polarized Slater exchange
    constexpr double C2 = 0.738558766382022;
    constexpr double C3 = 0.9847450218426965;
    constexpr double two_third = 2.0 / 3.0;

    double ex_u = (ru > 1e-30) ? -C2 * cbrt(2.0 * ru) : 0.0;
    double ex_d = (rd > 1e-30) ? -C2 * cbrt(2.0 * rd) : 0.0;
    double vx_u = (ru > 1e-30) ? -C3 * cbrt(2.0 * ru) : 0.0;
    double vx_d = (rd > 1e-30) ? -C3 * cbrt(2.0 * rd) : 0.0;
    double ex_total = 0.5 * (ru * ex_u + rd * ex_d) / rt;

    // PW92 spin-polarized correlation (full derivative)
    // Uses 3 parameterizations: unpolarized (ec0), fully polarized (ec1), spin-stiffness (mac)
    constexpr double rsfac = 0.6203504908994000;
    double sq_rsfac = sqrt(rsfac);
    double sq_rsfac_inv = 1.0 / sq_rsfac;

    double rhom1_3 = cbrt(1.0 / rt);
    double rhotmo6 = sqrt(rhom1_3);
    double rhoto6 = rt * rhom1_3 * rhom1_3 * rhotmo6;

    double rs = rsfac * rhom1_3;
    double sqr_rs = sq_rsfac * rhotmo6;
    double rsm1_2 = sq_rsfac_inv * rhoto6;

    // PW92 parameterization parameters
    constexpr double ec0_aa = 0.031091, ec0_a1 = 0.21370;
    constexpr double ec0_b1 = 7.5957, ec0_b2 = 3.5876, ec0_b3 = 1.6382, ec0_b4 = 0.49294;
    constexpr double ec1_aa = 0.015545, ec1_a1 = 0.20548;
    constexpr double ec1_b1 = 14.1189, ec1_b2 = 6.1977, ec1_b3 = 3.3662, ec1_b4 = 0.62517;
    constexpr double mac_aa = 0.016887, mac_a1 = 0.11125;
    constexpr double mac_b1 = 10.357, mac_b2 = 3.6231, mac_b3 = 0.88026, mac_b4 = 0.49671;

    // Inline pw92_G for unpolarized
    double ec0_q0 = -2.0 * ec0_aa * (1.0 + ec0_a1 * rs);
    double ec0_q1 = 2.0 * ec0_aa * (ec0_b1 * sqr_rs + ec0_b2 * rs + ec0_b3 * rs * sqr_rs + ec0_b4 * rs * rs);
    double ec0_q1p = ec0_aa * (ec0_b1 * rsm1_2 + 2.0 * ec0_b2 + 3.0 * ec0_b3 * sqr_rs + 4.0 * ec0_b4 * rs);
    double ec0_den = 1.0 / (ec0_q1 * ec0_q1 + ec0_q1);
    double ec0_log = -log(ec0_q1 * ec0_q1 * ec0_den);
    double ecrs0 = ec0_q0 * ec0_log;
    double decrs0_drs = -2.0 * ec0_aa * ec0_a1 * ec0_log - ec0_q0 * ec0_q1p * ec0_den;

    // Inline pw92_G for fully polarized
    double ec1_q0 = -2.0 * ec1_aa * (1.0 + ec1_a1 * rs);
    double ec1_q1 = 2.0 * ec1_aa * (ec1_b1 * sqr_rs + ec1_b2 * rs + ec1_b3 * rs * sqr_rs + ec1_b4 * rs * rs);
    double ec1_q1p = ec1_aa * (ec1_b1 * rsm1_2 + 2.0 * ec1_b2 + 3.0 * ec1_b3 * sqr_rs + 4.0 * ec1_b4 * rs);
    double ec1_den = 1.0 / (ec1_q1 * ec1_q1 + ec1_q1);
    double ec1_log = -log(ec1_q1 * ec1_q1 * ec1_den);
    double ecrs1 = ec1_q0 * ec1_log;
    double decrs1_drs = -2.0 * ec1_aa * ec1_a1 * ec1_log - ec1_q0 * ec1_q1p * ec1_den;

    // Inline pw92_G for spin stiffness (mac = -alpha_c)
    double mac_q0 = -2.0 * mac_aa * (1.0 + mac_a1 * rs);
    double mac_q1 = 2.0 * mac_aa * (mac_b1 * sqr_rs + mac_b2 * rs + mac_b3 * rs * sqr_rs + mac_b4 * rs * rs);
    double mac_q1p = mac_aa * (mac_b1 * rsm1_2 + 2.0 * mac_b2 + 3.0 * mac_b3 * sqr_rs + 4.0 * mac_b4 * rs);
    double mac_den = 1.0 / (mac_q1 * mac_q1 + mac_q1);
    double mac_log = -log(mac_q1 * mac_q1 * mac_den);
    double macrs = mac_q0 * mac_log;
    double dmacrs_drs = -2.0 * mac_aa * mac_a1 * mac_log - mac_q0 * mac_q1p * mac_den;

    // f(zeta) and derivatives
    double zeta = (ru - rd) / rt;
    zeta = fmin(fmax(zeta, -1.0), 1.0);
    double zetp = 1.0 + zeta;
    double zetm = 1.0 - zeta;
    double zetp_1_3 = cbrt(zetp);
    double zetm_1_3 = cbrt(zetm);
    constexpr double fsec_inv = 1.0 / 1.709920934161365;
    constexpr double factf_zeta = 1.709920934161365;
    constexpr double factfp_zeta = (4.0 / 3.0) * 1.709920934161365;
    double f_zeta = (zetp * zetp_1_3 + zetm * zetm_1_3 - 2.0) * fsec_inv;
    double fp_zeta = (zetp_1_3 - zetm_1_3) * (4.0 / 3.0) * fsec_inv;
    double zeta4 = zeta * zeta * zeta * zeta;

    // Interpolated correlation energy
    double gcrs = ecrs1 - ecrs0;
    double ecrs = ecrs0 + f_zeta * (zeta4 * gcrs - macrs) + macrs;
    double dgcrs_drs = decrs1_drs - decrs0_drs;
    double decrs_drs = decrs0_drs + f_zeta * (zeta4 * dgcrs_drs - dmacrs_drs) + dmacrs_drs;
    double decrs_dzeta = fp_zeta * (zeta4 * gcrs - macrs) + f_zeta * 4.0 * zeta * zeta * zeta * gcrs;

    exc[i] = ex_total + ecrs;

    // Full correlation potential
    constexpr double third = 1.0 / 3.0;
    double vxcadd = ecrs - (rs * third) * decrs_drs - zeta * decrs_dzeta;
    double vc_up = vxcadd + decrs_dzeta;
    double vc_dn = vxcadd - decrs_dzeta;
    vxc_up[i] = vx_u + vc_up;
    vxc_dn[i] = vx_d + vc_dn;
}

void lda_pw_spin_gpu(const double* d_rho_up, const double* d_rho_dn,
                      double* d_exc, double* d_vxc_up, double* d_vxc_dn, int N) {
    int bs = 256;
    int grid = (N + bs - 1) / bs;
    lda_pw_spin_kernel<<<grid, bs>>>(d_rho_up, d_rho_dn, d_exc, d_vxc_up, d_vxc_dn, N);
}

// ============================================================
// GGA PBE Exchange + Correlation — fused kernel (non-spin)
// Matches reference pbex() + pbec() exactly
// ============================================================
__global__ void gga_pbe_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ sigma,
    double* __restrict__ exc,
    double* __restrict__ vxc,
    double* __restrict__ v2xc,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double r = rho[i];
    if (r < 1e-30) {
        exc[i] = 0.0;
        vxc[i] = 0.0;
        v2xc[i] = 0.0;
        return;
    }

    // ============ PBE Exchange (pbex, iflag=1) ============
    constexpr double third = 1.0/3.0;
    constexpr double PI_val = 3.14159265358979323846;
    double threefourth_divpi = 0.75 / PI_val;
    double sixpi2_1_3 = cbrt(6.0 * PI_val * PI_val);
    double sixpi2m1_3 = 1.0 / sixpi2_1_3;

    constexpr double kappa = 0.804;
    constexpr double mu = 0.2195149727645171;
    double mu_divkappa = mu / kappa;

    double rho_updn = r * 0.5;
    double rho_updnm1_3 = pow(rho_updn, -third);
    double rhomot = rho_updnm1_3;
    double ex_lsd = -threefourth_divpi * sixpi2_1_3 * (rhomot * rhomot * rho_updn);

    double rho_inv_x = rhomot * rhomot * rhomot;
    double coeffss = 0.25 * sixpi2m1_3 * sixpi2m1_3 * (rho_inv_x * rho_inv_x * rhomot * rhomot);
    double ss = (sigma[i] * 0.25) * coeffss;

    double divss = 1.0 / (1.0 + mu_divkappa * ss);
    double dfxdss = mu * (divss * divss);
    double fx = 1.0 + kappa * (1.0 - divss);
    double dssdn = (-8.0/3.0) * (ss * rho_inv_x);
    double dfxdn = dfxdss * dssdn;
    double dssdg = 2.0 * coeffss;
    double dfxdg = dfxdss * dssdg;

    double ex_val = ex_lsd * fx;
    double vx_val = ex_lsd * ((4.0/3.0) * fx + rho_updn * dfxdn);
    double v2x_val = 0.5 * ex_lsd * rho_updn * dfxdg;

    // ============ PBE Correlation (pbec, iflag=1) ============
    constexpr double beta_pbe = 0.066725;
    double gamma_c = (1.0 - log(2.0)) / (PI_val * PI_val);
    double gamma_inv = 1.0 / gamma_c;
    constexpr double phi_zeta_inv = 1.0;
    constexpr double phi3_zeta = 1.0;
    double gamphi3inv = gamma_inv;

    double twom1_3 = pow(2.0, -third);
    constexpr double rsfac = 0.6203504908994000;
    double sq_rsfac = sqrt(rsfac);
    double sq_rsfac_inv = 1.0 / sq_rsfac;
    double coeff_tt = 1.0 / (16.0 / PI_val * cbrt(3.0 * PI_val * PI_val));

    constexpr double ec0_aa = 0.031091;
    constexpr double ec0_a1 = 0.21370;
    constexpr double ec0_b1 = 7.5957;
    constexpr double ec0_b2 = 3.5876;
    constexpr double ec0_b3 = 1.6382;
    constexpr double ec0_b4 = 0.49294;

    double rhom1_3 = twom1_3 * rho_updnm1_3;
    double rhotot_inv = rhom1_3 * rhom1_3 * rhom1_3;
    double rhotmo6 = sqrt(rhom1_3);
    double rhoto6 = r * rhom1_3 * rhom1_3 * rhotmo6;

    double rs = rsfac * rhom1_3;
    double sqr_rs = sq_rsfac * rhotmo6;
    double rsm1_2 = sq_rsfac_inv * rhoto6;

    // PW92 parameterization
    double ec0_q0 = -2.0 * ec0_aa * (1.0 + ec0_a1 * rs);
    double ec0_q1 = 2.0 * ec0_aa * (ec0_b1 * sqr_rs + ec0_b2 * rs + ec0_b3 * rs * sqr_rs + ec0_b4 * rs * rs);
    double ec0_q1p = ec0_aa * (ec0_b1 * rsm1_2 + 2.0 * ec0_b2 + 3.0 * ec0_b3 * sqr_rs + 4.0 * ec0_b4 * rs);
    double ec0_den = 1.0 / (ec0_q1 * ec0_q1 + ec0_q1);
    double ec0_log = -log(ec0_q1 * ec0_q1 * ec0_den);
    double ecrs = ec0_q0 * ec0_log;
    double decrs_drs = -2.0 * ec0_aa * ec0_a1 * ec0_log - ec0_q0 * ec0_q1p * ec0_den;

    double ec_val = ecrs;
    double vc_val = ecrs - (rs / 3.0) * decrs_drs;

    // GGA correlation: chain of variable substitutions
    // ec -> bb
    double bb = ecrs * gamphi3inv;
    double dbb_drs = decrs_drs * gamphi3inv;

    // bb -> cc
    double exp_pbe = exp(-bb);
    double cc = 1.0 / (exp_pbe - 1.0);
    double dcc_dbb = cc * cc * exp_pbe;
    double dcc_drs = dcc_dbb * dbb_drs;

    // cc -> aa
    double coeff_aa = beta_pbe * gamma_inv * phi_zeta_inv * phi_zeta_inv;
    double aa = coeff_aa * cc;
    double daa_drs = coeff_aa * dcc_drs;

    // Introduce tt
    double grrho2 = sigma[i];
    double dtt_dg = 2.0 * rhotot_inv * rhotot_inv * rhom1_3 * coeff_tt;
    double tt = 0.5 * grrho2 * dtt_dg;

    // tt,aa -> xx
    double xx = aa * tt;
    double dxx_drs = daa_drs * tt;
    double dxx_dtt = aa;

    // xx -> pade
    double pade_den = 1.0 / (1.0 + xx * (1.0 + xx));
    double pade = (1.0 + xx) * pade_den;
    double dpade_dxx = -xx * (2.0 + xx) * (pade_den * pade_den);
    double dpade_drs = dpade_dxx * dxx_drs;
    double dpade_dtt = dpade_dxx * dxx_dtt;

    // pade -> qq
    double coeff_qq = tt * phi_zeta_inv * phi_zeta_inv;
    double qq = coeff_qq * pade;
    double dqq_drs = coeff_qq * dpade_drs;
    double dqq_dtt = pade * phi_zeta_inv * phi_zeta_inv + coeff_qq * dpade_dtt;

    // qq -> rr
    double arg_rr = 1.0 + beta_pbe * gamma_inv * qq;
    double div_rr = 1.0 / arg_rr;
    double rr = gamma_c * log(arg_rr);
    double drr_dqq = beta_pbe * div_rr;
    double drr_drs = drr_dqq * dqq_drs;
    double drr_dtt = drr_dqq * dqq_dtt;

    // rr -> hh
    double hh = phi3_zeta * rr;
    double dhh_drs = phi3_zeta * drr_drs;
    double dhh_dtt = phi3_zeta * drr_dtt;

    // GGA correlation energy and potential
    ec_val += hh;
    double drhohh_drho = hh - third * rs * dhh_drs - (7.0/3.0) * tt * dhh_dtt;
    vc_val += drhohh_drho;

    double v2c_val = r * dtt_dg * dhh_dtt;

    // ============ Combine ============
    exc[i] = ex_val + ec_val;
    vxc[i] = vx_val + vc_val;
    v2xc[i] = v2x_val + v2c_val;
}

void gga_pbe_gpu(const double* d_rho, const double* d_sigma,
                  double* d_exc, double* d_vxc, double* d_v2xc, int N) {
    int bs = 256;
    int grid = (N + bs - 1) / bs;
    gga_pbe_kernel<<<grid, bs>>>(d_rho, d_sigma, d_exc, d_vxc, d_v2xc, N);
}

// ============================================================
// Utility kernels for GGA pipeline
// ============================================================

// sigma = |grad(rho)|^2 for orthogonal cells
__global__ void sigma_kernel(
    const double* __restrict__ Drho_x,
    const double* __restrict__ Drho_y,
    const double* __restrict__ Drho_z,
    double* __restrict__ sigma,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double dx = Drho_x[i], dy = Drho_y[i], dz = Drho_z[i];
        sigma[i] = dx*dx + dy*dy + dz*dz;
    }
}

// Multiply gradient by v2xc in place: Drho_dir[i] *= v2xc[i]
__global__ void v2xc_scale_kernel(
    double* __restrict__ f,
    const double* __restrict__ v2xc,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) f[i] *= v2xc[i];
}

// Add NLCC core density: rho_xc = max(rho + rho_core, 1e-14)
__global__ void nlcc_add_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ rho_core,
    double* __restrict__ rho_xc,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double val = rho[i] + rho_core[i];
        rho_xc[i] = (val > 1e-14) ? val : 1e-14;
    }
}

// Subtract divergence: Vxc[i] -= DDrho[i]
__global__ void divergence_sub_kernel(
    double* __restrict__ Vxc,
    const double* __restrict__ DDrho,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) Vxc[i] -= DDrho[i];
}

// Jacobi preconditioner: f[i] = scale * r[i]
__global__ void jacobi_scale_kernel(
    const double* __restrict__ r,
    double* __restrict__ f,
    double scale, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) f[i] = scale * r[i];
}

// ============================================================
// GGA PBE Exchange + Correlation — spin-polarized fused kernel
// Fuses pbex_spin + pbec_spin from CPU XCFunctional.cpp
// rho layout: [total(N) | up(N) | down(N)]
// sigma layout: [|grad rho_tot|^2(N) | |grad rho_up|^2(N) | |grad rho_down|^2(N)]
// vxc layout: [up(N) | down(N)]
// v2xc layout: [v2c(N) | v2x_up(N) | v2x_down(N)]
// ============================================================
__global__ void gga_pbe_spin_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ sigma,
    double* __restrict__ exc,
    double* __restrict__ vxc,
    double* __restrict__ v2xc,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double rho_tot = rho[i];
    if (rho_tot < 1e-30) {
        exc[i] = 0.0;
        vxc[i] = 0.0;
        vxc[N + i] = 0.0;
        v2xc[i] = 0.0;
        v2xc[N + i] = 0.0;
        v2xc[2*N + i] = 0.0;
        return;
    }

    // ============ PBE Exchange spin-polarized (pbex_spin) ============
    constexpr double third = 1.0 / 3.0;
    constexpr double PI_val = 3.14159265358979323846;
    double threefourth_divpi = 0.75 / PI_val;
    double sixpi2_1_3 = cbrt(6.0 * PI_val * PI_val);
    double sixpi2m1_3 = 1.0 / sixpi2_1_3;
    constexpr double kappa = 0.804;
    constexpr double mu = 0.2195149727645171;
    double mu_divkappa = mu / kappa;

    double rhom1_3_tot = pow(rho_tot, -third);
    double rhotot_inv = rhom1_3_tot * rhom1_3_tot * rhom1_3_tot;
    double extot = 0.0;

    // Process each spin channel for exchange
    double rho_spn[2], sigma_spn[2], vx_spn[2], v2x_spn[2];
    rho_spn[0] = rho[N + i];    // rho_up
    rho_spn[1] = rho[2*N + i];  // rho_down
    sigma_spn[0] = sigma[N + i];     // |grad rho_up|^2
    sigma_spn[1] = sigma[2*N + i];   // |grad rho_down|^2

    for (int spn = 0; spn < 2; spn++) {
        double rho_updn = rho_spn[spn];
        if (rho_updn < 1e-30) {
            vx_spn[spn] = 0.0;
            v2x_spn[spn] = 0.0;
            continue;
        }
        double rhomot = pow(rho_updn, -third);
        double ex_lsd = -threefourth_divpi * sixpi2_1_3 * (rhomot * rhomot * rho_updn);

        double rho_inv = rhomot * rhomot * rhomot;
        double coeffss = 0.25 * sixpi2m1_3 * sixpi2m1_3 * (rho_inv * rho_inv * rhomot * rhomot);
        double ss = sigma_spn[spn] * coeffss;

        double divss = 1.0 / (1.0 + mu_divkappa * ss);
        double dfxdss = mu * (divss * divss);
        double fx = 1.0 + kappa * (1.0 - divss);
        double dssdn = (-8.0 / 3.0) * (ss * rho_inv);
        double dfxdn = dfxdss * dssdn;
        double dssdg = 2.0 * coeffss;
        double dfxdg = dfxdss * dssdg;

        extot += ex_lsd * fx * rho_updn;
        vx_spn[spn] = ex_lsd * ((4.0 / 3.0) * fx + rho_updn * dfxdn);
        v2x_spn[spn] = ex_lsd * rho_updn * dfxdg;
    }
    double ex_val = extot * rhotot_inv;

    // ============ PBE Correlation spin-polarized (pbec_spin) ============
    constexpr double beta_pbe = 0.066725;
    double gamma_c = (1.0 - log(2.0)) / (PI_val * PI_val);
    double gamma_inv = 1.0 / gamma_c;
    constexpr double alpha_zeta2 = 1.0 - 1.0e-6;
    constexpr double alpha_zeta = 1.0 - 1.0e-6;
    constexpr double rsfac = 0.6203504908994000;
    double sq_rsfac = sqrt(rsfac);
    double sq_rsfac_inv = 1.0 / sq_rsfac;
    constexpr double fsec_inv = 1.0 / 1.709921;
    double factf_zeta = 1.0 / (pow(2.0, 4.0 / 3.0) - 2.0);
    double factfp_zeta = (4.0 / 3.0) * factf_zeta * alpha_zeta2;
    double coeff_tt = 1.0 / (16.0 / PI_val * cbrt(3.0 * PI_val * PI_val));

    // PW92 parameters
    constexpr double ec0_aa = 0.031091, ec0_a1 = 0.21370;
    constexpr double ec0_b1 = 7.5957, ec0_b2 = 3.5876, ec0_b3 = 1.6382, ec0_b4 = 0.49294;
    constexpr double ec1_aa = 0.015545, ec1_a1 = 0.20548;
    constexpr double ec1_b1 = 14.1189, ec1_b2 = 6.1977, ec1_b3 = 3.3662, ec1_b4 = 0.62517;
    constexpr double mac_aa = 0.016887, mac_a1 = 0.11125;
    constexpr double mac_b1 = 10.357, mac_b2 = 3.6231, mac_b3 = 0.88026, mac_b4 = 0.49671;

    double rhom1_3 = pow(rho_tot, -third);
    double rhotmo6 = sqrt(rhom1_3);
    double rhoto6 = rho_tot * rhom1_3 * rhom1_3 * rhotmo6;

    // Spin polarization
    double zeta = (rho_spn[0] - rho_spn[1]) * rhotot_inv;
    zeta = fmax(-1.0 + 1e-12, fmin(1.0 - 1e-12, zeta));
    double zetp = 1.0 + zeta * alpha_zeta;
    double zetm = 1.0 - zeta * alpha_zeta;
    double zetpm1_3 = pow(zetp, -third);
    double zetmm1_3 = pow(zetm, -third);

    double rs = rsfac * rhom1_3;
    double sqr_rs = sq_rsfac * rhotmo6;
    double rsm1_2 = sq_rsfac_inv * rhoto6;

    // PW92 for paramagnetic (ec0)
    double ec0_q0 = -2.0 * ec0_aa * (1.0 + ec0_a1 * rs);
    double ec0_q1 = 2.0 * ec0_aa * (ec0_b1 * sqr_rs + ec0_b2 * rs + ec0_b3 * rs * sqr_rs + ec0_b4 * rs * rs);
    double ec0_q1p = ec0_aa * (ec0_b1 * rsm1_2 + 2.0 * ec0_b2 + 3.0 * ec0_b3 * sqr_rs + 4.0 * ec0_b4 * rs);
    double ec0_den = 1.0 / (ec0_q1 * ec0_q1 + ec0_q1);
    double ec0_log = -log(ec0_q1 * ec0_q1 * ec0_den);
    double ecrs0 = ec0_q0 * ec0_log;
    double decrs0_drs = -2.0 * ec0_aa * ec0_a1 * ec0_log - ec0_q0 * ec0_q1p * ec0_den;

    // PW92 for ferromagnetic (ec1)
    double ec1_q0 = -2.0 * ec1_aa * (1.0 + ec1_a1 * rs);
    double ec1_q1 = 2.0 * ec1_aa * (ec1_b1 * sqr_rs + ec1_b2 * rs + ec1_b3 * rs * sqr_rs + ec1_b4 * rs * rs);
    double ec1_q1p = ec1_aa * (ec1_b1 * rsm1_2 + 2.0 * ec1_b2 + 3.0 * ec1_b3 * sqr_rs + 4.0 * ec1_b4 * rs);
    double ec1_den = 1.0 / (ec1_q1 * ec1_q1 + ec1_q1);
    double ec1_log = -log(ec1_q1 * ec1_q1 * ec1_den);
    double ecrs1 = ec1_q0 * ec1_log;
    double decrs1_drs = -2.0 * ec1_aa * ec1_a1 * ec1_log - ec1_q0 * ec1_q1p * ec1_den;

    // PW92 for MAC (alpha_c)
    double mac_q0 = -2.0 * mac_aa * (1.0 + mac_a1 * rs);
    double mac_q1 = 2.0 * mac_aa * (mac_b1 * sqr_rs + mac_b2 * rs + mac_b3 * rs * sqr_rs + mac_b4 * rs * rs);
    double mac_q1p = mac_aa * (mac_b1 * rsm1_2 + 2.0 * mac_b2 + 3.0 * mac_b3 * sqr_rs + 4.0 * mac_b4 * rs);
    double mac_den = 1.0 / (mac_q1 * mac_q1 + mac_q1);
    double mac_log = -log(mac_q1 * mac_q1 * mac_den);
    double macrs = mac_q0 * mac_log;
    double dmacrs_drs = -2.0 * mac_aa * mac_a1 * mac_log - mac_q0 * mac_q1p * mac_den;

    // f(zeta) and derivatives
    double zetp_1_3 = (1.0 + zeta * alpha_zeta2) * pow(zetpm1_3, 2.0);
    double zetm_1_3 = (1.0 - zeta * alpha_zeta2) * pow(zetmm1_3, 2.0);
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

    double ec_val = ecrs;
    double vxcadd = ecrs - rs * third * decrs_drs - zeta * decrs_dzeta;
    double vc_up = vxcadd + decrs_dzeta;
    double vc_dn = vxcadd - decrs_dzeta;

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
    double exp_pbe = exp(-bb);
    double cc = 1.0 / (exp_pbe - 1.0);
    double dcc_dbb = cc * cc * exp_pbe;
    double dcc_drs = dcc_dbb * dbb_drs;
    double dcc_dzeta = dcc_dbb * dbb_dzeta;

    // cc -> aa
    double coeff_aa = beta_pbe * gamma_inv * phi_zeta_inv * phi_zeta_inv;
    double aa = coeff_aa * cc;
    double daa_drs = coeff_aa * dcc_drs;
    double daa_dzeta = -2.0 * aa * phi_logder + coeff_aa * dcc_dzeta;

    // Reduced gradient t (uses total sigma)
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
    double arg_rr = 1.0 + beta_pbe * gamma_inv * qq;
    double div_rr = 1.0 / arg_rr;
    double rr = gamma_c * log(arg_rr);
    double drr_dqq = beta_pbe * div_rr;
    double drr_drs = drr_dqq * dqq_drs;
    double drr_dtt = drr_dqq * dqq_dtt;
    double drr_dzeta = drr_dqq * dqq_dzeta;

    // rr -> hh
    double hh = phi3_zeta * rr;
    double dhh_drs = phi3_zeta * drr_drs;
    double dhh_dtt = phi3_zeta * drr_dtt;
    double dhh_dzeta = phi3_zeta * (drr_dzeta + 3.0 * rr * phi_logder);

    // Final GGA correlation
    ec_val += hh;
    double drhohh_drho = hh - third * rs * dhh_drs - zeta * dhh_dzeta - (7.0 / 3.0) * tt * dhh_dtt;
    vc_up += drhohh_drho + dhh_dzeta;
    vc_dn += drhohh_drho - dhh_dzeta;

    // Derivative wrt total gradient
    double v2c_val = rho_tot * dtt_dg * dhh_dtt;

    // ============ Combine ============
    exc[i] = ex_val + ec_val;
    vxc[i] = vx_spn[0] + vc_up;
    vxc[N + i] = vx_spn[1] + vc_dn;
    v2xc[i] = v2c_val;
    v2xc[N + i] = v2x_spn[0];
    v2xc[2*N + i] = v2x_spn[1];
}

void gga_pbe_spin_gpu(const double* d_rho, const double* d_sigma,
                       double* d_exc, double* d_vxc, double* d_v2xc, int N) {
    int bs = 256;
    int grid = (N + bs - 1) / bs;
    gga_pbe_spin_kernel<<<grid, bs>>>(d_rho, d_sigma, d_exc, d_vxc, d_v2xc, N);
}

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
