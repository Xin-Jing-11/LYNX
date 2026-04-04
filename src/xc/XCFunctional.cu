#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <xc.h>
#include <xc_funcs.h>
#include <cublas_v2.h>
#include "core/gpu_common.cuh"
#include "core/GPUContext.cuh"
#include "core/LynxContext.hpp"
#include "xc/XCFunctional.hpp"
#include "xc/XCFunctional.cuh"
#include "parallel/HaloExchange.cuh"
#include "operators/Gradient.cuh"

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

void lda_pw_gpu(const double* d_rho, double* d_exc, double* d_vxc, int N, cudaStream_t stream) {
    int bs = 256;
    int grid = (N + bs - 1) / bs;
    lda_pw_kernel<<<grid, bs, 0, stream>>>(d_rho, d_exc, d_vxc, N);
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

void lda_pz_gpu(const double* d_rho, double* d_exc, double* d_vxc, int N, cudaStream_t stream) {
    int bs = 256;
    int grid = (N + bs - 1) / bs;
    lda_pz_kernel<<<grid, bs, 0, stream>>>(d_rho, d_exc, d_vxc, N);
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
                      double* d_exc, double* d_vxc_up, double* d_vxc_dn, int N,
                      cudaStream_t stream) {
    int bs = 256;
    int grid = (N + bs - 1) / bs;
    lda_pw_spin_kernel<<<grid, bs, 0, stream>>>(d_rho_up, d_rho_dn, d_exc, d_vxc_up, d_vxc_dn, N);
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
                  double* d_exc, double* d_vxc, double* d_v2xc, int N,
                  cudaStream_t stream) {
    int bs = 256;
    int grid = (N + bs - 1) / bs;
    gga_pbe_kernel<<<grid, bs, 0, stream>>>(d_rho, d_sigma, d_exc, d_vxc, d_v2xc, N);
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
                       double* d_exc, double* d_vxc, double* d_v2xc, int N,
                       cudaStream_t stream) {
    int bs = 256;
    int grid = (N + bs - 1) / bs;
    gga_pbe_spin_kernel<<<grid, bs, 0, stream>>>(d_rho, d_sigma, d_exc, d_vxc, d_v2xc, N);
}

// ============================================================
// SCAN metaGGA Exchange + Correlation — fused kernel (non-spin)
// Hand-coded from SCANFunctional.cpp.bak reference
// ============================================================
__global__ void mgga_scan_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ sigma,
    const double* __restrict__ tau,
    double* __restrict__ exc,
    double* __restrict__ vxc,
    double* __restrict__ v2xc,
    double* __restrict__ vtau,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double r = rho[i];
    if (r < 1e-30) {
        exc[i] = 0.0;
        vxc[i] = 0.0;
        v2xc[i] = 0.0;
        vtau[i] = 0.0;
        return;
    }

    double sig = sigma[i];
    if (sig < 1e-14) sig = 1e-14;
    double normDrho = sqrt(sig);
    double tau_val = tau[i];

    // ============ basic_MGGA_variables ============
    constexpr double PI_val = 3.14159265358979323846;
    double threeMPi2_1o3 = cbrt(3.0 * PI_val * PI_val);
    double threeMPi2_2o3 = threeMPi2_1o3 * threeMPi2_1o3;

    double rho_4o3 = r * cbrt(r);
    double s = normDrho / (2.0 * threeMPi2_1o3 * rho_4o3);
    double tauw = normDrho * normDrho / (8.0 * r);
    double tauUnif = 3.0 / 10.0 * threeMPi2_2o3 * r * cbrt(r * r); // rho^(5/3)
    double alpha = (tau_val - tauw) / tauUnif;

    double rho_7o3 = rho_4o3 * r;
    double dsdn = -2.0 * normDrho / (3.0 * threeMPi2_1o3 * rho_7o3);
    double dsddn = 1.0 / (2.0 * threeMPi2_1o3 * rho_4o3);
    double DtauwDn = -normDrho * normDrho / (8.0 * r * r);
    double DtauwDDn = normDrho / (4.0 * r);
    double rho_2o3 = cbrt(r * r);
    double DtauUnifDn = threeMPi2_2o3 / 2.0 * rho_2o3;
    double daddn = (-DtauwDDn) / tauUnif;
    double dadtau = 1.0 / tauUnif;
    double dadn = (-DtauwDn * tauUnif - (tau_val - tauw) * DtauUnifDn) / (tauUnif * tauUnif);

    // ============ Calculate_scanx (exchange) ============
    constexpr double k1 = 0.065;
    constexpr double mu_ak = 10.0 / 81.0;
    double b2_x = sqrt(5913.0 / 405000.0);
    double b1_x = 511.0 / 13500.0 / (2.0 * b2_x);
    constexpr double b3_x = 0.5;
    double b4_x = mu_ak * mu_ak / k1 - 1606.0 / 18225.0 - b1_x * b1_x;
    constexpr double hx0 = 1.174;
    constexpr double c1x = 0.667;
    constexpr double c2x = 0.8;
    constexpr double dx_x = 1.24;
    constexpr double a1_x = 4.9479;

    double s2 = s * s;
    double epsilon_xUnif = -3.0 / (4.0 * PI_val) * cbrt(3.0 * PI_val * PI_val * r);

    // compose h_x^1
    double term1 = 1.0 + b4_x * s2 / mu_ak * exp(-fabs(b4_x) * s2 / mu_ak);
    double xFir = mu_ak * s2 * term1;
    double term3 = 2.0 * (b1_x * s2 + b2_x * (1.0 - alpha) * exp(-b3_x * (1.0 - alpha) * (1.0 - alpha)));
    double xSec = (term3 / 2.0) * (term3 / 2.0);
    double hx1 = 1.0 + k1 - k1 / (1.0 + (xFir + xSec) / k1);

    double fx;
    if (fabs(alpha - 1.0) < 1e-14) {
        fx = 0.0;
    } else if (alpha > 1.0) {
        fx = -dx_x * exp(c2x / (1.0 - alpha));
    } else {
        fx = exp(-c1x * alpha / (1.0 - alpha));
    }

    double sqrt_s = sqrt(s);
    double gx = 1.0 - exp(-a1_x / (sqrt_s + 1e-30));
    double Fx = (hx1 + fx * (hx0 - hx1)) * gx;
    double epsilonx = epsilon_xUnif * Fx;

    // Derivatives of exchange
    double term2 = s2 * (b4_x / mu_ak * exp(-fabs(b4_x) * s2 / mu_ak) + b4_x * s2 / mu_ak * exp(-fabs(b4_x) * s2 / mu_ak) * (-fabs(b4_x) / mu_ak));
    double term4 = b2_x * (-exp(-b3_x * (1.0 - alpha) * (1.0 - alpha)) + (1.0 - alpha) * exp(-b3_x * (1.0 - alpha) * (1.0 - alpha)) * (2.0 * b3_x * (1.0 - alpha)));
    double DxDs = 2.0 * s * (mu_ak * (term1 + term2) + b1_x * term3);
    double DxDalpha = term3 * term4;
    double DxDn = dsdn * DxDs + dadn * DxDalpha;
    double DxDDn = dsddn * DxDs + daddn * DxDalpha;
    double DxDtau = dadtau * DxDalpha;

    double exp_gx = exp(-a1_x / (sqrt_s + 1e-30));
    double DgxDn = -exp_gx * (a1_x / 2.0 / (sqrt_s + 1e-30) / (s + 1e-30)) * dsdn;
    double DgxDDn = -exp_gx * (a1_x / 2.0 / (sqrt_s + 1e-30) / (s + 1e-30)) * dsddn;
    double Dhx1Dx = 1.0 / ((1.0 + (xFir + xSec) / k1) * (1.0 + (xFir + xSec) / k1));
    double Dhx1Dn = DxDn * Dhx1Dx;
    double Dhx1DDn = DxDDn * Dhx1Dx;
    double Dhx1Dtau = DxDtau * Dhx1Dx;

    double DfxDalpha;
    if (fabs(alpha - 1.0) < 1e-14) {
        DfxDalpha = 0.0;
    } else if (alpha > 1.0) {
        DfxDalpha = -dx_x * exp(c2x / (1.0 - alpha)) * (c2x / ((1.0 - alpha) * (1.0 - alpha)));
    } else {
        DfxDalpha = exp(-c1x * alpha / (1.0 - alpha)) * (-c1x / ((1.0 - alpha) * (1.0 - alpha)));
    }

    double DfxDn = DfxDalpha * dadn;
    double DfxDDn = DfxDalpha * daddn;
    double DfxDtau = DfxDalpha * dadtau;
    double DFxDn = (hx1 + fx * (hx0 - hx1)) * DgxDn + gx * (1.0 - fx) * Dhx1Dn + gx * (hx0 - hx1) * DfxDn;
    double DFxDDn = (hx1 + fx * (hx0 - hx1)) * DgxDDn + gx * (1.0 - fx) * Dhx1DDn + gx * (hx0 - hx1) * DfxDDn;
    double DFxDtau = gx * (1.0 - fx) * Dhx1Dtau + gx * (hx0 - hx1) * DfxDtau;

    double Depsilon_xUnifDn = -cbrt(3.0 * PI_val * PI_val) / (4.0 * PI_val) * pow(r, -2.0 / 3.0);
    double vx1 = (epsilon_xUnif + r * Depsilon_xUnifDn) * Fx + r * epsilon_xUnif * DFxDn;
    double vx2 = r * epsilon_xUnif * DFxDDn;
    double vx3 = r * epsilon_xUnif * DFxDtau;

    // ============ Calculate_scanc (correlation) ============
    constexpr double b1c = 0.0285764;
    constexpr double b2c = 0.0889;
    constexpr double b3c = 0.125541;
    constexpr double betaConst = 0.06672455060314922;
    constexpr double betaRsInf = betaConst * 0.1 / 0.1778;
    constexpr double f0 = -0.9;
    constexpr double AA = 0.0310907;
    constexpr double alpha1 = 0.21370;
    constexpr double beta1 = 7.5957;
    constexpr double beta2 = 3.5876;
    constexpr double beta3 = 1.6382;
    constexpr double beta4 = 0.49294;
    constexpr double r_c = 0.031091;
    constexpr double c1c = 0.64;
    constexpr double c2c = 1.5;
    constexpr double dc = 0.7;

    double rs = cbrt(0.75 / (PI_val * r));
    double sqrRs = sqrt(rs);

    // epsilon_c^0 (alpha approach 0)
    double ecLDA0 = -b1c / (1.0 + b2c * sqrRs + b3c * rs);
    double cx0 = -3.0 / (4.0 * PI_val) * cbrt(9.0 * PI_val / 4.0);
    // zeta=0, phi=1, dx=1 for non-spin
    double Gc = 1.0; // (1 - 2.3631*(1-1))*(1 - 0^12) = 1
    double w0 = exp(-ecLDA0 / b1c) - 1.0;
    double xiInf0 = cbrt(3.0 * PI_val * PI_val / 16.0) * cbrt(3.0 * PI_val * PI_val / 16.0) * (betaRsInf / (cx0 - f0));
    double gInf0s = pow(1.0 + 4.0 * xiInf0 * s2, -0.25);
    double H0 = b1c * log(1.0 + w0 * (1.0 - gInf0s));
    double ec0 = (ecLDA0 + H0) * Gc;

    // epsilon_c^1 (alpha approach 1)
    double beta_c = betaConst * (1.0 + 0.1 * rs) / (1.0 + 0.1778 * rs);
    double ec_q0 = -2.0 * AA * (1.0 + alpha1 * rs);
    double ec_q1 = 2.0 * AA * (beta1 * sqrRs + beta2 * rs + beta3 * rs * sqrRs + beta4 * rs * rs);
    double ec_den = 1.0 / (ec_q1 * ec_q1 + ec_q1);
    double ec_log = -log(ec_q1 * ec_q1 * ec_den);
    double ec_lsda1 = ec_q0 * ec_log;

    // H1
    double phi = 1.0; // no spin
    double rPhi3 = r_c * phi * phi * phi;
    double w1 = exp(-ec_lsda1 / rPhi3) - 1.0;
    double Ac = beta_c / (r_c * w1);
    double t = cbrt(3.0 * PI_val * PI_val / 16.0) * s / (phi * sqrRs);
    double g = pow(1.0 + 4.0 * Ac * t * t, -0.25);
    double H1 = rPhi3 * log(1.0 + w1 * (1.0 - g));
    double ec1 = ec_lsda1 + H1;

    // interpolate
    double fc;
    if (fabs(alpha - 1.0) < 1e-14) {
        fc = 0.0;
    } else if (alpha > 1.0) {
        fc = -dc * exp(c2c / (1.0 - alpha));
    } else {
        fc = exp(-c1c * alpha / (1.0 - alpha));
    }
    double epsilonc = ec1 + fc * (ec0 - ec1);

    // Derivatives of correlation
    double DrsDn = -4.0 * PI_val / 9.0 * pow(4.0 * PI_val / 3.0 * r, -4.0 / 3.0);
    // ec0 derivatives (non-spin: DzetaDn=0, DGcDn=0, Gc=1)
    double DgInf0sDs = -0.25 * pow(1.0 + 4.0 * xiInf0 * s2, -1.25) * (4.0 * xiInf0 * 2.0 * s);
    double DgInf0sDn = DgInf0sDs * dsdn;
    double DgInf0sDDn = DgInf0sDs * dsddn;
    double DecLDA0Dn = b1c * (0.5 * b2c / sqrRs + b3c) / ((1.0 + b2c * sqrRs + b3c * rs) * (1.0 + b2c * sqrRs + b3c * rs)) * DrsDn;
    double Dw0Dn = (w0 + 1.0) * (-DecLDA0Dn / b1c);
    double DH0Dn = b1c * (Dw0Dn * (1.0 - gInf0s) - w0 * DgInf0sDn) / (1.0 + w0 * (1.0 - gInf0s));
    double DH0DDn = b1c * (-w0 * DgInf0sDDn) / (1.0 + w0 * (1.0 - gInf0s));
    double Dec0Dn = (DecLDA0Dn + DH0Dn) * Gc;
    double Dec0DDn = DH0DDn * Gc;

    // ec1 derivatives
    double denominatorInLogLSDA1 = 2.0 * AA * (beta1 * sqrRs + beta2 * rs + beta3 * sqrRs * rs + beta4 * rs * rs);
    double Dec_lsda1Dn = -(rs / r / 3.0) * (-2.0 * AA * alpha1 * log(1.0 + 1.0 / denominatorInLogLSDA1)
        - ((-2.0 * AA * (1.0 + alpha1 * rs)) * (AA * (beta1 / sqrRs + 2.0 * beta2 + 3.0 * beta3 * sqrRs + 2.0 * 2.0 * beta4 * rs)))
        / (denominatorInLogLSDA1 * denominatorInLogLSDA1 + denominatorInLogLSDA1));
    double DbetaDn = 0.066725 * (0.1 * (1.0 + 0.1778 * rs) - 0.1778 * (1.0 + 0.1 * rs)) / ((1.0 + 0.1778 * rs) * (1.0 + 0.1778 * rs)) * DrsDn;
    // DphiDn = 0 (no spin), but keep DtDn
    double DtDn = cbrt(3.0 * PI_val * PI_val / 16.0) * (phi * sqrRs * dsdn - s * (phi * DrsDn / (2.0 * sqrRs))) / (phi * phi * rs);
    double DtDDn = t * dsddn / s;
    double Dw1Dn = (w1 + 1.0) * (-(rPhi3 * Dec_lsda1Dn) / (rPhi3 * rPhi3));
    double DADn = (w1 * DbetaDn - beta_c * Dw1Dn) / (r_c * w1 * w1);
    double DgDn = -0.25 * pow(1.0 + 4.0 * Ac * t * t, -1.25) * (4.0 * (DADn * t * t + 2.0 * Ac * t * DtDn));
    double DgDDn = -0.25 * pow(1.0 + 4.0 * Ac * t * t, -1.25) * (4.0 * 2.0 * Ac * t * DtDDn);
    double DH1Dn = rPhi3 * (Dw1Dn * (1.0 - g) - w1 * DgDn) / (1.0 + w1 * (1.0 - g));
    double DH1DDn = rPhi3 * (-w1 * DgDDn) / (1.0 + w1 * (1.0 - g));
    double Dec1Dn = Dec_lsda1Dn + DH1Dn;
    double Dec1DDn = DH1DDn;

    // fc derivatives
    double DfcDalpha;
    if (fabs(alpha - 1.0) < 1e-14) {
        DfcDalpha = 0.0;
    } else if (alpha > 1.0) {
        DfcDalpha = fc * (c2c / ((1.0 - alpha) * (1.0 - alpha)));
    } else {
        DfcDalpha = fc * (-c1c / ((1.0 - alpha) * (1.0 - alpha)));
    }
    double DfcDn = DfcDalpha * dadn;
    double DfcDDn = DfcDalpha * daddn;
    double DfcDtau = DfcDalpha * dadtau;
    double DepsiloncDn = Dec1Dn + fc * (Dec0Dn - Dec1Dn) + DfcDn * (ec0 - ec1);
    double DepsiloncDDn = Dec1DDn + fc * (Dec0DDn - Dec1DDn) + DfcDDn * (ec0 - ec1);
    double DepsiloncDtau = DfcDtau * (ec0 - ec1);

    double vc1 = epsilonc + r * DepsiloncDn;
    double vc2 = r * DepsiloncDDn;
    double vc3 = r * DepsiloncDtau;

    // ============ Combine and output ============
    exc[i] = epsilonx + epsilonc;
    vxc[i] = vx1 + vc1;
    // v2xc = (vx2 + vc2) / normDrho = d(n*eps)/d|grad n| / |grad n| = 2*vsigma
    v2xc[i] = (vx2 + vc2) / normDrho;
    vtau[i] = vx3 + vc3;
}

void mgga_scan_gpu(const double* d_rho, const double* d_sigma, const double* d_tau,
                    double* d_exc, double* d_vxc, double* d_v2xc, double* d_vtau, int N,
                    cudaStream_t stream) {
    int bs = 256;
    int grid = (N + bs - 1) / bs;
    mgga_scan_kernel<<<grid, bs, 0, stream>>>(d_rho, d_sigma, d_tau, d_exc, d_vxc, d_v2xc, d_vtau, N);
}

// ============================================================
// SCAN metaGGA Exchange + Correlation — fused kernel (spin-polarized)
// Hand-coded from SCANFunctional.cpp.bak reference
// ============================================================

// Helper device function: compute SCAN exchange for a single spin channel
// theRho = 2*rho_s, theNormDrho = 2*normDrho_s, theTau = 2*tau_s
// Returns: epsilonx_s (per particle of 2*rho_s), vx1, vx2, vx3 wrt (theRho, theNormDrho, theTau)
__device__ void scan_exchange_single(
    double theRho, double theNormDrho, double theTau,
    double &epsx, double &vx1_out, double &vx2_out, double &vx3_out)
{
    constexpr double PI_val = 3.14159265358979323846;
    double threeMPi2_1o3 = cbrt(3.0 * PI_val * PI_val);
    double threeMPi2_2o3 = threeMPi2_1o3 * threeMPi2_1o3;

    double rho_4o3 = theRho * cbrt(theRho);
    double s = theNormDrho / (2.0 * threeMPi2_1o3 * rho_4o3);
    double tauw = theNormDrho * theNormDrho / (8.0 * theRho);
    double tauUnif = 3.0 / 10.0 * threeMPi2_2o3 * theRho * cbrt(theRho * theRho);
    double alpha = (theTau - tauw) / tauUnif;

    double rho_7o3 = rho_4o3 * theRho;
    double dsdn = -2.0 * theNormDrho / (3.0 * threeMPi2_1o3 * rho_7o3);
    double dsddn = 1.0 / (2.0 * threeMPi2_1o3 * rho_4o3);
    double DtauwDn = -theNormDrho * theNormDrho / (8.0 * theRho * theRho);
    double DtauwDDn = theNormDrho / (4.0 * theRho);
    double DtauUnifDn = threeMPi2_2o3 / 2.0 * cbrt(theRho * theRho);
    double dadn = (-DtauwDn * tauUnif - (theTau - tauw) * DtauUnifDn) / (tauUnif * tauUnif);
    double daddn = (-DtauwDDn) / tauUnif;
    double dadtau = 1.0 / tauUnif;

    constexpr double k1 = 0.065;
    constexpr double mu_ak = 10.0 / 81.0;
    double b2_x = sqrt(5913.0 / 405000.0);
    double b1_x = 511.0 / 13500.0 / (2.0 * b2_x);
    constexpr double b3_x = 0.5;
    double b4_x = mu_ak * mu_ak / k1 - 1606.0 / 18225.0 - b1_x * b1_x;
    constexpr double hx0 = 1.174;
    constexpr double c1x = 0.667;
    constexpr double c2x = 0.8;
    constexpr double dx_x = 1.24;
    constexpr double a1_x = 4.9479;

    double s2 = s * s;
    double epsilon_xUnif = -3.0 / (4.0 * PI_val) * cbrt(3.0 * PI_val * PI_val * theRho);

    double term1 = 1.0 + b4_x * s2 / mu_ak * exp(-fabs(b4_x) * s2 / mu_ak);
    double xFir = mu_ak * s2 * term1;
    double term3 = 2.0 * (b1_x * s2 + b2_x * (1.0 - alpha) * exp(-b3_x * (1.0 - alpha) * (1.0 - alpha)));
    double xSec = (term3 / 2.0) * (term3 / 2.0);
    double hx1 = 1.0 + k1 - k1 / (1.0 + (xFir + xSec) / k1);

    double fx;
    if (fabs(alpha - 1.0) < 1e-14) {
        fx = 0.0;
    } else if (alpha > 1.0) {
        fx = -dx_x * exp(c2x / (1.0 - alpha));
    } else {
        fx = exp(-c1x * alpha / (1.0 - alpha));
    }

    double sqrt_s = sqrt(s);
    double gx = 1.0 - exp(-a1_x / (sqrt_s + 1e-30));
    double Fx = (hx1 + fx * (hx0 - hx1)) * gx;
    epsx = epsilon_xUnif * Fx;

    double term2 = s2 * (b4_x / mu_ak * exp(-fabs(b4_x) * s2 / mu_ak) + b4_x * s2 / mu_ak * exp(-fabs(b4_x) * s2 / mu_ak) * (-fabs(b4_x) / mu_ak));
    double term4 = b2_x * (-exp(-b3_x * (1.0 - alpha) * (1.0 - alpha)) + (1.0 - alpha) * exp(-b3_x * (1.0 - alpha) * (1.0 - alpha)) * (2.0 * b3_x * (1.0 - alpha)));
    double DxDs = 2.0 * s * (mu_ak * (term1 + term2) + b1_x * term3);
    double DxDalpha = term3 * term4;
    double DxDn = dsdn * DxDs + dadn * DxDalpha;
    double DxDDn = dsddn * DxDs + daddn * DxDalpha;
    double DxDtau = dadtau * DxDalpha;

    double exp_gx = exp(-a1_x / (sqrt_s + 1e-30));
    double DgxDn = -exp_gx * (a1_x / 2.0 / (sqrt_s + 1e-30) / (s + 1e-30)) * dsdn;
    double DgxDDn = -exp_gx * (a1_x / 2.0 / (sqrt_s + 1e-30) / (s + 1e-30)) * dsddn;
    double Dhx1Dx = 1.0 / ((1.0 + (xFir + xSec) / k1) * (1.0 + (xFir + xSec) / k1));
    double Dhx1Dn = DxDn * Dhx1Dx;
    double Dhx1DDn = DxDDn * Dhx1Dx;
    double Dhx1Dtau = DxDtau * Dhx1Dx;

    double DfxDalpha;
    if (fabs(alpha - 1.0) < 1e-14) {
        DfxDalpha = 0.0;
    } else if (alpha > 1.0) {
        DfxDalpha = -dx_x * exp(c2x / (1.0 - alpha)) * (c2x / ((1.0 - alpha) * (1.0 - alpha)));
    } else {
        DfxDalpha = exp(-c1x * alpha / (1.0 - alpha)) * (-c1x / ((1.0 - alpha) * (1.0 - alpha)));
    }

    double DfxDn = DfxDalpha * dadn;
    double DfxDDn = DfxDalpha * daddn;
    double DfxDtau = DfxDalpha * dadtau;
    double DFxDn = (hx1 + fx * (hx0 - hx1)) * DgxDn + gx * (1.0 - fx) * Dhx1Dn + gx * (hx0 - hx1) * DfxDn;
    double DFxDDn = (hx1 + fx * (hx0 - hx1)) * DgxDDn + gx * (1.0 - fx) * Dhx1DDn + gx * (hx0 - hx1) * DfxDDn;
    double DFxDtau = gx * (1.0 - fx) * Dhx1Dtau + gx * (hx0 - hx1) * DfxDtau;

    double Depsilon_xUnifDn = -cbrt(3.0 * PI_val * PI_val) / (4.0 * PI_val) * pow(theRho, -2.0 / 3.0);
    vx1_out = (epsilon_xUnif + theRho * Depsilon_xUnifDn) * Fx + theRho * epsilon_xUnif * DFxDn;
    vx2_out = theRho * epsilon_xUnif * DFxDDn;
    vx3_out = theRho * epsilon_xUnif * DFxDtau;
}

__global__ void mgga_scan_spin_kernel(
    const double* __restrict__ rho_up,
    const double* __restrict__ rho_dn,
    const double* __restrict__ sigma_uu,
    const double* __restrict__ sigma_dd,
    const double* __restrict__ sigma_tot,
    const double* __restrict__ tau_up,
    const double* __restrict__ tau_dn,
    double* __restrict__ exc,
    double* __restrict__ vxc_up,
    double* __restrict__ vxc_dn,
    double* __restrict__ v2xc_c,
    double* __restrict__ v2xc_x_up,
    double* __restrict__ v2xc_x_dn,
    double* __restrict__ vtau_up,
    double* __restrict__ vtau_dn,
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
        v2xc_c[i] = 0.0;
        v2xc_x_up[i] = 0.0;
        v2xc_x_dn[i] = 0.0;
        vtau_up[i] = 0.0;
        vtau_dn[i] = 0.0;
        return;
    }

    constexpr double PI_val = 3.14159265358979323846;

    double sig_uu = sigma_uu[i];
    double sig_dd = sigma_dd[i];
    double sig_tt = sigma_tot[i];
    if (sig_uu < 1e-14) sig_uu = 1e-14;
    if (sig_dd < 1e-14) sig_dd = 1e-14;
    if (sig_tt < 1e-14) sig_tt = 1e-14;
    double normDrho_up = sqrt(sig_uu);
    double normDrho_dn = sqrt(sig_dd);
    double normDrho_tot = sqrt(sig_tt);

    // ============ EXCHANGE (per spin channel, doubled density) ============
    double epsx_up, vx1_up, vx2_up, vx3_up;
    double epsx_dn, vx1_dn, vx2_dn, vx3_dn;

    if (ru > 1e-30) {
        scan_exchange_single(2.0 * ru, 2.0 * normDrho_up, 2.0 * tau_up[i],
                             epsx_up, vx1_up, vx2_up, vx3_up);
    } else {
        epsx_up = 0.0; vx1_up = 0.0; vx2_up = 0.0; vx3_up = 0.0;
    }

    if (rd > 1e-30) {
        scan_exchange_single(2.0 * rd, 2.0 * normDrho_dn, 2.0 * tau_dn[i],
                             epsx_dn, vx1_dn, vx2_dn, vx3_dn);
    } else {
        epsx_dn = 0.0; vx1_dn = 0.0; vx2_dn = 0.0; vx3_dn = 0.0;
    }

    // Exchange energy per particle of total density
    double ex_total = (ru * epsx_up + rd * epsx_dn) / rt;

    // Exchange v2: divide by normDrho per spin (like CPU line 607)
    double v2x_up_out = vx2_up / normDrho_up;
    double v2x_dn_out = vx2_dn / normDrho_dn;

    // ============ CORRELATION (spin-polarized) ============
    // basic_MGSGA_variables_correlation for total density
    double threeMPi2_1o3 = cbrt(3.0 * PI_val * PI_val);
    double threeMPi2_2o3 = threeMPi2_1o3 * threeMPi2_1o3;

    double rho_4o3 = rt * cbrt(rt);
    double s = normDrho_tot / (2.0 * threeMPi2_1o3 * rho_4o3);
    double zeta = (ru - rd) / rt;
    zeta = fmin(fmax(zeta, -1.0 + 1e-14), 1.0 - 1e-14);
    double ds_val = (pow(1.0 + zeta, 5.0 / 3.0) + pow(1.0 - zeta, 5.0 / 3.0)) / 2.0;
    double tauw = normDrho_tot * normDrho_tot / (8.0 * rt);
    double tauUnif = 3.0 / 10.0 * threeMPi2_2o3 * rt * cbrt(rt * rt) * ds_val;
    double tau_tot = tau_up[i] + tau_dn[i];
    double alpha = (tau_tot - tauw) / tauUnif;

    // derivatives: s, zeta, alpha wrt nup, ndn, |grad n|, tau
    double rho_7o3 = rho_4o3 * rt;
    double dsdn = -2.0 * normDrho_tot / (3.0 * threeMPi2_1o3 * rho_7o3);
    double dsddn = 1.0 / (2.0 * threeMPi2_1o3 * rho_4o3);
    double DzetaDnup = 2.0 * rd / (rt * rt);
    double DzetaDndn = -2.0 * ru / (rt * rt);
    double DtauwDn = -normDrho_tot * normDrho_tot / (8.0 * rt * rt);
    double DtauwDDn = normDrho_tot / (4.0 * rt);

    double zetp_2o3 = pow(1.0 + zeta, 2.0 / 3.0);
    double zetm_2o3 = pow(1.0 - zeta, 2.0 / 3.0);
    double DdsDnup = 5.0 / 3.0 * (zetp_2o3 - zetm_2o3) * DzetaDnup / 2.0;
    double DdsDndn = 5.0 / 3.0 * (zetp_2o3 - zetm_2o3) * DzetaDndn / 2.0;
    double rho_2o3 = cbrt(rt * rt);
    double rho_5o3 = rt * rho_2o3;
    double DtauUnifDnup = threeMPi2_2o3 / 2.0 * rho_2o3 * ds_val + 3.0 / 10.0 * threeMPi2_2o3 * rho_5o3 * DdsDnup;
    double DtauUnifDndn = threeMPi2_2o3 / 2.0 * rho_2o3 * ds_val + 3.0 / 10.0 * threeMPi2_2o3 * rho_5o3 * DdsDndn;
    double dadnup = (-DtauwDn * tauUnif - (tau_tot - tauw) * DtauUnifDnup) / (tauUnif * tauUnif);
    double dadndn = (-DtauwDn * tauUnif - (tau_tot - tauw) * DtauUnifDndn) / (tauUnif * tauUnif);
    double daddn_c = (-DtauwDDn) / tauUnif;
    double dadtau_c = 1.0 / tauUnif;

    // SCAN correlation constants
    constexpr double b1c = 0.0285764;
    constexpr double b2c = 0.0889;
    constexpr double b3c = 0.125541;
    constexpr double betaConst = 0.06672455060314922;
    constexpr double betaRsInf = betaConst * 0.1 / 0.1778;
    constexpr double f0 = -0.9;
    constexpr double r_c = 0.031091;
    constexpr double c1c = 0.64;
    constexpr double c2c = 1.5;
    constexpr double dc = 0.7;

    // PW92 spin-resolved LDA constants
    constexpr double AA0 = 0.0310907, alpha10 = 0.21370;
    constexpr double beta10 = 7.5957, beta20 = 3.5876, beta30 = 1.6382, beta40 = 0.49294;
    constexpr double AAac = 0.0168869, alpha1ac = 0.11125;
    constexpr double beta1ac = 10.357, beta2ac = 3.6231, beta3ac = 0.88026, beta4ac = 0.49671;
    constexpr double AA1 = 0.01554535, alpha11 = 0.20548;
    constexpr double beta11 = 14.1189, beta21 = 6.1977, beta31 = 3.3662, beta41 = 0.62517;
    constexpr double factf_inv = 1.0 / 1.709920934161365;

    double phi = (pow(1.0 + zeta, 2.0 / 3.0) + pow(1.0 - zeta, 2.0 / 3.0)) / 2.0;
    double dx_c = (pow(1.0 + zeta, 4.0 / 3.0) + pow(1.0 - zeta, 4.0 / 3.0)) / 2.0;

    double rs = cbrt(0.75 / (PI_val * rt));
    double sqrRs = sqrt(rs);
    double rsmHalf = 1.0 / sqrRs;

    // epsilon_c^0
    double ecLDA0 = -b1c / (1.0 + b2c * sqrRs + b3c * rs);
    double cx0 = -3.0 / (4.0 * PI_val) * cbrt(9.0 * PI_val / 4.0);
    double Gc = (1.0 - 2.3631 * (dx_c - 1.0)) * (1.0 - pow(zeta, 12.0));
    double w0 = exp(-ecLDA0 / b1c) - 1.0;
    double xiInf0 = pow(3.0 * PI_val * PI_val / 16.0, 2.0 / 3.0) * (betaRsInf / (cx0 - f0));
    double s2 = s * s;
    double gInf0s = pow(1.0 + 4.0 * xiInf0 * s2, -0.25);
    double H0 = b1c * log(1.0 + w0 * (1.0 - gInf0s));
    double ec0 = (ecLDA0 + H0) * Gc;

    // epsilon_c^1 (spin-resolved PW92)
    double beta_c = betaConst * (1.0 + 0.1 * rs) / (1.0 + 0.1778 * rs);

    // ecrs0 (unpolarized)
    double ecrs0_q0 = -2.0 * AA0 * (1.0 + alpha10 * rs);
    double ecrs0_q1 = 2.0 * AA0 * (beta10 * sqrRs + beta20 * rs + beta30 * rs * sqrRs + beta40 * rs * rs);
    double ecrs0_q1p = AA0 * (beta10 * rsmHalf + 2.0 * beta20 + 3.0 * beta30 * sqrRs + 4.0 * beta40 * rs);
    double ecrs0_den = 1.0 / (ecrs0_q1 * ecrs0_q1 + ecrs0_q1);
    double ecrs0_log = -log(ecrs0_q1 * ecrs0_q1 * ecrs0_den);
    double ecrs0 = ecrs0_q0 * ecrs0_log;
    double Decrs0_Drs = -2.0 * AA0 * alpha10 * ecrs0_log - ecrs0_q0 * ecrs0_q1p * ecrs0_den;

    // ac (spin stiffness)
    double ac_q0 = -2.0 * AAac * (1.0 + alpha1ac * rs);
    double ac_q1 = 2.0 * AAac * (beta1ac * sqrRs + beta2ac * rs + beta3ac * rs * sqrRs + beta4ac * rs * rs);
    double ac_q1p = AAac * (beta1ac * rsmHalf + 2.0 * beta2ac + 3.0 * beta3ac * sqrRs + 4.0 * beta4ac * rs);
    double ac_den = 1.0 / (ac_q1 * ac_q1 + ac_q1);
    double ac_log = -log(ac_q1 * ac_q1 * ac_den);
    double ac = ac_q0 * ac_log;
    double Dac_Drs = -2.0 * AAac * alpha1ac * ac_log - ac_q0 * ac_q1p * ac_den;

    // ecrs1 (fully polarized)
    double ecrs1_q0 = -2.0 * AA1 * (1.0 + alpha11 * rs);
    double ecrs1_q1 = 2.0 * AA1 * (beta11 * sqrRs + beta21 * rs + beta31 * rs * sqrRs + beta41 * rs * rs);
    double ecrs1_q1p = AA1 * (beta11 * rsmHalf + 2.0 * beta21 + 3.0 * beta31 * sqrRs + 4.0 * beta41 * rs);
    double ecrs1_den = 1.0 / (ecrs1_q1 * ecrs1_q1 + ecrs1_q1);
    double ecrs1_log = -log(ecrs1_q1 * ecrs1_q1 * ecrs1_den);
    double ecrs1 = ecrs1_q0 * ecrs1_log;
    double Decrs1_Drs = -2.0 * AA1 * alpha11 * ecrs1_log - ecrs1_q0 * ecrs1_q1p * ecrs1_den;

    double f_zeta = (pow(1.0 + zeta, 4.0 / 3.0) + pow(1.0 - zeta, 4.0 / 3.0) - 2.0) / (pow(2.0, 4.0 / 3.0) - 2.0);
    double fp_zeta = (pow(1.0 + zeta, 1.0 / 3.0) - pow(1.0 - zeta, 1.0 / 3.0)) * 4.0 / 3.0 / (pow(2.0, 4.0 / 3.0) - 2.0);
    double zeta4 = zeta * zeta * zeta * zeta;
    double gcrs = ecrs1 - ecrs0 + ac / 1.709921;
    double ec_lsda1 = ecrs0 + f_zeta * (zeta4 * gcrs - ac / 1.709921);

    // H1
    double rPhi3 = r_c * phi * phi * phi;
    double w1 = exp(-ec_lsda1 / rPhi3) - 1.0;
    double Ac = beta_c / (r_c * w1);
    double t = cbrt(3.0 * PI_val * PI_val / 16.0) * s / (phi * sqrRs);
    double g = pow(1.0 + 4.0 * Ac * t * t, -0.25);
    double H1 = rPhi3 * log(1.0 + w1 * (1.0 - g));
    double ec1 = ec_lsda1 + H1;

    // interpolate
    double fc;
    if (fabs(alpha - 1.0) < 1e-14) {
        fc = 0.0;
    } else if (alpha > 1.0) {
        fc = -dc * exp(c2c / (1.0 - alpha));
    } else {
        fc = exp(-c1c * alpha / (1.0 - alpha));
    }
    double epsilonc = ec1 + fc * (ec0 - ec1);

    // ---- Correlation derivatives ----
    double DrsDn = -4.0 * PI_val / 9.0 * pow(4.0 * PI_val / 3.0 * rt, -4.0 / 3.0);
    double zetp_1o3 = pow(1.0 + zeta, 1.0 / 3.0);
    double zetm_1o3 = pow(1.0 - zeta, 1.0 / 3.0);
    double Ddx_term = (4.0 / 3.0 * zetp_1o3 - 4.0 / 3.0 * zetm_1o3) / 2.0;
    double DdxDnup = Ddx_term * DzetaDnup;
    double DdxDndn = Ddx_term * DzetaDndn;
    double DGcDnup = -2.3631 * DdxDnup * (1.0 - pow(zeta, 12.0)) + (1.0 - 2.3631 * (dx_c - 1.0)) * (-12.0 * pow(zeta, 11.0) * DzetaDnup);
    double DGcDndn = -2.3631 * DdxDndn * (1.0 - pow(zeta, 12.0)) + (1.0 - 2.3631 * (dx_c - 1.0)) * (-12.0 * pow(zeta, 11.0) * DzetaDndn);
    double DgInf0sDs = -0.25 * pow(1.0 + 4.0 * xiInf0 * s2, -1.25) * (4.0 * xiInf0 * 2.0 * s);
    double DgInf0sDn = DgInf0sDs * dsdn;
    double DgInf0sDDn = DgInf0sDs * dsddn;
    double DecLDA0Dn = b1c * (0.5 * b2c / sqrRs + b3c) / ((1.0 + b2c * sqrRs + b3c * rs) * (1.0 + b2c * sqrRs + b3c * rs)) * DrsDn;
    double Dw0Dn = (w0 + 1.0) * (-DecLDA0Dn / b1c);
    double DH0Dn = b1c * (Dw0Dn * (1.0 - gInf0s) - w0 * DgInf0sDn) / (1.0 + w0 * (1.0 - gInf0s));
    double DH0DDn = b1c * (-w0 * DgInf0sDDn) / (1.0 + w0 * (1.0 - gInf0s));
    double Dec0Dnup = (DecLDA0Dn + DH0Dn) * Gc + (ecLDA0 + H0) * DGcDnup;
    double Dec0Dndn = (DecLDA0Dn + DH0Dn) * Gc + (ecLDA0 + H0) * DGcDndn;
    double Dec0DDn = DH0DDn * Gc;

    // ec1 derivatives
    double DgcrsDrs = Decrs1_Drs - Decrs0_Drs + Dac_Drs / 1.709921;
    double Dec_lsda1_Drs = Decrs0_Drs + f_zeta * (zeta4 * DgcrsDrs - Dac_Drs / 1.709921);
    double Dfzeta4_Dzeta = 4.0 * (zeta * zeta * zeta) * f_zeta + fp_zeta * zeta4;
    double Dec_lsda1_Dzeta = Dfzeta4_Dzeta * gcrs - fp_zeta * ac / 1.709921;
    double Dec_lsda1Dnup = (-rs / 3.0 * Dec_lsda1_Drs - zeta * Dec_lsda1_Dzeta + Dec_lsda1_Dzeta) / rt;
    double Dec_lsda1Dndn = (-rs / 3.0 * Dec_lsda1_Drs - zeta * Dec_lsda1_Dzeta - Dec_lsda1_Dzeta) / rt;

    double DbetaDn = 0.066725 * (0.1 * (1.0 + 0.1778 * rs) - 0.1778 * (1.0 + 0.1 * rs)) / ((1.0 + 0.1778 * rs) * (1.0 + 0.1778 * rs)) * DrsDn;
    double Dphi_term = 0.5 * (2.0 / 3.0 * pow(1.0 + zeta, -1.0 / 3.0) - 2.0 / 3.0 * pow(1.0 - zeta, -1.0 / 3.0));
    double DphiDnup = Dphi_term * DzetaDnup;
    double DphiDndn = Dphi_term * DzetaDndn;
    double threePi2_16_1o3 = cbrt(3.0 * PI_val * PI_val / 16.0);
    double DtDnup = threePi2_16_1o3 * (phi * sqrRs * dsdn - s * (DphiDnup * sqrRs + phi * DrsDn / (2.0 * sqrRs))) / (phi * phi * rs);
    double DtDndn = threePi2_16_1o3 * (phi * sqrRs * dsdn - s * (DphiDndn * sqrRs + phi * DrsDn / (2.0 * sqrRs))) / (phi * phi * rs);
    double DtDDn = t * dsddn / s;
    double Dw1Dnup = (w1 + 1.0) * (-(rPhi3 * Dec_lsda1Dnup - r_c * ec_lsda1 * (3.0 * phi * phi * DphiDnup)) / (rPhi3 * rPhi3));
    double Dw1Dndn = (w1 + 1.0) * (-(rPhi3 * Dec_lsda1Dndn - r_c * ec_lsda1 * (3.0 * phi * phi * DphiDndn)) / (rPhi3 * rPhi3));
    double DADnup = (w1 * DbetaDn - beta_c * Dw1Dnup) / (r_c * w1 * w1);
    double DADndn = (w1 * DbetaDn - beta_c * Dw1Dndn) / (r_c * w1 * w1);
    double DgDnup = -0.25 * pow(1.0 + 4.0 * Ac * t * t, -1.25) * (4.0 * (DADnup * t * t + 2.0 * Ac * t * DtDnup));
    double DgDndn = -0.25 * pow(1.0 + 4.0 * Ac * t * t, -1.25) * (4.0 * (DADndn * t * t + 2.0 * Ac * t * DtDndn));
    double DgDDn = -0.25 * pow(1.0 + 4.0 * Ac * t * t, -1.25) * (4.0 * 2.0 * Ac * t * DtDDn);
    double log_w1g = log(1.0 + w1 * (1.0 - g));
    double DH1Dnup = 3.0 * r_c * phi * phi * DphiDnup * log_w1g + rPhi3 * (Dw1Dnup * (1.0 - g) - w1 * DgDnup) / (1.0 + w1 * (1.0 - g));
    double DH1Dndn = 3.0 * r_c * phi * phi * DphiDndn * log_w1g + rPhi3 * (Dw1Dndn * (1.0 - g) - w1 * DgDndn) / (1.0 + w1 * (1.0 - g));
    double DH1DDn = rPhi3 * (-w1 * DgDDn) / (1.0 + w1 * (1.0 - g));
    double Dec1Dnup = Dec_lsda1Dnup + DH1Dnup;
    double Dec1Dndn = Dec_lsda1Dndn + DH1Dndn;
    double Dec1DDn = DH1DDn;

    // fc derivatives
    double DfcDalpha;
    if (fabs(alpha - 1.0) < 1e-14) {
        DfcDalpha = 0.0;
    } else if (alpha > 1.0) {
        DfcDalpha = fc * (c2c / ((1.0 - alpha) * (1.0 - alpha)));
    } else {
        DfcDalpha = fc * (-c1c / ((1.0 - alpha) * (1.0 - alpha)));
    }
    double DfcDnup = DfcDalpha * dadnup;
    double DfcDndn = DfcDalpha * dadndn;
    double DfcDDn = DfcDalpha * daddn_c;
    double DfcDtau = DfcDalpha * dadtau_c;
    double DepsiloncDnup = Dec1Dnup + fc * (Dec0Dnup - Dec1Dnup) + DfcDnup * (ec0 - ec1);
    double DepsiloncDndn = Dec1Dndn + fc * (Dec0Dndn - Dec1Dndn) + DfcDndn * (ec0 - ec1);
    double DepsiloncDDn = Dec1DDn + fc * (Dec0DDn - Dec1DDn) + DfcDDn * (ec0 - ec1);
    double DepsiloncDtau = DfcDtau * (ec0 - ec1);

    double vc1_up = epsilonc + rt * DepsiloncDnup;
    double vc1_dn = epsilonc + rt * DepsiloncDndn;
    double vc2_c = rt * DepsiloncDDn;
    double vc3 = rt * DepsiloncDtau;

    // Correlation v2: divide by normDrho_tot (like CPU line 630)
    double v2c_out = vc2_c / normDrho_tot;

    // ============ Combine outputs ============
    exc[i] = ex_total + epsilonc;
    vxc_up[i] = vx1_up + vc1_up;
    vxc_dn[i] = vx1_dn + vc1_dn;
    v2xc_c[i] = v2c_out;
    v2xc_x_up[i] = v2x_up_out;
    v2xc_x_dn[i] = v2x_dn_out;
    vtau_up[i] = vx3_up + vc3;
    vtau_dn[i] = vx3_dn + vc3;
}

void mgga_scan_spin_gpu(
    const double* d_rho_up, const double* d_rho_dn,
    const double* d_sigma_uu, const double* d_sigma_dd, const double* d_sigma_tot,
    const double* d_tau_up, const double* d_tau_dn,
    double* d_exc, double* d_vxc_up, double* d_vxc_dn,
    double* d_v2xc_c, double* d_v2xc_x_up, double* d_v2xc_x_dn,
    double* d_vtau_up, double* d_vtau_dn, int N,
    cudaStream_t stream) {
    int bs = 256;
    int grid = (N + bs - 1) / bs;
    mgga_scan_spin_kernel<<<grid, bs, 0, stream>>>(
        d_rho_up, d_rho_dn, d_sigma_uu, d_sigma_dd, d_sigma_tot,
        d_tau_up, d_tau_dn,
        d_exc, d_vxc_up, d_vxc_dn,
        d_v2xc_c, d_v2xc_x_up, d_v2xc_x_dn,
        d_vtau_up, d_vtau_dn, N);
}

// ============================================================
// rSCAN / r2SCAN GPU support via libxc CPU fallback
// Downloads density/sigma/tau from GPU, calls libxc on host,
// uploads exc/vxc/v2xc/vtau back to GPU.
// ============================================================

void mgga_libxc_gpu(int xc_x_id, int xc_c_id,
                     const double* d_rho, const double* d_sigma, const double* d_tau,
                     double* d_exc, double* d_vxc, double* d_v2xc, double* d_vtau, int N) {
    // Download from GPU
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    std::vector<double> h_rho(N), h_sigma(N), h_tau(N);
    CUDA_CHECK(cudaMemcpyAsync(h_rho.data(), d_rho, N * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_sigma.data(), d_sigma, N * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_tau.data(), d_tau, N * sizeof(double), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);  // CPU needs this data now

    // Apply sigma floor (matches CPU XCFunctional.cpp and SPARC)
    for (int i = 0; i < N; i++) {
        if (h_sigma[i] < 1e-14) h_sigma[i] = 1e-14;
    }

    // Evaluate via libxc
    xc_func_type func_x, func_c;
    xc_func_init(&func_x, xc_x_id, XC_UNPOLARIZED);
    xc_func_init(&func_c, xc_c_id, XC_UNPOLARIZED);

    std::vector<double> zk_x(N, 0.0), vrho_x(N, 0.0), vsigma_x(N, 0.0), vlapl_x(N, 0.0), vtau_x(N, 0.0);
    std::vector<double> zk_c(N, 0.0), vrho_c(N, 0.0), vsigma_c(N, 0.0), vlapl_c(N, 0.0), vtau_c(N, 0.0);
    std::vector<double> lapl(N, 0.0);

    xc_mgga_exc_vxc(&func_x, N, h_rho.data(), h_sigma.data(), lapl.data(), h_tau.data(),
                     zk_x.data(), vrho_x.data(), vsigma_x.data(), vlapl_x.data(), vtau_x.data());
    xc_mgga_exc_vxc(&func_c, N, h_rho.data(), h_sigma.data(), lapl.data(), h_tau.data(),
                     zk_c.data(), vrho_c.data(), vsigma_c.data(), vlapl_c.data(), vtau_c.data());

    xc_func_end(&func_x);
    xc_func_end(&func_c);

    // Combine and upload
    std::vector<double> h_exc(N), h_vxc(N), h_v2xc(N), h_vtau(N);
    for (int i = 0; i < N; i++) {
        h_exc[i] = zk_x[i] + zk_c[i];
        h_vxc[i] = vrho_x[i] + vrho_c[i];
        // v2xc = 2*vsigma (matches SCAN kernel convention: d(n*eps)/d|grad n|^2 * 2)
        h_v2xc[i] = 2.0 * (vsigma_x[i] + vsigma_c[i]);
        h_vtau[i] = vtau_x[i] + vtau_c[i];
    }

    CUDA_CHECK(cudaMemcpyAsync(d_exc, h_exc.data(), N * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vxc, h_vxc.data(), N * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_v2xc, h_v2xc.data(), N * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vtau, h_vtau.data(), N * sizeof(double), cudaMemcpyHostToDevice, stream));
}

void mgga_libxc_spin_gpu(int xc_x_id, int xc_c_id,
                          const double* d_rho_up, const double* d_rho_dn,
                          const double* d_sigma_uu, const double* d_sigma_dd, const double* d_sigma_tot,
                          const double* d_tau_up, const double* d_tau_dn,
                          double* d_exc, double* d_vxc_up, double* d_vxc_dn,
                          double* d_v2xc_c, double* d_v2xc_x_up, double* d_v2xc_x_dn,
                          double* d_vtau_up, double* d_vtau_dn, int N) {
    // Download from GPU
    std::vector<double> h_rho_up(N), h_rho_dn(N);
    std::vector<double> h_sigma_uu(N), h_sigma_dd(N), h_sigma_tot(N);
    std::vector<double> h_tau_up(N), h_tau_dn(N);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    CUDA_CHECK(cudaMemcpyAsync(h_rho_up.data(), d_rho_up, N * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_rho_dn.data(), d_rho_dn, N * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_sigma_uu.data(), d_sigma_uu, N * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_sigma_dd.data(), d_sigma_dd, N * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_sigma_tot.data(), d_sigma_tot, N * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_tau_up.data(), d_tau_up, N * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_tau_dn.data(), d_tau_dn, N * sizeof(double), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);  // CPU needs this data now

    // Apply sigma floor (matches CPU XCFunctional.cpp and SPARC)
    for (int i = 0; i < N; i++) {
        if (h_sigma_uu[i] < 1e-14) h_sigma_uu[i] = 1e-14;
        if (h_sigma_dd[i] < 1e-14) h_sigma_dd[i] = 1e-14;
        if (h_sigma_tot[i] < 1e-14) h_sigma_tot[i] = 1e-14;
    }

    // libxc expects: rho[2*N] = [up0,dn0,up1,dn1,...], sigma[3*N] = [uu0,ud0,dd0,uu1,ud1,dd1,...]
    // tau[2*N] = [up0,dn0,...], lapl[2*N] = [up0,dn0,...]
    std::vector<double> rho_2(2 * N), sigma_3(3 * N), tau_2(2 * N), lapl_2(2 * N, 0.0);
    for (int i = 0; i < N; i++) {
        rho_2[2*i]     = h_rho_up[i];
        rho_2[2*i + 1] = h_rho_dn[i];
        sigma_3[3*i]     = h_sigma_uu[i];
        // sigma_ud = (sigma_tot - sigma_uu - sigma_dd) / 2
        sigma_3[3*i + 1] = 0.5 * (h_sigma_tot[i] - h_sigma_uu[i] - h_sigma_dd[i]);
        sigma_3[3*i + 2] = h_sigma_dd[i];
        tau_2[2*i]     = h_tau_up[i];
        tau_2[2*i + 1] = h_tau_dn[i];
    }

    // Evaluate via libxc (polarized)
    xc_func_type func_x, func_c;
    xc_func_init(&func_x, xc_x_id, XC_POLARIZED);
    xc_func_init(&func_c, xc_c_id, XC_POLARIZED);

    std::vector<double> zk_x(N, 0.0), vrho_x(2*N, 0.0), vsigma_x(3*N, 0.0), vlapl_x(2*N, 0.0), vtau_x(2*N, 0.0);
    std::vector<double> zk_c(N, 0.0), vrho_c(2*N, 0.0), vsigma_c(3*N, 0.0), vlapl_c(2*N, 0.0), vtau_c(2*N, 0.0);

    xc_mgga_exc_vxc(&func_x, N, rho_2.data(), sigma_3.data(), lapl_2.data(), tau_2.data(),
                     zk_x.data(), vrho_x.data(), vsigma_x.data(), vlapl_x.data(), vtau_x.data());
    xc_mgga_exc_vxc(&func_c, N, rho_2.data(), sigma_3.data(), lapl_2.data(), tau_2.data(),
                     zk_c.data(), vrho_c.data(), vsigma_c.data(), vlapl_c.data(), vtau_c.data());

    xc_func_end(&func_x);
    xc_func_end(&func_c);

    // Unpack and upload
    // exc = zk_x + zk_c (per particle of total density)
    // vxc_up/dn = vrho_x[up/dn] + vrho_c[up/dn]
    // v2xc layout matches SCAN spin kernel: [v2xc_c | v2xc_x_up | v2xc_x_dn]
    //   v2xc_c = 2*vsigma_ud (cross term for correlation divergence on total grad)
    //   v2xc_x_up = 2*(vsigma_uu_x + vsigma_uu_c) (same-spin up)
    //   v2xc_x_dn = 2*(vsigma_dd_x + vsigma_dd_c) (same-spin dn)
    // But we need to match exactly how SCAN spin kernel packs v2xc.
    // From CPU XCFunctional.cpp evaluate_spin:
    //   Dxcdgrho[0..Nd-1] = 2*(vsigma_c_ud)  (correlation cross term for total density gradient)
    //   Dxcdgrho[Nd..2Nd-1] = 2*(vsigma_x_uu + vsigma_c_uu)  (exchange+corr same-spin up)
    //   Dxcdgrho[2Nd..3Nd-1] = 2*(vsigma_x_dd + vsigma_c_dd)  (exchange+corr same-spin dn)
    std::vector<double> h_exc(N), h_vxc_up(N), h_vxc_dn(N);
    std::vector<double> h_v2xc_c(N), h_v2xc_x_up(N), h_v2xc_x_dn(N);
    std::vector<double> h_vtau_up(N), h_vtau_dn(N);

    for (int i = 0; i < N; i++) {
        h_exc[i] = zk_x[i] + zk_c[i];
        h_vxc_up[i] = vrho_x[2*i] + vrho_c[2*i];
        h_vxc_dn[i] = vrho_x[2*i+1] + vrho_c[2*i+1];
        // v2xc_c: correlation cross-term (ud) for divergence on total density gradient
        h_v2xc_c[i] = 2.0 * (vsigma_x[3*i+1] + vsigma_c[3*i+1]);
        // v2xc_x_up: same-spin up (uu)
        h_v2xc_x_up[i] = 2.0 * (vsigma_x[3*i] + vsigma_c[3*i]);
        // v2xc_x_dn: same-spin dn (dd)
        h_v2xc_x_dn[i] = 2.0 * (vsigma_x[3*i+2] + vsigma_c[3*i+2]);
        h_vtau_up[i] = vtau_x[2*i] + vtau_c[2*i];
        h_vtau_dn[i] = vtau_x[2*i+1] + vtau_c[2*i+1];
    }

    CUDA_CHECK(cudaMemcpyAsync(d_exc, h_exc.data(), N * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vxc_up, h_vxc_up.data(), N * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vxc_dn, h_vxc_dn.data(), N * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_v2xc_c, h_v2xc_c.data(), N * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_v2xc_x_up, h_v2xc_x_up.data(), N * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_v2xc_x_dn, h_v2xc_x_dn.data(), N * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vtau_up, h_vtau_up.data(), N * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vtau_dn, h_vtau_dn.data(), N * sizeof(double), cudaMemcpyHostToDevice, stream));
}

} // namespace gpu

// ============================================================
// Kernels duplicated from GPUSCF.cu for XC Device-dispatch
// (file-static to avoid ODR violations)
// ============================================================

namespace {

__global__ void xc_sigma_nonorth_kernel(
    const double* __restrict__ Drho_x,
    const double* __restrict__ Drho_y,
    const double* __restrict__ Drho_z,
    double* __restrict__ sigma, int N,
    double L00, double L11, double L22, double L01, double L02, double L12)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double dx = Drho_x[i], dy = Drho_y[i], dz = Drho_z[i];
        sigma[i] = L00*dx*dx + L11*dy*dy + L22*dz*dz
                 + 2.0*L01*dx*dy + 2.0*L02*dx*dz + 2.0*L12*dy*dz;
    }
}

__global__ void xc_sigma_3col_kernel(
    const double* __restrict__ Drho_x,
    const double* __restrict__ Drho_y,
    const double* __restrict__ Drho_z,
    double* __restrict__ sigma, int N, int ncol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_N = N * ncol;
    if (idx < total_N) {
        double dx = Drho_x[idx], dy = Drho_y[idx], dz = Drho_z[idx];
        sigma[idx] = dx*dx + dy*dy + dz*dz;
    }
}

__global__ void xc_sigma_3col_nonorth_kernel(
    const double* __restrict__ Drho_x,
    const double* __restrict__ Drho_y,
    const double* __restrict__ Drho_z,
    double* __restrict__ sigma, int N, int ncol,
    double L00, double L11, double L22, double L01, double L02, double L12)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_N = N * ncol;
    if (idx < total_N) {
        double dx = Drho_x[idx], dy = Drho_y[idx], dz = Drho_z[idx];
        sigma[idx] = L00*dx*dx + L11*dy*dy + L22*dz*dz
                   + 2.0*L01*dx*dy + 2.0*L02*dx*dz + 2.0*L12*dy*dz;
    }
}

__global__ void xc_v2xc_scale_3col_kernel(
    double* __restrict__ f,
    const double* __restrict__ v2xc, int N, int ncol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_N = N * ncol;
    if (idx < total_N) f[idx] *= v2xc[idx];
}

__global__ void xc_nlcc_add_spin_kernel(
    const double* __restrict__ rho_up,
    const double* __restrict__ rho_dn,
    const double* __restrict__ rho_core,
    double* __restrict__ rho_xc, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double core_half = 0.5 * rho_core[i];
        double rup = rho_up[i] + core_half;
        double rdn = rho_dn[i] + core_half;
        double rtot = rup + rdn;
        rho_xc[i]       = (rtot > 1e-14) ? rtot : 1e-14;
        rho_xc[N + i]   = (rup > 1e-14) ? rup : 1e-14;
        rho_xc[2*N + i] = (rdn > 1e-14) ? rdn : 1e-14;
    }
}

__global__ void xc_rho_xc_spin_kernel(
    const double* __restrict__ rho_up,
    const double* __restrict__ rho_dn,
    double* __restrict__ rho_xc, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double rup = rho_up[i], rdn = rho_dn[i];
        double rtot = rup + rdn;
        rho_xc[i]       = (rtot > 1e-14) ? rtot : 1e-14;
        rho_xc[N + i]   = (rup > 1e-14) ? rup : 1e-14;
        rho_xc[2*N + i] = (rdn > 1e-14) ? rdn : 1e-14;
    }
}

__global__ void xc_spin_divergence_add_kernel(
    double* __restrict__ Vxc_up,
    double* __restrict__ Vxc_dn,
    const double* __restrict__ DDrho,
    int Nd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Nd) {
        Vxc_up[i] -= DDrho[i] + DDrho[Nd + i];
        Vxc_dn[i] -= DDrho[i] + DDrho[2*Nd + i];
    }
}

__global__ void xc_lapcT_flux_kernel(
    double* __restrict__ Drho_x,
    double* __restrict__ Drho_y,
    double* __restrict__ Drho_z,
    const double* __restrict__ v2xc, int N,
    double L00, double L01, double L02,
    double L10, double L11, double L12,
    double L20, double L21, double L22)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double dx = Drho_x[i], dy = Drho_y[i], dz = Drho_z[i];
        double v = v2xc[i];
        Drho_x[i] = v * (L00*dx + L01*dy + L02*dz);
        Drho_y[i] = v * (L10*dx + L11*dy + L12*dz);
        Drho_z[i] = v * (L20*dx + L21*dy + L22*dz);
    }
}

__global__ void xc_lapcT_flux_3col_kernel(
    double* __restrict__ Drho_x,
    double* __restrict__ Drho_y,
    double* __restrict__ Drho_z,
    const double* __restrict__ v2xc, int N, int ncol,
    double L00, double L01, double L02,
    double L10, double L11, double L12,
    double L20, double L21, double L22)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_N = N * ncol;
    if (idx < total_N) {
        double dx = Drho_x[idx], dy = Drho_y[idx], dz = Drho_z[idx];
        double v = v2xc[idx];
        Drho_x[idx] = v * (L00*dx + L01*dy + L02*dz);
        Drho_y[idx] = v * (L10*dx + L11*dy + L12*dz);
        Drho_z[idx] = v * (L20*dx + L21*dy + L22*dz);
    }
}

} // anonymous namespace

// ============================================================
// Helper: get libxc functional IDs for mGGA XC types
// ============================================================
static void get_mgga_libxc_ids(XCType xc_type, int& xc_x_id, int& xc_c_id) {
    switch (xc_type) {
        case XCType::MGGA_SCAN:   xc_x_id = XC_MGGA_X_SCAN;   xc_c_id = XC_MGGA_C_SCAN;   break;
        case XCType::MGGA_RSCAN:  xc_x_id = XC_MGGA_X_RSCAN;  xc_c_id = XC_MGGA_C_RSCAN;  break;
        case XCType::MGGA_R2SCAN: xc_x_id = XC_MGGA_X_R2SCAN; xc_c_id = XC_MGGA_C_R2SCAN; break;
        default:                  xc_x_id = XC_MGGA_X_SCAN;    xc_c_id = XC_MGGA_C_SCAN;   break;
    }
}

// ============================================================
// GPUXCState — GPU-side data for Device-dispatching evaluate()
// ============================================================

struct GPUXCState {
    // Grid parameters
    int nx = 0, ny = 0, nz = 0, FDn = 0, Nd = 0;
    bool is_orth = true;
    bool has_mixed_deriv = false;

    // Metric tensor (row-major 3x3)
    double lapcT[9] = {};

    // XC type flags
    XCType xc_type = XCType::GGA_PBE;
    bool is_gga = false;
    bool is_mgga = false;

    // NLCC
    bool has_nlcc = false;
    double* d_rho_core = nullptr;

    // tau pointer (for mGGA, set externally)
    double* d_tau = nullptr;
    double* d_vtau = nullptr;
    bool tau_valid = false;

    // NOTE: GGA/mGGA workspace buffers (grad_rho, sigma, v2xc, x_ex, rho_xc)
    // are NOT owned here — they come from GPUContext::buf or scratch_pool,
    // allocated on use in evaluate()/evaluate_spin().
};

// ── setup_gpu / cleanup_gpu ──────────────────────────

void XCFunctional::setup_gpu(const LynxContext& ctx, int Nspin) {
    if (!gpu_state_raw_)
        gpu_state_raw_ = new GPUXCState();
    auto* gs = static_cast<GPUXCState*>(gpu_state_raw_);

    const auto& grid    = ctx.grid();
    const auto& domain  = ctx.domain();
    const auto& stencil = ctx.stencil();

    gs->nx  = grid.Nx();
    gs->ny  = grid.Ny();
    gs->nz  = grid.Nz();
    gs->FDn = stencil.FDn();
    gs->Nd  = domain.Nd_d();
    gs->is_orth         = grid.lattice().is_orthogonal();
    gs->has_mixed_deriv = !gs->is_orth;

    gs->xc_type = type_;
    gs->is_gga  = is_gga() || is_mgga();
    gs->is_mgga = is_mgga();

    // Metric tensor
    {
        const auto& L = grid.lattice().lapc_T();
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                gs->lapcT[i * 3 + j] = L(i, j);
    }

    // GGA/mGGA workspace is NOT allocated here — it comes from
    // GPUContext::buf (non-spin) or scratch_pool (spin) at call time.
}

void XCFunctional::cleanup_gpu() {
    if (!gpu_state_raw_) return;
    auto* gs = static_cast<GPUXCState*>(gpu_state_raw_);

    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    auto safe_free = [stream](auto*& p) { if (p) { cudaFreeAsync(p, stream); p = nullptr; } };

    safe_free(gs->d_rho_core);

    delete gs;
    gpu_state_raw_ = nullptr;
}

XCFunctional::~XCFunctional() {
    cleanup_gpu();
}

void XCFunctional::set_gpu_nlcc(const double* rho_core, int Nd) {
    auto* gs = static_cast<GPUXCState*>(gpu_state_raw_);
    if (!gs) return;
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    auto safe_free = [stream](auto*& p) { if (p) { cudaFreeAsync(p, stream); p = nullptr; } };
    safe_free(gs->d_rho_core);
    if (rho_core) {
        gs->has_nlcc = true;
        CUDA_CHECK(cudaMallocAsync(&gs->d_rho_core, Nd * sizeof(double), stream));
        CUDA_CHECK(cudaMemcpyAsync(gs->d_rho_core, rho_core, Nd * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    } else {
        gs->has_nlcc = false;
    }
}

void XCFunctional::set_gpu_tau_valid(bool valid) {
    auto* gs = static_cast<GPUXCState*>(gpu_state_raw_);
    if (gs) gs->tau_valid = valid;
}

// ============================================================
// Device-dispatching evaluate() — non-spin
// ============================================================

void XCFunctional::evaluate(const double* rho, double* Vxc, double* exc, int Nd_d,
                             Device dev,
                             double* Dxcdgrho,
                             const double* tau, double* vtau) const {
    if (dev == Device::CPU) {
        evaluate(rho, Vxc, exc, Nd_d, Dxcdgrho, tau, vtau);
        return;
    }

    // GPU path — mirrors GPUSCF::gpu_xc_evaluate()
    auto* gs = static_cast<GPUXCState*>(gpu_state_raw_);
    auto& ctx = gpu::GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;
    int bs = 256;
    int grid_sz = gpu::ceildiv(Nd_d, bs);
    int nx_ex = gs->nx + 2 * gs->FDn, ny_ex = gs->ny + 2 * gs->FDn;

    bool use_gga = gs->is_gga || gs->is_mgga;
    if (use_gga) {
        // GGA / mGGA path
        double* d_Drho_x = ctx.buf.grad_rho;
        double* d_Drho_y = ctx.buf.grad_rho + Nd_d;
        double* d_Drho_z = ctx.buf.grad_rho + 2 * Nd_d;
        double* d_sigma  = ctx.buf.aar_r;      // reuse
        double* d_v2xc   = ctx.buf.Dxcdgrho;
        double* d_x_ex   = ctx.buf.aar_x_ex;

        // NLCC: rho + rho_core
        double* d_rho_xc;
        if (gs->has_nlcc && gs->d_rho_core) {
            d_rho_xc = ctx.buf.b;
            gpu::nlcc_add_kernel<<<grid_sz, bs, 0, stream>>>(
                rho, gs->d_rho_core, d_rho_xc, Nd_d);
        } else {
            d_rho_xc = const_cast<double*>(rho);
        }

        // Gradient
        gpu::halo_exchange_gpu(d_rho_xc, d_x_ex, gs->nx, gs->ny, gs->nz,
                               gs->FDn, 1, true, true, true, stream);
        gpu::gradient_gpu(d_x_ex, d_Drho_x, gs->nx, gs->ny, gs->nz,
                          gs->FDn, nx_ex, ny_ex, 0, 1, stream);
        gpu::gradient_gpu(d_x_ex, d_Drho_y, gs->nx, gs->ny, gs->nz,
                          gs->FDn, nx_ex, ny_ex, 1, 1, stream);
        gpu::gradient_gpu(d_x_ex, d_Drho_z, gs->nx, gs->ny, gs->nz,
                          gs->FDn, nx_ex, ny_ex, 2, 1, stream);

        // sigma = |nabla rho|^2
        if (gs->is_orth) {
            gpu::sigma_kernel<<<grid_sz, bs, 0, stream>>>(
                d_Drho_x, d_Drho_y, d_Drho_z, d_sigma, Nd_d);
        } else {
            xc_sigma_nonorth_kernel<<<grid_sz, bs, 0, stream>>>(
                d_Drho_x, d_Drho_y, d_Drho_z, d_sigma, Nd_d,
                gs->lapcT[0], gs->lapcT[4], gs->lapcT[8],
                gs->lapcT[1], gs->lapcT[2], gs->lapcT[5]);
        }

        // XC kernel dispatch
        if (gs->is_mgga && gs->tau_valid && tau) {
            if (gs->xc_type == XCType::MGGA_SCAN) {
                gpu::mgga_scan_gpu(d_rho_xc, d_sigma, tau, exc, Vxc, d_v2xc,
                                   const_cast<double*>(vtau), Nd_d, stream);
            } else {
                int xc_x_id, xc_c_id;
                get_mgga_libxc_ids(gs->xc_type, xc_x_id, xc_c_id);
                gpu::mgga_libxc_gpu(xc_x_id, xc_c_id, d_rho_xc, d_sigma, tau,
                                    exc, Vxc, d_v2xc, const_cast<double*>(vtau), Nd_d);
            }
        } else {
            gpu::gga_pbe_gpu(d_rho_xc, d_sigma, exc, Vxc, d_v2xc, Nd_d, stream);
        }

        // Divergence correction
        if (gs->is_orth) {
            gpu::v2xc_scale_kernel<<<grid_sz, bs, 0, stream>>>(d_Drho_x, d_v2xc, Nd_d);
            gpu::v2xc_scale_kernel<<<grid_sz, bs, 0, stream>>>(d_Drho_y, d_v2xc, Nd_d);
            gpu::v2xc_scale_kernel<<<grid_sz, bs, 0, stream>>>(d_Drho_z, d_v2xc, Nd_d);
        } else {
            xc_lapcT_flux_kernel<<<grid_sz, bs, 0, stream>>>(
                d_Drho_x, d_Drho_y, d_Drho_z, d_v2xc, Nd_d,
                gs->lapcT[0], gs->lapcT[1], gs->lapcT[2],
                gs->lapcT[3], gs->lapcT[4], gs->lapcT[5],
                gs->lapcT[6], gs->lapcT[7], gs->lapcT[8]);
        }

        double* d_DDrho = d_sigma;  // reuse
        // x-direction
        gpu::halo_exchange_gpu(d_Drho_x, d_x_ex, gs->nx, gs->ny, gs->nz,
                               gs->FDn, 1, true, true, true, stream);
        gpu::gradient_gpu(d_x_ex, d_DDrho, gs->nx, gs->ny, gs->nz,
                          gs->FDn, nx_ex, ny_ex, 0, 1, stream);
        gpu::divergence_sub_kernel<<<grid_sz, bs, 0, stream>>>(Vxc, d_DDrho, Nd_d);
        // y-direction
        gpu::halo_exchange_gpu(d_Drho_y, d_x_ex, gs->nx, gs->ny, gs->nz,
                               gs->FDn, 1, true, true, true, stream);
        gpu::gradient_gpu(d_x_ex, d_DDrho, gs->nx, gs->ny, gs->nz,
                          gs->FDn, nx_ex, ny_ex, 1, 1, stream);
        gpu::divergence_sub_kernel<<<grid_sz, bs, 0, stream>>>(Vxc, d_DDrho, Nd_d);
        // z-direction
        gpu::halo_exchange_gpu(d_Drho_z, d_x_ex, gs->nx, gs->ny, gs->nz,
                               gs->FDn, 1, true, true, true, stream);
        gpu::gradient_gpu(d_x_ex, d_DDrho, gs->nx, gs->ny, gs->nz,
                          gs->FDn, nx_ex, ny_ex, 2, 1, stream);
        gpu::divergence_sub_kernel<<<grid_sz, bs, 0, stream>>>(Vxc, d_DDrho, Nd_d);

    } else {
        // LDA path
        double* d_rho_xc;
        if (gs->has_nlcc && gs->d_rho_core) {
            d_rho_xc = ctx.buf.b;
            gpu::nlcc_add_kernel<<<grid_sz, bs, 0, stream>>>(
                rho, gs->d_rho_core, d_rho_xc, Nd_d);
        } else {
            d_rho_xc = const_cast<double*>(rho);
        }
        gpu::lda_pw_gpu(d_rho_xc, exc, Vxc, Nd_d, stream);
    }
}

// ============================================================
// Device-dispatching evaluate_spin() — spin-polarized
// ============================================================

void XCFunctional::evaluate_spin(const double* rho, double* Vxc, double* exc, int Nd_d,
                                  Device dev,
                                  double* Dxcdgrho,
                                  const double* tau, double* vtau) const {
    if (dev == Device::CPU) {
        evaluate_spin(rho, Vxc, exc, Nd_d, Dxcdgrho, tau, vtau);
        return;
    }

    // GPU path — mirrors GPUSCF::gpu_xc_evaluate_spin()
    auto* gs = static_cast<GPUXCState*>(gpu_state_raw_);
    auto& ctx = gpu::GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;
    int bs = 256;
    int grid_sz = gpu::ceildiv(Nd_d, bs);
    int nx_ex = gs->nx + 2 * gs->FDn, ny_ex = gs->ny + 2 * gs->FDn;

    // rho layout: [up(Nd)|dn(Nd)], Vxc layout: [up(Nd)|dn(Nd)]
    const double* d_rho_up = rho;
    const double* d_rho_dn = rho + Nd_d;

    bool use_gga = gs->is_gga || gs->is_mgga;
    if (use_gga) {
        // Spin GGA / mGGA path
        auto& sp = ctx.scratch_pool;
        size_t sp_cp = sp.checkpoint();

        double* d_rho_xc  = sp.alloc<double>(3 * Nd_d);
        double* d_sigma    = sp.alloc<double>(3 * Nd_d);
        double* d_Drho_x   = sp.alloc<double>(3 * Nd_d);
        double* d_Drho_y   = sp.alloc<double>(3 * Nd_d);
        double* d_Drho_z   = sp.alloc<double>(3 * Nd_d);
        double* d_v2xc     = ctx.buf.Dxcdgrho;  // [3*Nd]
        double* d_x_ex_tmp = ctx.buf.aar_x_ex;

        // Build rho_xc = [total|up|dn]
        if (gs->has_nlcc && gs->d_rho_core) {
            xc_nlcc_add_spin_kernel<<<grid_sz, bs, 0, stream>>>(
                d_rho_up, d_rho_dn, gs->d_rho_core, d_rho_xc, Nd_d);
        } else {
            xc_rho_xc_spin_kernel<<<grid_sz, bs, 0, stream>>>(
                d_rho_up, d_rho_dn, d_rho_xc, Nd_d);
        }

        // Gradient of all 3 columns
        for (int col = 0; col < 3; col++) {
            gpu::halo_exchange_gpu(d_rho_xc + col * Nd_d, d_x_ex_tmp,
                                   gs->nx, gs->ny, gs->nz, gs->FDn,
                                   1, true, true, true, stream);
            gpu::gradient_gpu(d_x_ex_tmp, d_Drho_x + col * Nd_d,
                              gs->nx, gs->ny, gs->nz, gs->FDn,
                              nx_ex, ny_ex, 0, 1, stream);
            gpu::gradient_gpu(d_x_ex_tmp, d_Drho_y + col * Nd_d,
                              gs->nx, gs->ny, gs->nz, gs->FDn,
                              nx_ex, ny_ex, 1, 1, stream);
            gpu::gradient_gpu(d_x_ex_tmp, d_Drho_z + col * Nd_d,
                              gs->nx, gs->ny, gs->nz, gs->FDn,
                              nx_ex, ny_ex, 2, 1, stream);
        }

        // sigma for 3 columns
        int grid3 = gpu::ceildiv(3 * Nd_d, bs);
        if (gs->is_orth) {
            xc_sigma_3col_kernel<<<grid3, bs, 0, stream>>>(
                d_Drho_x, d_Drho_y, d_Drho_z, d_sigma, Nd_d, 3);
        } else {
            xc_sigma_3col_nonorth_kernel<<<grid3, bs, 0, stream>>>(
                d_Drho_x, d_Drho_y, d_Drho_z, d_sigma, Nd_d, 3,
                gs->lapcT[0], gs->lapcT[4], gs->lapcT[8],
                gs->lapcT[1], gs->lapcT[2], gs->lapcT[5]);
        }

        // XC kernel dispatch
        if (gs->is_mgga && gs->tau_valid && tau) {
            if (gs->xc_type == XCType::MGGA_SCAN) {
                gpu::mgga_scan_spin_gpu(
                    d_rho_xc + Nd_d, d_rho_xc + 2 * Nd_d,
                    d_sigma + Nd_d, d_sigma + 2 * Nd_d, d_sigma,
                    tau, tau + Nd_d,
                    exc, Vxc, Vxc + Nd_d,
                    d_v2xc, d_v2xc + Nd_d, d_v2xc + 2 * Nd_d,
                    const_cast<double*>(vtau), const_cast<double*>(vtau) + Nd_d,
                    Nd_d, stream);
            } else {
                int xc_x_id, xc_c_id;
                get_mgga_libxc_ids(gs->xc_type, xc_x_id, xc_c_id);
                gpu::mgga_libxc_spin_gpu(xc_x_id, xc_c_id,
                    d_rho_xc + Nd_d, d_rho_xc + 2 * Nd_d,
                    d_sigma + Nd_d, d_sigma + 2 * Nd_d, d_sigma,
                    tau, tau + Nd_d,
                    exc, Vxc, Vxc + Nd_d,
                    d_v2xc, d_v2xc + Nd_d, d_v2xc + 2 * Nd_d,
                    const_cast<double*>(vtau), const_cast<double*>(vtau) + Nd_d,
                    Nd_d);
            }
        } else {
            gpu::gga_pbe_spin_gpu(d_rho_xc, d_sigma, exc, Vxc, d_v2xc, Nd_d, stream);
        }

        // Divergence correction for 3 columns
        if (gs->is_orth) {
            xc_v2xc_scale_3col_kernel<<<grid3, bs, 0, stream>>>(d_Drho_x, d_v2xc, Nd_d, 3);
            xc_v2xc_scale_3col_kernel<<<grid3, bs, 0, stream>>>(d_Drho_y, d_v2xc, Nd_d, 3);
            xc_v2xc_scale_3col_kernel<<<grid3, bs, 0, stream>>>(d_Drho_z, d_v2xc, Nd_d, 3);
        } else {
            xc_lapcT_flux_3col_kernel<<<grid3, bs, 0, stream>>>(
                d_Drho_x, d_Drho_y, d_Drho_z, d_v2xc, Nd_d, 3,
                gs->lapcT[0], gs->lapcT[1], gs->lapcT[2],
                gs->lapcT[3], gs->lapcT[4], gs->lapcT[5],
                gs->lapcT[6], gs->lapcT[7], gs->lapcT[8]);
        }

        // Accumulate divergence
        double* d_DDrho = d_sigma;  // reuse
        CUDA_CHECK(cudaMemsetAsync(d_DDrho, 0, 3 * Nd_d * sizeof(double), stream));

        for (int dir = 0; dir < 3; dir++) {
            double* d_Drho_dir = (dir == 0) ? d_Drho_x : (dir == 1) ? d_Drho_y : d_Drho_z;
            for (int col = 0; col < 3; col++) {
                double* d_DDcol = sp.alloc<double>(Nd_d);
                gpu::halo_exchange_gpu(d_Drho_dir + col * Nd_d, d_x_ex_tmp,
                                       gs->nx, gs->ny, gs->nz, gs->FDn,
                                       1, true, true, true, stream);
                gpu::gradient_gpu(d_x_ex_tmp, d_DDcol,
                                  gs->nx, gs->ny, gs->nz, gs->FDn,
                                  nx_ex, ny_ex, dir, 1, stream);
                double one = 1.0;
                cublasDaxpy(ctx.cublas, Nd_d, &one, d_DDcol, 1,
                            d_DDrho + col * Nd_d, 1);
                sp.restore(sp.checkpoint() - Nd_d * sizeof(double));
            }
        }

        // Apply spin divergence
        xc_spin_divergence_add_kernel<<<grid_sz, bs, 0, stream>>>(
            Vxc, Vxc + Nd_d, d_DDrho, Nd_d);

        sp.restore(sp_cp);
    } else {
        // LDA spin path
        if (gs->has_nlcc && gs->d_rho_core) {
            auto& sp = ctx.scratch_pool;
            size_t sp_cp = sp.checkpoint();
            double* d_rho_xc_3 = sp.alloc<double>(3 * Nd_d);
            xc_nlcc_add_spin_kernel<<<grid_sz, bs, 0, stream>>>(
                d_rho_up, d_rho_dn, gs->d_rho_core, d_rho_xc_3, Nd_d);
            gpu::lda_pw_spin_gpu(d_rho_xc_3 + Nd_d, d_rho_xc_3 + 2 * Nd_d,
                                 exc, Vxc, Vxc + Nd_d, Nd_d, stream);
            sp.restore(sp_cp);
        } else {
            gpu::lda_pw_spin_gpu(d_rho_up, d_rho_dn, exc, Vxc, Vxc + Nd_d,
                                 Nd_d, stream);
        }
    }
}

} // namespace lynx

#endif // USE_CUDA
