// High-occupancy SCAN XC kernel benchmark
// Problem: mgga_scan_kernel uses 48 regs → only 5 blocks/SM (83% max occupancy)
// Fix strategies:
//   V2: __launch_bounds__(256, 6) to force ≤40 regs
//   V3: Split into exchange + correlation passes (halves live variables per pass)
//   V4: Smaller block size (128) to allow more blocks with same register budget
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

__host__ __device__ int ceildiv(int a, int b) { return (a + b - 1) / b; }

// ============================================================
// BASELINE: Original SCAN kernel (no launch bounds, 48 regs)
// ============================================================
__global__ void mgga_scan_kernel_orig(
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
        exc[i] = 0.0; vxc[i] = 0.0; v2xc[i] = 0.0; vtau[i] = 0.0;
        return;
    }

    double sig = sigma[i];
    if (sig < 1e-14) sig = 1e-14;
    double normDrho = sqrt(sig);
    double tau_val = tau[i];

    constexpr double PI_val = 3.14159265358979323846;
    double threeMPi2_1o3 = cbrt(3.0 * PI_val * PI_val);
    double threeMPi2_2o3 = threeMPi2_1o3 * threeMPi2_1o3;

    double rho_4o3 = r * cbrt(r);
    double s = normDrho / (2.0 * threeMPi2_1o3 * rho_4o3);
    double tauw = normDrho * normDrho / (8.0 * r);
    double tauUnif = 3.0 / 10.0 * threeMPi2_2o3 * r * cbrt(r * r);
    double alpha = (tau_val - tauw) / tauUnif;

    // Exchange
    constexpr double k1 = 0.065;
    constexpr double mu_ak = 10.0 / 81.0;
    double b2_x = sqrt(5913.0 / 405000.0);
    double b1_x = 511.0 / 13500.0 / (2.0 * b2_x);
    constexpr double b3_x = 0.5;
    double b4_x = mu_ak * mu_ak / k1 - 1606.0 / 18225.0 - b1_x * b1_x;
    constexpr double hx0 = 1.174, c1x = 0.667, c2x = 0.8, dx_x = 1.24, a1_x = 4.9479;

    double s2 = s * s;
    double epsilon_xUnif = -3.0 / (4.0 * PI_val) * cbrt(3.0 * PI_val * PI_val * r);

    double term1 = 1.0 + b4_x * s2 / mu_ak * exp(-fabs(b4_x) * s2 / mu_ak);
    double xFir = mu_ak * s2 * term1;
    double term3 = 2.0 * (b1_x * s2 + b2_x * (1.0 - alpha) * exp(-b3_x * (1.0 - alpha) * (1.0 - alpha)));
    double xSec = (term3 / 2.0) * (term3 / 2.0);
    double hx1 = 1.0 + k1 - k1 / (1.0 + (xFir + xSec) / k1);

    double fx;
    if (fabs(alpha - 1.0) < 1e-14) fx = 0.0;
    else if (alpha > 1.0) fx = -dx_x * exp(c2x / (1.0 - alpha));
    else fx = exp(-c1x * alpha / (1.0 - alpha));

    double sqrt_s = sqrt(s);
    double gx = 1.0 - exp(-a1_x / (sqrt_s + 1e-30));
    double Fx = (hx1 + fx * (hx0 - hx1)) * gx;
    double epsilonx = epsilon_xUnif * Fx;

    // Correlation (PW92 + SCAN interpolation)
    constexpr double b1c = 0.0285764, b2c = 0.0889, b3c = 0.125541;
    constexpr double betaConst = 0.06672455060314922;
    constexpr double betaRsInf = betaConst * 0.1 / 0.1778;
    constexpr double f0 = -0.9;
    constexpr double AA = 0.0310907, alpha1 = 0.21370;
    constexpr double beta1_c = 7.5957, beta2_c = 3.5876, beta3_c = 1.6382, beta4_c = 0.49294;
    constexpr double r_c = 0.031091, c1c = 0.64, c2c = 1.5, dc = 0.7;

    double rs = cbrt(0.75 / (PI_val * r));
    double sqrRs = sqrt(rs);
    double ecLDA0 = -b1c / (1.0 + b2c * sqrRs + b3c * rs);
    double cx0 = -3.0 / (4.0 * PI_val) * cbrt(9.0 * PI_val / 4.0);
    double w0 = exp(-ecLDA0 / b1c) - 1.0;
    double xiInf0 = cbrt(3.0 * PI_val * PI_val / 16.0) * cbrt(3.0 * PI_val * PI_val / 16.0) * (betaRsInf / (cx0 - f0));
    double gInf0s = pow(1.0 + 4.0 * xiInf0 * s2, -0.25);
    double H0 = b1c * log(1.0 + w0 * (1.0 - gInf0s));
    double ec0 = ecLDA0 + H0;

    double beta_c_val = betaConst * (1.0 + 0.1 * rs) / (1.0 + 0.1778 * rs);
    double ec_q0 = -2.0 * AA * (1.0 + alpha1 * rs);
    double ec_q1 = 2.0 * AA * (beta1_c * sqrRs + beta2_c * rs + beta3_c * rs * sqrRs + beta4_c * rs * rs);
    double ec_den = 1.0 / (ec_q1 * ec_q1 + ec_q1);
    double ec_lsda1 = ec_q0 * (-log(ec_q1 * ec_q1 * ec_den));

    double w1 = exp(-ec_lsda1 / r_c) - 1.0;
    double Ac = beta_c_val / (r_c * w1);
    double t = cbrt(3.0 * PI_val * PI_val / 16.0) * s / sqrRs;
    double g = pow(1.0 + 4.0 * Ac * t * t, -0.25);
    double H1 = r_c * log(1.0 + w1 * (1.0 - g));
    double ec1 = ec_lsda1 + H1;

    double fc;
    if (fabs(alpha - 1.0) < 1e-14) fc = 0.0;
    else if (alpha > 1.0) fc = -dc * exp(c2c / (1.0 - alpha));
    else fc = exp(-c1c * alpha / (1.0 - alpha));

    double epsilonc = ec1 + fc * (ec0 - ec1);
    exc[i] = epsilonx + epsilonc;
    vxc[i] = epsilonx * 1.33 + epsilonc * 1.1;
    v2xc[i] = epsilonx * 0.5;
    vtau[i] = epsilonc * 0.2;
}

// ============================================================
// V2: Same kernel with __launch_bounds__(256, 6)
// Forces compiler to use ≤40 registers (may spill to local memory)
// ============================================================
__global__ __launch_bounds__(256, 6)
void mgga_scan_kernel_v2(
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
        exc[i] = 0.0; vxc[i] = 0.0; v2xc[i] = 0.0; vtau[i] = 0.0;
        return;
    }

    double sig = sigma[i];
    if (sig < 1e-14) sig = 1e-14;
    double normDrho = sqrt(sig);
    double tau_val = tau[i];

    constexpr double PI_val = 3.14159265358979323846;
    double threeMPi2_1o3 = cbrt(3.0 * PI_val * PI_val);
    double threeMPi2_2o3 = threeMPi2_1o3 * threeMPi2_1o3;

    double rho_4o3 = r * cbrt(r);
    double s = normDrho / (2.0 * threeMPi2_1o3 * rho_4o3);
    double tauw = normDrho * normDrho / (8.0 * r);
    double tauUnif = 3.0 / 10.0 * threeMPi2_2o3 * r * cbrt(r * r);
    double alpha = (tau_val - tauw) / tauUnif;

    // Exchange
    constexpr double k1 = 0.065;
    constexpr double mu_ak = 10.0 / 81.0;
    double b2_x = sqrt(5913.0 / 405000.0);
    double b1_x = 511.0 / 13500.0 / (2.0 * b2_x);
    constexpr double b3_x = 0.5;
    double b4_x = mu_ak * mu_ak / k1 - 1606.0 / 18225.0 - b1_x * b1_x;
    constexpr double hx0 = 1.174, c1x = 0.667, c2x = 0.8, dx_x = 1.24, a1_x = 4.9479;

    double s2 = s * s;
    double epsilon_xUnif = -3.0 / (4.0 * PI_val) * cbrt(3.0 * PI_val * PI_val * r);

    double term1 = 1.0 + b4_x * s2 / mu_ak * exp(-fabs(b4_x) * s2 / mu_ak);
    double xFir = mu_ak * s2 * term1;
    double term3 = 2.0 * (b1_x * s2 + b2_x * (1.0 - alpha) * exp(-b3_x * (1.0 - alpha) * (1.0 - alpha)));
    double xSec = (term3 / 2.0) * (term3 / 2.0);
    double hx1 = 1.0 + k1 - k1 / (1.0 + (xFir + xSec) / k1);

    double fx;
    if (fabs(alpha - 1.0) < 1e-14) fx = 0.0;
    else if (alpha > 1.0) fx = -dx_x * exp(c2x / (1.0 - alpha));
    else fx = exp(-c1x * alpha / (1.0 - alpha));

    double sqrt_s = sqrt(s);
    double gx = 1.0 - exp(-a1_x / (sqrt_s + 1e-30));
    double Fx = (hx1 + fx * (hx0 - hx1)) * gx;
    double epsilonx = epsilon_xUnif * Fx;

    // Correlation
    constexpr double b1c = 0.0285764, b2c = 0.0889, b3c = 0.125541;
    constexpr double betaConst = 0.06672455060314922;
    constexpr double betaRsInf = betaConst * 0.1 / 0.1778;
    constexpr double f0 = -0.9;
    constexpr double AA = 0.0310907, alpha1 = 0.21370;
    constexpr double beta1_c = 7.5957, beta2_c = 3.5876, beta3_c = 1.6382, beta4_c = 0.49294;
    constexpr double r_c = 0.031091, c1c = 0.64, c2c = 1.5, dc = 0.7;

    double rs = cbrt(0.75 / (PI_val * r));
    double sqrRs = sqrt(rs);
    double ecLDA0 = -b1c / (1.0 + b2c * sqrRs + b3c * rs);
    double cx0 = -3.0 / (4.0 * PI_val) * cbrt(9.0 * PI_val / 4.0);
    double w0 = exp(-ecLDA0 / b1c) - 1.0;
    double xiInf0 = cbrt(3.0 * PI_val * PI_val / 16.0) * cbrt(3.0 * PI_val * PI_val / 16.0) * (betaRsInf / (cx0 - f0));
    double gInf0s = pow(1.0 + 4.0 * xiInf0 * s2, -0.25);
    double H0 = b1c * log(1.0 + w0 * (1.0 - gInf0s));
    double ec0 = ecLDA0 + H0;

    double beta_c_val = betaConst * (1.0 + 0.1 * rs) / (1.0 + 0.1778 * rs);
    double ec_q0 = -2.0 * AA * (1.0 + alpha1 * rs);
    double ec_q1 = 2.0 * AA * (beta1_c * sqrRs + beta2_c * rs + beta3_c * rs * sqrRs + beta4_c * rs * rs);
    double ec_den = 1.0 / (ec_q1 * ec_q1 + ec_q1);
    double ec_lsda1 = ec_q0 * (-log(ec_q1 * ec_q1 * ec_den));

    double w1 = exp(-ec_lsda1 / r_c) - 1.0;
    double Ac = beta_c_val / (r_c * w1);
    double t = cbrt(3.0 * PI_val * PI_val / 16.0) * s / sqrRs;
    double g = pow(1.0 + 4.0 * Ac * t * t, -0.25);
    double H1 = r_c * log(1.0 + w1 * (1.0 - g));
    double ec1 = ec_lsda1 + H1;

    double fc;
    if (fabs(alpha - 1.0) < 1e-14) fc = 0.0;
    else if (alpha > 1.0) fc = -dc * exp(c2c / (1.0 - alpha));
    else fc = exp(-c1c * alpha / (1.0 - alpha));

    double epsilonc = ec1 + fc * (ec0 - ec1);
    exc[i] = epsilonx + epsilonc;
    vxc[i] = epsilonx * 1.33 + epsilonc * 1.1;
    v2xc[i] = epsilonx * 0.5;
    vtau[i] = epsilonc * 0.2;
}

// ============================================================
// V3: Two-pass split (exchange, then correlation)
// Each pass has far fewer live variables → lower register pressure
// Cost: reads rho/sigma/tau twice, writes intermediate to global memory
// ============================================================

// Pass 1: Exchange only
__global__ __launch_bounds__(256, 6)
void mgga_scan_exchange_v3(
    const double* __restrict__ rho,
    const double* __restrict__ sigma,
    const double* __restrict__ tau,
    double* __restrict__ exc,     // output: exchange energy
    double* __restrict__ s2_out,  // output: s² for correlation pass
    double* __restrict__ alpha_out, // output: alpha for correlation pass
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double r = rho[i];
    if (r < 1e-30) {
        exc[i] = 0.0; s2_out[i] = 0.0; alpha_out[i] = 0.0;
        return;
    }

    double sig = sigma[i];
    if (sig < 1e-14) sig = 1e-14;
    double normDrho = sqrt(sig);
    double tau_val = tau[i];

    constexpr double PI_val = 3.14159265358979323846;
    double threeMPi2_1o3 = cbrt(3.0 * PI_val * PI_val);
    double threeMPi2_2o3 = threeMPi2_1o3 * threeMPi2_1o3;

    double rho_4o3 = r * cbrt(r);
    double s = normDrho / (2.0 * threeMPi2_1o3 * rho_4o3);
    double tauw = normDrho * normDrho / (8.0 * r);
    double tauUnif = 3.0 / 10.0 * threeMPi2_2o3 * r * cbrt(r * r);
    double alpha = (tau_val - tauw) / tauUnif;

    double s2 = s * s;
    s2_out[i] = s2;
    alpha_out[i] = alpha;

    // Exchange
    constexpr double k1 = 0.065;
    constexpr double mu_ak = 10.0 / 81.0;
    double b2_x = sqrt(5913.0 / 405000.0);
    double b1_x = 511.0 / 13500.0 / (2.0 * b2_x);
    constexpr double b3_x = 0.5;
    double b4_x = mu_ak * mu_ak / k1 - 1606.0 / 18225.0 - b1_x * b1_x;
    constexpr double hx0 = 1.174, c1x = 0.667, c2x = 0.8, dx_x = 1.24, a1_x = 4.9479;

    double epsilon_xUnif = -3.0 / (4.0 * PI_val) * cbrt(3.0 * PI_val * PI_val * r);

    double term1 = 1.0 + b4_x * s2 / mu_ak * exp(-fabs(b4_x) * s2 / mu_ak);
    double xFir = mu_ak * s2 * term1;
    double term3 = 2.0 * (b1_x * s2 + b2_x * (1.0 - alpha) * exp(-b3_x * (1.0 - alpha) * (1.0 - alpha)));
    double xSec = (term3 / 2.0) * (term3 / 2.0);
    double hx1 = 1.0 + k1 - k1 / (1.0 + (xFir + xSec) / k1);

    double fx;
    if (fabs(alpha - 1.0) < 1e-14) fx = 0.0;
    else if (alpha > 1.0) fx = -dx_x * exp(c2x / (1.0 - alpha));
    else fx = exp(-c1x * alpha / (1.0 - alpha));

    double sqrt_s = sqrt(s);
    double gx = 1.0 - exp(-a1_x / (sqrt_s + 1e-30));
    double Fx = (hx1 + fx * (hx0 - hx1)) * gx;
    exc[i] = epsilon_xUnif * Fx;
}

// Pass 2: Correlation only (reads precomputed s², alpha)
__global__ __launch_bounds__(256, 6)
void mgga_scan_correlation_v3(
    const double* __restrict__ rho,
    const double* __restrict__ s2_in,
    const double* __restrict__ alpha_in,
    double* __restrict__ exc,     // in/out: add correlation to exchange
    double* __restrict__ vxc,
    double* __restrict__ v2xc,
    double* __restrict__ vtau,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double r = rho[i];
    if (r < 1e-30) {
        vxc[i] = 0.0; v2xc[i] = 0.0; vtau[i] = 0.0;
        return;
    }

    double s2 = s2_in[i];
    double alpha = alpha_in[i];

    constexpr double PI_val = 3.14159265358979323846;
    constexpr double b1c = 0.0285764, b2c = 0.0889, b3c = 0.125541;
    constexpr double betaConst = 0.06672455060314922;
    constexpr double betaRsInf = betaConst * 0.1 / 0.1778;
    constexpr double f0 = -0.9;
    constexpr double AA = 0.0310907, alpha1 = 0.21370;
    constexpr double beta1_c = 7.5957, beta2_c = 3.5876, beta3_c = 1.6382, beta4_c = 0.49294;
    constexpr double r_c = 0.031091, c1c = 0.64, c2c = 1.5, dc = 0.7;

    double rs = cbrt(0.75 / (PI_val * r));
    double sqrRs = sqrt(rs);
    double ecLDA0 = -b1c / (1.0 + b2c * sqrRs + b3c * rs);
    double cx0 = -3.0 / (4.0 * PI_val) * cbrt(9.0 * PI_val / 4.0);
    double w0 = exp(-ecLDA0 / b1c) - 1.0;
    double xiInf0 = cbrt(3.0 * PI_val * PI_val / 16.0) * cbrt(3.0 * PI_val * PI_val / 16.0) * (betaRsInf / (cx0 - f0));
    double gInf0s = pow(1.0 + 4.0 * xiInf0 * s2, -0.25);
    double H0 = b1c * log(1.0 + w0 * (1.0 - gInf0s));
    double ec0 = ecLDA0 + H0;

    double beta_c_val = betaConst * (1.0 + 0.1 * rs) / (1.0 + 0.1778 * rs);
    double ec_q0 = -2.0 * AA * (1.0 + alpha1 * rs);
    double ec_q1 = 2.0 * AA * (beta1_c * sqrRs + beta2_c * rs + beta3_c * rs * sqrRs + beta4_c * rs * rs);
    double ec_den = 1.0 / (ec_q1 * ec_q1 + ec_q1);
    double ec_lsda1 = ec_q0 * (-log(ec_q1 * ec_q1 * ec_den));

    double s = sqrt(s2);
    double w1 = exp(-ec_lsda1 / r_c) - 1.0;
    double Ac = beta_c_val / (r_c * w1);
    double t = cbrt(3.0 * PI_val * PI_val / 16.0) * s / sqrRs;
    double g = pow(1.0 + 4.0 * Ac * t * t, -0.25);
    double H1 = r_c * log(1.0 + w1 * (1.0 - g));
    double ec1 = ec_lsda1 + H1;

    double fc;
    if (fabs(alpha - 1.0) < 1e-14) fc = 0.0;
    else if (alpha > 1.0) fc = -dc * exp(c2c / (1.0 - alpha));
    else fc = exp(-c1c * alpha / (1.0 - alpha));

    double epsilonc = ec1 + fc * (ec0 - ec1);
    double epsilonx = exc[i];  // exchange from pass 1
    exc[i] = epsilonx + epsilonc;
    vxc[i] = epsilonx * 1.33 + epsilonc * 1.1;
    v2xc[i] = epsilonx * 0.5;
    vtau[i] = epsilonc * 0.2;
}

// ============================================================
// V4: __launch_bounds__(128, 10) — smaller block, more blocks
// 128 threads = 4 warps; at 40 regs: 65536/(40*128) = 12 blocks
// but capped by warps: 48/4 = 12 blocks → 48 warps → 100%
// More blocks means better SM utilization for small grids
// ============================================================
__global__ __launch_bounds__(128, 10)
void mgga_scan_kernel_v4(
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
        exc[i] = 0.0; vxc[i] = 0.0; v2xc[i] = 0.0; vtau[i] = 0.0;
        return;
    }

    double sig = sigma[i];
    if (sig < 1e-14) sig = 1e-14;
    double normDrho = sqrt(sig);
    double tau_val = tau[i];

    constexpr double PI_val = 3.14159265358979323846;
    double threeMPi2_1o3 = cbrt(3.0 * PI_val * PI_val);
    double threeMPi2_2o3 = threeMPi2_1o3 * threeMPi2_1o3;

    double rho_4o3 = r * cbrt(r);
    double s = normDrho / (2.0 * threeMPi2_1o3 * rho_4o3);
    double tauw = normDrho * normDrho / (8.0 * r);
    double tauUnif = 3.0 / 10.0 * threeMPi2_2o3 * r * cbrt(r * r);
    double alpha = (tau_val - tauw) / tauUnif;

    constexpr double k1 = 0.065;
    constexpr double mu_ak = 10.0 / 81.0;
    double b2_x = sqrt(5913.0 / 405000.0);
    double b1_x = 511.0 / 13500.0 / (2.0 * b2_x);
    constexpr double b3_x = 0.5;
    double b4_x = mu_ak * mu_ak / k1 - 1606.0 / 18225.0 - b1_x * b1_x;
    constexpr double hx0 = 1.174, c1x = 0.667, c2x = 0.8, dx_x = 1.24, a1_x = 4.9479;

    double s2 = s * s;
    double epsilon_xUnif = -3.0 / (4.0 * PI_val) * cbrt(3.0 * PI_val * PI_val * r);

    double term1 = 1.0 + b4_x * s2 / mu_ak * exp(-fabs(b4_x) * s2 / mu_ak);
    double xFir = mu_ak * s2 * term1;
    double term3 = 2.0 * (b1_x * s2 + b2_x * (1.0 - alpha) * exp(-b3_x * (1.0 - alpha) * (1.0 - alpha)));
    double xSec = (term3 / 2.0) * (term3 / 2.0);
    double hx1 = 1.0 + k1 - k1 / (1.0 + (xFir + xSec) / k1);

    double fx;
    if (fabs(alpha - 1.0) < 1e-14) fx = 0.0;
    else if (alpha > 1.0) fx = -dx_x * exp(c2x / (1.0 - alpha));
    else fx = exp(-c1x * alpha / (1.0 - alpha));

    double sqrt_s = sqrt(s);
    double gx = 1.0 - exp(-a1_x / (sqrt_s + 1e-30));
    double Fx = (hx1 + fx * (hx0 - hx1)) * gx;
    double epsilonx = epsilon_xUnif * Fx;

    constexpr double b1c = 0.0285764, b2c = 0.0889, b3c = 0.125541;
    constexpr double betaConst = 0.06672455060314922;
    constexpr double betaRsInf = betaConst * 0.1 / 0.1778;
    constexpr double f0 = -0.9;
    constexpr double AA = 0.0310907, alpha1 = 0.21370;
    constexpr double beta1_c = 7.5957, beta2_c = 3.5876, beta3_c = 1.6382, beta4_c = 0.49294;
    constexpr double r_c = 0.031091, c1c = 0.64, c2c = 1.5, dc = 0.7;

    double rs = cbrt(0.75 / (PI_val * r));
    double sqrRs = sqrt(rs);
    double ecLDA0 = -b1c / (1.0 + b2c * sqrRs + b3c * rs);
    double cx0 = -3.0 / (4.0 * PI_val) * cbrt(9.0 * PI_val / 4.0);
    double w0 = exp(-ecLDA0 / b1c) - 1.0;
    double xiInf0 = cbrt(3.0 * PI_val * PI_val / 16.0) * cbrt(3.0 * PI_val * PI_val / 16.0) * (betaRsInf / (cx0 - f0));
    double gInf0s = pow(1.0 + 4.0 * xiInf0 * s2, -0.25);
    double H0 = b1c * log(1.0 + w0 * (1.0 - gInf0s));
    double ec0 = ecLDA0 + H0;

    double beta_c_val = betaConst * (1.0 + 0.1 * rs) / (1.0 + 0.1778 * rs);
    double ec_q0 = -2.0 * AA * (1.0 + alpha1 * rs);
    double ec_q1 = 2.0 * AA * (beta1_c * sqrRs + beta2_c * rs + beta3_c * rs * sqrRs + beta4_c * rs * rs);
    double ec_den = 1.0 / (ec_q1 * ec_q1 + ec_q1);
    double ec_lsda1 = ec_q0 * (-log(ec_q1 * ec_q1 * ec_den));

    double w1 = exp(-ec_lsda1 / r_c) - 1.0;
    double Ac = beta_c_val / (r_c * w1);
    double t = cbrt(3.0 * PI_val * PI_val / 16.0) * s / sqrRs;
    double g = pow(1.0 + 4.0 * Ac * t * t, -0.25);
    double H1 = r_c * log(1.0 + w1 * (1.0 - g));
    double ec1 = ec_lsda1 + H1;

    double fc;
    if (fabs(alpha - 1.0) < 1e-14) fc = 0.0;
    else if (alpha > 1.0) fc = -dc * exp(c2c / (1.0 - alpha));
    else fc = exp(-c1c * alpha / (1.0 - alpha));

    double epsilonc = ec1 + fc * (ec0 - ec1);
    exc[i] = epsilonx + epsilonc;
    vxc[i] = epsilonx * 1.33 + epsilonc * 1.1;
    v2xc[i] = epsilonx * 0.5;
    vtau[i] = epsilonc * 0.2;
}

int main() {
    const int N = 25 * 26 * 27;  // ~17550 grid points
    const int NREPS = 500;

    double *d_rho, *d_sigma, *d_tau, *d_exc, *d_vxc, *d_v2xc, *d_vtau;
    double *d_exc2, *d_vxc2, *d_v2xc2, *d_vtau2;
    double *d_s2_tmp, *d_alpha_tmp;

    CUDA_CHECK(cudaMalloc(&d_rho, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sigma, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tau, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_exc, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vxc, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v2xc, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vtau, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_exc2, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vxc2, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v2xc2, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vtau2, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_s2_tmp, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_alpha_tmp, N * sizeof(double)));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniformDouble(gen, d_rho, N);
    curandGenerateUniformDouble(gen, d_sigma, N);
    curandGenerateUniformDouble(gen, d_tau, N);
    curandDestroyGenerator(gen);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("=== SCAN XC High-Occupancy Benchmark: N=%d (NREPS=%d) ===\n\n", N, NREPS);

    // Orig (no launch bounds)
    {
        int bs = 256, grid = ceildiv(N, bs);
        printf("  Orig: grid=%d blocks, bs=%d\n", grid, bs);
        for (int i = 0; i < 10; i++)
            mgga_scan_kernel_orig<<<grid, bs>>>(d_rho, d_sigma, d_tau, d_exc, d_vxc, d_v2xc, d_vtau, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++)
            mgga_scan_kernel_orig<<<grid, bs>>>(d_rho, d_sigma, d_tau, d_exc, d_vxc, d_v2xc, d_vtau, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        printf("  Orig (256, no LB):  %8.4f ms\n", ms / NREPS);
    }

    // V2: launch_bounds(256, 6)
    {
        int bs = 256, grid = ceildiv(N, bs);
        for (int i = 0; i < 10; i++)
            mgga_scan_kernel_v2<<<grid, bs>>>(d_rho, d_sigma, d_tau, d_exc, d_vxc, d_v2xc, d_vtau, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++)
            mgga_scan_kernel_v2<<<grid, bs>>>(d_rho, d_sigma, d_tau, d_exc, d_vxc, d_v2xc, d_vtau, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        printf("  V2 (256, LB 6):    %8.4f ms\n", ms / NREPS);
    }

    // V3: Two-pass split
    {
        int bs = 256, grid = ceildiv(N, bs);
        for (int i = 0; i < 10; i++) {
            mgga_scan_exchange_v3<<<grid, bs>>>(d_rho, d_sigma, d_tau, d_exc2, d_s2_tmp, d_alpha_tmp, N);
            mgga_scan_correlation_v3<<<grid, bs>>>(d_rho, d_s2_tmp, d_alpha_tmp, d_exc2, d_vxc2, d_v2xc2, d_vtau2, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++) {
            mgga_scan_exchange_v3<<<grid, bs>>>(d_rho, d_sigma, d_tau, d_exc2, d_s2_tmp, d_alpha_tmp, N);
            mgga_scan_correlation_v3<<<grid, bs>>>(d_rho, d_s2_tmp, d_alpha_tmp, d_exc2, d_vxc2, d_v2xc2, d_vtau2, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        printf("  V3 (2-pass split): %8.4f ms\n", ms / NREPS);
    }

    // V4: smaller block size (128)
    {
        int bs = 128, grid = ceildiv(N, bs);
        printf("  V4: grid=%d blocks, bs=%d\n", grid, bs);
        for (int i = 0; i < 10; i++)
            mgga_scan_kernel_v4<<<grid, bs>>>(d_rho, d_sigma, d_tau, d_exc, d_vxc, d_v2xc, d_vtau, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++)
            mgga_scan_kernel_v4<<<grid, bs>>>(d_rho, d_sigma, d_tau, d_exc, d_vxc, d_v2xc, d_vtau, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        printf("  V4 (128, LB 10):   %8.4f ms\n", ms / NREPS);
    }

    // Correctness check: V3 split vs Orig
    {
        int bs = 256, grid = ceildiv(N, bs);
        mgga_scan_kernel_orig<<<grid, bs>>>(d_rho, d_sigma, d_tau, d_exc, d_vxc, d_v2xc, d_vtau, N);
        mgga_scan_exchange_v3<<<grid, bs>>>(d_rho, d_sigma, d_tau, d_exc2, d_s2_tmp, d_alpha_tmp, N);
        mgga_scan_correlation_v3<<<grid, bs>>>(d_rho, d_s2_tmp, d_alpha_tmp, d_exc2, d_vxc2, d_v2xc2, d_vtau2, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        double *h1 = (double*)malloc(N * sizeof(double));
        double *h2 = (double*)malloc(N * sizeof(double));
        CUDA_CHECK(cudaMemcpy(h1, d_exc, N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h2, d_exc2, N * sizeof(double), cudaMemcpyDeviceToHost));
        double maxerr = 0.0;
        for (int i = 0; i < N; i++) {
            double err = fabs(h1[i] - h2[i]);
            if (err > maxerr) maxerr = err;
        }
        printf("\n  Orig vs V3-split max error: %.2e %s\n", maxerr, maxerr < 1e-12 ? "(PASS)" : "(FAIL)");
        free(h1); free(h2);
    }

    printf("\n");
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_rho); cudaFree(d_sigma); cudaFree(d_tau);
    cudaFree(d_exc); cudaFree(d_vxc); cudaFree(d_v2xc); cudaFree(d_vtau);
    cudaFree(d_exc2); cudaFree(d_vxc2); cudaFree(d_v2xc2); cudaFree(d_vtau2);
    cudaFree(d_s2_tmp); cudaFree(d_alpha_tmp);
    return 0;
}
