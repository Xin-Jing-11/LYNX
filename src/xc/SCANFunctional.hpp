#pragma once

namespace lynx {

// Hand-coded SCAN metaGGA functional — ported from SPARC mGGAscan.c
// Reference: Sun, Ruzsinszky, Perdew, PRL 115, 036402 (2015)
//
// Conventions (same as SPARC):
//   Non-spin inputs:  rho(Nd_d), sigma(Nd_d) = |grad rho|^2, tau(Nd_d)
//   Non-spin outputs: ex/ec(Nd_d), v1(Nd_d) = d(n*eps)/dn,
//                     v2(Nd_d) = d(n*eps)/d|grad n| / |grad n|,
//                     v3(Nd_d) = d(n*eps)/d(tau)
//
//   Spin inputs:  rho[total|up|dn](3*Nd_d), sigma[total|up|dn](3*Nd_d), tau[total|up|dn](3*Nd_d)
//   Spin exchange outputs: ex(Nd_d), vx1[up|dn](2*Nd_d), vx2[up|dn](2*Nd_d), vx3[up|dn](2*Nd_d)
//   Spin correlation outputs: ec(Nd_d), vc1[up|dn](2*Nd_d), vc2(Nd_d), vc3(Nd_d)

namespace scan {

void scanx(int DMnd, const double* rho, const double* sigma, const double* tau,
           double* ex, double* vx, double* v2x, double* v3x);
void scanc(int DMnd, const double* rho, const double* sigma, const double* tau,
           double* ec, double* vc, double* v2c, double* v3c);
void scanx_spin(int DMnd, const double* rho, const double* sigma, const double* tau,
                double* ex, double* vx, double* v2x, double* v3x);
void scanc_spin(int DMnd, const double* rho, const double* sigma, const double* tau,
                double* ec, double* vc, double* v2c, double* v3c);

} // namespace scan
} // namespace lynx
