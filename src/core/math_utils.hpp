#pragma once

#include <cmath>

namespace lynx {

// Real spherical harmonics Y_lm(x, y, z, r) for l = 0..6.
// Matches reference LYNX (tools.c RealSphericalHarmonic).
// All expressions use unit vector (xn,yn,zn) = (x,y,z)/r for efficiency.
// (x, y, z) is the position vector; r = ||(x,y,z)|| (may be pre-computed).
inline double spherical_harmonic(int l, int m, double x, double y, double z, double r) {
    // l = 0: Y_00 = 0.5*sqrt(1/pi)
    if (l == 0) return 0.282094791773878;

    // For l > 0, direction is undefined at r=0
    if (r < 1e-14) return 0.0;

    double invr = 1.0 / r;
    double xn = x * invr, yn = y * invr, zn = z * invr;

    // l = 1
    if (l == 1) {
        constexpr double C = 0.488602511902920; // sqrt(3/(4*pi))
        if (m == -1) return C * yn;
        if (m ==  0) return C * zn;
        if (m ==  1) return C * xn;
        return 0.0;
    }

    double x2 = xn * xn, y2 = yn * yn, z2 = zn * zn;

    // l = 2
    if (l == 2) {
        switch (m) {
            case -2: return 1.092548430592079 * xn * yn;           // 0.5*sqrt(15/pi)
            case -1: return 1.092548430592079 * yn * zn;
            case  0: return 0.315391565252520 * (3.0*z2 - 1.0);    // 0.25*sqrt(5/pi)
            case  1: return 1.092548430592079 * xn * zn;
            case  2: return 0.546274215296040 * (x2 - y2);         // 0.25*sqrt(15/pi)
        }
        return 0.0;
    }

    // l = 3
    if (l == 3) {
        switch (m) {
            case -3: return 0.590043589926644 * yn * (3.0*x2 - y2);
            case -2: return 2.890611442640554 * xn * yn * zn;
            case -1: return 0.457045799464466 * yn * (5.0*z2 - 1.0);
            case  0: return 0.373176332590115 * zn * (5.0*z2 - 3.0);
            case  1: return 0.457045799464466 * xn * (5.0*z2 - 1.0);
            case  2: return 1.445305721320277 * zn * (x2 - y2);
            case  3: return 0.590043589926644 * xn * (x2 - 3.0*y2);
        }
        return 0.0;
    }

    // l = 4
    if (l == 4) {
        // r² terms eliminated: xn²+yn²+zn²=1
        switch (m) {
            case -4: return 2.503342941796705 * xn*yn*(x2 - y2);
            case -3: return 1.770130769779930 * yn*zn*(3.0*x2 - y2);
            case -2: return 0.946174695757560 * xn*yn*(7.0*z2 - 1.0);
            case -1: return 0.669046543557289 * yn*zn*(7.0*z2 - 3.0);
            case  0: return 0.105785546915204 * (35.0*z2*z2 - 30.0*z2 + 3.0);
            case  1: return 0.669046543557289 * xn*zn*(7.0*z2 - 3.0);
            case  2: return 0.473087347878780 * (x2 - y2)*(7.0*z2 - 1.0);
            case  3: return 1.770130769779930 * xn*zn*(x2 - 3.0*y2);
            case  4: return 0.625835735449176 * (x2*(x2 - 3.0*y2) - y2*(3.0*x2 - y2));
        }
        return 0.0;
    }

    double x3 = x2 * xn, y3 = y2 * yn, z3 = z2 * zn;
    double x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;

    // l = 5
    if (l == 5) {
        // pn² = xn²+yn² = 1-zn²
        double pn2 = x2 + y2;
        switch (m) {
            case -5: return 0.656382056840170 * (8.0*x4*yn - 4.0*x2*y3 + 4.0*y4*yn - 3.0*yn*pn2*pn2);
            case -4: return 2.075662314881042 * (4.0*x3*yn - 4.0*xn*y3) * zn;
            case -3: return 0.489238299435250 * (3.0*yn*pn2 - 4.0*y3) * (9.0*z2 - 1.0);
            case -2: return 4.793536784973324 * xn*yn * (3.0*z3 - zn);
            case -1: return 0.452946651195697 * yn * (21.0*z4 - 14.0*z2 + 1.0);
            case  0: return 0.116950322453424 * zn * (63.0*z4 - 70.0*z2 + 15.0);
            case  1: return 0.452946651195697 * xn * (21.0*z4 - 14.0*z2 + 1.0);
            case  2: return 2.396768392486662 * (x2 - y2) * (3.0*z3 - zn);
            case  3: return 0.489238299435250 * (4.0*x3 - 3.0*xn*pn2) * (9.0*z2 - 1.0);
            case  4: return 2.075662314881042 * (4.0*x4 + 4.0*y4 - 3.0*pn2*pn2) * zn;
            case  5: return 0.656382056840170 * (4.0*x4*xn + 8.0*xn*y4 - 4.0*x3*y2 - 3.0*xn*pn2*pn2);
        }
        return 0.0;
    }

    // l = 6
    if (l == 6) {
        double x5 = x4 * xn, y5 = y4 * yn, z5 = z4 * zn;
        double z6 = z4 * z2;
        double pn2 = x2 + y2;
        double pn4 = pn2 * pn2;
        switch (m) {
            case -6: return 0.683184105191914 * (12.0*x5*yn + 12.0*xn*y5 - 8.0*x3*y3 - 6.0*xn*yn*pn4);
            case -5: return 2.366619162231752 * (8.0*x4*yn - 4.0*x2*y3 + 4.0*y5 - 3.0*yn*pn4) * zn;
            case -4: return 0.504564900728724 * (4.0*x3*yn - 4.0*xn*y3) * (11.0*z2 - 1.0);
            case -3: return 0.921205259514923 * (-4.0*y3 + 3.0*yn*pn2) * (11.0*z3 - 3.0*zn);
            case -2: return 0.460602629757462 * 2.0*xn*yn * (33.0*z4 - 18.0*z2 + 1.0);
            case -1: return 0.582621362518731 * yn * (33.0*z5 - 30.0*z3 + 5.0*zn);
            case  0: return 0.0635692022676284 * (231.0*z6 - 315.0*z4 + 105.0*z2 - 5.0);
            case  1: return 0.582621362518731 * xn * (33.0*z5 - 30.0*z3 + 5.0*zn);
            case  2: return 0.460602629757462 * (x2 - y2) * (33.0*z4 - 18.0*z2 + 1.0);
            case  3: return 0.921205259514923 * (4.0*x3 - 3.0*xn*pn2) * (11.0*z3 - 3.0*zn);
            case  4: return 0.504564900728724 * (4.0*x4 + 4.0*y4 - 3.0*pn4) * (11.0*z2 - 1.0);
            case  5: return 2.366619162231752 * (4.0*x5 + 8.0*xn*y4 - 4.0*x3*y2 - 3.0*xn*pn4) * zn;
            case  6: return 0.683184105191914 * (4.0*x5*xn - 4.0*y5*yn + 12.0*x2*y4 - 12.0*x4*y2 + 3.0*y2*pn4 - 3.0*x2*pn4);
        }
        return 0.0;
    }

    return 0.0; // l >= 7 not implemented
}

} // namespace lynx
