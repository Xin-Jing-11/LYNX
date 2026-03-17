#pragma once

#include <cmath>
#include <complex>

namespace lynx {

// Complex spherical harmonics Y_lm(x, y, z, r) for l = 0..6.
// Matches SPARC's ComplexSphericalHarmonic (Condon-Shortley phase convention).
// For m > 0: Y_l^m = (-1)^m * C_lm * (x+iy)^m * P_l^m(z/r) / r^l
// For m < 0: Y_l^{-|m|} = C_l|m| * (x-iy)^|m| * P_l^|m|(z/r) / r^l
// For m = 0: Y_l^0 = C_l0 * P_l(z/r)
inline std::complex<double> complex_spherical_harmonic(int l, int m, double x, double y, double z, double r) {
    using C = std::complex<double>;

    // l = 0
    if (l == 0) return C(0.282094791773878, 0.0);

    if (r < 1e-10) return C(0.0, 0.0);

    // Coefficients matching SPARC (Condon-Shortley phase)
    // l=1
    constexpr double C11 = 0.345494149471335;
    constexpr double C10 = 0.488602511902920;
    // l=2
    constexpr double C22 = 0.386274202023190;
    constexpr double C21 = 0.772548404046379;
    constexpr double C20 = 0.315391565252520;
    // l=3
    constexpr double C33 = 0.417223823632784;
    constexpr double C32 = 1.021985476433282;
    constexpr double C31 = 0.323180184114151;
    constexpr double C30 = 0.373176332590115;
    // l=4
    constexpr double C44 = 0.442532692444983;
    constexpr double C43 = 1.251671470898352;
    constexpr double C42 = 0.334523271778645;
    constexpr double C41 = 0.473087347878780;
    constexpr double C40 = 0.105785546915204;
    // l=5
    constexpr double C55 = 0.464132203440858;
    constexpr double C54 = 1.467714898305751;
    constexpr double C53 = 0.345943719146840;
    constexpr double C52 = 1.694771183260899;
    constexpr double C51 = 0.320281648576215;
    constexpr double C50 = 0.116950322453424;
    // l=6
    constexpr double C66 = 0.483084113580066;
    constexpr double C65 = 1.673452458100098;
    constexpr double C64 = 0.356781262853998;
    constexpr double C63 = 0.651390485867716;
    constexpr double C62 = 0.325695242933858;
    constexpr double C61 = 0.411975516301141;
    constexpr double C60 = 0.063569202267628;

    // (x +/- iy) — matching SPARC's (x[i] +/- I*y[i])
    C xpiy(x, y);   // x + iy
    C xmiy(x, -y);  // x - iy

    // Use division by r products exactly as SPARC does (not precomputed inverse)
    if (l == 1) {
        switch (m) {
            case -1: return C11 * (xmiy / r);
            case  0: return C(C10 * (z / r), 0.0);
            case  1: return -C11 * (xpiy / r);
        }
        return C(0.0, 0.0);
    }

    double rr = r * r;  // r^2
    C xmiy2 = xmiy * xmiy;
    C xpiy2 = xpiy * xpiy;

    if (l == 2) {
        switch (m) {
            case -2: return C22 * (xmiy2) / rr;
            case -1: return C21 * (xmiy * z) / rr;
            case  0: return C(C20 * (2.0*z*z - x*x - y*y) / rr, 0.0);
            case  1: return -C21 * (xpiy * z) / rr;
            case  2: return C22 * (xpiy2) / rr;
        }
        return C(0.0, 0.0);
    }

    double r3 = rr * r;
    C xmiy3 = xmiy2 * xmiy;
    C xpiy3 = xpiy2 * xpiy;

    if (l == 3) {
        switch (m) {
            case -3: return C33 * xmiy3 / r3;
            case -2: return C32 * (xmiy2 * z) / r3;
            case -1: return C31 * (xmiy * (4.0*z*z - x*x - y*y)) / r3;
            case  0: return C(C30 * z * (2.0*z*z - 3.0*x*x - 3.0*y*y) / r3, 0.0);
            case  1: return -C31 * (xpiy * (4.0*z*z - x*x - y*y)) / r3;
            case  2: return C32 * (xpiy2 * z) / r3;
            case  3: return -C33 * xpiy3 / r3;
        }
        return C(0.0, 0.0);
    }

    double r4 = rr * rr;
    C xmiy4 = xmiy2 * xmiy2;
    C xpiy4 = xpiy2 * xpiy2;

    if (l == 4) {
        switch (m) {
            case -4: return C44 * xmiy4 / r4;
            case -3: return C43 * (xmiy3 * z) / r4;
            case -2: return C42 * (xmiy2 * (7.0*z*z - rr)) / r4;
            case -1: return C41 * (xmiy * z * (7.0*z*z - 3.0*rr)) / r4;
            case  0: return C(C40 * (35.0*z*z*z*z - 30.0*z*z*rr + 3.0*rr*rr) / r4, 0.0);
            case  1: return -C41 * (xpiy * z * (7.0*z*z - 3.0*rr)) / r4;
            case  2: return C42 * (xpiy2 * (7.0*z*z - rr)) / r4;
            case  3: return -C43 * (xpiy3 * z) / r4;
            case  4: return C44 * xpiy4 / r4;
        }
        return C(0.0, 0.0);
    }

    double r5 = r4 * r;
    C xmiy5 = xmiy4 * xmiy;
    C xpiy5 = xpiy4 * xpiy;
    double z2 = z * z, z3 = z2 * z, z4 = z2 * z2, z5 = z4 * z;

    if (l == 5) {
        switch (m) {
            case -5: return C55 * xmiy5 / r5;
            case -4: return C54 * (xmiy4 * z) / r5;
            case -3: return C53 * (xmiy3 * (9.0*z2 - rr)) / r5;
            case -2: return C52 * (xmiy2 * (3.0*z3 - z*rr)) / r5;
            case -1: return C51 * (xmiy * (21.0*z4 - 14.0*z2*rr + r4)) / r5;
            case  0: return C(C50 * (63.0*z5 - 70.0*z3*rr + 15.0*z*r4) / r5, 0.0);
            case  1: return -C51 * (xpiy * (21.0*z4 - 14.0*z2*rr + r4)) / r5;
            case  2: return C52 * (xpiy2 * (3.0*z3 - z*rr)) / r5;
            case  3: return -C53 * (xpiy3 * (9.0*z2 - rr)) / r5;
            case  4: return C54 * (xpiy4 * z) / r5;
            case  5: return -C55 * xpiy5 / r5;
        }
        return C(0.0, 0.0);
    }

    double r6 = r4 * rr;
    C xmiy6 = xmiy5 * xmiy;
    C xpiy6 = xpiy5 * xpiy;
    double z6 = z4 * z2;

    if (l == 6) {
        switch (m) {
            case -6: return C66 * xmiy6 / r6;
            case -5: return C65 * (xmiy5 * z) / r6;
            case -4: return C64 * (xmiy4 * (11.0*z2 - rr)) / r6;
            case -3: return C63 * (xmiy3 * (11.0*z3 - 3.0*z*rr)) / r6;
            case -2: return C62 * (xmiy2 * (33.0*z4 - 18.0*z2*rr + r4)) / r6;
            case -1: return C61 * (xmiy * (33.0*z5 - 30.0*z3*rr + 5.0*z*r4)) / r6;
            case  0: return C(C60 * (231.0*z6 - 315.0*z4*rr + 105.0*z2*r4 - 5.0*r6) / r6, 0.0);
            case  1: return -C61 * (xpiy * (33.0*z5 - 30.0*z3*rr + 5.0*z*r4)) / r6;
            case  2: return C62 * (xpiy2 * (33.0*z4 - 18.0*z2*rr + r4)) / r6;
            case  3: return -C63 * (xpiy3 * (11.0*z3 - 3.0*z*rr)) / r6;
            case  4: return C64 * (xpiy4 * (11.0*z2 - rr)) / r6;
            case  5: return -C65 * (xpiy5 * z) / r6;
            case  6: return C66 * xpiy6 / r6;
        }
        return C(0.0, 0.0);
    }

    return C(0.0, 0.0); // l >= 7 not implemented
}

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
