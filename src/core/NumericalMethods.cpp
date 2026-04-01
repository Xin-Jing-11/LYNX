#include "core/NumericalMethods.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cstring>

namespace lynx {

// ============================================================================
// Brent's root-finding method
// Exact port from Occupation.cpp find_fermi_level (lines 78-135)
// ============================================================================
double brent_root(std::function<double(double)> f, double a, double b,
                  double tol, int max_iter) {
    double fa = f(a);
    double fb = f(b);

    if (fa * fb > 0) {
        throw std::runtime_error("brent_root: f(a) and f(b) must have opposite signs");
    }

    double c = a, fc = fa;
    double d = b - a, e = d;

    for (int iter = 0; iter < max_iter; ++iter) {
        if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
            c = a; fc = fa;
            d = e = b - a;
        }
        if (std::abs(fc) < std::abs(fb)) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }
        double tol1 = 2.0 * std::numeric_limits<double>::epsilon() * std::abs(b) + 0.5 * tol;
        double xm = 0.5 * (c - b);

        if (std::abs(xm) <= tol1 || fb == 0.0) {
            return b;
        }

        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            double s = fb / fa;
            double p, q;
            if (a == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                q = fa / fc;
                double r = fb / fc;
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0) q = -q;
            p = std::abs(p);
            if (2.0 * p < std::min(3.0 * xm * q - std::abs(tol1 * q), std::abs(e * q))) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        if (std::abs(d) > tol1) {
            b += d;
        } else {
            b += (xm >= 0) ? tol1 : -tol1;
        }
        fb = f(b);
    }

    return b;
}

// ============================================================================
// Gaussian elimination with partial pivoting
// Exact port from Mixer.cpp lines 213-245 and LinearSolver.cpp lines 107-135
// ============================================================================
void gauss_solve_inplace(double* A, double* b, int n) {
    // Forward elimination with partial pivoting
    for (int k = 0; k < n; ++k) {
        int pivot = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(A[i * n + k]) > std::abs(A[pivot * n + k]))
                pivot = i;
        }
        if (pivot != k) {
            for (int j = 0; j < n; ++j)
                std::swap(A[k * n + j], A[pivot * n + j]);
            std::swap(b[k], b[pivot]);
        }
        double diag = A[k * n + k];
        if (std::abs(diag) < 1e-14) continue;
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i * n + k] / diag;
            for (int j = k + 1; j < n; ++j)
                A[i * n + j] -= factor * A[k * n + j];
            b[i] -= factor * b[k];
        }
    }
    // Back substitution
    for (int k = n - 1; k >= 0; --k) {
        if (std::abs(A[k * n + k]) < 1e-14) continue;
        for (int j = k + 1; j < n; ++j)
            b[k] -= A[k * n + j] * b[j];
        b[k] /= A[k * n + k];
    }
}

void gauss_solve(const double* A, const double* b, double* x, int n) {
    // Copy A and b, then solve in-place
    std::vector<double> A_copy(A, A + n * n);
    std::memcpy(x, b, n * sizeof(double));
    gauss_solve_inplace(A_copy.data(), x, n);
}

// ============================================================================
// Cubic Hermite spline
// Exact port from Pseudopotential.cpp spline_deriv (lines 366-426)
// and spline_interp / spline_interp_single (lines 429-532)
// ============================================================================

SplineData spline_setup(const std::vector<double>& x, const std::vector<double>& y) {
    SplineData spline;
    spline.x = x;
    spline.y = y;

    int n = static_cast<int>(x.size());
    spline.dydx.resize(n, 0.0);
    if (n < 2) return spline;
    if (n == 2) {
        spline.dydx[0] = spline.dydx[1] = (y[1] - y[0]) / (x[1] - x[0]);
        return spline;
    }

    // Build tridiagonal system for first derivatives
    std::vector<double> A(n, 0.0), B(n, 0.0), C(n, 0.0);

    // First row (non-standard BC matching reference getYD_gen)
    double h0 = x[1] - x[0];
    double h1 = x[2] - x[1];
    double r0 = (y[1] - y[0]) / h0;
    double r1 = (y[2] - y[1]) / h1;
    B[0] = h1 * (h0 + h1);
    C[0] = (h0 + h1) * (h0 + h1);
    spline.dydx[0] = r0 * (3.0 * h0 * h1 + 2.0 * h1 * h1) + r1 * h0 * h0;

    // Interior rows
    for (int i = 1; i < n - 1; ++i) {
        h0 = x[i] - x[i-1];
        h1 = x[i+1] - x[i];
        r0 = (y[i] - y[i-1]) / h0;
        r1 = (y[i+1] - y[i]) / h1;
        A[i] = h1;
        B[i] = 2.0 * (h0 + h1);
        C[i] = h0;
        spline.dydx[i] = 3.0 * (r0 * h1 + r1 * h0);
    }

    // Last row
    int last = n - 1;
    h0 = x[last-1] - x[last-2];
    h1 = x[last] - x[last-1];
    r0 = (y[last-1] - y[last-2]) / h0;
    r1 = (y[last] - y[last-1]) / h1;
    A[last] = (h0 + h1) * (h0 + h1);
    B[last] = h0 * (h0 + h1);
    spline.dydx[last] = r0 * h1 * h1 + r1 * (3.0 * h0 * h1 + 2.0 * h0 * h0);

    // Solve tridiagonal system (Gauss elimination)
    // Forward sweep
    std::vector<double> F(n, 0.0);
    double bval = B[0];
    spline.dydx[0] = spline.dydx[0] / bval;
    for (int j = 1; j < n; ++j) {
        F[j] = C[j-1] / bval;
        bval = B[j] - A[j] * F[j];
        if (std::abs(bval) < 1e-30) bval = 1e-30;
        spline.dydx[j] = (spline.dydx[j] - A[j] * spline.dydx[j-1]) / bval;
    }
    // Back substitution
    for (int j = n - 2; j >= 0; --j) {
        spline.dydx[j] -= F[j+1] * spline.dydx[j+1];
    }

    return spline;
}

double spline_eval(const SplineData& spline, double xi) {
    int n = static_cast<int>(spline.x.size());
    if (xi <= spline.x[0]) return spline.y[0];
    if (xi >= spline.x[n - 1]) return spline.y[n - 1];

    // Find interval (uniform grid fast path)
    double delta_x = spline.x[1] - spline.x[0];
    int j = static_cast<int>((xi - spline.x[0]) / delta_x);
    if (j >= n - 1) j = n - 2;
    // Verify uniform grid assumption; if not, binary search
    if (xi < spline.x[j] || xi > spline.x[j + 1]) {
        int lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (spline.x[mid] > xi) hi = mid;
            else lo = mid;
        }
        j = lo;
    }

    double p1 = spline.x[j];
    double p3 = spline.x[j + 1];
    double dx = 1.0 / (p3 - p1);
    double dy = (spline.y[j + 1] - spline.y[j]) * dx;
    double A0 = spline.y[j];
    double A1 = spline.dydx[j];
    double A2 = dx * (3.0 * dy - 2.0 * spline.dydx[j] - spline.dydx[j + 1]);
    double A3 = dx * dx * (-2.0 * dy + spline.dydx[j] + spline.dydx[j + 1]);

    double x = xi - p1;
    return ((A3 * x + A2) * x + A1) * x + A0;
}

void spline_eval_array(const SplineData& spline, const double* xi, double* yi, int m) {
    int n = static_cast<int>(spline.x.size());

    // Check if grid is uniform
    bool is_uniform = true;
    if (n > 2) {
        double delta = spline.x[1] - spline.x[0];
        for (int i = 2; i < n; ++i) {
            if (std::abs((spline.x[i] - spline.x[i-1]) - delta) > 1e-10 * std::max(1.0, delta)) {
                is_uniform = false;
                break;
            }
        }
    }

    double delta_x = (n >= 2) ? (spline.x[1] - spline.x[0]) : 1.0;
    double x1_max = spline.x[n - 1];

    for (int i = 0; i < m; ++i) {
        double r = xi[i];

        // Clamp to grid range
        if (r <= spline.x[0]) {
            yi[i] = spline.y[0];
            continue;
        }
        if (r >= x1_max) {
            yi[i] = spline.y[n - 1];
            continue;
        }

        // Find interval
        int j;
        if (is_uniform) {
            j = static_cast<int>((r - spline.x[0]) / delta_x);
            if (j >= n - 1) j = n - 2;
        } else {
            // Binary search
            int lo = 0, hi = n - 1;
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (spline.x[mid] > r) hi = mid;
                else lo = mid;
            }
            j = lo;
        }

        // Hermite interpolation coefficients
        double p1 = spline.x[j];
        double p3 = spline.x[j + 1];
        double dx = 1.0 / (p3 - p1);
        double dy = (spline.y[j + 1] - spline.y[j]) * dx;
        double A0 = spline.y[j];
        double A1 = spline.dydx[j];
        double A2 = dx * (3.0 * dy - 2.0 * spline.dydx[j] - spline.dydx[j + 1]);
        double A3 = dx * dx * (-2.0 * dy + spline.dydx[j] + spline.dydx[j + 1]);

        // Horner's rule
        double x = r - p1;
        yi[i] = ((A3 * x + A2) * x + A1) * x + A0;
    }
}

} // namespace lynx
