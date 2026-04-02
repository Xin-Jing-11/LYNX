#pragma once

#include <vector>
#include <functional>

namespace lynx {

// ============================================================================
// General-purpose numerical algorithms
// ============================================================================

// --- Brent's root-finding method ---
// Find root of f(x) = 0 in [a, b] using Brent's method.
// Requires f(a) and f(b) have opposite signs.
// Returns x such that |f(x)| < tol (or interval width < tol).
double brent_root(std::function<double(double)> f, double a, double b,
                  double tol = 1e-14, int max_iter = 200);

// --- Gaussian elimination with partial pivoting ---
// Solve A*x = b where A is n×n (row-major), b is n×1.
// Solution stored in x. A and b are not modified.
void gauss_solve(const double* A, const double* b, double* x, int n);

// In-place variant: on entry A is n×n (row-major), b is n×1.
// On exit, b contains the solution. A is destroyed.
void gauss_solve_inplace(double* A, double* b, int n);

// --- Cubic Hermite spline interpolation ---
struct SplineData {
    std::vector<double> x;     // knot positions
    std::vector<double> y;     // function values at knots
    std::vector<double> dydx;  // first derivatives at knots
};

// Compute derivatives for cubic Hermite spline (matches getYD_gen algorithm).
// x: knot positions, y: function values. Returns SplineData with derivatives.
SplineData spline_setup(const std::vector<double>& x, const std::vector<double>& y);

// Interpolate at a single point xi.
double spline_eval(const SplineData& spline, double xi);

// Interpolate at an array of points.
void spline_eval_array(const SplineData& spline, const double* xi, double* yi, int n);

} // namespace lynx
