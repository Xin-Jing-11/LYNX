#pragma once

#include <vector>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <complex>
#include <type_traits>

// LAPACK tridiagonal eigenvalue solver
extern "C" {
    void dsterf_(const int* n, double* d, double* e, int* info);
}

namespace lynx {

using Complex = std::complex<double>;

namespace lanczos_detail {

// Type-aware dot product: Re(<x, y>)
// For double: sum(x[i]*y[i])
// For Complex: Re(sum(conj(x[i])*y[i]))
template<typename T>
inline double real_dot(const T* x, const T* y, int N) {
    if constexpr (std::is_same_v<T, double>) {
        double sum = 0.0;
        for (int i = 0; i < N; ++i) sum += x[i] * y[i];
        return sum;
    } else {
        // Complex case
        Complex sum(0.0, 0.0);
        for (int i = 0; i < N; ++i)
            sum += std::conj(x[i]) * y[i];
        return sum.real();
    }
}

// Type-aware norm: sqrt(Re(<x, x>))
template<typename T>
inline double norm(const T* x, int N) {
    if constexpr (std::is_same_v<T, double>) {
        double sum = 0.0;
        for (int i = 0; i < N; ++i) sum += x[i] * x[i];
        return std::sqrt(sum);
    } else {
        double sum = 0.0;
        for (int i = 0; i < N; ++i)
            sum += std::norm(x[i]);  // |z|^2
        return std::sqrt(sum);
    }
}

// Type-aware random initialization
template<typename T>
inline void random_init(T* v, int N) {
    if constexpr (std::is_same_v<T, double>) {
        for (int i = 0; i < N; ++i)
            v[i] = (double)std::rand() / RAND_MAX;
    } else {
        for (int i = 0; i < N; ++i)
            v[i] = Complex((double)std::rand() / RAND_MAX,
                           (double)std::rand() / RAND_MAX);
    }
}

// Type-aware random vector in [-1, 1] (for invariant subspace fallback)
template<typename T>
inline void random_init_symmetric(T* v, int N) {
    if constexpr (std::is_same_v<T, double>) {
        for (int i = 0; i < N; ++i)
            v[i] = -1.0 + 2.0 * ((double)std::rand() / RAND_MAX);
    } else {
        for (int i = 0; i < N; ++i)
            v[i] = Complex(-1.0 + 2.0 * ((double)std::rand() / RAND_MAX),
                           -1.0 + 2.0 * ((double)std::rand() / RAND_MAX));
    }
}

} // namespace lanczos_detail

// Estimate eigenvalue bounds [eigval_min, eigval_max] of a linear operator
// using the Lanczos algorithm with tolerance-based convergence.
//
// matvec: callback that computes y = A*x, signature void(const T* x, T* y)
// Nd_d: dimension of the vectors
// eigval_min, eigval_max: output eigenvalue bounds
// tol: convergence tolerance for eigenvalue changes
// max_iter: maximum Lanczos iterations
// rand_seed: seed for random initial vector
template<typename T>
void lanczos_bounds(
    std::function<void(const T*, T*)> matvec,
    int Nd_d,
    double& eigval_min, double& eigval_max,
    double tol = 1e-2, int max_iter = 1000,
    int rand_seed = 1)
{
    using namespace lanczos_detail;

    std::vector<T> V_j(Nd_d), V_jm1(Nd_d), V_jp1(Nd_d);
    std::vector<double> a(max_iter + 1, 0.0), b(max_iter + 1, 0.0);

    // Initial random vector
    std::srand(rand_seed);
    random_init(V_jm1.data(), Nd_d);

    // Normalize
    double vscal = 1.0 / norm(V_jm1.data(), Nd_d);
    for (int i = 0; i < Nd_d; ++i) V_jm1[i] *= vscal;

    // First H*v (warm-up)
    matvec(V_jm1.data(), V_j.data());

    // a[0] = Re(<V_jm1, V_j>)
    a[0] = real_dot(V_jm1.data(), V_j.data(), Nd_d);

    // V_j -= a[0] * V_jm1
    for (int i = 0; i < Nd_d; ++i)
        V_j[i] -= a[0] * V_jm1[i];

    // b[0] = ||V_j||
    b[0] = norm(V_j.data(), Nd_d);

    if (b[0] == 0.0) {
        // Invariant subspace; pick random vector orthogonal to V_jm1
        random_init_symmetric(V_j.data(), Nd_d);
        double dot_re = real_dot(V_j.data(), V_jm1.data(), Nd_d);
        if constexpr (std::is_same_v<T, double>) {
            for (int i = 0; i < Nd_d; ++i)
                V_j[i] -= dot_re * V_jm1[i];
        } else {
            // For complex, need to subtract the full complex projection
            Complex dot_val(0.0, 0.0);
            for (int i = 0; i < Nd_d; ++i)
                dot_val += std::conj(V_j[i]) * V_jm1[i];
            for (int i = 0; i < Nd_d; ++i)
                V_j[i] -= dot_val * V_jm1[i];
        }
        b[0] = norm(V_j.data(), Nd_d);
    }

    // Scale V_j
    vscal = (b[0] == 0.0) ? 1.0 : (1.0 / b[0]);
    for (int i = 0; i < Nd_d; ++i) V_j[i] *= vscal;

    eigval_min = 0.0;
    eigval_max = 0.0;
    double eigmin_pre = 0.0, eigmax_pre = 0.0;
    double err_eigmin = tol + 1.0;
    double err_eigmax = tol + 1.0;

    int j = 0;
    while ((err_eigmin > tol || err_eigmax > tol) && j < max_iter) {
        // V_{j+1} = H * V_j
        matvec(V_j.data(), V_jp1.data());

        // a[j+1] = Re(<V_j, V_{j+1}>)
        a[j + 1] = real_dot(V_j.data(), V_jp1.data(), Nd_d);

        // V_{j+1} = V_{j+1} - a[j+1]*V_j - b[j]*V_{j-1}
        for (int i = 0; i < Nd_d; ++i) {
            V_jp1[i] -= (a[j + 1] * V_j[i] + b[j] * V_jm1[i]);
            V_jm1[i] = V_j[i];
        }

        b[j + 1] = norm(V_jp1.data(), Nd_d);
        if (b[j + 1] == 0.0) break;

        vscal = 1.0 / b[j + 1];
        for (int i = 0; i < Nd_d; ++i) V_j[i] = V_jp1[i] * vscal;

        // Solve tridiagonal eigenvalue problem using dsterf_
        int n = j + 2;
        std::vector<double> d(n), e_vec(n);
        for (int kk = 0; kk < n; ++kk) { d[kk] = a[kk]; e_vec[kk] = b[kk]; }

        int info;
        dsterf_(&n, d.data(), e_vec.data(), &info);
        if (info == 0) {
            eigval_min = d[0];
            eigval_max = d[n - 1];
        } else {
            break;
        }

        err_eigmin = std::abs(eigval_min - eigmin_pre);
        err_eigmax = std::abs(eigval_max - eigmax_pre);
        eigmin_pre = eigval_min;
        eigmax_pre = eigval_max;

        j++;
    }

    // Apply safety margins (reference: eigmax *= 1.01, eigmin -= 0.1)
    eigval_max *= 1.01;
    eigval_min -= 0.1;
}

} // namespace lynx
