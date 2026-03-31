// Standalone benchmark: Chebyshev filter vector operations
// Compares BASELINE vs OPTIMIZED for the vector update kernel
// The Chebyshev filter HOT loop: Y_new = gamma*(HY - c*Y) - ss*Y_old
// From src/solvers/EigenSolver.cpp

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <omp.h>

// ==================== BASELINE ====================
// Simple OMP parallel for
static void cheb_update_baseline(const double* __restrict__ HY,
                                 const double* __restrict__ Y,
                                 const double* __restrict__ Y_old,
                                 double* __restrict__ Y_new,
                                 double gamma, double c, double ss, int total)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total; ++i)
        Y_new[i] = gamma * (HY[i] - c * Y[i]) - ss * Y_old[i];
}

// ==================== OPTIMIZED ====================
// 1. Fused multiply-subtract pattern for better FMA utilization
// 2. #pragma omp simd aligned for vectorization
// 3. Pre-computed gamma*c to reduce operations
static void cheb_update_optimized(const double* __restrict__ HY,
                                  const double* __restrict__ Y,
                                  const double* __restrict__ Y_old,
                                  double* __restrict__ Y_new,
                                  double gamma, double c, double ss, int total)
{
    const double gc = gamma * c;  // pre-compute
    #pragma omp parallel for simd schedule(static) aligned(HY, Y, Y_old, Y_new: 64)
    for (int i = 0; i < total; ++i)
        Y_new[i] = gamma * HY[i] - gc * Y[i] - ss * Y_old[i];
}

// ==================== Full Chebyshev loop (vector ops only) ====================
static void cheb_full_baseline(double* X, double* Y, double* Y_old, double* Y_new,
                               const double* HY, int total, int degree,
                               double sigma_1, double e, double c) {
    double scale = sigma_1 / e;
    // First step
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total; ++i)
        Y[i] = scale * (HY[i] - c * X[i]);
    memcpy(Y_old, X, total * sizeof(double));

    double sigma = sigma_1;
    for (int k = 2; k <= degree; ++k) {
        double sigma_new = 1.0 / (2.0 / sigma_1 - sigma);
        double gamma = 2.0 * sigma_new / e;
        double ss = sigma * sigma_new;
        cheb_update_baseline(HY, Y, Y_old, Y_new, gamma, c, ss, total);
        memcpy(Y_old, Y, total * sizeof(double));
        memcpy(Y, Y_new, total * sizeof(double));
        sigma = sigma_new;
    }
}

static void cheb_full_optimized(double* X, double* Y, double* Y_old, double* Y_new,
                                const double* HY, int total, int degree,
                                double sigma_1, double e, double c) {
    double scale = sigma_1 / e;
    double gc = scale * c;
    #pragma omp parallel for simd schedule(static) aligned(Y, HY, X: 64)
    for (int i = 0; i < total; ++i)
        Y[i] = scale * HY[i] - gc * X[i];
    memcpy(Y_old, X, total * sizeof(double));

    double sigma = sigma_1;
    for (int k = 2; k <= degree; ++k) {
        double sigma_new = 1.0 / (2.0 / sigma_1 - sigma);
        double gamma = 2.0 * sigma_new / e;
        double ss = sigma * sigma_new;
        cheb_update_optimized(HY, Y, Y_old, Y_new, gamma, c, ss, total);
        // Pointer swap instead of memcpy
        double* tmp = Y_old;
        Y_old = Y;
        Y = Y_new;
        Y_new = tmp;
        sigma = sigma_new;
    }
}

int main(int argc, char** argv) {
    int Nd_d = 25 * 26 * 27;
    int Nband = 40;
    int degree = 25;
    int NREPS = 100;

    if (argc > 1) Nband = atoi(argv[1]);
    if (argc > 2) degree = atoi(argv[2]);
    if (argc > 3) NREPS = atoi(argv[3]);

    int total = Nd_d * Nband;

    printf("=== Chebyshev Filter Vector Ops Benchmark ===\n");
    printf("Nd_d=%d, Nband=%d, total=%d, degree=%d\n", Nd_d, Nband, total, degree);
    printf("Threads: %d\n", omp_get_max_threads());

    double* X     = (double*)aligned_alloc(64, total * sizeof(double));
    double* Y     = (double*)aligned_alloc(64, total * sizeof(double));
    double* Y_old = (double*)aligned_alloc(64, total * sizeof(double));
    double* Y_new = (double*)aligned_alloc(64, total * sizeof(double));
    double* HY    = (double*)aligned_alloc(64, total * sizeof(double));
    double* Y_b   = (double*)aligned_alloc(64, total * sizeof(double));
    double* Yo_b  = (double*)aligned_alloc(64, total * sizeof(double));
    double* Yn_b  = (double*)aligned_alloc(64, total * sizeof(double));

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < total; i++) {
        X[i] = dist(rng); Y[i] = dist(rng);
        Y_old[i] = dist(rng); HY[i] = dist(rng);
    }

    double eigval_max = 50.0, eigval_min = -2.0, lambda_cutoff = 5.0;
    double e = (eigval_max - lambda_cutoff) / 2.0;
    double c = (eigval_max + lambda_cutoff) / 2.0;
    double sigma_1 = e / (eigval_min - c);
    double gamma = 2.0 * sigma_1 / e;
    double ss = sigma_1 * sigma_1;

    // ---- Benchmark 1: Single vector update step ----
    {
        cheb_update_baseline(HY, Y, Y_old, Y_new, gamma, c, ss, total);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < NREPS * degree; rep++)
            cheb_update_baseline(HY, Y, Y_old, Y_new, gamma, c, ss, total);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_base = std::chrono::duration<double, std::milli>(t1 - t0).count() / (NREPS * degree);

        cheb_update_optimized(HY, Y, Y_old, Yn_b, gamma, c, ss, total);
        t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < NREPS * degree; rep++)
            cheb_update_optimized(HY, Y, Y_old, Yn_b, gamma, c, ss, total);
        t1 = std::chrono::high_resolution_clock::now();
        double ms_opt = std::chrono::duration<double, std::milli>(t1 - t0).count() / (NREPS * degree);

        double max_diff = 0.0;
        for (int i = 0; i < total; i++)
            max_diff = std::max(max_diff, std::abs(Y_new[i] - Yn_b[i]));

        double bytes = 4.0 * total * sizeof(double);
        printf("\n--- Single vector update: Y_new = gamma*(HY-c*Y) - ss*Y_old ---\n");
        printf("%-12s %10s %10s %10s\n", "Version", "Time(ms)", "GB/s", "Speedup");
        printf("%-12s %10.4f %10.1f %10s\n", "Baseline", ms_base, bytes/(ms_base*1e6), "1.00x");
        printf("%-12s %10.4f %10.1f %9.2fx\n", "Optimized", ms_opt, bytes/(ms_opt*1e6), ms_base/ms_opt);
        printf("Max |diff|: %.2e\n", max_diff);
    }

    // ---- Benchmark 2: Full Chebyshev loop ----
    {
        // Reset
        for (int i = 0; i < total; i++) { Y[i] = X[i]; Y_b[i] = X[i]; }

        cheb_full_baseline(X, Y, Y_old, Y_new, HY, total, degree, sigma_1, e, c);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < NREPS; rep++) {
            memcpy(Y, X, total * sizeof(double));
            cheb_full_baseline(X, Y, Y_old, Y_new, HY, total, degree, sigma_1, e, c);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_base = std::chrono::duration<double, std::milli>(t1 - t0).count() / NREPS;
        memcpy(Y_b, Y, total * sizeof(double));

        for (int i = 0; i < total; i++) Y[i] = X[i];
        cheb_full_optimized(X, Y, Y_old, Y_new, HY, total, degree, sigma_1, e, c);
        t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < NREPS; rep++) {
            memcpy(Y, X, total * sizeof(double));
            cheb_full_optimized(X, Y, Y_old, Y_new, HY, total, degree, sigma_1, e, c);
        }
        t1 = std::chrono::high_resolution_clock::now();
        double ms_opt = std::chrono::duration<double, std::milli>(t1 - t0).count() / NREPS;

        printf("\n--- Full Chebyshev loop (vector ops, degree=%d) ---\n", degree);
        printf("%-12s %10s %10s\n", "Version", "Time(ms)", "Speedup");
        printf("%-12s %10.3f %10s\n", "Baseline", ms_base, "1.00x");
        printf("%-12s %10.3f %9.2fx\n", "Optimized", ms_opt, ms_base/ms_opt);
    }

    free(X); free(Y); free(Y_old); free(Y_new); free(HY);
    free(Y_b); free(Yo_b); free(Yn_b);
    return 0;
}
