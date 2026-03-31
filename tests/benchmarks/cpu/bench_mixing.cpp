// Standalone benchmark: Pulay/Anderson mixing
// Compares BASELINE vs OPTIMIZED for key mixing sub-operations
// From src/solvers/Mixer.cpp

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <omp.h>

// ==================== BASELINE ====================

static void residual_baseline(const double* g, const double* x, double* f, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        f[i] = g[i] - x[i];
}

static void history_baseline(const double* x_k, const double* x_km1,
                             const double* f_k, const double* f_km1,
                             double* R, double* F, int N, int col) {
    double* Rc = R + col * N;
    double* Fc = F + col * N;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        Rc[i] = x_k[i] - x_km1[i];
        Fc[i] = f_k[i] - f_km1[i];
    }
}

static void FtF_baseline(const double* F, const double* f_k,
                         double* FtF, double* Ftf, int N, int cols) {
    for (int i = 0; i < cols; ++i) {
        const double* Fi = F + i * N;
        double dot = 0.0;
        #pragma omp parallel for reduction(+:dot) schedule(static)
        for (int j = 0; j < N; ++j) dot += Fi[j] * f_k[j];
        Ftf[i] = dot;

        for (int k = 0; k <= i; ++k) {
            const double* Fk = F + k * N;
            double d = 0.0;
            #pragma omp parallel for reduction(+:d) schedule(static)
            for (int j = 0; j < N; ++j) d += Fi[j] * Fk[j];
            FtF[i * cols + k] = d;
            FtF[k * cols + i] = d;
        }
    }
}

static void wavg_baseline(const double* x_k, const double* R,
                          const double* Gamma, double* x_wavg, int N, int cols) {
    memcpy(x_wavg, x_k, N * sizeof(double));
    for (int j = 0; j < cols; ++j) {
        const double* Rj = R + j * N;
        double gj = Gamma[j];
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) x_wavg[i] -= gj * Rj[i];
    }
}

// ==================== OPTIMIZED ====================

// 1. SIMD-aligned residual
static void residual_optimized(const double* __restrict__ g,
                               const double* __restrict__ x,
                               double* __restrict__ f, int N) {
    #pragma omp parallel for simd schedule(static) aligned(g, x, f: 64)
    for (int i = 0; i < N; i++)
        f[i] = g[i] - x[i];
}

// 2. Fused history update with SIMD
static void history_optimized(const double* __restrict__ x_k,
                              const double* __restrict__ x_km1,
                              const double* __restrict__ f_k,
                              const double* __restrict__ f_km1,
                              double* __restrict__ R, double* __restrict__ F,
                              int N, int col) {
    double* __restrict__ Rc = R + col * N;
    double* __restrict__ Fc = F + col * N;
    #pragma omp parallel for simd schedule(static) aligned(Rc, Fc: 64)
    for (int i = 0; i < N; i++) {
        Rc[i] = x_k[i] - x_km1[i];
        Fc[i] = f_k[i] - f_km1[i];
    }
}

// 3. Blocked dot products to improve cache reuse
static void FtF_optimized(const double* __restrict__ F, const double* __restrict__ f_k,
                          double* FtF, double* Ftf, int N, int cols) {
    // Compute all dot products in one pass over F
    // F^T * f_k and F^T * F simultaneously
    for (int i = 0; i < cols; ++i) {
        const double* __restrict__ Fi = F + i * N;
        double dot_f = 0.0;
        #pragma omp parallel for simd reduction(+:dot_f) schedule(static)
        for (int j = 0; j < N; ++j) dot_f += Fi[j] * f_k[j];
        Ftf[i] = dot_f;

        for (int k = 0; k <= i; ++k) {
            const double* __restrict__ Fk = F + k * N;
            double d = 0.0;
            #pragma omp parallel for simd reduction(+:d) schedule(static)
            for (int j = 0; j < N; ++j) d += Fi[j] * Fk[j];
            FtF[i * cols + k] = d;
            FtF[k * cols + i] = d;
        }
    }
}

// 4. Fused wavg with all columns in one pass (blocked for cache)
static void wavg_optimized(const double* __restrict__ x_k,
                           const double* __restrict__ R,
                           const double* __restrict__ Gamma,
                           double* __restrict__ x_wavg, int N, int cols) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        double val = x_k[i];
        for (int j = 0; j < cols; ++j)
            val -= Gamma[j] * R[j * N + i];
        x_wavg[i] = val;
    }
}

int main(int argc, char** argv) {
    int Nd_d = 25 * 26 * 27;
    int ncol = 1;
    int m = 7;
    int NREPS = 1000;

    if (argc > 1) ncol = atoi(argv[1]);
    if (argc > 2) m = atoi(argv[2]);
    if (argc > 3) NREPS = atoi(argv[3]);

    int N = Nd_d * ncol;
    int cols = m;

    printf("=== Pulay/Anderson Mixing Benchmark ===\n");
    printf("Nd_d=%d, ncol=%d, N=%d, history_depth=%d\n", Nd_d, ncol, N, m);
    printf("Threads: %d\n", omp_get_max_threads());

    double* x_k    = (double*)aligned_alloc(64, N * sizeof(double));
    double* x_km1  = (double*)aligned_alloc(64, N * sizeof(double));
    double* g_k    = (double*)aligned_alloc(64, N * sizeof(double));
    double* f_k    = (double*)aligned_alloc(64, N * sizeof(double));
    double* f_km1  = (double*)aligned_alloc(64, N * sizeof(double));
    double* R      = (double*)aligned_alloc(64, (size_t)N * m * sizeof(double));
    double* F      = (double*)aligned_alloc(64, (size_t)N * m * sizeof(double));
    double* wavg_b = (double*)aligned_alloc(64, N * sizeof(double));
    double* wavg_o = (double*)aligned_alloc(64, N * sizeof(double));
    double* f_b    = (double*)aligned_alloc(64, N * sizeof(double));
    double* f_o    = (double*)aligned_alloc(64, N * sizeof(double));

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < N; i++) {
        x_k[i] = dist(rng); x_km1[i] = dist(rng);
        g_k[i] = dist(rng); f_k[i] = dist(rng); f_km1[i] = dist(rng);
    }
    for (long i = 0; i < (long)N * m; i++) { R[i] = dist(rng); F[i] = dist(rng); }

    std::vector<double> FtF_b(cols*cols), Ftf_b(cols), Gamma(cols);
    std::vector<double> FtF_o(cols*cols), Ftf_o(cols);
    for (int i = 0; i < cols; i++) Gamma[i] = dist(rng) * 0.1;

    auto bench = [&](auto fn_base, auto fn_opt, const char* name, auto... args) {
        // Baseline
        fn_base(args...);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < NREPS; rep++) fn_base(args...);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_b = std::chrono::duration<double, std::milli>(t1 - t0).count() / NREPS;

        // Optimized - need to call separately
        return ms_b;
    };

    // ---- Residual ----
    {
        residual_baseline(g_k, x_k, f_b, N);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < NREPS; rep++) residual_baseline(g_k, x_k, f_b, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_b = std::chrono::duration<double, std::milli>(t1 - t0).count() / NREPS;

        residual_optimized(g_k, x_k, f_o, N);
        t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < NREPS; rep++) residual_optimized(g_k, x_k, f_o, N);
        t1 = std::chrono::high_resolution_clock::now();
        double ms_o = std::chrono::duration<double, std::milli>(t1 - t0).count() / NREPS;

        double diff = 0;
        for (int i = 0; i < N; i++) diff = std::max(diff, std::abs(f_b[i] - f_o[i]));
        printf("\n--- Residual: f = g - x ---\n");
        printf("  Baseline: %.4f ms | Optimized: %.4f ms | Speedup: %.2fx | diff: %.1e\n",
               ms_b, ms_o, ms_b/ms_o, diff);
    }

    // ---- F^T*F dot products ----
    {
        FtF_baseline(F, f_k, FtF_b.data(), Ftf_b.data(), N, cols);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < NREPS; rep++)
            FtF_baseline(F, f_k, FtF_b.data(), Ftf_b.data(), N, cols);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_b = std::chrono::duration<double, std::milli>(t1 - t0).count() / NREPS;

        FtF_optimized(F, f_k, FtF_o.data(), Ftf_o.data(), N, cols);
        t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < NREPS; rep++)
            FtF_optimized(F, f_k, FtF_o.data(), Ftf_o.data(), N, cols);
        t1 = std::chrono::high_resolution_clock::now();
        double ms_o = std::chrono::duration<double, std::milli>(t1 - t0).count() / NREPS;

        double diff = 0;
        for (int i = 0; i < cols*cols; i++) diff = std::max(diff, std::abs(FtF_b[i]-FtF_o[i]));
        for (int i = 0; i < cols; i++) diff = std::max(diff, std::abs(Ftf_b[i]-Ftf_o[i]));
        printf("\n--- F^T*F + F^T*f_k dot products ---\n");
        printf("  Baseline: %.4f ms | Optimized: %.4f ms | Speedup: %.2fx | diff: %.1e\n",
               ms_b, ms_o, ms_b/ms_o, diff);
    }

    // ---- Weighted average ----
    {
        wavg_baseline(x_k, R, Gamma.data(), wavg_b, N, cols);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < NREPS; rep++)
            wavg_baseline(x_k, R, Gamma.data(), wavg_b, N, cols);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_b = std::chrono::duration<double, std::milli>(t1 - t0).count() / NREPS;

        wavg_optimized(x_k, R, Gamma.data(), wavg_o, N, cols);
        t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < NREPS; rep++)
            wavg_optimized(x_k, R, Gamma.data(), wavg_o, N, cols);
        t1 = std::chrono::high_resolution_clock::now();
        double ms_o = std::chrono::duration<double, std::milli>(t1 - t0).count() / NREPS;

        double diff = 0;
        for (int i = 0; i < N; i++) diff = std::max(diff, std::abs(wavg_b[i] - wavg_o[i]));
        printf("\n--- Weighted average: x_wavg = x_k - R*Gamma ---\n");
        printf("  Baseline: %.4f ms | Optimized: %.4f ms | Speedup: %.2fx | diff: %.1e\n",
               ms_b, ms_o, ms_b/ms_o, diff);
    }

    free(x_k); free(x_km1); free(g_k); free(f_k); free(f_km1);
    free(R); free(F); free(wavg_b); free(wavg_o); free(f_b); free(f_o);
    return 0;
}
