// Standalone benchmark: FD Gradient (1st derivative, all 3 directions)
// Compares BASELINE vs OPTIMIZED (flattened OMP + SIMD)
// From src/operators/Gradient.cpp: apply_impl

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <omp.h>

// ==================== BASELINE ====================
static void gradient_baseline(const double* __restrict__ x_ex,
                              double* __restrict__ y,
                              const double* __restrict__ coeff,
                              int stride,
                              int nx, int ny, int nz, int FDn, int ncol)
{
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nxny_ex = nx_ex * ny_ex;
    int nd_ex = nxny_ex * (nz + 2 * FDn);
    int nd = nx * ny * nz;

    #pragma omp parallel for schedule(static)
    for (int col = 0; col < ncol; ++col) {
        const double* xc = x_ex + col * nd_ex;
        double* yc = y + col * nd;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
                    double val = 0.0;
                    for (int p = 1; p <= FDn; ++p)
                        val += coeff[p] * (xc[idx + p * stride] - xc[idx - p * stride]);
                    yc[i + j * nx + k * nx * ny] = val;
                }
            }
        }
    }
}

// ==================== OPTIMIZED ====================
// 1. Flattened ncol*nz outer loop
// 2. #pragma omp simd on inner i-loop
// 3. Pre-computed base indices
static void gradient_optimized(const double* __restrict__ x_ex,
                               double* __restrict__ y,
                               const double* __restrict__ coeff,
                               int stride,
                               int nx, int ny, int nz, int FDn, int ncol)
{
    const int nx_ex = nx + 2 * FDn;
    const int ny_ex = ny + 2 * FDn;
    const int nxny_ex = nx_ex * ny_ex;
    const int nd_ex = nxny_ex * (nz + 2 * FDn);
    const int nd = nx * ny * nz;

    const int total_slabs = ncol * nz;

    #pragma omp parallel for schedule(static)
    for (int ck = 0; ck < total_slabs; ++ck) {
        const int col = ck / nz;
        const int k = ck % nz;
        const double* __restrict__ xc = x_ex + col * nd_ex;
        double* __restrict__ yc = y + col * nd;

        for (int j = 0; j < ny; ++j) {
            const int idx_base = FDn + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
            const int loc_base = j * nx + k * nx * ny;

            #pragma omp simd
            for (int i = 0; i < nx; ++i) {
                const int idx = idx_base + i;
                double val = 0.0;
                for (int p = 1; p <= FDn; ++p)
                    val += coeff[p] * (xc[idx + p * stride] - xc[idx - p * stride]);
                yc[loc_base + i] = val;
            }
        }
    }
}

typedef void (*grad_fn)(const double*, double*, const double*, int,
                        int, int, int, int, int);

static double bench_3dir(grad_fn fn, const double* x_ex, double* y,
                         const double* d1x, const double* d1y, const double* d1z,
                         int sx, int sy, int sz,
                         int nx, int ny, int nz, int FDn, int ncol, int nreps) {
    fn(x_ex, y, d1x, sx, nx, ny, nz, FDn, ncol);
    fn(x_ex, y, d1y, sy, nx, ny, nz, FDn, ncol);
    fn(x_ex, y, d1z, sz, nx, ny, nz, FDn, ncol);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < nreps; rep++) {
        fn(x_ex, y, d1x, sx, nx, ny, nz, FDn, ncol);
        fn(x_ex, y, d1y, sy, nx, ny, nz, FDn, ncol);
        fn(x_ex, y, d1z, sz, nx, ny, nz, FDn, ncol);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / nreps;
}

int main(int argc, char** argv) {
    int nx = 25, ny = 26, nz = 27, ncol = 40, FDn = 6, NREPS = 100;
    if (argc > 1) ncol = atoi(argv[1]);
    if (argc > 2) NREPS = atoi(argv[2]);

    int nd = nx * ny * nz;
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nd_ex = nx_ex * ny_ex * (nz + 2 * FDn);

    printf("=== Gradient FD Stencil Benchmark (3 directions) ===\n");
    printf("Grid: %dx%dx%d = %d, ncol=%d, FDn=%d\n", nx, ny, nz, nd, ncol, FDn);
    printf("Threads: %d\n", omp_get_max_threads());

    double* x_ex   = (double*)aligned_alloc(64, (size_t)nd_ex * ncol * sizeof(double));
    double* y_base = (double*)aligned_alloc(64, (size_t)nd * ncol * sizeof(double));
    double* y_opt  = (double*)aligned_alloc(64, (size_t)nd * ncol * sizeof(double));

    double d1x[7] = {0.0, 1.1905, -0.0992, 0.0132, -0.0012, 0.0001, -0.0000};
    double d1y[7] = {0.0, 1.1538, -0.0962, 0.0128, -0.0011, 0.0001, -0.0000};
    double d1z[7] = {0.0, 1.1190, -0.0933, 0.0124, -0.0011, 0.0001, -0.0000};

    int sx = 1, sy = nx_ex, sz = nx_ex * ny_ex;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (long i = 0; i < (long)nd_ex * ncol; i++) x_ex[i] = dist(rng);

    double ms_base = bench_3dir(gradient_baseline, x_ex, y_base, d1x, d1y, d1z,
                                sx, sy, sz, nx, ny, nz, FDn, ncol, NREPS);
    double ms_opt  = bench_3dir(gradient_optimized, x_ex, y_opt, d1x, d1y, d1z,
                                sx, sy, sz, nx, ny, nz, FDn, ncol, NREPS);

    double max_diff = 0.0;
    for (long i = 0; i < (long)nd * ncol; i++)
        max_diff = std::max(max_diff, std::abs(y_base[i] - y_opt[i]));

    double reads_per_pt = 3.0 * 2.0 * FDn;
    double bytes = (double)nd * ncol * (reads_per_pt + 3.0) * sizeof(double);
    double flops = (double)nd * ncol * 3.0 * 2.0 * FDn;

    printf("\n%-12s %10s %10s %10s %10s\n", "Version", "Time(ms)", "GB/s", "GFLOPS", "Speedup");
    printf("%-12s %10.3f %10.1f %10.2f %10s\n", "Baseline", ms_base,
           bytes/(ms_base*1e6), flops/(ms_base*1e6), "1.00x");
    printf("%-12s %10.3f %10.1f %10.2f %9.2fx\n", "Optimized", ms_opt,
           bytes/(ms_opt*1e6), flops/(ms_opt*1e6), ms_base/ms_opt);
    printf("\nMax |diff|: %.2e (should be < 1e-12)\n", max_diff);

    free(x_ex); free(y_base); free(y_opt);
    return 0;
}
