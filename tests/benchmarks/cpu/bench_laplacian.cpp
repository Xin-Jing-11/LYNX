// Standalone benchmark: Orthogonal Laplacian FD stencil (12th-order, FDn=6)
// Compares BASELINE (original LYNX) vs OPTIMIZED (flattened OMP + SIMD + pre-scaled coeffs)
// Kernel: y[i,j,k] = a*sum_p c[p]*(x[i+p]+x[i-p]) per direction + c*x
// From src/operators/Laplacian.cpp: apply_orth_impl

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <omp.h>

// ==================== BASELINE (original LYNX code) ====================
// OMP over ncol only, repeated a*cx[p] multiply in inner loop
static void laplacian_baseline(const double* __restrict__ x_ex,
                               double* __restrict__ y,
                               const double* __restrict__ cx,
                               const double* __restrict__ cy,
                               const double* __restrict__ cz,
                               int nx, int ny, int nz, int FDn, int ncol,
                               double a, double c)
{
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nxny_ex = nx_ex * ny_ex;
    int nd_ex = nxny_ex * (nz + 2 * FDn);
    int nd = nx * ny * nz;

    double diag_coeff = a * (cx[0] + cy[0] + cz[0]) + c;

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < ncol; ++n) {
        const double* xn = x_ex + n * nd_ex;
        double* yn = y + n * nd;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
                    int loc = i + j * nx + k * nx * ny;

                    double val = diag_coeff * xn[idx];

                    for (int p = 1; p <= FDn; ++p) {
                        val += a * cx[p] * (xn[idx + p] + xn[idx - p]);
                        val += a * cy[p] * (xn[idx + p * nx_ex] + xn[idx - p * nx_ex]);
                        val += a * cz[p] * (xn[idx + p * nxny_ex] + xn[idx - p * nxny_ex]);
                    }

                    yn[loc] = val;
                }
            }
        }
    }
}

// ==================== OPTIMIZED ====================
// 1. Flattened ncol*nz outer loop for better thread distribution
// 2. Pre-scaled coefficients (acx[p] = a*cx[p]) — avoids repeated multiply
// 3. #pragma omp simd on inner i-loop
// 4. __restrict__ + const qualifiers throughout
static void laplacian_optimized(const double* __restrict__ x_ex,
                                double* __restrict__ y,
                                const double* __restrict__ cx,
                                const double* __restrict__ cy,
                                const double* __restrict__ cz,
                                int nx, int ny, int nz, int FDn, int ncol,
                                double a, double c)
{
    const int nx_ex = nx + 2 * FDn;
    const int ny_ex = ny + 2 * FDn;
    const int nxny_ex = nx_ex * ny_ex;
    const int nd_ex = nxny_ex * (nz + 2 * FDn);
    const int nd = nx * ny * nz;

    const double diag_coeff = a * (cx[0] + cy[0] + cz[0]) + c;

    // Pre-scale coefficients to avoid repeated multiply by 'a' in inner loop
    double acx[7], acy[7], acz[7];
    for (int p = 1; p <= FDn; ++p) {
        acx[p] = a * cx[p];
        acy[p] = a * cy[p];
        acz[p] = a * cz[p];
    }

    // Flatten ncol*nz for better thread distribution (ncol*nz >> ncol)
    const int total_slabs = ncol * nz;

    #pragma omp parallel for schedule(static)
    for (int nk = 0; nk < total_slabs; ++nk) {
        const int n = nk / nz;
        const int k = nk % nz;
        const double* __restrict__ xn = x_ex + n * nd_ex;
        double* __restrict__ yn = y + n * nd;

        for (int j = 0; j < ny; ++j) {
            const int idx_base = FDn + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
            const int loc_base = j * nx + k * nx * ny;

            #pragma omp simd
            for (int i = 0; i < nx; ++i) {
                const int idx = idx_base + i;
                double val = diag_coeff * xn[idx];

                for (int p = 1; p <= FDn; ++p) {
                    val += acx[p] * (xn[idx + p] + xn[idx - p]);
                    val += acy[p] * (xn[idx + p * nx_ex] + xn[idx - p * nx_ex]);
                    val += acz[p] * (xn[idx + p * nxny_ex] + xn[idx - p * nxny_ex]);
                }

                yn[loc_base + i] = val;
            }
        }
    }
}

// ==================== Benchmark harness ====================
typedef void (*laplacian_fn)(const double*, double*, const double*, const double*,
                             const double*, int, int, int, int, int, double, double);

static double bench(laplacian_fn fn, const double* x_ex, double* y,
                    const double* cx, const double* cy, const double* cz,
                    int nx, int ny, int nz, int FDn, int ncol,
                    double a, double c, int nreps) {
    // warm up
    fn(x_ex, y, cx, cy, cz, nx, ny, nz, FDn, ncol, a, c);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < nreps; rep++)
        fn(x_ex, y, cx, cy, cz, nx, ny, nz, FDn, ncol, a, c);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / nreps;
}

int main(int argc, char** argv) {
    int nx = 25, ny = 26, nz = 27;
    int ncol = 40;
    int FDn = 6;
    int NREPS = 100;

    if (argc > 1) ncol = atoi(argv[1]);
    if (argc > 2) NREPS = atoi(argv[2]);

    int nd = nx * ny * nz;
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nz_ex = nz + 2 * FDn;
    int nd_ex = nx_ex * ny_ex * nz_ex;

    printf("=== Laplacian FD Stencil Benchmark ===\n");
    printf("Grid: %dx%dx%d = %d, ncol=%d, FDn=%d (order %d)\n",
           nx, ny, nz, nd, ncol, FDn, 2*FDn);
    printf("Threads: %d\n", omp_get_max_threads());

    double* x_ex = (double*)aligned_alloc(64, (size_t)nd_ex * ncol * sizeof(double));
    double* y_base = (double*)aligned_alloc(64, (size_t)nd * ncol * sizeof(double));
    double* y_opt  = (double*)aligned_alloc(64, (size_t)nd * ncol * sizeof(double));

    double cx[7] = {-5.7083, 0.9375, -0.1563, 0.0208, -0.0018, 0.0001, -0.0000};
    double cy[7] = {-5.3846, 0.8846, -0.1474, 0.0196, -0.0017, 0.0001, -0.0000};
    double cz[7] = {-5.0794, 0.8349, -0.1391, 0.0185, -0.0016, 0.0001, -0.0000};

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (long i = 0; i < (long)nd_ex * ncol; i++) x_ex[i] = dist(rng);

    // Benchmark
    double ms_base = bench(laplacian_baseline, x_ex, y_base, cx, cy, cz,
                           nx, ny, nz, FDn, ncol, -0.5, 0.0, NREPS);
    double ms_opt  = bench(laplacian_optimized, x_ex, y_opt, cx, cy, cz,
                           nx, ny, nz, FDn, ncol, -0.5, 0.0, NREPS);

    // Verify correctness: max absolute difference
    double max_diff = 0.0;
    for (long i = 0; i < (long)nd * ncol; i++)
        max_diff = std::max(max_diff, std::abs(y_base[i] - y_opt[i]));

    // Metrics
    double reads_per_pt = 1.0 + 2.0 * 3.0 * FDn;
    double bytes = (double)nd * ncol * (reads_per_pt + 1.0) * sizeof(double);
    double flops = (double)nd * ncol * (1.0 + FDn * 9.0);

    printf("\n%-12s %10s %10s %10s %10s\n", "Version", "Time(ms)", "GB/s", "GFLOPS", "Speedup");
    printf("%-12s %10.3f %10.1f %10.2f %10s\n", "Baseline", ms_base,
           bytes/(ms_base*1e6), flops/(ms_base*1e6), "1.00x");
    printf("%-12s %10.3f %10.1f %10.2f %9.2fx\n", "Optimized", ms_opt,
           bytes/(ms_opt*1e6), flops/(ms_opt*1e6), ms_base/ms_opt);
    printf("\nMax |diff|: %.2e (should be < 1e-12)\n", max_diff);

    free(x_ex); free(y_base); free(y_opt);
    return 0;
}
