// Standalone benchmark: Hamiltonian local = -0.5*Laplacian + Veff*psi
// Compares BASELINE vs OPTIMIZED (flattened OMP + SIMD)
// From src/operators/Hamiltonian.cpp: lap_plus_diag_orth_impl

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <omp.h>

// ==================== BASELINE ====================
static void hamiltonian_baseline(const double* __restrict__ x_ex,
                                 const double* __restrict__ Veff,
                                 double* __restrict__ y,
                                 const double* __restrict__ cx,
                                 const double* __restrict__ cy,
                                 const double* __restrict__ cz,
                                 int nx, int ny, int nz, int FDn, int ncol)
{
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nxny_ex = nx_ex * ny_ex;
    int nd_ex = nxny_ex * (nz + 2 * FDn);
    int nd = nx * ny * nz;

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < ncol; ++n) {
        const double* xn = x_ex + n * nd_ex;
        double* yn = y + n * nd;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx_ex = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
                    int idx_loc = i + j * nx + k * nx * ny;

                    double lap = (cx[0] + cy[0] + cz[0]) * xn[idx_ex];

                    for (int p = 1; p <= FDn; ++p) {
                        lap += cx[p] * (xn[idx_ex + p] + xn[idx_ex - p]);
                        lap += cy[p] * (xn[idx_ex + p * nx_ex] + xn[idx_ex - p * nx_ex]);
                        lap += cz[p] * (xn[idx_ex + p * nxny_ex] + xn[idx_ex - p * nxny_ex]);
                    }

                    yn[idx_loc] = -0.5 * lap + Veff[idx_loc] * xn[idx_ex];
                }
            }
        }
    }
}

// ==================== OPTIMIZED ====================
// 1. Flattened ncol*nz for better thread distribution
// 2. Pre-computed diag_lap constant
// 3. #pragma omp simd on inner i-loop
static void hamiltonian_optimized(const double* __restrict__ x_ex,
                                  const double* __restrict__ Veff,
                                  double* __restrict__ y,
                                  const double* __restrict__ cx,
                                  const double* __restrict__ cy,
                                  const double* __restrict__ cz,
                                  int nx, int ny, int nz, int FDn, int ncol)
{
    const int nx_ex = nx + 2 * FDn;
    const int ny_ex = ny + 2 * FDn;
    const int nxny_ex = nx_ex * ny_ex;
    const int nd_ex = nxny_ex * (nz + 2 * FDn);
    const int nd = nx * ny * nz;

    const double diag_lap = cx[0] + cy[0] + cz[0];

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
                const int idx_ex = idx_base + i;
                const int idx_loc = loc_base + i;

                double lap = diag_lap * xn[idx_ex];

                for (int p = 1; p <= FDn; ++p) {
                    lap += cx[p] * (xn[idx_ex + p] + xn[idx_ex - p]);
                    lap += cy[p] * (xn[idx_ex + p * nx_ex] + xn[idx_ex - p * nx_ex]);
                    lap += cz[p] * (xn[idx_ex + p * nxny_ex] + xn[idx_ex - p * nxny_ex]);
                }

                yn[idx_loc] = -0.5 * lap + Veff[idx_loc] * xn[idx_ex];
            }
        }
    }
}

typedef void (*ham_fn)(const double*, const double*, double*, const double*,
                       const double*, const double*, int, int, int, int, int);

static double bench(ham_fn fn, const double* x_ex, const double* Veff, double* y,
                    const double* cx, const double* cy, const double* cz,
                    int nx, int ny, int nz, int FDn, int ncol, int nreps) {
    fn(x_ex, Veff, y, cx, cy, cz, nx, ny, nz, FDn, ncol);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < nreps; rep++)
        fn(x_ex, Veff, y, cx, cy, cz, nx, ny, nz, FDn, ncol);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / nreps;
}

int main(int argc, char** argv) {
    int nx = 25, ny = 26, nz = 27, ncol = 40, FDn = 6, NREPS = 100;
    if (argc > 1) ncol = atoi(argv[1]);
    if (argc > 2) NREPS = atoi(argv[2]);

    int nd = nx * ny * nz;
    int nd_ex = (nx+2*FDn) * (ny+2*FDn) * (nz+2*FDn);

    printf("=== Hamiltonian Local (Lap+Veff) Benchmark ===\n");
    printf("Grid: %dx%dx%d = %d, ncol=%d, FDn=%d\n", nx, ny, nz, nd, ncol, FDn);
    printf("Threads: %d\n", omp_get_max_threads());

    double* x_ex = (double*)aligned_alloc(64, (size_t)nd_ex * ncol * sizeof(double));
    double* Veff = (double*)aligned_alloc(64, (size_t)nd * sizeof(double));
    double* y_base = (double*)aligned_alloc(64, (size_t)nd * ncol * sizeof(double));
    double* y_opt  = (double*)aligned_alloc(64, (size_t)nd * ncol * sizeof(double));

    double cx[7] = {-5.7083, 0.9375, -0.1563, 0.0208, -0.0018, 0.0001, -0.0000};
    double cy[7] = {-5.3846, 0.8846, -0.1474, 0.0196, -0.0017, 0.0001, -0.0000};
    double cz[7] = {-5.0794, 0.8349, -0.1391, 0.0185, -0.0016, 0.0001, -0.0000};

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (long i = 0; i < (long)nd_ex * ncol; i++) x_ex[i] = dist(rng);
    for (int i = 0; i < nd; i++) Veff[i] = dist(rng) * 10.0;

    double ms_base = bench(hamiltonian_baseline, x_ex, Veff, y_base, cx, cy, cz,
                           nx, ny, nz, FDn, ncol, NREPS);
    double ms_opt  = bench(hamiltonian_optimized, x_ex, Veff, y_opt, cx, cy, cz,
                           nx, ny, nz, FDn, ncol, NREPS);

    double max_diff = 0.0;
    for (long i = 0; i < (long)nd * ncol; i++)
        max_diff = std::max(max_diff, std::abs(y_base[i] - y_opt[i]));

    double reads_per_pt = 1.0 + 2.0 * 3.0 * FDn + 1.0;
    double bytes = (double)nd * ncol * (reads_per_pt + 1.0) * sizeof(double);
    double flops = (double)nd * ncol * (1.0 + FDn * 9.0 + 3.0);

    printf("\n%-12s %10s %10s %10s %10s\n", "Version", "Time(ms)", "GB/s", "GFLOPS", "Speedup");
    printf("%-12s %10.3f %10.1f %10.2f %10s\n", "Baseline", ms_base,
           bytes/(ms_base*1e6), flops/(ms_base*1e6), "1.00x");
    printf("%-12s %10.3f %10.1f %10.2f %9.2fx\n", "Optimized", ms_opt,
           bytes/(ms_opt*1e6), flops/(ms_opt*1e6), ms_base/ms_opt);
    printf("\nMax |diff|: %.2e (should be < 1e-12)\n", max_diff);

    free(x_ex); free(Veff); free(y_base); free(y_opt);
    return 0;
}
