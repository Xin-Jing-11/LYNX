// Benchmark: SPARC vs LYNX Laplacian Stencil Patterns (multi-column, multi-thread)
//
// Compares five implementations:
//   A: LYNX current    — separate cx/cy/cz arrays, no SIMD hint
//   B: SPARC simd-only — interleaved coefficients, #pragma omp simd on x-loop
//   C: SPARC+OMP cols  — B + #pragma omp parallel for on columns
//   D: Flat OMP+SIMD   — flattened ncol*nz loop + simd x-loop
//   E: SPARC+AVX2      — explicit _mm256_fmadd_pd on x-loop
//
// Compile:
//   g++ -O3 -march=native -mavx2 -mfma -fopenmp -o bench_sparc_stencil bench_sparc_stencil.cpp
//   Add -fopt-info-vec-optimized to verify vectorization
//
// Run:
//   for t in 1 2 4 8; do OMP_NUM_THREADS=$t ./bench_sparc_stencil; done

#include <immintrin.h>
#include <omp.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>

static constexpr int FDn = 6;

// ============================================================
// Version A: LYNX current (separate coeff arrays, no SIMD hint)
// ============================================================
__attribute__((noinline))
void laplacian_lynx(const double* __restrict__ x_ex, double* __restrict__ y,
                    const double* cx, const double* cy, const double* cz,
                    double diag_coeff, double a,
                    int nx, int ny, int nz, int nx_ex, int ny_ex,
                    int nd, int nd_ex, int ncol) {
    int nxny = nx * ny;
    int nxny_ex = nx_ex * ny_ex;
    for (int n = 0; n < ncol; ++n) {
        const double* xn = x_ex + n * nd_ex;
        double* yn = y + n * nd;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
                    int loc = i + j * nx + k * nxny;
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

// ============================================================
// Version B: SPARC pattern (interleaved coeffs, omp simd on x-loop)
// ============================================================
__attribute__((noinline))
void laplacian_sparc_simd(const double* __restrict__ x_ex, double* __restrict__ y,
                          const double* stencil_coefs, double coef_0,
                          int nx, int ny, int nz,
                          int stride_y, int stride_y_ex,
                          int stride_z, int stride_z_ex,
                          int nd, int nd_ex, int ncol) {
    for (int n = 0; n < ncol; ++n) {
        const double* xn = x_ex + n * nd_ex;
        double* yn = y + n * nd;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                int offset = k * stride_z + j * stride_y;
                int offset_ex = (k + FDn) * stride_z_ex + (j + FDn) * stride_y_ex;
                #pragma omp simd
                for (int i = 0; i < nx; ++i) {
                    int idx_ex = offset_ex + i + FDn;
                    double res = coef_0 * xn[idx_ex];
                    for (int r = 1; r <= 6; r++) {
                        int stride_y_r = r * stride_y_ex;
                        int stride_z_r = r * stride_z_ex;
                        res += (xn[idx_ex - r]          + xn[idx_ex + r])          * stencil_coefs[3*r];
                        res += (xn[idx_ex - stride_y_r] + xn[idx_ex + stride_y_r]) * stencil_coefs[3*r+1];
                        res += (xn[idx_ex - stride_z_r] + xn[idx_ex + stride_z_r]) * stencil_coefs[3*r+2];
                    }
                    yn[offset + i] = res;
                }
            }
        }
    }
}

// ============================================================
// Version C: SPARC + OpenMP parallel for on columns
// ============================================================
__attribute__((noinline))
void laplacian_sparc_omp_cols(const double* __restrict__ x_ex, double* __restrict__ y,
                              const double* stencil_coefs, double coef_0,
                              int nx, int ny, int nz,
                              int stride_y, int stride_y_ex,
                              int stride_z, int stride_z_ex,
                              int nd, int nd_ex, int ncol) {
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < ncol; ++n) {
        const double* xn = x_ex + n * nd_ex;
        double* yn = y + n * nd;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                int offset = k * stride_z + j * stride_y;
                int offset_ex = (k + FDn) * stride_z_ex + (j + FDn) * stride_y_ex;
                #pragma omp simd
                for (int i = 0; i < nx; ++i) {
                    int idx_ex = offset_ex + i + FDn;
                    double res = coef_0 * xn[idx_ex];
                    for (int r = 1; r <= 6; r++) {
                        int stride_y_r = r * stride_y_ex;
                        int stride_z_r = r * stride_z_ex;
                        res += (xn[idx_ex - r]          + xn[idx_ex + r])          * stencil_coefs[3*r];
                        res += (xn[idx_ex - stride_y_r] + xn[idx_ex + stride_y_r]) * stencil_coefs[3*r+1];
                        res += (xn[idx_ex - stride_z_r] + xn[idx_ex + stride_z_r]) * stencil_coefs[3*r+2];
                    }
                    yn[offset + i] = res;
                }
            }
        }
    }
}

// ============================================================
// Version D: Flattened OMP on ncol*nz + SIMD on x
// ============================================================
__attribute__((noinline))
void laplacian_flat_omp(const double* __restrict__ x_ex, double* __restrict__ y,
                        const double* stencil_coefs, double coef_0,
                        int nx, int ny, int nz,
                        int stride_y, int stride_y_ex,
                        int stride_z, int stride_z_ex,
                        int nd, int nd_ex, int ncol) {
    int total_nk = ncol * nz;
    #pragma omp parallel for schedule(static)
    for (int nk = 0; nk < total_nk; ++nk) {
        int n = nk / nz;
        int k = nk % nz;
        const double* xn = x_ex + n * nd_ex;
        double* yn = y + n * nd;
        for (int j = 0; j < ny; ++j) {
            int offset = k * stride_z + j * stride_y;
            int offset_ex = (k + FDn) * stride_z_ex + (j + FDn) * stride_y_ex;
            #pragma omp simd
            for (int i = 0; i < nx; ++i) {
                int idx_ex = offset_ex + i + FDn;
                double res = coef_0 * xn[idx_ex];
                for (int r = 1; r <= 6; r++) {
                    int stride_y_r = r * stride_y_ex;
                    int stride_z_r = r * stride_z_ex;
                    res += (xn[idx_ex - r]          + xn[idx_ex + r])          * stencil_coefs[3*r];
                    res += (xn[idx_ex - stride_y_r] + xn[idx_ex + stride_y_r]) * stencil_coefs[3*r+1];
                    res += (xn[idx_ex - stride_z_r] + xn[idx_ex + stride_z_r]) * stencil_coefs[3*r+2];
                }
                yn[offset + i] = res;
            }
        }
    }
}

// ============================================================
// Version E: SPARC + AVX2 intrinsics on x-loop
// ============================================================
__attribute__((noinline))
void laplacian_sparc_avx2(const double* __restrict__ x_ex, double* __restrict__ y,
                          const double* stencil_coefs, double coef_0,
                          int nx, int ny, int nz,
                          int stride_y, int stride_y_ex,
                          int stride_z, int stride_z_ex,
                          int nd, int nd_ex, int ncol) {
    // Pre-broadcast coefficients
    __m256d vcoef0 = _mm256_set1_pd(coef_0);
    __m256d vcx[7], vcy[7], vcz[7];
    for (int r = 1; r <= 6; r++) {
        vcx[r] = _mm256_set1_pd(stencil_coefs[3*r]);
        vcy[r] = _mm256_set1_pd(stencil_coefs[3*r+1]);
        vcz[r] = _mm256_set1_pd(stencil_coefs[3*r+2]);
    }

    for (int n = 0; n < ncol; ++n) {
        const double* xn = x_ex + n * nd_ex;
        double* yn = y + n * nd;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                int offset = k * stride_z + j * stride_y;
                int offset_ex = (k + FDn) * stride_z_ex + (j + FDn) * stride_y_ex + FDn;
                int i = 0;
                int nx4 = nx & ~3;
                for (; i < nx4; i += 4) {
                    int idx_ex = offset_ex + i;
                    __m256d center = _mm256_loadu_pd(&xn[idx_ex]);
                    __m256d val = _mm256_mul_pd(vcoef0, center);
                    for (int r = 1; r <= 6; r++) {
                        int sy = r * stride_y_ex;
                        int sz = r * stride_z_ex;
                        __m256d fwd_x = _mm256_loadu_pd(&xn[idx_ex + r]);
                        __m256d bwd_x = _mm256_loadu_pd(&xn[idx_ex - r]);
                        val = _mm256_fmadd_pd(vcx[r], _mm256_add_pd(fwd_x, bwd_x), val);
                        __m256d fwd_y = _mm256_loadu_pd(&xn[idx_ex + sy]);
                        __m256d bwd_y = _mm256_loadu_pd(&xn[idx_ex - sy]);
                        val = _mm256_fmadd_pd(vcy[r], _mm256_add_pd(fwd_y, bwd_y), val);
                        __m256d fwd_z = _mm256_loadu_pd(&xn[idx_ex + sz]);
                        __m256d bwd_z = _mm256_loadu_pd(&xn[idx_ex - sz]);
                        val = _mm256_fmadd_pd(vcz[r], _mm256_add_pd(fwd_z, bwd_z), val);
                    }
                    _mm256_storeu_pd(&yn[offset + i], val);
                }
                // Scalar remainder
                for (; i < nx; ++i) {
                    int idx_ex = offset_ex + i;
                    double res = coef_0 * xn[idx_ex];
                    for (int r = 1; r <= 6; r++) {
                        int sy = r * stride_y_ex;
                        int sz = r * stride_z_ex;
                        res += (xn[idx_ex - r]  + xn[idx_ex + r])  * stencil_coefs[3*r];
                        res += (xn[idx_ex - sy]  + xn[idx_ex + sy]) * stencil_coefs[3*r+1];
                        res += (xn[idx_ex - sz]  + xn[idx_ex + sz]) * stencil_coefs[3*r+2];
                    }
                    yn[offset + i] = res;
                }
            }
        }
    }
}

// ============================================================
// Version E+OMP: AVX2 + OpenMP parallel for on columns
// ============================================================
__attribute__((noinline))
void laplacian_sparc_avx2_omp(const double* __restrict__ x_ex, double* __restrict__ y,
                              const double* stencil_coefs, double coef_0,
                              int nx, int ny, int nz,
                              int stride_y, int stride_y_ex,
                              int stride_z, int stride_z_ex,
                              int nd, int nd_ex, int ncol) {
    __m256d vcoef0 = _mm256_set1_pd(coef_0);
    __m256d vcx[7], vcy[7], vcz[7];
    for (int r = 1; r <= 6; r++) {
        vcx[r] = _mm256_set1_pd(stencil_coefs[3*r]);
        vcy[r] = _mm256_set1_pd(stencil_coefs[3*r+1]);
        vcz[r] = _mm256_set1_pd(stencil_coefs[3*r+2]);
    }
    int nx4 = nx & ~3;

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < ncol; ++n) {
        const double* xn = x_ex + n * nd_ex;
        double* yn = y + n * nd;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                int offset = k * stride_z + j * stride_y;
                int offset_ex = (k + FDn) * stride_z_ex + (j + FDn) * stride_y_ex + FDn;
                int i = 0;
                for (; i < nx4; i += 4) {
                    int idx_ex = offset_ex + i;
                    __m256d center = _mm256_loadu_pd(&xn[idx_ex]);
                    __m256d val = _mm256_mul_pd(vcoef0, center);
                    for (int r = 1; r <= 6; r++) {
                        int sy = r * stride_y_ex;
                        int sz = r * stride_z_ex;
                        val = _mm256_fmadd_pd(vcx[r], _mm256_add_pd(
                            _mm256_loadu_pd(&xn[idx_ex + r]), _mm256_loadu_pd(&xn[idx_ex - r])), val);
                        val = _mm256_fmadd_pd(vcy[r], _mm256_add_pd(
                            _mm256_loadu_pd(&xn[idx_ex + sy]), _mm256_loadu_pd(&xn[idx_ex - sy])), val);
                        val = _mm256_fmadd_pd(vcz[r], _mm256_add_pd(
                            _mm256_loadu_pd(&xn[idx_ex + sz]), _mm256_loadu_pd(&xn[idx_ex - sz])), val);
                    }
                    _mm256_storeu_pd(&yn[offset + i], val);
                }
                for (; i < nx; ++i) {
                    int idx_ex = offset_ex + i;
                    double res = coef_0 * xn[idx_ex];
                    for (int r = 1; r <= 6; r++) {
                        int sy = r * stride_y_ex;
                        int sz = r * stride_z_ex;
                        res += (xn[idx_ex - r] + xn[idx_ex + r]) * stencil_coefs[3*r];
                        res += (xn[idx_ex - sy] + xn[idx_ex + sy]) * stencil_coefs[3*r+1];
                        res += (xn[idx_ex - sz] + xn[idx_ex + sz]) * stencil_coefs[3*r+2];
                    }
                    yn[offset + i] = res;
                }
            }
        }
    }
}

// ============================================================
// Helpers
// ============================================================
struct BenchResult {
    const char* name;
    double time_ms;
    double max_err;
};

void run_bench(const char* grid_label, int nx, int ny, int nz, int ncol, int nreps) {
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nz_ex = nz + 2 * FDn;
    int nxny = nx * ny;
    int nxny_ex = nx_ex * ny_ex;
    int nd = nx * ny * nz;
    int nd_ex = nx_ex * ny_ex * nz_ex;
    int stride_y = nx, stride_y_ex = nx_ex;
    int stride_z = nxny, stride_z_ex = nxny_ex;
    int nthreads = 1;
    #pragma omp parallel
    { nthreads = omp_get_num_threads(); }

    printf("=== Laplacian Stencil Benchmark: %s ===\n", grid_label);
    printf("Grid: %d x %d x %d = %d pts, ncol=%d, FDn=%d\n", nx, ny, nz, nd, ncol, FDn);
    printf("Threads: %d, Reps: %d\n\n", nthreads, nreps);

    // Separate coefficient arrays for LYNX (Version A)
    double cx[FDn + 1] = {-3.05, 1.7, -0.2833, 0.07524, -0.02121, 0.005040, -0.0006944};
    double cy[FDn + 1] = {-2.95, 1.6, -0.2733, 0.07024, -0.02021, 0.004940, -0.0006844};
    double cz[FDn + 1] = {-2.85, 1.5, -0.2633, 0.06524, -0.01921, 0.004840, -0.0006744};
    double a = 1.0;
    double diag_coeff = a * (cx[0] + cy[0] + cz[0]);

    // SPARC interleaved coefficients: [_, _, _, cx1, cy1, cz1, cx2, cy2, cz2, ...]
    // coef_0 = diag_coeff (pre-scaled with 'a')
    double stencil_coefs[3 * (FDn + 1)];
    memset(stencil_coefs, 0, sizeof(stencil_coefs));
    for (int r = 1; r <= FDn; r++) {
        stencil_coefs[3*r]   = a * cx[r];  // x-direction
        stencil_coefs[3*r+1] = a * cy[r];  // y-direction
        stencil_coefs[3*r+2] = a * cz[r];  // z-direction
    }
    double coef_0 = diag_coeff;

    // Allocate arrays
    size_t alloc_ex = (size_t)ncol * nd_ex * sizeof(double);
    size_t alloc_y  = (size_t)ncol * nd * sizeof(double);
    double* x_ex  = (double*)aligned_alloc(64, alloc_ex);
    double* y_ref = (double*)aligned_alloc(64, alloc_y);
    double* y_out = (double*)aligned_alloc(64, alloc_y);

    // Fill with random data
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < (size_t)ncol * nd_ex; ++i) x_ex[i] = dist(rng);

    // Reference: Version A
    laplacian_lynx(x_ex, y_ref, cx, cy, cz, diag_coeff, a, nx, ny, nz, nx_ex, ny_ex, nd, nd_ex, ncol);

    // Define versions to benchmark
    struct Version {
        const char* name;
        void (*fn_lynx)(const double*, double*, const double*, const double*, const double*,
                        double, double, int, int, int, int, int, int, int, int);
        void (*fn_sparc)(const double*, double*, const double*, double,
                         int, int, int, int, int, int, int, int, int, int);
    };

    // Benchmark each version
    auto bench_one = [&](const char* name, auto fn, auto... args) -> BenchResult {
        // Warm up
        fn(args...);
        // Correctness
        double max_err = 0.0;
        for (size_t i = 0; i < (size_t)ncol * nd; ++i)
            max_err = std::max(max_err, std::abs(y_out[i] - y_ref[i]));
        // Time
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < nreps; ++r) fn(args...);
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double, std::milli>(t1 - t0).count() / nreps;
        return {name, dt, max_err};
    };

    BenchResult results[6];

    // A: LYNX current
    results[0] = bench_one("A: LYNX current",
        laplacian_lynx, x_ex, y_out, cx, cy, cz, diag_coeff, a,
        nx, ny, nz, nx_ex, ny_ex, nd, nd_ex, ncol);

    // B: SPARC simd-only
    results[1] = bench_one("B: SPARC simd",
        laplacian_sparc_simd, x_ex, y_out, stencil_coefs, coef_0,
        nx, ny, nz, stride_y, stride_y_ex, stride_z, stride_z_ex, nd, nd_ex, ncol);

    // C: SPARC+OMP cols
    results[2] = bench_one("C: SPARC+OMP cols",
        laplacian_sparc_omp_cols, x_ex, y_out, stencil_coefs, coef_0,
        nx, ny, nz, stride_y, stride_y_ex, stride_z, stride_z_ex, nd, nd_ex, ncol);

    // D: Flat OMP+SIMD
    results[3] = bench_one("D: Flat OMP+SIMD",
        laplacian_flat_omp, x_ex, y_out, stencil_coefs, coef_0,
        nx, ny, nz, stride_y, stride_y_ex, stride_z, stride_z_ex, nd, nd_ex, ncol);

    // E: SPARC+AVX2 (single-threaded)
    results[4] = bench_one("E: SPARC+AVX2",
        laplacian_sparc_avx2, x_ex, y_out, stencil_coefs, coef_0,
        nx, ny, nz, stride_y, stride_y_ex, stride_z, stride_z_ex, nd, nd_ex, ncol);

    // E+OMP: AVX2 + OMP
    results[5] = bench_one("E+OMP: AVX2+OMP",
        laplacian_sparc_avx2_omp, x_ex, y_out, stencil_coefs, coef_0,
        nx, ny, nz, stride_y, stride_y_ex, stride_z, stride_z_ex, nd, nd_ex, ncol);

    // Bandwidth and FLOP calculation
    // Per point: reads center + 2*FDn*3 neighbors = 37 doubles read, 1 write = 38 * 8 bytes
    // FLOPs per point: 1 mul (center) + FDn*(1 add + 1 fma)*3 = 1 + 6*2*3 = 37
    //   Actually: center mul + per radius: 3 adds(pairs) + 3 fma = 6 flops, so 6*6=36+1=37
    double bytes_per_call = (double)ncol * (1.0 + 2.0 * FDn * 3.0 + 1.0) * nd * 8.0;
    double flops_per_call = (double)ncol * (1.0 + FDn * 6.0) * nd;

    printf("%-20s %10s %10s %10s %10s %8s\n",
           "Version", "Time(ms)", "Speedup", "GB/s", "GFLOPS", "Err");
    printf("%-20s %10s %10s %10s %10s %8s\n",
           "-------", "--------", "-------", "----", "------", "---");
    double base_time = results[0].time_ms;
    for (int i = 0; i < 6; i++) {
        auto& r = results[i];
        printf("%-20s %10.4f %9.2fx %10.2f %10.2f %8.1e %s\n",
               r.name, r.time_ms, base_time / r.time_ms,
               bytes_per_call / r.time_ms / 1e6,
               flops_per_call / r.time_ms / 1e6,
               r.max_err,
               r.max_err < 1e-14 ? "PASS" : (r.max_err < 1e-12 ? "WARN" : "FAIL"));
    }
    printf("\n");

    free(x_ex); free(y_ref); free(y_out);
}

int main() {
    printf("================================================================\n");
    printf("  SPARC vs LYNX Laplacian Stencil Benchmark\n");
    printf("================================================================\n\n");

    // Small grid (Si4-like)
    run_bench("Si4 (25x26x27, ncol=40)", 25, 26, 27, 40, 100);

    // Large grid
    run_bench("Large (48x48x48, ncol=80)", 48, 48, 48, 80, 50);

    return 0;
}
