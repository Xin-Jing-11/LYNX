// Benchmark: SPARC vs LYNX Gradient Stencil Patterns (multi-column, multi-thread)
//
// Gradient uses antisymmetric stencil: coef[p] * (in[i+p*stride] - in[i-p*stride])
//
// Versions:
//   A: LYNX current    — separate coeff array, no SIMD hint, recompute idx
//   B: SPARC simd-only — pre-computed offsets, #pragma omp simd on x-loop
//   C: SPARC+OMP cols  — B + parallel for on columns
//   D: Flat OMP+SIMD   — flattened ncol*nz + simd x-loop
//   E: SPARC+AVX2      — explicit _mm256_fmadd_pd
//
// Compile:
//   g++ -O3 -march=native -mavx2 -mfma -fopenmp -o bench_sparc_gradient bench_sparc_gradient.cpp
//
// Run:
//   for t in 1 2 4 8; do OMP_NUM_THREADS=$t ./bench_sparc_gradient; done

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
// Version A: LYNX-style gradient (x-direction)
// ============================================================
__attribute__((noinline))
void gradient_lynx(const double* __restrict__ x_ex, double* __restrict__ Dx,
                   const double* D1_coeff, double c,
                   int nx, int ny, int nz, int nx_ex, int ny_ex,
                   int nd, int nd_ex, int ncol) {
    int nxny = nx * ny;
    int nxny_ex = nx_ex * ny_ex;
    for (int n = 0; n < ncol; ++n) {
        const double* xn = x_ex + n * nd_ex;
        double* dxn = Dx + n * nd;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    // For x-direction gradient: extended only in x
                    int idx = (i + FDn) + j * nx_ex + k * nxny_ex;
                    int loc = i + j * nx + k * nxny;
                    double val = c * xn[idx];
                    for (int p = 1; p <= FDn; ++p) {
                        val += D1_coeff[p] * (xn[idx + p] - xn[idx - p]);
                    }
                    dxn[loc] = val;
                }
            }
        }
    }
}

// ============================================================
// Version B: SPARC-style gradient (x-direction, simd on x-loop)
// ============================================================
__attribute__((noinline))
void gradient_sparc_simd(const double* __restrict__ x_ex, double* __restrict__ Dx,
                         const double* D1_coeff, double c, int stride_X,
                         int nx, int ny, int nz,
                         int stride_y, int stride_y_ex,
                         int stride_z, int stride_z_ex,
                         int nd, int nd_ex, int ncol) {
    for (int n = 0; n < ncol; ++n) {
        const double* xn = x_ex + n * nd_ex;
        double* dxn = Dx + n * nd;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                int offset = k * stride_z + j * stride_y;
                int offset_ex = k * stride_z_ex + j * stride_y_ex;
                #pragma omp simd
                for (int i = 0; i < nx; ++i) {
                    int idx_ex = offset_ex + i + FDn;
                    double temp = xn[idx_ex] * c;
                    for (int r = 1; r <= 6; r++) {
                        int stride_r = r * stride_X;
                        temp += (xn[idx_ex + stride_r] - xn[idx_ex - stride_r]) * D1_coeff[r];
                    }
                    dxn[offset + i] = temp;
                }
            }
        }
    }
}

// ============================================================
// Version C: SPARC + OMP parallel for on columns
// ============================================================
__attribute__((noinline))
void gradient_sparc_omp_cols(const double* __restrict__ x_ex, double* __restrict__ Dx,
                             const double* D1_coeff, double c, int stride_X,
                             int nx, int ny, int nz,
                             int stride_y, int stride_y_ex,
                             int stride_z, int stride_z_ex,
                             int nd, int nd_ex, int ncol) {
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < ncol; ++n) {
        const double* xn = x_ex + n * nd_ex;
        double* dxn = Dx + n * nd;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                int offset = k * stride_z + j * stride_y;
                int offset_ex = k * stride_z_ex + j * stride_y_ex;
                #pragma omp simd
                for (int i = 0; i < nx; ++i) {
                    int idx_ex = offset_ex + i + FDn;
                    double temp = xn[idx_ex] * c;
                    for (int r = 1; r <= 6; r++) {
                        int stride_r = r * stride_X;
                        temp += (xn[idx_ex + stride_r] - xn[idx_ex - stride_r]) * D1_coeff[r];
                    }
                    dxn[offset + i] = temp;
                }
            }
        }
    }
}

// ============================================================
// Version D: Flattened OMP on ncol*nz + SIMD x
// ============================================================
__attribute__((noinline))
void gradient_flat_omp(const double* __restrict__ x_ex, double* __restrict__ Dx,
                       const double* D1_coeff, double c, int stride_X,
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
        double* dxn = Dx + n * nd;
        for (int j = 0; j < ny; ++j) {
            int offset = k * stride_z + j * stride_y;
            int offset_ex = k * stride_z_ex + j * stride_y_ex;
            #pragma omp simd
            for (int i = 0; i < nx; ++i) {
                int idx_ex = offset_ex + i + FDn;
                double temp = xn[idx_ex] * c;
                for (int r = 1; r <= 6; r++) {
                    int stride_r = r * stride_X;
                    temp += (xn[idx_ex + stride_r] - xn[idx_ex - stride_r]) * D1_coeff[r];
                }
                dxn[offset + i] = temp;
            }
        }
    }
}

// ============================================================
// Version E: SPARC + AVX2 intrinsics on x-loop
// ============================================================
__attribute__((noinline))
void gradient_sparc_avx2(const double* __restrict__ x_ex, double* __restrict__ Dx,
                         const double* D1_coeff, double c, int stride_X,
                         int nx, int ny, int nz,
                         int stride_y, int stride_y_ex,
                         int stride_z, int stride_z_ex,
                         int nd, int nd_ex, int ncol) {
    __m256d vc = _mm256_set1_pd(c);
    __m256d vcoef[7];
    for (int r = 1; r <= 6; r++)
        vcoef[r] = _mm256_set1_pd(D1_coeff[r]);
    int nx4 = nx & ~3;

    for (int n = 0; n < ncol; ++n) {
        const double* xn = x_ex + n * nd_ex;
        double* dxn = Dx + n * nd;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                int offset = k * stride_z + j * stride_y;
                int offset_ex = k * stride_z_ex + j * stride_y_ex + FDn;
                int i = 0;
                for (; i < nx4; i += 4) {
                    int idx_ex = offset_ex + i;
                    __m256d val = _mm256_mul_pd(vc, _mm256_loadu_pd(&xn[idx_ex]));
                    for (int r = 1; r <= 6; r++) {
                        int sr = r * stride_X;
                        __m256d diff = _mm256_sub_pd(
                            _mm256_loadu_pd(&xn[idx_ex + sr]),
                            _mm256_loadu_pd(&xn[idx_ex - sr]));
                        val = _mm256_fmadd_pd(vcoef[r], diff, val);
                    }
                    _mm256_storeu_pd(&dxn[offset + i], val);
                }
                for (; i < nx; ++i) {
                    int idx_ex = offset_ex + i;
                    double temp = xn[idx_ex] * c;
                    for (int r = 1; r <= 6; r++) {
                        int sr = r * stride_X;
                        temp += (xn[idx_ex + sr] - xn[idx_ex - sr]) * D1_coeff[r];
                    }
                    dxn[offset + i] = temp;
                }
            }
        }
    }
}

// ============================================================
// Version E+OMP: AVX2 + OMP
// ============================================================
__attribute__((noinline))
void gradient_sparc_avx2_omp(const double* __restrict__ x_ex, double* __restrict__ Dx,
                             const double* D1_coeff, double c, int stride_X,
                             int nx, int ny, int nz,
                             int stride_y, int stride_y_ex,
                             int stride_z, int stride_z_ex,
                             int nd, int nd_ex, int ncol) {
    __m256d vc = _mm256_set1_pd(c);
    __m256d vcoef[7];
    for (int r = 1; r <= 6; r++)
        vcoef[r] = _mm256_set1_pd(D1_coeff[r]);
    int nx4 = nx & ~3;

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < ncol; ++n) {
        const double* xn = x_ex + n * nd_ex;
        double* dxn = Dx + n * nd;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                int offset = k * stride_z + j * stride_y;
                int offset_ex = k * stride_z_ex + j * stride_y_ex + FDn;
                int i = 0;
                for (; i < nx4; i += 4) {
                    int idx_ex = offset_ex + i;
                    __m256d val = _mm256_mul_pd(vc, _mm256_loadu_pd(&xn[idx_ex]));
                    for (int r = 1; r <= 6; r++) {
                        int sr = r * stride_X;
                        val = _mm256_fmadd_pd(vcoef[r], _mm256_sub_pd(
                            _mm256_loadu_pd(&xn[idx_ex + sr]),
                            _mm256_loadu_pd(&xn[idx_ex - sr])), val);
                    }
                    _mm256_storeu_pd(&dxn[offset + i], val);
                }
                for (; i < nx; ++i) {
                    int idx_ex = offset_ex + i;
                    double temp = xn[idx_ex] * c;
                    for (int r = 1; r <= 6; r++) {
                        int sr = r * stride_X;
                        temp += (xn[idx_ex + sr] - xn[idx_ex - sr]) * D1_coeff[r];
                    }
                    dxn[offset + i] = temp;
                }
            }
        }
    }
}

// ============================================================
struct BenchResult {
    const char* name;
    double time_ms;
    double max_err;
};

void run_bench(const char* grid_label, int nx, int ny, int nz, int ncol, int nreps) {
    // For x-direction gradient: extended only in x
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny;
    int nz_ex = nz;
    int nxny = nx * ny;
    int nxny_ex = nx_ex * ny_ex;
    int nd = nx * ny * nz;
    int nd_ex = nx_ex * ny_ex * nz_ex;
    int stride_y = nx, stride_y_ex = nx_ex;
    int stride_z = nxny, stride_z_ex = nxny_ex;
    int stride_X = 1; // x-direction stride in extended array
    int nthreads = 1;
    #pragma omp parallel
    { nthreads = omp_get_num_threads(); }

    printf("=== Gradient (x-dir) Stencil Benchmark: %s ===\n", grid_label);
    printf("Grid: %d x %d x %d = %d pts, ncol=%d, FDn=%d\n", nx, ny, nz, nd, ncol, FDn);
    printf("Threads: %d, Reps: %d\n\n", nthreads, nreps);

    // D1 coefficients (typical 12th order central difference, antisymmetric)
    double D1_coeff[FDn + 1] = {0.0, 0.8, -0.2, 0.038095, -0.0035714, -0.00025253, 0.0000125};
    double c = 0.0; // no diagonal shift for pure gradient

    size_t alloc_ex = (size_t)ncol * nd_ex * sizeof(double);
    size_t alloc_y  = (size_t)ncol * nd * sizeof(double);
    double* x_ex  = (double*)aligned_alloc(64, alloc_ex);
    double* y_ref = (double*)aligned_alloc(64, alloc_y);
    double* y_out = (double*)aligned_alloc(64, alloc_y);

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < (size_t)ncol * nd_ex; ++i) x_ex[i] = dist(rng);

    // Reference
    gradient_lynx(x_ex, y_ref, D1_coeff, c, nx, ny, nz, nx_ex, ny_ex, nd, nd_ex, ncol);

    auto bench_one = [&](const char* name, auto fn, auto... args) -> BenchResult {
        fn(args...);
        double max_err = 0.0;
        for (size_t i = 0; i < (size_t)ncol * nd; ++i)
            max_err = std::max(max_err, std::abs(y_out[i] - y_ref[i]));
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < nreps; ++r) fn(args...);
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double, std::milli>(t1 - t0).count() / nreps;
        return {name, dt, max_err};
    };

    BenchResult results[6];

    results[0] = bench_one("A: LYNX current",
        gradient_lynx, x_ex, y_out, D1_coeff, c,
        nx, ny, nz, nx_ex, ny_ex, nd, nd_ex, ncol);

    results[1] = bench_one("B: SPARC simd",
        gradient_sparc_simd, x_ex, y_out, D1_coeff, c, stride_X,
        nx, ny, nz, stride_y, stride_y_ex, stride_z, stride_z_ex, nd, nd_ex, ncol);

    results[2] = bench_one("C: SPARC+OMP cols",
        gradient_sparc_omp_cols, x_ex, y_out, D1_coeff, c, stride_X,
        nx, ny, nz, stride_y, stride_y_ex, stride_z, stride_z_ex, nd, nd_ex, ncol);

    results[3] = bench_one("D: Flat OMP+SIMD",
        gradient_flat_omp, x_ex, y_out, D1_coeff, c, stride_X,
        nx, ny, nz, stride_y, stride_y_ex, stride_z, stride_z_ex, nd, nd_ex, ncol);

    results[4] = bench_one("E: SPARC+AVX2",
        gradient_sparc_avx2, x_ex, y_out, D1_coeff, c, stride_X,
        nx, ny, nz, stride_y, stride_y_ex, stride_z, stride_z_ex, nd, nd_ex, ncol);

    results[5] = bench_one("E+OMP: AVX2+OMP",
        gradient_sparc_avx2_omp, x_ex, y_out, D1_coeff, c, stride_X,
        nx, ny, nz, stride_y, stride_y_ex, stride_z, stride_z_ex, nd, nd_ex, ncol);

    // Gradient: reads center + 2*FDn neighbors, writes 1
    double bytes_per_call = (double)ncol * (1.0 + 2.0 * FDn + 1.0) * nd * 8.0;
    double flops_per_call = (double)ncol * (1.0 + FDn * 3.0) * nd; // mul + FDn*(sub + fma)

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
    printf("  SPARC vs LYNX Gradient Stencil Benchmark (x-direction)\n");
    printf("================================================================\n\n");

    run_bench("Si4 (25x26x27, ncol=40)", 25, 26, 27, 40, 100);
    run_bench("Large (48x48x48, ncol=80)", 48, 48, 48, 80, 50);

    return 0;
}
