// Standalone benchmark: Nonlocal projector (Chi^T * psi gather + Chi * alpha scatter)
// Compares BASELINE vs OPTIMIZED (BLAS-like blocked gather + restrict)
// From src/operators/NonlocalProjector.cpp

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <omp.h>

// ==================== BASELINE ====================
// Direct loop implementation matching LYNX original
static void nonlocal_baseline(const double* __restrict__ psi,
                              double* __restrict__ Hpsi,
                              const double* __restrict__ Chi,
                              const int* __restrict__ gpos,
                              const double* __restrict__ Gamma,
                              int Nd_d, int ncol,
                              int ndc, int nproj, int n_atom, double dV)
{
    std::vector<double> alpha(nproj * ncol * n_atom, 0.0);

    for (int iat = 0; iat < n_atom; ++iat) {
        double* a_off = alpha.data() + iat * nproj * ncol;

        #pragma omp parallel for schedule(static)
        for (int n = 0; n < ncol; ++n) {
            const double* psi_n = psi + n * Nd_d;
            for (int jp = 0; jp < nproj; ++jp) {
                double dot = 0.0;
                for (int ig = 0; ig < ndc; ++ig)
                    dot += Chi[ig + jp * ndc] * psi_n[gpos[ig]];
                a_off[n * nproj + jp] = dot * dV;
            }
        }
    }

    for (int iat = 0; iat < n_atom; ++iat) {
        double* a_off = alpha.data() + iat * nproj * ncol;
        for (int n = 0; n < ncol; ++n)
            for (int jp = 0; jp < nproj; ++jp)
                a_off[n * nproj + jp] *= Gamma[jp];
    }

    for (int iat = 0; iat < n_atom; ++iat) {
        const double* a_off = alpha.data() + iat * nproj * ncol;

        #pragma omp parallel for schedule(static)
        for (int n = 0; n < ncol; ++n) {
            double* Hpsi_n = Hpsi + n * Nd_d;
            for (int ig = 0; ig < ndc; ++ig) {
                double val = 0.0;
                for (int jp = 0; jp < nproj; ++jp)
                    val += Chi[ig + jp * ndc] * a_off[n * nproj + jp];
                Hpsi_n[gpos[ig]] += val;
            }
        }
    }
}

// ==================== OPTIMIZED ====================
// 1. Gather psi into contiguous buffer before GEMV (cache-friendly)
// 2. Loop reorder: proj-major inner loop for better vectorization
// 3. Fused Gamma scaling into scatter
static void nonlocal_optimized(const double* __restrict__ psi,
                               double* __restrict__ Hpsi,
                               const double* __restrict__ Chi,
                               const int* __restrict__ gpos,
                               const double* __restrict__ Gamma,
                               int Nd_d, int ncol,
                               int ndc, int nproj, int n_atom, double dV)
{
    for (int iat = 0; iat < n_atom; ++iat) {
        // Gather + inner product + Gamma scale + scatter, per atom
        // alpha[nproj x ncol] = dV * Chi^T * psi_gathered

        #pragma omp parallel
        {
            std::vector<double> psi_buf(ndc);  // thread-local gather buffer

            #pragma omp for schedule(static)
            for (int n = 0; n < ncol; ++n) {
                const double* __restrict__ psi_n = psi + n * Nd_d;
                double* __restrict__ Hpsi_n = Hpsi + n * Nd_d;

                // Gather psi into contiguous buffer
                for (int ig = 0; ig < ndc; ++ig)
                    psi_buf[ig] = psi_n[gpos[ig]];

                // Inner product: alpha[jp] = dV * Chi[:,jp] . psi_buf
                double alpha_local[32];  // nproj <= 32 for all practical cases
                for (int jp = 0; jp < nproj; ++jp) {
                    const double* __restrict__ chi_jp = Chi + jp * ndc;
                    double dot = 0.0;
                    #pragma omp simd reduction(+:dot)
                    for (int ig = 0; ig < ndc; ++ig)
                        dot += chi_jp[ig] * psi_buf[ig];
                    alpha_local[jp] = dot * dV * Gamma[jp];  // fused Gamma scale
                }

                // Scatter: Hpsi[gpos[ig]] += Chi[ig,:] . alpha
                for (int ig = 0; ig < ndc; ++ig) {
                    double val = 0.0;
                    for (int jp = 0; jp < nproj; ++jp)
                        val += Chi[ig + jp * ndc] * alpha_local[jp];
                    Hpsi_n[gpos[ig]] += val;
                }
            }
        }
    }
}

typedef void (*vnl_fn)(const double*, double*, const double*, const int*,
                       const double*, int, int, int, int, int, double);

static double bench(vnl_fn fn, const double* psi, double* Hpsi,
                    const double* Chi, const int* gpos, const double* Gamma,
                    int Nd_d, int ncol, int ndc, int nproj, int n_atom, double dV,
                    int nreps) {
    memset(Hpsi, 0, (size_t)Nd_d * ncol * sizeof(double));
    fn(psi, Hpsi, Chi, gpos, Gamma, Nd_d, ncol, ndc, nproj, n_atom, dV);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < nreps; rep++) {
        memset(Hpsi, 0, (size_t)Nd_d * ncol * sizeof(double));
        fn(psi, Hpsi, Chi, gpos, Gamma, Nd_d, ncol, ndc, nproj, n_atom, dV);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / nreps;
}

int main(int argc, char** argv) {
    int Nd_d = 25 * 26 * 27, ncol = 40, ndc = 200, nproj = 9, n_atom = 4;
    int NREPS = 500;
    double dV = 0.064;

    if (argc > 1) ncol = atoi(argv[1]);
    if (argc > 2) ndc = atoi(argv[2]);
    if (argc > 3) NREPS = atoi(argv[3]);

    printf("=== Nonlocal Projector Benchmark ===\n");
    printf("Nd_d=%d, ncol=%d, ndc=%d, nproj=%d, n_atom=%d\n",
           Nd_d, ncol, ndc, nproj, n_atom);
    printf("Threads: %d\n", omp_get_max_threads());

    double* psi     = (double*)aligned_alloc(64, (size_t)Nd_d * ncol * sizeof(double));
    double* Hpsi_b  = (double*)aligned_alloc(64, (size_t)Nd_d * ncol * sizeof(double));
    double* Hpsi_o  = (double*)aligned_alloc(64, (size_t)Nd_d * ncol * sizeof(double));
    double* Chi     = (double*)aligned_alloc(64, (size_t)ndc * nproj * sizeof(double));
    int* gpos       = (int*)aligned_alloc(64, ndc * sizeof(int));
    double* Gamma   = (double*)aligned_alloc(64, nproj * sizeof(double));

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (long i = 0; i < (long)Nd_d * ncol; i++) psi[i] = dist(rng);
    for (int i = 0; i < ndc * nproj; i++) Chi[i] = dist(rng);
    for (int i = 0; i < nproj; i++) Gamma[i] = dist(rng);
    std::uniform_int_distribution<int> idist(0, Nd_d - 1);
    for (int i = 0; i < ndc; i++) gpos[i] = idist(rng);

    double ms_base = bench(nonlocal_baseline, psi, Hpsi_b, Chi, gpos, Gamma,
                           Nd_d, ncol, ndc, nproj, n_atom, dV, NREPS);
    double ms_opt  = bench(nonlocal_optimized, psi, Hpsi_o, Chi, gpos, Gamma,
                           Nd_d, ncol, ndc, nproj, n_atom, dV, NREPS);

    double max_diff = 0.0;
    for (long i = 0; i < (long)Nd_d * ncol; i++)
        max_diff = std::max(max_diff, std::abs(Hpsi_b[i] - Hpsi_o[i]));

    double total_flops = 2.0 * n_atom * ncol * nproj * ndc * 2.0;
    printf("\n%-12s %10s %10s %10s\n", "Version", "Time(ms)", "GFLOPS", "Speedup");
    printf("%-12s %10.4f %10.2f %10s\n", "Baseline", ms_base,
           total_flops/(ms_base*1e6), "1.00x");
    printf("%-12s %10.4f %10.2f %9.2fx\n", "Optimized", ms_opt,
           total_flops/(ms_opt*1e6), ms_base/ms_opt);
    printf("\nMax |diff|: %.2e (should be < 1e-10)\n", max_diff);

    free(psi); free(Hpsi_b); free(Hpsi_o); free(Chi); free(gpos); free(Gamma);
    return 0;
}
