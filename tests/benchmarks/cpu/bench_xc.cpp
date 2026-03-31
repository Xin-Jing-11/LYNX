// Standalone benchmark: XC functional evaluation (density -> exc, Vxc)
// Compares BASELINE (serial libxc) vs OPTIMIZED (thread-parallel libxc)
// From src/xc/XCFunctional.cpp

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <xc.h>
#include <xc_funcs.h>

// ==================== BASELINE (original LYNX) ====================
// Single-threaded libxc calls, OMP only on combine loop
static void xc_lda_baseline(const double* rho, double* Vxc, double* exc, int Nd_d) {
    xc_func_type func_x, func_c;
    xc_func_init(&func_x, XC_LDA_X, XC_UNPOLARIZED);
    xc_func_init(&func_c, XC_LDA_C_PW, XC_UNPOLARIZED);

    std::vector<double> zk_x(Nd_d), vrho_x(Nd_d);
    std::vector<double> zk_c(Nd_d), vrho_c(Nd_d);

    xc_lda_exc_vxc(&func_x, Nd_d, rho, zk_x.data(), vrho_x.data());
    xc_lda_exc_vxc(&func_c, Nd_d, rho, zk_c.data(), vrho_c.data());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Nd_d; i++) {
        exc[i] = zk_x[i] + zk_c[i];
        Vxc[i] = vrho_x[i] + vrho_c[i];
    }

    xc_func_end(&func_x);
    xc_func_end(&func_c);
}

static void xc_gga_baseline(const double* rho, const double* sigma,
                             double* Vxc, double* exc, double* Dxcdgrho, int Nd_d) {
    xc_func_type func_x, func_c;
    xc_func_init(&func_x, XC_GGA_X_PBE, XC_UNPOLARIZED);
    xc_func_init(&func_c, XC_GGA_C_PBE, XC_UNPOLARIZED);

    std::vector<double> zk_x(Nd_d), vrho_x(Nd_d), vsigma_x(Nd_d);
    std::vector<double> zk_c(Nd_d), vrho_c(Nd_d), vsigma_c(Nd_d);

    xc_gga_exc_vxc(&func_x, Nd_d, rho, sigma, zk_x.data(), vrho_x.data(), vsigma_x.data());
    xc_gga_exc_vxc(&func_c, Nd_d, rho, sigma, zk_c.data(), vrho_c.data(), vsigma_c.data());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Nd_d; i++) {
        exc[i] = zk_x[i] + zk_c[i];
        Vxc[i] = vrho_x[i] + vrho_c[i];
        Dxcdgrho[i] = 2.0 * (vsigma_x[i] + vsigma_c[i]);
    }

    xc_func_end(&func_x);
    xc_func_end(&func_c);
}

// ==================== OPTIMIZED ====================
// Each OMP thread creates its own xc_func_type and processes a chunk
static void xc_lda_optimized(const double* rho, double* Vxc, double* exc, int Nd_d) {
    #pragma omp parallel
    {
        xc_func_type tfx, tfc;
        xc_func_init(&tfx, XC_LDA_X, XC_UNPOLARIZED);
        xc_func_init(&tfc, XC_LDA_C_PW, XC_UNPOLARIZED);

        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int chunk = (Nd_d + nt - 1) / nt;
        int start = tid * chunk;
        int end = std::min(start + chunk, Nd_d);
        int len = end - start;

        if (len > 0) {
            std::vector<double> zk_x(len), vrho_x(len);
            std::vector<double> zk_c(len), vrho_c(len);

            xc_lda_exc_vxc(&tfx, len, rho + start, zk_x.data(), vrho_x.data());
            xc_lda_exc_vxc(&tfc, len, rho + start, zk_c.data(), vrho_c.data());

            for (int i = 0; i < len; i++) {
                exc[start + i] = zk_x[i] + zk_c[i];
                Vxc[start + i] = vrho_x[i] + vrho_c[i];
            }
        }

        xc_func_end(&tfx);
        xc_func_end(&tfc);
    }
}

static void xc_gga_optimized(const double* rho, const double* sigma,
                              double* Vxc, double* exc, double* Dxcdgrho, int Nd_d) {
    #pragma omp parallel
    {
        xc_func_type tfx, tfc;
        xc_func_init(&tfx, XC_GGA_X_PBE, XC_UNPOLARIZED);
        xc_func_init(&tfc, XC_GGA_C_PBE, XC_UNPOLARIZED);

        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int chunk = (Nd_d + nt - 1) / nt;
        int start = tid * chunk;
        int end = std::min(start + chunk, Nd_d);
        int len = end - start;

        if (len > 0) {
            std::vector<double> zk_x(len), vrho_x(len), vsigma_x(len);
            std::vector<double> zk_c(len), vrho_c(len), vsigma_c(len);

            xc_gga_exc_vxc(&tfx, len, rho + start, sigma + start,
                           zk_x.data(), vrho_x.data(), vsigma_x.data());
            xc_gga_exc_vxc(&tfc, len, rho + start, sigma + start,
                           zk_c.data(), vrho_c.data(), vsigma_c.data());

            for (int i = 0; i < len; i++) {
                exc[start + i] = zk_x[i] + zk_c[i];
                Vxc[start + i] = vrho_x[i] + vrho_c[i];
                Dxcdgrho[start + i] = 2.0 * (vsigma_x[i] + vsigma_c[i]);
            }
        }

        xc_func_end(&tfx);
        xc_func_end(&tfc);
    }
}

static void compute_sigma(const double* Dx, const double* Dy, const double* Dz,
                          double* sigma, int Nd_d) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Nd_d; i++) {
        sigma[i] = Dx[i]*Dx[i] + Dy[i]*Dy[i] + Dz[i]*Dz[i];
        if (sigma[i] < 1e-14) sigma[i] = 1e-14;
    }
}

int main(int argc, char** argv) {
    int Nd_d = 25 * 26 * 27;
    int NREPS = 200;

    if (argc > 1) Nd_d = atoi(argv[1]);
    if (argc > 2) NREPS = atoi(argv[2]);

    printf("=== XC Functional Evaluation Benchmark ===\n");
    printf("Nd_d=%d\n", Nd_d);
    printf("Threads: %d\n", omp_get_max_threads());

    double* rho     = (double*)aligned_alloc(64, Nd_d * sizeof(double));
    double* sigma   = (double*)aligned_alloc(64, Nd_d * sizeof(double));
    double* Vxc_b   = (double*)aligned_alloc(64, Nd_d * sizeof(double));
    double* Vxc_o   = (double*)aligned_alloc(64, Nd_d * sizeof(double));
    double* exc_b   = (double*)aligned_alloc(64, Nd_d * sizeof(double));
    double* exc_o   = (double*)aligned_alloc(64, Nd_d * sizeof(double));
    double* Dxc_b   = (double*)aligned_alloc(64, Nd_d * sizeof(double));
    double* Dxc_o   = (double*)aligned_alloc(64, Nd_d * sizeof(double));
    double* Drho_x  = (double*)aligned_alloc(64, Nd_d * sizeof(double));
    double* Drho_y  = (double*)aligned_alloc(64, Nd_d * sizeof(double));
    double* Drho_z  = (double*)aligned_alloc(64, Nd_d * sizeof(double));

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.001, 2.0);
    std::uniform_real_distribution<double> gdist(-1.0, 1.0);
    for (int i = 0; i < Nd_d; i++) {
        rho[i] = dist(rng);
        Drho_x[i] = gdist(rng); Drho_y[i] = gdist(rng); Drho_z[i] = gdist(rng);
    }
    compute_sigma(Drho_x, Drho_y, Drho_z, sigma, Nd_d);

    // ---- LDA ----
    {
        xc_lda_baseline(rho, Vxc_b, exc_b, Nd_d);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < NREPS; rep++)
            xc_lda_baseline(rho, Vxc_b, exc_b, Nd_d);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_base = std::chrono::duration<double, std::milli>(t1 - t0).count() / NREPS;

        xc_lda_optimized(rho, Vxc_o, exc_o, Nd_d);
        t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < NREPS; rep++)
            xc_lda_optimized(rho, Vxc_o, exc_o, Nd_d);
        t1 = std::chrono::high_resolution_clock::now();
        double ms_opt = std::chrono::duration<double, std::milli>(t1 - t0).count() / NREPS;

        double max_diff = 0.0;
        for (int i = 0; i < Nd_d; i++) {
            max_diff = std::max(max_diff, std::abs(Vxc_b[i] - Vxc_o[i]));
            max_diff = std::max(max_diff, std::abs(exc_b[i] - exc_o[i]));
        }

        printf("\n--- LDA (PW) ---\n");
        printf("%-12s %10s %10s\n", "Version", "Time(ms)", "Speedup");
        printf("%-12s %10.3f %10s\n", "Baseline", ms_base, "1.00x");
        printf("%-12s %10.3f %9.2fx\n", "Optimized", ms_opt, ms_base/ms_opt);
        printf("Max |diff|: %.2e\n", max_diff);
    }

    // ---- GGA (PBE) ----
    {
        xc_gga_baseline(rho, sigma, Vxc_b, exc_b, Dxc_b, Nd_d);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < NREPS; rep++)
            xc_gga_baseline(rho, sigma, Vxc_b, exc_b, Dxc_b, Nd_d);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_base = std::chrono::duration<double, std::milli>(t1 - t0).count() / NREPS;

        xc_gga_optimized(rho, sigma, Vxc_o, exc_o, Dxc_o, Nd_d);
        t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < NREPS; rep++)
            xc_gga_optimized(rho, sigma, Vxc_o, exc_o, Dxc_o, Nd_d);
        t1 = std::chrono::high_resolution_clock::now();
        double ms_opt = std::chrono::duration<double, std::milli>(t1 - t0).count() / NREPS;

        double max_diff = 0.0;
        for (int i = 0; i < Nd_d; i++) {
            max_diff = std::max(max_diff, std::abs(Vxc_b[i] - Vxc_o[i]));
            max_diff = std::max(max_diff, std::abs(exc_b[i] - exc_o[i]));
            max_diff = std::max(max_diff, std::abs(Dxc_b[i] - Dxc_o[i]));
        }

        printf("\n--- GGA (PBE) ---\n");
        printf("%-12s %10s %10s\n", "Version", "Time(ms)", "Speedup");
        printf("%-12s %10.3f %10s\n", "Baseline", ms_base, "1.00x");
        printf("%-12s %10.3f %9.2fx\n", "Optimized", ms_opt, ms_base/ms_opt);
        printf("Max |diff|: %.2e\n", max_diff);
    }

    free(rho); free(sigma); free(Vxc_b); free(Vxc_o); free(exc_b); free(exc_o);
    free(Dxc_b); free(Dxc_o); free(Drho_x); free(Drho_y); free(Drho_z);
    return 0;
}
