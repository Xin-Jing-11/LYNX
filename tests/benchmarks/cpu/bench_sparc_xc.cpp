// Benchmark: Serial vs Thread-Parallel libxc evaluation
//
// Compares:
//   A: Serial libxc    — single call to xc_*_exc_vxc on full array (SPARC style)
//   B: OMP chunked     — parallel for with per-thread xc_func_type, chunk evaluation
//   C: OMP per-point   — parallel for where each thread calls libxc on a chunk
//
// Tests LDA (Slater X + PW C), GGA (PBE), and mGGA (SCAN) functionals
//
// Compile:
//   g++ -O3 -march=native -fopenmp \
//       -I/home/xx/Desktop/LYNX/external/libxc/src \
//       -I/home/xx/Desktop/LYNX/build/external/libxc \
//       -o bench_sparc_xc bench_sparc_xc.cpp \
//       /home/xx/Desktop/LYNX/build/external/libxc/libxc.a -lm
//
// Run:
//   for t in 1 2 4 8; do OMP_NUM_THREADS=$t ./bench_sparc_xc; done

#include <omp.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>

#include "xc.h"
#include "xc_funcs.h"

struct BenchResult {
    const char* name;
    double time_ms;
    double max_err_exc;
    double max_err_vrho;
};

// ============================================================
// LDA benchmark
// ============================================================
void bench_lda(int Nd, int nreps) {
    int nthreads = 1;
    #pragma omp parallel
    { nthreads = omp_get_num_threads(); }

    printf("--- LDA (Slater X + PW C), Nd=%d, threads=%d, reps=%d ---\n", Nd, nthreads, nreps);

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.01, 2.0);
    std::vector<double> rho(Nd);
    for (int i = 0; i < Nd; i++) rho[i] = dist(rng);

    // Reference: serial
    std::vector<double> exc_ref(Nd), vrho_ref(Nd);
    {
        xc_func_type func_x, func_c;
        xc_func_init(&func_x, XC_LDA_X, XC_UNPOLARIZED);
        xc_func_init(&func_c, XC_LDA_C_PW, XC_UNPOLARIZED);
        std::vector<double> zk_x(Nd), zk_c(Nd), vr_x(Nd), vr_c(Nd);
        xc_lda_exc_vxc(&func_x, Nd, rho.data(), zk_x.data(), vr_x.data());
        xc_lda_exc_vxc(&func_c, Nd, rho.data(), zk_c.data(), vr_c.data());
        for (int i = 0; i < Nd; i++) {
            exc_ref[i] = zk_x[i] + zk_c[i];
            vrho_ref[i] = vr_x[i] + vr_c[i];
        }
        xc_func_end(&func_x);
        xc_func_end(&func_c);
    }

    // A: Serial (SPARC style)
    auto serial_lda = [&](std::vector<double>& exc_out, std::vector<double>& vrho_out) {
        xc_func_type func_x, func_c;
        xc_func_init(&func_x, XC_LDA_X, XC_UNPOLARIZED);
        xc_func_init(&func_c, XC_LDA_C_PW, XC_UNPOLARIZED);
        std::vector<double> zk_x(Nd), zk_c(Nd), vr_x(Nd), vr_c(Nd);
        xc_lda_exc_vxc(&func_x, Nd, rho.data(), zk_x.data(), vr_x.data());
        xc_lda_exc_vxc(&func_c, Nd, rho.data(), zk_c.data(), vr_c.data());
        for (int i = 0; i < Nd; i++) {
            exc_out[i] = zk_x[i] + zk_c[i];
            vrho_out[i] = vr_x[i] + vr_c[i];
        }
        xc_func_end(&func_x);
        xc_func_end(&func_c);
    };

    // B: Thread-parallel chunks
    auto parallel_lda = [&](std::vector<double>& exc_out, std::vector<double>& vrho_out) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int chunk = (Nd + nt - 1) / nt;
            int start = tid * chunk;
            int end = std::min(start + chunk, Nd);
            int np = end - start;
            if (np <= 0) np = 0;

            xc_func_type func_x, func_c;
            xc_func_init(&func_x, XC_LDA_X, XC_UNPOLARIZED);
            xc_func_init(&func_c, XC_LDA_C_PW, XC_UNPOLARIZED);

            std::vector<double> zk_x(np), zk_c(np), vr_x(np), vr_c(np);
            if (np > 0) {
                xc_lda_exc_vxc(&func_x, np, &rho[start], zk_x.data(), vr_x.data());
                xc_lda_exc_vxc(&func_c, np, &rho[start], zk_c.data(), vr_c.data());
                for (int i = 0; i < np; i++) {
                    exc_out[start + i] = zk_x[i] + zk_c[i];
                    vrho_out[start + i] = vr_x[i] + vr_c[i];
                }
            }
            xc_func_end(&func_x);
            xc_func_end(&func_c);
        }
    };

    // Benchmark
    std::vector<double> exc_out(Nd), vrho_out(Nd);

    // Serial
    serial_lda(exc_out, vrho_out);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < nreps; r++) serial_lda(exc_out, vrho_out);
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt_serial = std::chrono::duration<double, std::milli>(t1 - t0).count() / nreps;

    // Parallel
    parallel_lda(exc_out, vrho_out);
    double max_err_exc = 0, max_err_vrho = 0;
    for (int i = 0; i < Nd; i++) {
        max_err_exc = std::max(max_err_exc, std::abs(exc_out[i] - exc_ref[i]));
        max_err_vrho = std::max(max_err_vrho, std::abs(vrho_out[i] - vrho_ref[i]));
    }

    t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < nreps; r++) parallel_lda(exc_out, vrho_out);
    t1 = std::chrono::high_resolution_clock::now();
    double dt_parallel = std::chrono::duration<double, std::milli>(t1 - t0).count() / nreps;

    printf("  %-20s %10.4f ms\n", "A: Serial", dt_serial);
    printf("  %-20s %10.4f ms  speedup=%.2fx  err(exc)=%.1e err(vrho)=%.1e %s\n",
           "B: OMP parallel", dt_parallel, dt_serial / dt_parallel,
           max_err_exc, max_err_vrho,
           (max_err_exc < 1e-14 && max_err_vrho < 1e-14) ? "PASS" : "CHECK");
    printf("\n");
}

// ============================================================
// GGA benchmark (PBE)
// ============================================================
void bench_gga(int Nd, int nreps) {
    int nthreads = 1;
    #pragma omp parallel
    { nthreads = omp_get_num_threads(); }

    printf("--- GGA PBE, Nd=%d, threads=%d, reps=%d ---\n", Nd, nthreads, nreps);

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist_rho(0.01, 2.0);
    std::uniform_real_distribution<double> dist_sig(0.001, 1.0);
    std::vector<double> rho(Nd), sigma(Nd);
    for (int i = 0; i < Nd; i++) { rho[i] = dist_rho(rng); sigma[i] = dist_sig(rng); }

    // Reference
    std::vector<double> exc_ref(Nd), vrho_ref(Nd), vsigma_ref(Nd);
    {
        xc_func_type func_x, func_c;
        xc_func_init(&func_x, XC_GGA_X_PBE, XC_UNPOLARIZED);
        xc_func_init(&func_c, XC_GGA_C_PBE, XC_UNPOLARIZED);
        std::vector<double> zk_x(Nd), zk_c(Nd), vr_x(Nd), vr_c(Nd), vs_x(Nd), vs_c(Nd);
        xc_gga_exc_vxc(&func_x, Nd, rho.data(), sigma.data(), zk_x.data(), vr_x.data(), vs_x.data());
        xc_gga_exc_vxc(&func_c, Nd, rho.data(), sigma.data(), zk_c.data(), vr_c.data(), vs_c.data());
        for (int i = 0; i < Nd; i++) {
            exc_ref[i] = zk_x[i] + zk_c[i];
            vrho_ref[i] = vr_x[i] + vr_c[i];
            vsigma_ref[i] = vs_x[i] + vs_c[i];
        }
        xc_func_end(&func_x);
        xc_func_end(&func_c);
    }

    auto serial_gga = [&](std::vector<double>& exc_out, std::vector<double>& vrho_out, std::vector<double>& vsigma_out) {
        xc_func_type func_x, func_c;
        xc_func_init(&func_x, XC_GGA_X_PBE, XC_UNPOLARIZED);
        xc_func_init(&func_c, XC_GGA_C_PBE, XC_UNPOLARIZED);
        std::vector<double> zk_x(Nd), zk_c(Nd), vr_x(Nd), vr_c(Nd), vs_x(Nd), vs_c(Nd);
        xc_gga_exc_vxc(&func_x, Nd, rho.data(), sigma.data(), zk_x.data(), vr_x.data(), vs_x.data());
        xc_gga_exc_vxc(&func_c, Nd, rho.data(), sigma.data(), zk_c.data(), vr_c.data(), vs_c.data());
        for (int i = 0; i < Nd; i++) {
            exc_out[i] = zk_x[i] + zk_c[i];
            vrho_out[i] = vr_x[i] + vr_c[i];
            vsigma_out[i] = vs_x[i] + vs_c[i];
        }
        xc_func_end(&func_x);
        xc_func_end(&func_c);
    };

    auto parallel_gga = [&](std::vector<double>& exc_out, std::vector<double>& vrho_out, std::vector<double>& vsigma_out) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int chunk = (Nd + nt - 1) / nt;
            int start = tid * chunk;
            int end = std::min(start + chunk, Nd);
            int np = end - start;
            if (np <= 0) np = 0;

            xc_func_type func_x, func_c;
            xc_func_init(&func_x, XC_GGA_X_PBE, XC_UNPOLARIZED);
            xc_func_init(&func_c, XC_GGA_C_PBE, XC_UNPOLARIZED);

            std::vector<double> zk_x(np), zk_c(np), vr_x(np), vr_c(np), vs_x(np), vs_c(np);
            if (np > 0) {
                xc_gga_exc_vxc(&func_x, np, &rho[start], &sigma[start],
                               zk_x.data(), vr_x.data(), vs_x.data());
                xc_gga_exc_vxc(&func_c, np, &rho[start], &sigma[start],
                               zk_c.data(), vr_c.data(), vs_c.data());
                for (int i = 0; i < np; i++) {
                    exc_out[start + i] = zk_x[i] + zk_c[i];
                    vrho_out[start + i] = vr_x[i] + vr_c[i];
                    vsigma_out[start + i] = vs_x[i] + vs_c[i];
                }
            }
            xc_func_end(&func_x);
            xc_func_end(&func_c);
        }
    };

    std::vector<double> exc_out(Nd), vrho_out(Nd), vsigma_out(Nd);

    serial_gga(exc_out, vrho_out, vsigma_out);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < nreps; r++) serial_gga(exc_out, vrho_out, vsigma_out);
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt_serial = std::chrono::duration<double, std::milli>(t1 - t0).count() / nreps;

    parallel_gga(exc_out, vrho_out, vsigma_out);
    double max_err_exc = 0, max_err_vrho = 0;
    for (int i = 0; i < Nd; i++) {
        max_err_exc = std::max(max_err_exc, std::abs(exc_out[i] - exc_ref[i]));
        max_err_vrho = std::max(max_err_vrho, std::abs(vrho_out[i] - vrho_ref[i]));
    }

    t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < nreps; r++) parallel_gga(exc_out, vrho_out, vsigma_out);
    t1 = std::chrono::high_resolution_clock::now();
    double dt_parallel = std::chrono::duration<double, std::milli>(t1 - t0).count() / nreps;

    printf("  %-20s %10.4f ms\n", "A: Serial", dt_serial);
    printf("  %-20s %10.4f ms  speedup=%.2fx  err(exc)=%.1e err(vrho)=%.1e %s\n",
           "B: OMP parallel", dt_parallel, dt_serial / dt_parallel,
           max_err_exc, max_err_vrho,
           (max_err_exc < 1e-14 && max_err_vrho < 1e-14) ? "PASS" : "CHECK");
    printf("\n");
}

// ============================================================
// mGGA benchmark (SCAN)
// ============================================================
void bench_mgga(int Nd, int nreps) {
    int nthreads = 1;
    #pragma omp parallel
    { nthreads = omp_get_num_threads(); }

    printf("--- mGGA SCAN, Nd=%d, threads=%d, reps=%d ---\n", Nd, nthreads, nreps);

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist_rho(0.01, 2.0);
    std::uniform_real_distribution<double> dist_sig(0.001, 1.0);
    std::uniform_real_distribution<double> dist_tau(0.01, 5.0);
    std::vector<double> rho(Nd), sigma(Nd), lapl(Nd, 0.0), tau(Nd);
    for (int i = 0; i < Nd; i++) {
        rho[i] = dist_rho(rng);
        sigma[i] = dist_sig(rng);
        tau[i] = dist_tau(rng);
    }

    // Reference
    std::vector<double> exc_ref(Nd), vrho_ref(Nd);
    {
        xc_func_type func_x, func_c;
        xc_func_init(&func_x, XC_MGGA_X_SCAN, XC_UNPOLARIZED);
        xc_func_init(&func_c, XC_MGGA_C_SCAN, XC_UNPOLARIZED);
        std::vector<double> zk_x(Nd), zk_c(Nd), vr_x(Nd), vr_c(Nd);
        std::vector<double> vs_x(Nd), vs_c(Nd), vl_x(Nd), vl_c(Nd), vt_x(Nd), vt_c(Nd);
        xc_mgga_exc_vxc(&func_x, Nd, rho.data(), sigma.data(), lapl.data(), tau.data(),
                        zk_x.data(), vr_x.data(), vs_x.data(), vl_x.data(), vt_x.data());
        xc_mgga_exc_vxc(&func_c, Nd, rho.data(), sigma.data(), lapl.data(), tau.data(),
                        zk_c.data(), vr_c.data(), vs_c.data(), vl_c.data(), vt_c.data());
        for (int i = 0; i < Nd; i++) {
            exc_ref[i] = zk_x[i] + zk_c[i];
            vrho_ref[i] = vr_x[i] + vr_c[i];
        }
        xc_func_end(&func_x);
        xc_func_end(&func_c);
    }

    auto serial_mgga = [&](std::vector<double>& exc_out, std::vector<double>& vrho_out) {
        xc_func_type func_x, func_c;
        xc_func_init(&func_x, XC_MGGA_X_SCAN, XC_UNPOLARIZED);
        xc_func_init(&func_c, XC_MGGA_C_SCAN, XC_UNPOLARIZED);
        std::vector<double> zk_x(Nd), zk_c(Nd), vr_x(Nd), vr_c(Nd);
        std::vector<double> vs_x(Nd), vs_c(Nd), vl_x(Nd), vl_c(Nd), vt_x(Nd), vt_c(Nd);
        xc_mgga_exc_vxc(&func_x, Nd, rho.data(), sigma.data(), lapl.data(), tau.data(),
                        zk_x.data(), vr_x.data(), vs_x.data(), vl_x.data(), vt_x.data());
        xc_mgga_exc_vxc(&func_c, Nd, rho.data(), sigma.data(), lapl.data(), tau.data(),
                        zk_c.data(), vr_c.data(), vs_c.data(), vl_c.data(), vt_c.data());
        for (int i = 0; i < Nd; i++) {
            exc_out[i] = zk_x[i] + zk_c[i];
            vrho_out[i] = vr_x[i] + vr_c[i];
        }
        xc_func_end(&func_x);
        xc_func_end(&func_c);
    };

    auto parallel_mgga = [&](std::vector<double>& exc_out, std::vector<double>& vrho_out) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int chunk = (Nd + nt - 1) / nt;
            int start = tid * chunk;
            int end = std::min(start + chunk, Nd);
            int np = end - start;
            if (np <= 0) np = 0;

            xc_func_type func_x, func_c;
            xc_func_init(&func_x, XC_MGGA_X_SCAN, XC_UNPOLARIZED);
            xc_func_init(&func_c, XC_MGGA_C_SCAN, XC_UNPOLARIZED);
            std::vector<double> zk_x(np), zk_c(np), vr_x(np), vr_c(np);
            std::vector<double> vs_x(np), vs_c(np), vl_x(np), vl_c(np), vt_x(np), vt_c(np);
            if (np > 0) {
                xc_mgga_exc_vxc(&func_x, np, &rho[start], &sigma[start], &lapl[start], &tau[start],
                                zk_x.data(), vr_x.data(), vs_x.data(), vl_x.data(), vt_x.data());
                xc_mgga_exc_vxc(&func_c, np, &rho[start], &sigma[start], &lapl[start], &tau[start],
                                zk_c.data(), vr_c.data(), vs_c.data(), vl_c.data(), vt_c.data());
                for (int i = 0; i < np; i++) {
                    exc_out[start + i] = zk_x[i] + zk_c[i];
                    vrho_out[start + i] = vr_x[i] + vr_c[i];
                }
            }
            xc_func_end(&func_x);
            xc_func_end(&func_c);
        }
    };

    std::vector<double> exc_out(Nd), vrho_out(Nd);

    serial_mgga(exc_out, vrho_out);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < nreps; r++) serial_mgga(exc_out, vrho_out);
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt_serial = std::chrono::duration<double, std::milli>(t1 - t0).count() / nreps;

    parallel_mgga(exc_out, vrho_out);
    double max_err_exc = 0, max_err_vrho = 0;
    for (int i = 0; i < Nd; i++) {
        max_err_exc = std::max(max_err_exc, std::abs(exc_out[i] - exc_ref[i]));
        max_err_vrho = std::max(max_err_vrho, std::abs(vrho_out[i] - vrho_ref[i]));
    }

    t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < nreps; r++) parallel_mgga(exc_out, vrho_out);
    t1 = std::chrono::high_resolution_clock::now();
    double dt_parallel = std::chrono::duration<double, std::milli>(t1 - t0).count() / nreps;

    printf("  %-20s %10.4f ms\n", "A: Serial", dt_serial);
    printf("  %-20s %10.4f ms  speedup=%.2fx  err(exc)=%.1e err(vrho)=%.1e %s\n",
           "B: OMP parallel", dt_parallel, dt_serial / dt_parallel,
           max_err_exc, max_err_vrho,
           (max_err_exc < 1e-14 && max_err_vrho < 1e-14) ? "PASS" : "CHECK");
    printf("\n");
}

int main() {
    printf("================================================================\n");
    printf("  Serial vs Thread-Parallel libxc Benchmark\n");
    printf("================================================================\n\n");

    int Nd_small = 25 * 26 * 27;  // 17550 (Si4-like)
    int Nd_large = 48 * 48 * 48;  // 110592

    bench_lda(Nd_small, 100);
    bench_lda(Nd_large, 50);

    bench_gga(Nd_small, 100);
    bench_gga(Nd_large, 50);

    bench_mgga(Nd_small, 50);
    bench_mgga(Nd_large, 20);

    return 0;
}
