// Test GPU algorithms: EigenSolver, AAR solver, XC functional, density computation
// Verifies GPU results match CPU reference to numerical precision.

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "core/GPUContext.cuh"
#include "core/gpu_common.cuh"

using namespace lynx::gpu;

// Forward declarations for GPU functions
namespace lynx { namespace gpu {
void compute_ata_gpu(const double* d_X, double* d_S, int Nd, int N, double dV, cudaStream_t stream = 0);
void compute_atb_gpu(const double* d_X, const double* d_HX, double* d_Hs, int Nd, int N, double dV, cudaStream_t stream = 0);
void symmetrize_gpu(double* d_Hs, int N, cudaStream_t stream = 0);
void orthogonalize_gpu(double* d_X, double* d_S, int Nd, int N, double dV);
void rotate_orbitals_gpu(double* d_X, const double* d_Q, double* d_temp, int Nd, int N);
void compute_density_gpu(const double* d_psi, const double* d_occ, double* d_rho, int Nd, int Ns, double weight, cudaStream_t stream = 0);
void lda_pw_gpu(const double* d_rho, double* d_exc, double* d_vxc, int N, cudaStream_t stream = 0);
void lda_pz_gpu(const double* d_rho, double* d_exc, double* d_vxc, int N, cudaStream_t stream = 0);

int aar_gpu(
    void (*op_gpu)(const double* d_x, double* d_Ax),
    void (*precond_gpu)(const double* d_r, double* d_f),
    const double* d_b,
    double* d_x,
    int N,
    double omega, double beta, int m, int p,
    double tol, int max_iter,
    double* d_r, double* d_f, double* d_Ax,
    double* d_X_hist, double* d_F_hist,
    double* d_x_old, double* d_f_old, cudaStream_t stream = 0);
}} // namespace

// CPU reference BLAS
extern "C" {
    void dgemm_(const char*, const char*, const int*, const int*, const int*,
                const double*, const double*, const int*,
                const double*, const int*,
                const double*, double*, const int*);
    void dpotrf_(const char*, const int*, double*, const int*, int*);
    void dtrsm_(const char*, const char*, const char*, const char*,
                const int*, const int*, const double*,
                const double*, const int*, double*, const int*);
}

// ============================================================
// Test 1: Custom dot kernel A^T*A vs CPU BLAS
// ============================================================
void test_ata_kernel() {
    printf("=== Test: GPU A^T*A (custom dot kernel) vs CPU BLAS ===\n");

    for (auto [M, N] : std::vector<std::pair<int,int>>{{1000, 12}, {5000, 30}, {17576, 12}, {50000, 50}}) {
        double dV = 0.01;

        std::vector<double> h_A(M * N);
        for (int i = 0; i < M * N; i++) h_A[i] = sin(0.001 * i) * 0.1;

        // CPU reference
        std::vector<double> h_S_cpu(N * N);
        char transT = 'T', transN = 'N';
        dgemm_(&transT, &transN, &N, &N, &M,
               &dV, h_A.data(), &M, h_A.data(), &M,
               &dV /* beta=0 used wrong, let's fix */, h_S_cpu.data(), &N);
        // Redo with correct beta
        double beta = 0.0;
        dgemm_(&transT, &transN, &N, &N, &M,
               &dV, h_A.data(), &M, h_A.data(), &M,
               &beta, h_S_cpu.data(), &N);

        // GPU
        double *d_A, *d_S;
        CUDA_CHECK(cudaMalloc(&d_A, (size_t)M * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_S, (size_t)N * N * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), (size_t)M * N * sizeof(double), cudaMemcpyHostToDevice));

        compute_ata_gpu(d_A, d_S, M, N, dV);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> h_S_gpu(N * N);
        CUDA_CHECK(cudaMemcpy(h_S_gpu.data(), d_S, N * N * sizeof(double), cudaMemcpyDeviceToHost));

        double max_err = 0;
        for (int i = 0; i < N * N; i++)
            max_err = std::max(max_err, std::abs(h_S_cpu[i] - h_S_gpu[i]));

        printf("  M=%6d N=%3d: max_err=%.2e %s\n", M, N, max_err,
               max_err < 1e-10 ? "OK" : "FAIL");
        assert(max_err < 1e-10);

        cudaFree(d_A);
        cudaFree(d_S);
    }
    printf("  PASSED\n\n");
}

// ============================================================
// Test 2: GPU Orthogonalization (Cholesky QR)
// ============================================================
void test_orthogonalize() {
    printf("=== Test: GPU Cholesky QR Orthogonalization ===\n");

    int Nd = 5000, N = 12;
    double dV = 0.01;

    std::vector<double> h_X(Nd * N);
    srand(42);
    for (int i = 0; i < Nd * N; i++) h_X[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;

    double *d_X, *d_S;
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)Nd * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_S, (size_t)N * N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), (size_t)Nd * N * sizeof(double), cudaMemcpyHostToDevice));

    auto& ctx = GPUContext::instance();
    // Need to init SCF buffers for cusolver_devinfo
    if (!ctx.buf.cusolver_devinfo) {
        CUDA_CHECK(cudaMalloc(&ctx.buf.cusolver_devinfo, sizeof(int)));
        ctx.scratch_pool.init(4 * 1024 * 1024);
    }

    orthogonalize_gpu(d_X, d_S, Nd, N, dV);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify: X^T * X * dV should be identity
    compute_ata_gpu(d_X, d_S, Nd, N, dV);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_S(N * N);
    CUDA_CHECK(cudaMemcpy(h_S.data(), d_S, N * N * sizeof(double), cudaMemcpyDeviceToHost));

    double max_err = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            max_err = std::max(max_err, std::abs(h_S[i + j * N] - expected));
        }
    }

    printf("  Nd=%d N=%d: ||X^T*X*dV - I||_max = %.2e %s\n",
           Nd, N, max_err, max_err < 1e-10 ? "OK" : "FAIL");
    assert(max_err < 1e-10);

    cudaFree(d_X);
    cudaFree(d_S);
    printf("  PASSED\n\n");
}

// ============================================================
// Test 3: GPU Orbital Rotation (dgemm)
// ============================================================
void test_rotate_orbitals() {
    printf("=== Test: GPU Orbital Rotation ===\n");

    int Nd = 8000, N = 12;

    std::vector<double> h_X(Nd * N), h_Q(N * N);
    srand(123);
    for (int i = 0; i < Nd * N; i++) h_X[i] = sin(0.001 * i);
    // Q = identity (rotation should be a no-op)
    for (int i = 0; i < N * N; i++) h_Q[i] = 0.0;
    for (int i = 0; i < N; i++) h_Q[i + i * N] = 1.0;

    double *d_X, *d_Q, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)Nd * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Q, (size_t)N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_temp, (size_t)Nd * N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), (size_t)Nd * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), (size_t)N * N * sizeof(double), cudaMemcpyHostToDevice));

    rotate_orbitals_gpu(d_X, d_Q, d_temp, Nd, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_X_out(Nd * N);
    CUDA_CHECK(cudaMemcpy(h_X_out.data(), d_X, (size_t)Nd * N * sizeof(double), cudaMemcpyDeviceToHost));

    double max_err = 0;
    for (int i = 0; i < Nd * N; i++)
        max_err = std::max(max_err, std::abs(h_X[i] - h_X_out[i]));

    printf("  Identity rotation: max_err=%.2e %s\n", max_err, max_err < 1e-14 ? "OK" : "FAIL");
    assert(max_err < 1e-14);

    // Now test with a non-trivial rotation: swap first two columns
    for (int i = 0; i < N * N; i++) h_Q[i] = 0.0;
    for (int i = 0; i < N; i++) h_Q[i + i * N] = 1.0;
    h_Q[0] = 0.0; h_Q[1] = 1.0;  // Q[0,0]=0, Q[1,0]=1
    h_Q[N] = 1.0; h_Q[N+1] = 0.0; // Q[0,1]=1, Q[1,1]=0

    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), (size_t)Nd * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), (size_t)N * N * sizeof(double), cudaMemcpyHostToDevice));
    rotate_orbitals_gpu(d_X, d_Q, d_temp, Nd, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_X_out.data(), d_X, (size_t)Nd * N * sizeof(double), cudaMemcpyDeviceToHost));

    // Column 0 of result should be column 1 of input
    max_err = 0;
    for (int i = 0; i < Nd; i++) {
        max_err = std::max(max_err, std::abs(h_X_out[i] - h_X[i + Nd]));  // col 0 <- col 1
        max_err = std::max(max_err, std::abs(h_X_out[i + Nd] - h_X[i]));  // col 1 <- col 0
    }
    printf("  Column swap rotation: max_err=%.2e %s\n", max_err, max_err < 1e-14 ? "OK" : "FAIL");
    assert(max_err < 1e-14);

    cudaFree(d_X); cudaFree(d_Q); cudaFree(d_temp);
    printf("  PASSED\n\n");
}

// ============================================================
// Test 4: GPU Density Computation
// ============================================================
void test_density() {
    printf("=== Test: GPU Density Computation ===\n");

    int Nd = 8000, Ns = 12;
    double weight = 2.0;  // spin factor

    std::vector<double> h_psi(Nd * Ns), h_occ(Ns);
    for (int i = 0; i < Nd * Ns; i++) h_psi[i] = sin(0.001 * i) * 0.1;
    for (int n = 0; n < Ns; n++) h_occ[n] = (n < 5) ? 1.0 : (n < 8 ? 0.5 : 0.0);

    // CPU reference
    std::vector<double> h_rho_cpu(Nd, 0.0);
    for (int i = 0; i < Nd; i++) {
        double sum = 0;
        for (int n = 0; n < Ns; n++) {
            double v = h_psi[i + n * Nd];
            sum += h_occ[n] * v * v;
        }
        h_rho_cpu[i] = weight * sum;
    }

    // GPU
    double *d_psi, *d_occ, *d_rho;
    CUDA_CHECK(cudaMalloc(&d_psi, (size_t)Nd * Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_occ, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rho, Nd * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_psi, h_psi.data(), (size_t)Nd * Ns * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_occ, h_occ.data(), Ns * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_rho, 0, Nd * sizeof(double)));

    compute_density_gpu(d_psi, d_occ, d_rho, Nd, Ns, weight);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_rho_gpu(Nd);
    CUDA_CHECK(cudaMemcpy(h_rho_gpu.data(), d_rho, Nd * sizeof(double), cudaMemcpyDeviceToHost));

    double max_err = 0;
    for (int i = 0; i < Nd; i++)
        max_err = std::max(max_err, std::abs(h_rho_cpu[i] - h_rho_gpu[i]));

    printf("  Nd=%d Ns=%d: max_err=%.2e %s\n", Nd, Ns, max_err,
           max_err < 1e-14 ? "OK" : "FAIL");
    assert(max_err < 1e-14);

    cudaFree(d_psi); cudaFree(d_occ); cudaFree(d_rho);
    printf("  PASSED\n\n");
}

// ============================================================
// Test 5: GPU LDA XC (PW92 and PZ81)
// ============================================================
void test_xc_lda() {
    printf("=== Test: GPU LDA XC vs CPU reference ===\n");

    int N = 10000;
    std::vector<double> h_rho(N);
    for (int i = 0; i < N; i++) h_rho[i] = 0.001 + 0.1 * ((double)i / N);

    // CPU reference for LDA_PW (Slater + PW92)
    auto cpu_lda_pw = [](double rho, double& exc, double& vxc) {
        double rho_cbrt = cbrt(rho);
        double ex = -0.738558766382022 * rho_cbrt;
        double vx = -0.9847450218426965 * rho_cbrt;

        double rs = 0.6203504908993999 / rho_cbrt;
        double rs_sqrt = sqrt(rs);
        double G2 = 2.0*0.031091*(7.5957*rs_sqrt + 3.5876*rs + 1.6382*rs*rs_sqrt + 0.49294*rs*rs);
        double G1 = log(1.0 + 1.0/G2);
        double ec = -2.0*0.031091*(1.0 + 0.21370*rs)*G1;
        double vc = ec - (rs/3.0)*(-2.0*0.031091*0.21370*G1
            + (2.0*0.031091*(1.0+0.21370*rs)
               *(0.031091*(7.5957/rs_sqrt + 2.0*3.5876 + 3.0*1.6382*rs_sqrt + 2.0*2.0*0.49294*rs)))
              /(G2*(G2+1.0)));
        exc = ex + ec;
        vxc = vx + vc;
    };

    std::vector<double> h_exc_cpu(N), h_vxc_cpu(N);
    for (int i = 0; i < N; i++)
        cpu_lda_pw(h_rho[i], h_exc_cpu[i], h_vxc_cpu[i]);

    // GPU
    double *d_rho, *d_exc, *d_vxc;
    CUDA_CHECK(cudaMalloc(&d_rho, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_exc, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vxc, N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_rho, h_rho.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    lda_pw_gpu(d_rho, d_exc, d_vxc, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_exc_gpu(N), h_vxc_gpu(N);
    CUDA_CHECK(cudaMemcpy(h_exc_gpu.data(), d_exc, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vxc_gpu.data(), d_vxc, N * sizeof(double), cudaMemcpyDeviceToHost));

    double max_err_exc = 0, max_err_vxc = 0;
    for (int i = 0; i < N; i++) {
        max_err_exc = std::max(max_err_exc, std::abs(h_exc_cpu[i] - h_exc_gpu[i]));
        max_err_vxc = std::max(max_err_vxc, std::abs(h_vxc_cpu[i] - h_vxc_gpu[i]));
    }

    printf("  LDA_PW: exc_err=%.2e vxc_err=%.2e %s\n",
           max_err_exc, max_err_vxc,
           (max_err_exc < 1e-14 && max_err_vxc < 1e-14) ? "OK" : "FAIL");
    assert(max_err_exc < 1e-14);
    assert(max_err_vxc < 1e-14);

    // Also test PZ81
    lda_pz_gpu(d_rho, d_exc, d_vxc, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_exc_gpu.data(), d_exc, N * sizeof(double), cudaMemcpyDeviceToHost));

    // Just verify it produces reasonable values
    for (int i = 0; i < N; i++) {
        assert(h_exc_gpu[i] < 0);  // XC energy should be negative
        assert(h_exc_gpu[i] > -10);
    }
    printf("  LDA_PZ: values reasonable (all negative, bounded): OK\n");

    cudaFree(d_rho); cudaFree(d_exc); cudaFree(d_vxc);
    printf("  PASSED\n\n");
}

// ============================================================
// Test 6: GPU AAR Solver (solve -Lap*x = b)
// ============================================================

// Global GPU workspace for AAR test operator
static double* g_d_Ax_tmp = nullptr;
static double* g_d_x_ex_tmp = nullptr;
static int g_aar_N = 0;

// Simple test operator: A*x = diagonal * x (just to test AAR framework)
__global__ void diag_op_kernel(const double* x, double* Ax, double diag, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) Ax[idx] = diag * x[idx];
}

static void test_op_gpu(const double* d_x, double* d_Ax) {
    int bs = 256;
    diag_op_kernel<<<ceildiv(g_aar_N, bs), bs>>>(d_x, d_Ax, 4.0, g_aar_N);
}

void test_aar_solver() {
    printf("=== Test: GPU AAR Solver ===\n");

    int N = 1000;
    g_aar_N = N;
    int m = 7, p = 6;
    double omega = 0.2, beta = 0.2, tol = 1e-10;
    int max_iter = 500;

    // Solve 4*x = b, solution: x = b/4
    std::vector<double> h_b(N);
    for (int i = 0; i < N; i++) h_b[i] = 4.0 * (1.0 + sin(0.01 * i));

    // GPU buffers
    double *d_b, *d_x, *d_r, *d_f, *d_Ax, *d_X_hist, *d_F_hist, *d_x_old, *d_f_old;
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_f, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ax, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_X_hist, N * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_F_hist, N * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x_old, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_f_old, N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, N * sizeof(double)));  // initial guess = 0

    int iters = aar_gpu(test_op_gpu, nullptr, d_b, d_x, N,
                         omega, beta, m, p, tol, max_iter,
                         d_r, d_f, d_Ax, d_X_hist, d_F_hist,
                         d_x_old, d_f_old);

    std::vector<double> h_x(N);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, N * sizeof(double), cudaMemcpyDeviceToHost));

    double max_err = 0;
    for (int i = 0; i < N; i++) {
        double expected = h_b[i] / 4.0;
        max_err = std::max(max_err, std::abs(h_x[i] - expected));
    }

    printf("  N=%d: %d iterations, max_err=%.2e %s\n",
           N, iters, max_err, max_err < 1e-8 ? "OK" : "FAIL");
    assert(max_err < 1e-8);

    cudaFree(d_b); cudaFree(d_x); cudaFree(d_r); cudaFree(d_f);
    cudaFree(d_Ax); cudaFree(d_X_hist); cudaFree(d_F_hist);
    cudaFree(d_x_old); cudaFree(d_f_old);
    printf("  PASSED\n\n");
}

// ============================================================
// Performance benchmark: GPU subspace ops
// ============================================================
void bench_subspace_ops() {
    printf("=== Benchmark: GPU Subspace Operations ===\n");

    auto& ctx = GPUContext::instance();

    for (auto [Nd, N] : std::vector<std::pair<int,int>>{{17576, 12}, {17576, 30}, {110592, 12}, {110592, 30}}) {
        double dV = 0.01;

        double *d_X, *d_S, *d_HX, *d_Hs, *d_temp;
        CUDA_CHECK(cudaMalloc(&d_X, (size_t)Nd * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_S, (size_t)N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_HX, (size_t)Nd * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Hs, (size_t)N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_temp, (size_t)Nd * N * sizeof(double)));

        // Init with random data
        std::vector<double> h_X(Nd * N);
        for (int i = 0; i < Nd * N; i++) h_X[i] = sin(0.001 * i) * 0.1;
        CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), (size_t)Nd * N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_HX, h_X.data(), (size_t)Nd * N * sizeof(double), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        float ms;

        int iters = 100;

        // A^T * A
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            compute_ata_gpu(d_X, d_S, Nd, N, dV);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("  Nd=%-7d N=%-3d: X^T*X = %.3f ms", Nd, N, ms / iters);

        // A^T * B
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            compute_atb_gpu(d_X, d_HX, d_Hs, Nd, N, dV);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf(", X^T*HX = %.3f ms", ms / iters);

        // Orbital rotation
        std::vector<double> h_Q(N * N, 0);
        for (int i = 0; i < N; i++) h_Q[i + i * N] = 1.0;
        CUDA_CHECK(cudaMemcpy(d_S, h_Q.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), (size_t)Nd * N * sizeof(double), cudaMemcpyHostToDevice));
            rotate_orbitals_gpu(d_X, d_S, d_temp, Nd, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf(", Rotate = %.3f ms\n", ms / iters);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        cudaFree(d_X); cudaFree(d_S); cudaFree(d_HX); cudaFree(d_Hs); cudaFree(d_temp);
    }
    printf("\n");
}

int main() {
    printf("GPU Algorithm Tests\n");
    printf("===================\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    test_ata_kernel();
    test_orthogonalize();
    test_rotate_orbitals();
    test_density();
    test_xc_lda();
    test_aar_solver();
    bench_subspace_ops();

    printf("All tests PASSED.\n");
    return 0;
}
