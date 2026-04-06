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
#include <xc.h>
#include <xc_funcs.h>
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
    printf("=== Test: GPU LDA XC (libxc CUDA) vs CPU reference ===\n");

    int N = 10000;
    std::vector<double> h_rho(N);
    for (int i = 0; i < N; i++) h_rho[i] = 0.001 + 0.1 * ((double)i / N);

    // CPU reference using libxc (host mode)
    xc_func_type cpu_x, cpu_c;
    xc_func_init_flags(&cpu_x, XC_LDA_X, XC_UNPOLARIZED, XC_FLAGS_ON_HOST);
    xc_func_init_flags(&cpu_c, XC_LDA_C_PW, XC_UNPOLARIZED, XC_FLAGS_ON_HOST);
    std::vector<double> h_zk_x(N), h_zk_c(N), h_vr_x(N), h_vr_c(N);
    xc_lda_exc_vxc(&cpu_x, N, h_rho.data(), h_zk_x.data(), h_vr_x.data());
    xc_lda_exc_vxc(&cpu_c, N, h_rho.data(), h_zk_c.data(), h_vr_c.data());
    std::vector<double> h_exc_cpu(N), h_vxc_cpu(N);
    for (int i = 0; i < N; i++) {
        h_exc_cpu[i] = h_zk_x[i] + h_zk_c[i];
        h_vxc_cpu[i] = h_vr_x[i] + h_vr_c[i];
    }
    xc_func_end(&cpu_x);
    xc_func_end(&cpu_c);

    // GPU: libxc with device pointers
    double *d_rho, *d_exc, *d_vxc, *d_exc_c, *d_vxc_c;
    CUDA_CHECK(cudaMalloc(&d_rho, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_exc, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vxc, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_exc_c, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vxc_c, N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_rho, h_rho.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    xc_func_type gpu_x, gpu_c;
    xc_func_init_flags(&gpu_x, XC_LDA_X, XC_UNPOLARIZED, XC_FLAGS_ON_DEVICE);
    xc_func_init_flags(&gpu_c, XC_LDA_C_PW, XC_UNPOLARIZED, XC_FLAGS_ON_DEVICE);
    xc_lda_exc_vxc(&gpu_x, N, d_rho, d_exc, d_vxc);
    xc_lda_exc_vxc(&gpu_c, N, d_rho, d_exc_c, d_vxc_c);
    // Accumulate on GPU
    auto& ctx = lynx::gpu::GPUContext::instance();
    double one = 1.0;
    cublasDaxpy(ctx.cublas, N, &one, d_exc_c, 1, d_exc, 1);
    cublasDaxpy(ctx.cublas, N, &one, d_vxc_c, 1, d_vxc, 1);
    xc_func_end(&gpu_x);
    xc_func_end(&gpu_c);
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
           (max_err_exc < 1e-12 && max_err_vxc < 1e-12) ? "OK" : "FAIL");
    assert(max_err_exc < 1e-12);
    assert(max_err_vxc < 1e-12);

    // Also test PZ81 via libxc
    xc_func_type gpu_pz_x, gpu_pz_c;
    xc_func_init_flags(&gpu_pz_x, XC_LDA_X, XC_UNPOLARIZED, XC_FLAGS_ON_DEVICE);
    xc_func_init_flags(&gpu_pz_c, XC_LDA_C_PZ, XC_UNPOLARIZED, XC_FLAGS_ON_DEVICE);
    xc_lda_exc_vxc(&gpu_pz_x, N, d_rho, d_exc, d_vxc);
    xc_lda_exc_vxc(&gpu_pz_c, N, d_rho, d_exc_c, d_vxc_c);
    cublasDaxpy(ctx.cublas, N, &one, d_exc_c, 1, d_exc, 1);
    xc_func_end(&gpu_pz_x);
    xc_func_end(&gpu_pz_c);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_exc_gpu.data(), d_exc, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        assert(h_exc_gpu[i] < 0);
        assert(h_exc_gpu[i] > -10);
    }
    printf("  LDA_PZ: values reasonable (all negative, bounded): OK\n");

    cudaFree(d_rho); cudaFree(d_exc); cudaFree(d_vxc); cudaFree(d_exc_c); cudaFree(d_vxc_c);
    printf("  PASSED\n\n");
}

// ============================================================
// Test 6: GPU AAR Solver (solve -Lap*x = b)
// ============================================================

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
    // test_aar_solver removed: standalone gpu::aar_gpu deleted (AAR loop now in Mixer/PoissonSolver .cpp)
    bench_subspace_ops();

    printf("All tests PASSED.\n");
    return 0;
}
