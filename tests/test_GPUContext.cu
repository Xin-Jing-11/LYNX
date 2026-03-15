// Test GPUContext: memory pool allocation, cuBLAS/cuSOLVER handles, SCF buffer init
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "core/GPUContext.cuh"

using namespace sparc::gpu;

// Simple kernel to verify GPU memory is accessible
__global__ void fill_kernel(double* data, int n, double val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = val;
}

__global__ void check_kernel(const double* data, int n, double expected, int* errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && fabs(data[idx] - expected) > 1e-14) {
        atomicAdd(errors, 1);
    }
}

void test_memory_pool() {
    printf("=== Test GPUMemoryPool ===\n");

    GPUMemoryPool pool;
    pool.init(64 * 1024 * 1024);  // 64 MB
    assert(pool.capacity() == 64 * 1024 * 1024);
    assert(pool.used() == 0);

    // Allocate a few buffers
    double* a = pool.alloc<double>(1000);
    double* b = pool.alloc<double>(2000);
    double* c = pool.alloc<double>(5000);
    assert(a != nullptr);
    assert(b != nullptr);
    assert(c != nullptr);
    assert(a != b && b != c);
    printf("  Alloc 3 buffers: used=%zu bytes\n", pool.used());

    // Verify GPU memory is accessible
    int bs = 256;
    fill_kernel<<<ceildiv(1000, bs), bs>>>(a, 1000, 3.14);
    fill_kernel<<<ceildiv(2000, bs), bs>>>(b, 2000, 2.71);
    fill_kernel<<<ceildiv(5000, bs), bs>>>(c, 5000, 1.41);
    CUDA_CHECK(cudaDeviceSynchronize());

    int* d_errors;
    CUDA_CHECK(cudaMalloc(&d_errors, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_errors, 0, sizeof(int)));
    check_kernel<<<ceildiv(1000, bs), bs>>>(a, 1000, 3.14, d_errors);
    check_kernel<<<ceildiv(2000, bs), bs>>>(b, 2000, 2.71, d_errors);
    check_kernel<<<ceildiv(5000, bs), bs>>>(c, 5000, 1.41, d_errors);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));
    assert(h_errors == 0);
    printf("  GPU memory read/write: OK\n");

    // Test checkpoint/restore
    size_t cp = pool.checkpoint();
    double* d = pool.alloc<double>(10000);
    assert(d != nullptr);
    printf("  After extra alloc: used=%zu\n", pool.used());
    pool.restore(cp);
    printf("  After restore: used=%zu\n", pool.used());

    // Test reset
    pool.reset();
    assert(pool.used() == 0);
    printf("  After reset: used=%zu\n", pool.used());

    // Re-alloc should reuse same memory
    double* a2 = pool.alloc<double>(1000);
    assert(a2 == a);  // Same base pointer
    printf("  Re-alloc reuses memory: OK\n");

    cudaFree(d_errors);
    printf("  PASSED\n\n");
}

void test_pinned_pool() {
    printf("=== Test PinnedMemoryPool ===\n");

    PinnedMemoryPool pool;
    pool.init(1024 * 1024);  // 1 MB

    double* h_buf = pool.alloc<double>(256);
    for (int i = 0; i < 256; i++) h_buf[i] = (double)i;

    // Async copy to GPU and back
    double* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, 256 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyAsync(d_buf, h_buf, 256 * sizeof(double),
                                cudaMemcpyHostToDevice, 0));

    double* h_buf2 = pool.alloc<double>(256);
    CUDA_CHECK(cudaMemcpyAsync(h_buf2, d_buf, 256 * sizeof(double),
                                cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < 256; i++) assert(h_buf2[i] == (double)i);
    printf("  Pinned async H2D + D2H: OK\n");

    cudaFree(d_buf);
    printf("  PASSED\n\n");
}

void test_scf_buffers() {
    printf("=== Test SCF Buffer Init ===\n");

    auto& ctx = GPUContext::instance();

    // Typical BaTiO3: 26^3 grid, 12 bands, FDn=6
    int nx = 26, ny = 26, nz = 26;
    int Nd = nx * ny * nz;  // 17576
    int FDn = 6;
    int Ns = 12, Ns_global = 12, Nspin = 1;
    int aar_m = 7, mix_m = 7, mix_ncol = 1;
    int total_nproj = 48;
    size_t chi_size = 5000;  // approximate
    size_t gpos_size = 3000;

    ctx.init_scf_buffers(Nd, nx, ny, nz, FDn,
                          Ns, Ns_global, Nspin,
                          aar_m, mix_m, mix_ncol,
                          total_nproj, chi_size, gpos_size,
                          false, false);

    ctx.print_memory_info();

    // Verify buffers are non-null and distinct
    assert(ctx.buf.psi != nullptr);
    assert(ctx.buf.Hpsi != nullptr);
    assert(ctx.buf.Veff != nullptr);
    assert(ctx.buf.phi != nullptr);
    assert(ctx.buf.rho != nullptr);
    assert(ctx.buf.x_ex != nullptr);
    assert(ctx.buf.aar_r != nullptr);
    assert(ctx.buf.Chi_flat != nullptr);
    assert(ctx.buf.alpha != nullptr);
    assert(ctx.buf.psi != ctx.buf.Hpsi);
    printf("  All buffers allocated: OK\n");

    // Verify we can write to all major buffers
    int bs = 256;
    fill_kernel<<<ceildiv(Nd, bs), bs>>>(ctx.buf.psi, Nd, 1.0);
    fill_kernel<<<ceildiv(Nd, bs), bs>>>(ctx.buf.Veff, Nd, -0.5);
    fill_kernel<<<ceildiv(Nd, bs), bs>>>(ctx.buf.rho, Nd, 0.01);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back one value to verify
    double val;
    CUDA_CHECK(cudaMemcpy(&val, ctx.buf.Veff, sizeof(double), cudaMemcpyDeviceToHost));
    assert(val == -0.5);
    printf("  Buffer read/write: OK\n");

    // Test cuBLAS with pre-allocated workspace
    // Simple DGEMM: C = A^T * A (N=12, M=100)
    int M = 100, N = 12;
    double* d_A = ctx.scratch_pool.alloc<double>(M * N);
    double* d_C = ctx.scratch_pool.alloc<double>(N * N);

    // Fill A with some values
    std::vector<double> h_A(M * N);
    for (int i = 0; i < M * N; i++) h_A[i] = sin(0.01 * i);
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * N * sizeof(double), cudaMemcpyHostToDevice));

    double alpha = 1.0, beta = 0.0;
    cublasStatus_t stat = cublasDgemm(ctx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                       N, N, M, &alpha, d_A, M, d_A, M, &beta, d_C, N);
    assert(stat == CUBLAS_STATUS_SUCCESS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back and verify C is symmetric positive semi-definite
    std::vector<double> h_C(N * N);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        assert(h_C[i * N + i] > 0);  // Diagonal must be positive
        for (int j = 0; j < N; j++) {
            assert(fabs(h_C[i * N + j] - h_C[j * N + i]) < 1e-12);  // Symmetric
        }
    }
    printf("  cuBLAS DGEMM with workspace: OK\n");

    // Test cuSOLVER dsyevd (eigenvalue decomposition)
    int lwork = 0;
    cusolverDnDsyevd_bufferSize(ctx.cusolver, CUSOLVER_EIG_MODE_VECTOR,
                                 CUBLAS_FILL_MODE_UPPER, N, d_C, N,
                                 ctx.buf.eigenvalues, &lwork);
    double* d_work = ctx.scratch_pool.alloc<double>(lwork);
    cusolverStatus_t cs = cusolverDnDsyevd(ctx.cusolver, CUSOLVER_EIG_MODE_VECTOR,
                                            CUBLAS_FILL_MODE_UPPER, N, d_C, N,
                                            ctx.buf.eigenvalues, d_work, lwork,
                                            ctx.buf.cusolver_devinfo);
    assert(cs == CUSOLVER_STATUS_SUCCESS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check devinfo
    int devinfo;
    CUDA_CHECK(cudaMemcpy(&devinfo, ctx.buf.cusolver_devinfo, sizeof(int), cudaMemcpyDeviceToHost));
    assert(devinfo == 0);

    // Read eigenvalues — should be non-negative (A^T*A is PSD)
    std::vector<double> h_eig(N);
    CUDA_CHECK(cudaMemcpy(h_eig.data(), ctx.buf.eigenvalues, N * sizeof(double), cudaMemcpyDeviceToHost));
    printf("  cuSOLVER dsyevd eigenvalues: [");
    for (int i = 0; i < N; i++) {
        printf("%.4f%s", h_eig[i], i < N-1 ? ", " : "");
        assert(h_eig[i] >= -1e-10);
    }
    printf("]\n");
    printf("  cuSOLVER dsyevd: OK\n");

    ctx.scratch_pool.reset();
    printf("  PASSED\n\n");
}

void test_dual_stream() {
    printf("=== Test Dual Stream Overlap ===\n");

    auto& ctx = GPUContext::instance();
    int N = 100000;

    // Alloc from scratch
    double* d_a = ctx.scratch_pool.alloc<double>(N);
    double* d_b = ctx.scratch_pool.alloc<double>(N);

    // Launch kernel on compute_stream while doing memcpy on copy_stream
    int bs = 256;
    fill_kernel<<<ceildiv(N, bs), bs, 0, ctx.compute_stream>>>(d_a, N, 42.0);

    // Simultaneously copy something else on copy_stream
    std::vector<double> h_b(N, 7.0);
    CUDA_CHECK(cudaMemcpyAsync(d_b, h_b.data(), N * sizeof(double),
                                cudaMemcpyHostToDevice, ctx.copy_stream));

    CUDA_CHECK(cudaStreamSynchronize(ctx.compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.copy_stream));

    // Verify both
    double v1, v2;
    CUDA_CHECK(cudaMemcpy(&v1, d_a + N/2, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&v2, d_b + N/2, sizeof(double), cudaMemcpyDeviceToHost));
    assert(v1 == 42.0);
    assert(v2 == 7.0);
    printf("  Dual stream overlap: OK\n");

    ctx.scratch_pool.reset();
    printf("  PASSED\n\n");
}

int main() {
    printf("GPUContext Infrastructure Tests\n");
    printf("===============================\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    test_memory_pool();
    test_pinned_pool();
    test_scf_buffers();
    test_dual_stream();

    printf("All tests PASSED.\n");
    return 0;
}
