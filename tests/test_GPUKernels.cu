// GPU kernel correctness + performance tests
// Build: cmake -B build -DUSE_CUDA=ON -DBUILD_TESTS=ON && cmake --build build
// Run:   ./build/tests/lynx_gpu_tests

#ifdef USE_CUDA

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <cstdio>
#include <numeric>

#include "core/gpu_common.cuh"

// Forward declarations for GPU functions
namespace lynx { namespace gpu {
    void upload_stencil_coefficients(
        const double* D2x, const double* D2y, const double* D2z,
        const double* D1x, const double* D1y, const double* D1z,
        const double* D2xy, const double* D2xz, const double* D2yz,
        int FDn);
    void halo_exchange_gpu(const double* d_x, double* d_x_ex,
                           int nx, int ny, int nz, int FDn, int ncol,
                           bool px, bool py, bool pz);
    void halo_exchange_batched_gpu(const double* d_x, double* d_x_ex,
                                    int nx, int ny, int nz, int FDn, int ncol,
                                    bool px, bool py, bool pz);
    void halo_exchange_batched_nomemset_gpu(const double* d_x, double* d_x_ex,
                                             int nx, int ny, int nz, int FDn, int ncol,
                                             bool px, bool py, bool pz);
    void laplacian_orth_gpu(const double* d_x_ex, const double* d_V, double* d_y,
                            int nx, int ny, int nz, int FDn,
                            int nx_ex, int ny_ex,
                            double a, double b, double c,
                            double diag_coeff, int ncol);
    void laplacian_nonorth_gpu(const double* d_x_ex, const double* d_V, double* d_y,
                               int nx, int ny, int nz, int FDn,
                               int nx_ex, int ny_ex,
                               double a, double b, double c,
                               double diag_coeff,
                               bool has_xy, bool has_xz, bool has_yz,
                               int ncol);
    void gradient_gpu(const double* d_x_ex, double* d_y,
                      int nx, int ny, int nz, int FDn,
                      int nx_ex, int ny_ex,
                      int direction, int ncol);
    void hamiltonian_apply_local_gpu(
        const double* d_psi, const double* d_Veff, double* d_Hpsi,
        double* d_x_ex,
        int nx, int ny, int nz, int FDn, int ncol, double c,
        bool is_orthogonal,
        bool periodic_x, bool periodic_y, bool periodic_z,
        double diag_coeff,
        bool has_xy, bool has_xz, bool has_yz);
    // Device-metadata interface (hot path, 3 kernel launches)
    void nonlocal_projector_apply_gpu(
        const double* d_psi, double* d_Hpsi,
        const double* d_Chi_flat, const int* d_gpos_flat,
        const int* d_gpos_offsets, const int* d_chi_offsets,
        const int* d_ndc_arr, const int* d_nproj_arr,
        const int* d_IP_displ, const double* d_Gamma,
        double* d_alpha,
        int Nd, int ncol, double dV,
        int n_atoms, int total_nproj,
        int max_ndc, int max_nproj);
    // Convenience wrapper: takes host-side metadata, uploads to device
    void nonlocal_projector_apply_gpu(
        const double* d_psi, double* d_Hpsi,
        const double* d_Chi_flat, const int* d_gpos_flat,
        const double* d_Gamma, double* d_alpha,
        int Nd, int ncol, double dV,
        int n_atoms, int total_nproj,
        const int* h_gpos_offsets, const int* h_chi_offsets,
        const int* h_ndc_arr, const int* h_nproj_arr, const int* h_IP_displ,
        int max_ndc, int max_nproj);
    // V2: Template FDn + multi-column batching
    void laplacian_orth_v2_gpu(const double* d_x_ex, const double* d_V, double* d_y,
                                int nx, int ny, int nz, int FDn,
                                int nx_ex, int ny_ex,
                                double a, double b, double c,
                                double diag_coeff, int ncol);
    // V3: Shared memory 2D tiling + z-sweep (uses x_ex)
    void laplacian_orth_v3_gpu(const double* d_x_ex, const double* d_V, double* d_y,
                                int nx, int ny, int nz, int FDn,
                                int nx_ex, int ny_ex,
                                double a, double b, double c,
                                double diag_coeff, int ncol);
    // V4: Fused periodic (no x_ex, no halo exchange)
    void laplacian_orth_fused_gpu(const double* d_psi, const double* d_V, double* d_y,
                                   int nx, int ny, int nz, int FDn,
                                   double a, double b, double c,
                                   double diag_coeff, int ncol);
    // V5: Fused periodic + shared memory
    void laplacian_orth_v5_gpu(const double* d_psi, const double* d_V, double* d_y,
                                int nx, int ny, int nz, int FDn,
                                double a, double b, double c,
                                double diag_coeff, int ncol);
    // V6: One column per block (low register pressure)
    void laplacian_orth_v6_gpu(const double* d_x_ex, const double* d_V, double* d_y,
                                int nx, int ny, int nz, int FDn,
                                int nx_ex, int ny_ex,
                                double a, double b, double c,
                                double diag_coeff, int ncol);
    // V7: V6 + precomputed a*coeff for FMA
    void upload_precomputed_coefficients(const double* D2x, const double* D2y, const double* D2z,
                                          double a, int FDn);
    void laplacian_orth_v7_gpu(const double* d_x_ex, const double* d_V, double* d_y,
                                int nx, int ny, int nz, int FDn,
                                int nx_ex, int ny_ex,
                                double a, double b, double c,
                                double diag_coeff, int ncol);
    // V8: Multi-column loop + precomputed a*coeff (best of V2+V7)
    void laplacian_orth_v8_gpu(const double* d_x_ex, const double* d_V, double* d_y,
                                int nx, int ny, int nz, int FDn,
                                int nx_ex, int ny_ex,
                                double a, double b, double c,
                                double diag_coeff, int ncol);
    // Gradient V2: template FDn + multi-column batching
    void gradient_v2_gpu(const double* d_x_ex, double* d_y,
                          int nx, int ny, int nz, int FDn,
                          int nx_ex, int ny_ex,
                          int direction, int ncol);
}}

using namespace lynx::gpu;

// ============================================================
// CPU reference implementations (self-contained, no project deps)
// ============================================================
namespace cpu_ref {

void halo_copy_interior(const double* x, double* x_ex,
                        int nx, int ny, int nz, int FDn) {
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nxny_ex = nx_ex * ny_ex;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int loc = i + j * nx + k * nx * ny;
                int idx_ex = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
                x_ex[idx_ex] = x[loc];
            }
}

void halo_periodic_bc(double* x_ex, int nx, int ny, int nz, int FDn) {
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nz_ex = nz + 2 * FDn;
    int nxny_ex = nx_ex * ny_ex;

    // Z
    for (int k = 0; k < FDn; ++k) {
        std::memcpy(x_ex + k * nxny_ex, x_ex + (nz + k) * nxny_ex, nxny_ex * sizeof(double));
        std::memcpy(x_ex + (nz + FDn + k) * nxny_ex, x_ex + (FDn + k) * nxny_ex, nxny_ex * sizeof(double));
    }
    // Y
    for (int k = 0; k < nz_ex; ++k)
        for (int j = 0; j < FDn; ++j) {
            std::memcpy(x_ex + j * nx_ex + k * nxny_ex,
                       x_ex + (ny + j) * nx_ex + k * nxny_ex, nx_ex * sizeof(double));
            std::memcpy(x_ex + (ny + FDn + j) * nx_ex + k * nxny_ex,
                       x_ex + (FDn + j) * nx_ex + k * nxny_ex, nx_ex * sizeof(double));
        }
    // X
    for (int k = 0; k < nz_ex; ++k)
        for (int j = 0; j < ny_ex; ++j) {
            int base = j * nx_ex + k * nxny_ex;
            for (int i = 0; i < FDn; ++i) {
                x_ex[base + i] = x_ex[base + nx + i];
                x_ex[base + nx + FDn + i] = x_ex[base + FDn + i];
            }
        }
}

void halo_execute(const double* x, double* x_ex, int nx, int ny, int nz, int FDn) {
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nz_ex = nz + 2 * FDn;
    int nd_ex = nx_ex * ny_ex * nz_ex;
    std::memset(x_ex, 0, nd_ex * sizeof(double));
    halo_copy_interior(x, x_ex, nx, ny, nz, FDn);
    halo_periodic_bc(x_ex, nx, ny, nz, FDn);
}

void laplacian_orth(const double* x_ex, const double* V, double* y,
                    const double* cx, const double* cy, const double* cz,
                    int nx, int ny, int nz, int FDn,
                    double a, double b, double c) {
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nxny_ex = nx_ex * ny_ex;
    double diag = a * (cx[0] + cy[0] + cz[0]) + c;

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int idx = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
                int loc = i + j * nx + k * nx * ny;
                double val = diag * x_ex[idx];
                for (int p = 1; p <= FDn; ++p) {
                    val += a * cx[p] * (x_ex[idx + p] + x_ex[idx - p]);
                    val += a * cy[p] * (x_ex[idx + p * nx_ex] + x_ex[idx - p * nx_ex]);
                    val += a * cz[p] * (x_ex[idx + p * nxny_ex] + x_ex[idx - p * nxny_ex]);
                }
                if (V) val += b * V[loc] * x_ex[idx];
                y[loc] = val;
            }
}

void gradient(const double* x_ex, double* y, const double* coeff,
              int nx, int ny, int nz, int FDn, int stride) {
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nxny_ex = nx_ex * ny_ex;

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int idx = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nxny_ex;
                int loc = i + j * nx + k * nx * ny;
                double val = 0.0;
                for (int p = 1; p <= FDn; ++p)
                    val += coeff[p] * (x_ex[idx + p * stride] - x_ex[idx - p * stride]);
                y[loc] = val;
            }
}

} // namespace cpu_ref

// ============================================================
// Test fixture
// ============================================================
class GPUKernelTest : public ::testing::Test {
protected:
    // Typical BaTiO3-like grid: 26x26x26, FD order 12 (FDn=6)
    static constexpr int NX = 26, NY = 26, NZ = 26, FDN = 6;
    static constexpr int NX_EX = NX + 2 * FDN;
    static constexpr int NY_EX = NY + 2 * FDN;
    static constexpr int NZ_EX = NZ + 2 * FDN;
    static constexpr int ND = NX * NY * NZ;
    static constexpr int ND_EX = NX_EX * NY_EX * NZ_EX;

    // FD coefficients for order 12 (from LYNX reference)
    double D2_w[7] = {-3.277777777778, 1.777777777778, -0.311111111111,
                       0.075396825397, -0.017676767677, 0.003480963480, -0.000462962963};
    // Scale for uniform grid h=1.0 (tests don't need physical spacing)
    double D2x[7], D2y[7], D2z[7];
    double D1_w[7] = {0.0, 0.797979797980, -0.131313131313, 0.036630036630,
                      -0.011655011655, 0.003534003534, -0.000667000667};
    double D1x[7], D1y[7], D1z[7];

    std::vector<double> h_x, h_x_ex, h_y_cpu, h_y_gpu;
    std::vector<double> h_V;

    void SetUp() override {
        // Use spacing h=1.0 so coefficients = weights directly
        for (int i = 0; i < 7; ++i) {
            D2x[i] = D2_w[i]; D2y[i] = D2_w[i]; D2z[i] = D2_w[i];
            D1x[i] = D1_w[i]; D1y[i] = D1_w[i]; D1z[i] = D1_w[i];
        }

        // Upload to constant memory
        double zeros[7] = {};
        upload_stencil_coefficients(D2x, D2y, D2z, D1x, D1y, D1z, zeros, zeros, zeros, FDN);

        // Initialize test data: sin function on grid
        h_x.resize(ND);
        h_V.resize(ND);
        for (int k = 0; k < NZ; ++k)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i) {
                    int loc = i + j * NX + k * NX * NY;
                    double xi = 2.0 * M_PI * i / NX;
                    double yj = 2.0 * M_PI * j / NY;
                    double zk = 2.0 * M_PI * k / NZ;
                    h_x[loc] = std::sin(xi) * std::sin(yj) * std::sin(zk);
                    h_V[loc] = 1.0 + 0.1 * std::cos(xi);
                }

        h_x_ex.resize(ND_EX, 0.0);
        h_y_cpu.resize(ND);
        h_y_gpu.resize(ND);
    }
};

// ============================================================
// Test: HaloExchange GPU vs CPU
// ============================================================
TEST_F(GPUKernelTest, HaloExchange_Correctness) {
    // CPU reference
    std::vector<double> cpu_x_ex(ND_EX, 0.0);
    cpu_ref::halo_execute(h_x.data(), cpu_x_ex.data(), NX, NY, NZ, FDN);

    // GPU
    double *d_x, *d_x_ex;
    CUDA_CHECK(cudaMalloc(&d_x, ND * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x_ex, ND_EX * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), ND * sizeof(double), cudaMemcpyHostToDevice));

    halo_exchange_gpu(d_x, d_x_ex, NX, NY, NZ, FDN, 1, true, true, true);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> gpu_x_ex(ND_EX);
    CUDA_CHECK(cudaMemcpy(gpu_x_ex.data(), d_x_ex, ND_EX * sizeof(double), cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (int i = 0; i < ND_EX; ++i) {
        max_err = std::max(max_err, std::abs(gpu_x_ex[i] - cpu_x_ex[i]));
    }
    EXPECT_LT(max_err, 1e-14) << "HaloExchange GPU vs CPU max error: " << max_err;

    cudaFree(d_x);
    cudaFree(d_x_ex);
}

// ============================================================
// Test: Laplacian (orthogonal) GPU vs CPU
// ============================================================
TEST_F(GPUKernelTest, LaplacianOrth_Correctness) {
    // Prepare extended array on CPU
    cpu_ref::halo_execute(h_x.data(), h_x_ex.data(), NX, NY, NZ, FDN);

    double a = 1.0, b = 0.5, c = 0.1;
    double diag = a * (D2x[0] + D2y[0] + D2z[0]) + c;

    // CPU reference
    cpu_ref::laplacian_orth(h_x_ex.data(), h_V.data(), h_y_cpu.data(),
                            D2x, D2y, D2z, NX, NY, NZ, FDN, a, b, c);

    // GPU
    double *d_x_ex, *d_V, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x_ex, ND_EX * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, ND * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, ND * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x_ex, h_x_ex.data(), ND_EX * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), ND * sizeof(double), cudaMemcpyHostToDevice));

    laplacian_orth_gpu(d_x_ex, d_V, d_y, NX, NY, NZ, FDN, NX_EX, NY_EX,
                       a, b, c, diag, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, ND * sizeof(double), cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (int i = 0; i < ND; ++i)
        max_err = std::max(max_err, std::abs(h_y_gpu[i] - h_y_cpu[i]));

    EXPECT_LT(max_err, 1e-12) << "Laplacian orth GPU vs CPU max error: " << max_err;

    cudaFree(d_x_ex);
    cudaFree(d_V);
    cudaFree(d_y);
}

// ============================================================
// Test: Laplacian without diagonal (V=nullptr)
// ============================================================
TEST_F(GPUKernelTest, LaplacianOrth_NoDiag) {
    cpu_ref::halo_execute(h_x.data(), h_x_ex.data(), NX, NY, NZ, FDN);

    double a = -0.5, b = 0.0, c = 0.0;
    double diag = a * (D2x[0] + D2y[0] + D2z[0]) + c;

    cpu_ref::laplacian_orth(h_x_ex.data(), nullptr, h_y_cpu.data(),
                            D2x, D2y, D2z, NX, NY, NZ, FDN, a, b, c);

    double *d_x_ex, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x_ex, ND_EX * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, ND * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x_ex, h_x_ex.data(), ND_EX * sizeof(double), cudaMemcpyHostToDevice));

    laplacian_orth_gpu(d_x_ex, nullptr, d_y, NX, NY, NZ, FDN, NX_EX, NY_EX,
                       a, b, c, diag, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, ND * sizeof(double), cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (int i = 0; i < ND; ++i)
        max_err = std::max(max_err, std::abs(h_y_gpu[i] - h_y_cpu[i]));
    EXPECT_LT(max_err, 1e-12);

    cudaFree(d_x_ex);
    cudaFree(d_y);
}

// ============================================================
// Test: Gradient GPU vs CPU (all 3 directions)
// ============================================================
TEST_F(GPUKernelTest, Gradient_Correctness) {
    cpu_ref::halo_execute(h_x.data(), h_x_ex.data(), NX, NY, NZ, FDN);

    double *d_x_ex, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x_ex, ND_EX * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, ND * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x_ex, h_x_ex.data(), ND_EX * sizeof(double), cudaMemcpyHostToDevice));

    int strides[3] = {1, NX_EX, NX_EX * NY_EX};
    const double* coeffs[3] = {D1x, D1y, D1z};

    for (int dir = 0; dir < 3; ++dir) {
        // CPU reference
        cpu_ref::gradient(h_x_ex.data(), h_y_cpu.data(), coeffs[dir],
                          NX, NY, NZ, FDN, strides[dir]);

        // GPU
        gradient_gpu(d_x_ex, d_y, NX, NY, NZ, FDN, NX_EX, NY_EX, dir, 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, ND * sizeof(double), cudaMemcpyDeviceToHost));

        double max_err = 0.0;
        for (int i = 0; i < ND; ++i)
            max_err = std::max(max_err, std::abs(h_y_gpu[i] - h_y_cpu[i]));

        EXPECT_LT(max_err, 1e-13)
            << "Gradient dir=" << dir << " GPU vs CPU max error: " << max_err;
    }

    cudaFree(d_x_ex);
    cudaFree(d_y);
}

// ============================================================
// Test: Multi-column Laplacian
// ============================================================
TEST_F(GPUKernelTest, LaplacianOrth_MultiCol) {
    int ncol = 4;
    std::vector<double> h_x_multi(ND * ncol);
    std::vector<double> h_x_ex_multi(ND_EX * ncol);
    std::vector<double> h_y_cpu_multi(ND * ncol);
    std::vector<double> h_y_gpu_multi(ND * ncol);

    // Fill with different data per column
    for (int n = 0; n < ncol; ++n)
        for (int i = 0; i < ND; ++i)
            h_x_multi[n * ND + i] = h_x[i] * (n + 1);

    // CPU: halo + laplacian per column
    for (int n = 0; n < ncol; ++n) {
        cpu_ref::halo_execute(h_x_multi.data() + n * ND,
                              h_x_ex_multi.data() + n * ND_EX, NX, NY, NZ, FDN);
        cpu_ref::laplacian_orth(h_x_ex_multi.data() + n * ND_EX, nullptr,
                                h_y_cpu_multi.data() + n * ND,
                                D2x, D2y, D2z, NX, NY, NZ, FDN, 1.0, 0.0, 0.0);
    }

    // GPU
    double diag = D2x[0] + D2y[0] + D2z[0];
    double *d_x_ex, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x_ex, ND_EX * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, ND * ncol * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x_ex, h_x_ex_multi.data(), ND_EX * ncol * sizeof(double), cudaMemcpyHostToDevice));

    laplacian_orth_gpu(d_x_ex, nullptr, d_y, NX, NY, NZ, FDN, NX_EX, NY_EX,
                       1.0, 0.0, 0.0, diag, ncol);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_y_gpu_multi.data(), d_y, ND * ncol * sizeof(double), cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (int i = 0; i < ND * ncol; ++i)
        max_err = std::max(max_err, std::abs(h_y_gpu_multi[i] - h_y_cpu_multi[i]));
    EXPECT_LT(max_err, 1e-12) << "Multi-col Laplacian max error: " << max_err;

    cudaFree(d_x_ex);
    cudaFree(d_y);
}

// ============================================================
// Test: Hamiltonian local (halo + laplacian fused)
// ============================================================
TEST_F(GPUKernelTest, HamiltonianLocal_Correctness) {
    double c_shift = 0.05;

    // CPU reference: halo then laplacian with a=-0.5, b=1.0
    cpu_ref::halo_execute(h_x.data(), h_x_ex.data(), NX, NY, NZ, FDN);
    // Manual H*psi = -0.5*Lap*psi + V*psi + c*psi
    {
        int nx_ex = NX_EX, ny_ex = NY_EX;
        int nxny_ex = nx_ex * ny_ex;
        double diag_lap = -0.5 * (D2x[0] + D2y[0] + D2z[0]);
        for (int k = 0; k < NZ; ++k)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i) {
                    int idx_ex = (i + FDN) + (j + FDN) * nx_ex + (k + FDN) * nxny_ex;
                    int loc = i + j * NX + k * NX * NY;
                    double lap = (D2x[0] + D2y[0] + D2z[0]) * h_x_ex[idx_ex];
                    for (int p = 1; p <= FDN; ++p) {
                        lap += D2x[p] * (h_x_ex[idx_ex + p] + h_x_ex[idx_ex - p]);
                        lap += D2y[p] * (h_x_ex[idx_ex + p * nx_ex] + h_x_ex[idx_ex - p * nx_ex]);
                        lap += D2z[p] * (h_x_ex[idx_ex + p * nxny_ex] + h_x_ex[idx_ex - p * nxny_ex]);
                    }
                    h_y_cpu[loc] = -0.5 * lap + (h_V[loc] + c_shift) * h_x_ex[idx_ex];
                }
    }

    // GPU
    double diag_coeff = -0.5 * (D2x[0] + D2y[0] + D2z[0]) + c_shift;
    double *d_psi, *d_Veff, *d_Hpsi, *d_x_ex;
    CUDA_CHECK(cudaMalloc(&d_psi, ND * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Veff, ND * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Hpsi, ND * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x_ex, ND_EX * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_psi, h_x.data(), ND * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Veff, h_V.data(), ND * sizeof(double), cudaMemcpyHostToDevice));

    hamiltonian_apply_local_gpu(d_psi, d_Veff, d_Hpsi, d_x_ex,
                                NX, NY, NZ, FDN, 1, c_shift,
                                true, true, true, true,
                                diag_coeff, false, false, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_Hpsi, ND * sizeof(double), cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (int i = 0; i < ND; ++i)
        max_err = std::max(max_err, std::abs(h_y_gpu[i] - h_y_cpu[i]));
    EXPECT_LT(max_err, 1e-12) << "Hamiltonian local GPU vs CPU max error: " << max_err;

    cudaFree(d_psi);
    cudaFree(d_Veff);
    cudaFree(d_Hpsi);
    cudaFree(d_x_ex);
}

// ============================================================
// Helper: build NonlocalProjector test data with proper Chi layout
// Chi_flat is stored as concatenated per-atom (ndc, nproj) blocks.
// chi_offsets[iat] gives the offset into Chi_flat for atom iat.
// gpos_offsets[iat] gives the offset into gpos_flat for atom iat.
// ============================================================
struct NlpTestData {
    int n_influence, ncol, Nd;
    double dV;
    std::vector<int> ndc_arr, nproj_arr, IP_displ;
    std::vector<int> gpos_offsets;   // CSR into gpos_flat
    std::vector<int> chi_offsets;    // CSR into Chi_flat
    std::vector<int> gpos_flat;
    std::vector<double> Chi_flat;
    std::vector<double> Gamma;
    std::vector<double> psi;
    int total_nproj;
    int total_gpos;
    int total_chi;
    int max_ndc;

    void build(int n_infl, const int* ndcs, const int* nprojs, int ncol_, int Nd_, double dV_, unsigned seed) {
        n_influence = n_infl; ncol = ncol_; Nd = Nd_; dV = dV_;
        ndc_arr.assign(ndcs, ndcs + n_influence);
        nproj_arr.assign(nprojs, nprojs + n_influence);

        gpos_offsets.resize(n_influence + 1, 0);
        chi_offsets.resize(n_influence + 1, 0);
        IP_displ.resize(n_influence);
        total_nproj = 0; max_ndc = 0;
        for (int i = 0; i < n_influence; ++i) {
            gpos_offsets[i + 1] = gpos_offsets[i] + ndc_arr[i];
            chi_offsets[i + 1] = chi_offsets[i] + ndc_arr[i] * nproj_arr[i];
            IP_displ[i] = total_nproj;
            total_nproj += nproj_arr[i];
            max_ndc = std::max(max_ndc, ndc_arr[i]);
        }
        total_gpos = gpos_offsets[n_influence];
        total_chi = chi_offsets[n_influence];

        gpos_flat.resize(total_gpos);
        Chi_flat.resize(total_chi);
        Gamma.resize(total_nproj);

        srand(seed);
        for (int i = 0; i < total_gpos; ++i)
            gpos_flat[i] = rand() % Nd;
        for (int i = 0; i < total_chi; ++i)
            Chi_flat[i] = 0.01 * ((rand() % 2000) - 1000);
        for (int i = 0; i < total_nproj; ++i)
            Gamma[i] = 0.5 + 0.1 * (i % 5);

        psi.resize(Nd * ncol);
        for (int i = 0; i < Nd * ncol; ++i)
            psi[i] = std::sin(0.01 * i) * ((i / Nd) + 1);
    }

    // CPU reference: returns Hpsi
    std::vector<double> cpu_apply() const {
        std::vector<double> alpha(total_nproj * ncol, 0.0);
        // Step 1: Chi^T * psi
        for (int iat = 0; iat < n_influence; ++iat) {
            int ndc = ndc_arr[iat];
            int np = nproj_arr[iat];
            int goff = gpos_offsets[iat];
            int coff = chi_offsets[iat];
            int abase = IP_displ[iat];
            for (int n = 0; n < ncol; ++n)
                for (int jp = 0; jp < np; ++jp) {
                    double dot = 0.0;
                    for (int ig = 0; ig < ndc; ++ig)
                        dot += Chi_flat[coff + ig + jp * ndc] * psi[gpos_flat[goff + ig] + n * Nd];
                    alpha[(abase + jp) * ncol + n] = dot * dV;
                }
        }
        // Step 2: Gamma
        for (int ip = 0; ip < total_nproj; ++ip)
            for (int n = 0; n < ncol; ++n)
                alpha[ip * ncol + n] *= Gamma[ip];
        // Step 3: Chi * alpha
        std::vector<double> Hpsi(Nd * ncol, 0.0);
        for (int iat = 0; iat < n_influence; ++iat) {
            int ndc = ndc_arr[iat];
            int np = nproj_arr[iat];
            int goff = gpos_offsets[iat];
            int coff = chi_offsets[iat];
            int abase = IP_displ[iat];
            for (int n = 0; n < ncol; ++n)
                for (int ig = 0; ig < ndc; ++ig) {
                    double val = 0.0;
                    for (int jp = 0; jp < np; ++jp)
                        val += Chi_flat[coff + ig + jp * ndc] * alpha[(abase + jp) * ncol + n];
                    Hpsi[gpos_flat[goff + ig] + n * Nd] += val;
                }
        }
        return Hpsi;
    }

    // Upload to device and call GPU kernel, return Hpsi
    std::vector<double> gpu_apply() const {
        double *d_psi_d, *d_Hpsi_d, *d_Chi_d, *d_Gamma_d, *d_alpha_d;
        int *d_gpos_d;
        CUDA_CHECK(cudaMalloc(&d_psi_d, Nd * ncol * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Hpsi_d, Nd * ncol * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Chi_d, total_chi * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_gpos_d, total_gpos * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_Gamma_d, total_nproj * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_alpha_d, total_nproj * ncol * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_psi_d, psi.data(), Nd * ncol * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_Hpsi_d, 0, Nd * ncol * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_Chi_d, Chi_flat.data(), total_chi * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gpos_d, gpos_flat.data(), total_gpos * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Gamma_d, Gamma.data(), total_nproj * sizeof(double), cudaMemcpyHostToDevice));

        // Compute max_nproj
        int max_np = 0;
        for (int i = 0; i < n_influence; ++i)
            max_np = std::max(max_np, nproj_arr[i]);

        nonlocal_projector_apply_gpu(
            d_psi_d, d_Hpsi_d, d_Chi_d, d_gpos_d, d_Gamma_d, d_alpha_d,
            Nd, ncol, dV, n_influence, total_nproj,
            gpos_offsets.data(), chi_offsets.data(),
            ndc_arr.data(), nproj_arr.data(), IP_displ.data(),
            max_ndc, max_np);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> Hpsi(Nd * ncol);
        CUDA_CHECK(cudaMemcpy(Hpsi.data(), d_Hpsi_d, Nd * ncol * sizeof(double), cudaMemcpyDeviceToHost));

        cudaFree(d_psi_d); cudaFree(d_Hpsi_d); cudaFree(d_Chi_d);
        cudaFree(d_gpos_d); cudaFree(d_Gamma_d); cudaFree(d_alpha_d);
        return Hpsi;
    }
};

// ============================================================
// Test: NonlocalProjector GPU vs CPU (varying atom sizes)
// ============================================================
TEST_F(GPUKernelTest, NonlocalProjector_Correctness) {
    // 4 influence atoms with different ndc and nproj
    int ndcs[] = {100, 80, 150, 60};
    int nprojs[] = {3, 9, 4, 9};
    NlpTestData data;
    data.build(4, ndcs, nprojs, 2, ND, 0.01, 42);

    auto Hpsi_cpu = data.cpu_apply();
    auto Hpsi_gpu = data.gpu_apply();

    double max_err = 0.0, max_val = 0.0;
    for (int i = 0; i < ND * 2; ++i) {
        max_err = std::max(max_err, std::abs(Hpsi_gpu[i] - Hpsi_cpu[i]));
        max_val = std::max(max_val, std::abs(Hpsi_cpu[i]));
    }
    double rel_err = max_err / (max_val + 1e-30);
    EXPECT_LT(rel_err, 1e-10)
        << "NonlocalProjector GPU vs CPU relative error: " << rel_err
        << " (abs: " << max_err << ", max: " << max_val << ")";
}

// ============================================================
// Multi-scale performance benchmarks
// Grid sizes: 26³ (BaTiO3), 48³ (Si supercell), 72³, 100³
// ============================================================

static void bench_laplacian(int N, int ncol, const double* D2x, const double* D2y, const double* D2z,
                            const double* D1x, const double* D1y, const double* D1z) {
    int FDn = 6;
    int nd = N * N * N;
    int nx_ex = N + 2 * FDn, ny_ex = N + 2 * FDn, nz_ex = N + 2 * FDn;
    int nd_ex = nx_ex * ny_ex * nz_ex;
    double diag = D2x[0] + D2y[0] + D2z[0];

    // Prepare extended data on CPU
    std::vector<double> h_x_loc(nd);
    for (int i = 0; i < nd; ++i)
        h_x_loc[i] = std::sin(0.01 * i);

    std::vector<double> h_x_ex_multi(nd_ex * ncol);
    for (int n = 0; n < ncol; ++n) {
        std::vector<double> xn(nd);
        for (int i = 0; i < nd; ++i) xn[i] = h_x_loc[i] * (n + 1);
        cpu_ref::halo_execute(xn.data(), h_x_ex_multi.data() + n * nd_ex, N, N, N, FDn);
    }

    // CPU timing
    std::vector<double> y_cpu(nd);
    int cpu_iters = std::max(1, 500000 / (nd * ncol));  // scale iters to problem size
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < cpu_iters; ++it)
        for (int n = 0; n < ncol; ++n)
            cpu_ref::laplacian_orth(h_x_ex_multi.data() + n * nd_ex, nullptr,
                                    y_cpu.data(), D2x, D2y, D2z, N, N, N, FDn, 1.0, 0.0, 0.0);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / cpu_iters;

    // GPU timing
    double *d_x_ex, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x_ex, (size_t)nd_ex * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)nd * ncol * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x_ex, h_x_ex_multi.data(), (size_t)nd_ex * ncol * sizeof(double), cudaMemcpyHostToDevice));

    for (int it = 0; it < 10; ++it)
        laplacian_orth_gpu(d_x_ex, nullptr, d_y, N, N, N, FDn, nx_ex, ny_ex,
                           1.0, 0.0, 0.0, diag, ncol);
    CUDA_CHECK(cudaDeviceSynchronize());

    int gpu_iters = std::max(10, 50000000 / (nd * ncol));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int it = 0; it < gpu_iters; ++it)
        laplacian_orth_gpu(d_x_ex, nullptr, d_y, N, N, N, FDn, nx_ex, ny_ex,
                           1.0, 0.0, 0.0, diag, ncol);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float gpu_ms_total; cudaEventElapsedTime(&gpu_ms_total, start, stop);
    double gpu_ms = gpu_ms_total / gpu_iters;

    double bytes = ncol * ((double)nd_ex + nd) * sizeof(double);
    std::printf("  %3d³ × %2d cols: CPU %8.3f ms  GPU %8.3f ms  %6.1fx  BW %.0f GB/s\n",
                N, ncol, cpu_ms, gpu_ms, cpu_ms / gpu_ms, bytes / (gpu_ms * 1e-3) / 1e9);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_x_ex); cudaFree(d_y);
}

static void bench_hamiltonian(int N, int ncol, const double* D2x, const double* D2y, const double* D2z) {
    int FDn = 6;
    int nd = N * N * N;
    int nx_ex = N + 2 * FDn, ny_ex = N + 2 * FDn, nz_ex = N + 2 * FDn;
    int nd_ex = nx_ex * ny_ex * nz_ex;
    double diag_coeff = -0.5 * (D2x[0] + D2y[0] + D2z[0]);

    std::vector<double> h_psi(nd * ncol), h_V(nd);
    for (int i = 0; i < nd * ncol; ++i) h_psi[i] = std::sin(0.01 * i);
    for (int i = 0; i < nd; ++i) h_V[i] = 1.0 + 0.1 * std::cos(0.02 * i);

    // CPU
    std::vector<double> cpu_x_ex(nd_ex), cpu_y(nd);
    int cpu_iters = std::max(1, 200000 / (nd * ncol));
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < cpu_iters; ++it)
        for (int n = 0; n < ncol; ++n) {
            cpu_ref::halo_execute(h_psi.data() + n * nd, cpu_x_ex.data(), N, N, N, FDn);
            cpu_ref::laplacian_orth(cpu_x_ex.data(), h_V.data(), cpu_y.data(),
                                    D2x, D2y, D2z, N, N, N, FDn, -0.5, 1.0, 0.0);
        }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / cpu_iters;

    // GPU
    double *d_psi, *d_Veff, *d_Hpsi, *d_x_ex;
    CUDA_CHECK(cudaMalloc(&d_psi, (size_t)nd * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Veff, (size_t)nd * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Hpsi, (size_t)nd * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x_ex, (size_t)nd_ex * ncol * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_psi, h_psi.data(), (size_t)nd * ncol * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Veff, h_V.data(), (size_t)nd * sizeof(double), cudaMemcpyHostToDevice));

    for (int it = 0; it < 5; ++it)
        hamiltonian_apply_local_gpu(d_psi, d_Veff, d_Hpsi, d_x_ex,
                                    N, N, N, FDn, ncol, 0.0,
                                    true, true, true, true,
                                    diag_coeff, false, false, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    int gpu_iters = std::max(10, 20000000 / (nd * ncol));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int it = 0; it < gpu_iters; ++it)
        hamiltonian_apply_local_gpu(d_psi, d_Veff, d_Hpsi, d_x_ex,
                                    N, N, N, FDn, ncol, 0.0,
                                    true, true, true, true,
                                    diag_coeff, false, false, false);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float gpu_ms_total; cudaEventElapsedTime(&gpu_ms_total, start, stop);
    double gpu_ms = gpu_ms_total / gpu_iters;

    std::printf("  %3d³ × %2d cols: CPU %8.3f ms  GPU %8.3f ms  %6.1fx\n",
                N, ncol, cpu_ms, gpu_ms, cpu_ms / gpu_ms);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_psi); cudaFree(d_Veff); cudaFree(d_Hpsi); cudaFree(d_x_ex);
}

static void bench_nonlocal(int Nd, int n_influence, int ndc_per_atom, int nproj_per_atom, int ncol) {
    NlpTestData data;
    std::vector<int> ndcs(n_influence, ndc_per_atom);
    std::vector<int> nprojs(n_influence, nproj_per_atom);
    data.build(n_influence, ndcs.data(), nprojs.data(), ncol, Nd, 0.01, 123);

    // CPU timing
    int cpu_iters = std::max(1, 2000000 / (n_influence * ndc_per_atom * nproj_per_atom * ncol));
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < cpu_iters; ++it)
        data.cpu_apply();
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / cpu_iters;

    // GPU: allocate once, upload metadata to device for hot-path interface
    double *d_psi, *d_Hpsi, *d_Chi, *d_Gamma_d, *d_alpha;
    int *d_gpos, *d_gpos_off, *d_chi_off, *d_ndc_d, *d_nproj_d, *d_ip_d;
    CUDA_CHECK(cudaMalloc(&d_psi, (size_t)Nd * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Hpsi, (size_t)Nd * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Chi, data.total_chi * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gpos, data.total_gpos * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Gamma_d, data.total_nproj * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_alpha, data.total_nproj * ncol * sizeof(double)));
    // Device metadata
    CUDA_CHECK(cudaMalloc(&d_gpos_off, (n_influence + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_chi_off, (n_influence + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ndc_d, n_influence * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nproj_d, n_influence * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ip_d, n_influence * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_psi, data.psi.data(), (size_t)Nd * ncol * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Chi, data.Chi_flat.data(), data.total_chi * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gpos, data.gpos_flat.data(), data.total_gpos * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Gamma_d, data.Gamma.data(), data.total_nproj * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gpos_off, data.gpos_offsets.data(), (n_influence + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chi_off, data.chi_offsets.data(), (n_influence + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ndc_d, data.ndc_arr.data(), n_influence * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nproj_d, data.nproj_arr.data(), n_influence * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ip_d, data.IP_displ.data(), n_influence * sizeof(int), cudaMemcpyHostToDevice));

    // Warmup
    for (int it = 0; it < 5; ++it) {
        CUDA_CHECK(cudaMemset(d_Hpsi, 0, (size_t)Nd * ncol * sizeof(double)));
        nonlocal_projector_apply_gpu(
            d_psi, d_Hpsi, d_Chi, d_gpos,
            d_gpos_off, d_chi_off, d_ndc_d, d_nproj_d, d_ip_d,
            d_Gamma_d, d_alpha,
            Nd, ncol, data.dV, n_influence, data.total_nproj,
            data.max_ndc, nproj_per_atom);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    int gpu_iters = std::max(10, 20000000 / (n_influence * ndc_per_atom * nproj_per_atom * ncol));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int it = 0; it < gpu_iters; ++it) {
        CUDA_CHECK(cudaMemset(d_Hpsi, 0, (size_t)Nd * ncol * sizeof(double)));
        nonlocal_projector_apply_gpu(
            d_psi, d_Hpsi, d_Chi, d_gpos,
            d_gpos_off, d_chi_off, d_ndc_d, d_nproj_d, d_ip_d,
            d_Gamma_d, d_alpha,
            Nd, ncol, data.dV, n_influence, data.total_nproj,
            data.max_ndc, nproj_per_atom);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float gpu_ms_total; cudaEventElapsedTime(&gpu_ms_total, start, stop);
    double gpu_ms = gpu_ms_total / gpu_iters;

    std::printf("  Nd=%7d, %3d atoms × %3d pts × %d proj, %2d cols: CPU %8.3f ms  GPU %8.3f ms  %6.1fx\n",
                Nd, n_influence, ndc_per_atom, nproj_per_atom, ncol,
                cpu_ms, gpu_ms, cpu_ms / gpu_ms);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_psi); cudaFree(d_Hpsi); cudaFree(d_Chi);
    cudaFree(d_gpos); cudaFree(d_Gamma_d); cudaFree(d_alpha);
    cudaFree(d_gpos_off); cudaFree(d_chi_off);
    cudaFree(d_ndc_d); cudaFree(d_nproj_d); cudaFree(d_ip_d);
}

// ============================================================
// Laplacian benchmark: multiple grid sizes
// ============================================================
TEST_F(GPUKernelTest, PERF_Laplacian_MultiScale) {
    std::printf("\n[PERF] Laplacian orth (FD order 12):\n");
    bench_laplacian(26,  12, D2x, D2y, D2z, D1x, D1y, D1z);  // BaTiO3
    bench_laplacian(48,  12, D2x, D2y, D2z, D1x, D1y, D1z);  // Si supercell
    bench_laplacian(72,  12, D2x, D2y, D2z, D1x, D1y, D1z);  // medium
    bench_laplacian(100, 12, D2x, D2y, D2z, D1x, D1y, D1z);  // large
    bench_laplacian(100, 30, D2x, D2y, D2z, D1x, D1y, D1z);  // large + many bands
}

// ============================================================
// Hamiltonian benchmark: multiple grid sizes
// ============================================================
TEST_F(GPUKernelTest, PERF_Hamiltonian_MultiScale) {
    std::printf("\n[PERF] Hamiltonian local (halo + Lap + Veff, FD order 12):\n");
    bench_hamiltonian(26,  12, D2x, D2y, D2z);
    bench_hamiltonian(48,  12, D2x, D2y, D2z);
    bench_hamiltonian(72,  12, D2x, D2y, D2z);
    bench_hamiltonian(100, 12, D2x, D2y, D2z);
    bench_hamiltonian(100, 30, D2x, D2y, D2z);
}

// ============================================================
// Hamiltonian halo variant comparison: memset vs no-memset
// ============================================================
static void bench_hamiltonian_halo_variants(int N, int ncol, const double* D2x, const double* D2y, const double* D2z) {
    int FDn = 6;
    int nd = N * N * N;
    int nx_ex = N + 2 * FDn, ny_ex = N + 2 * FDn, nz_ex = N + 2 * FDn;
    int nd_ex = nx_ex * ny_ex * nz_ex;
    double diag_coeff = -0.5 * (D2x[0] + D2y[0] + D2z[0]);

    std::vector<double> h_psi(nd * ncol), h_V(nd);
    for (int i = 0; i < nd * ncol; ++i) h_psi[i] = std::sin(0.01 * i);
    for (int i = 0; i < nd; ++i) h_V[i] = 1.0 + 0.1 * std::cos(0.02 * i);

    double *d_psi, *d_Veff, *d_Hpsi, *d_Hpsi2, *d_x_ex;
    CUDA_CHECK(cudaMalloc(&d_psi, (size_t)nd * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Veff, (size_t)nd * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Hpsi, (size_t)nd * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Hpsi2, (size_t)nd * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x_ex, (size_t)nd_ex * ncol * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_psi, h_psi.data(), (size_t)nd * ncol * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Veff, h_V.data(), (size_t)nd * sizeof(double), cudaMemcpyHostToDevice));

    int gpu_iters = std::max(10, 20000000 / (nd * ncol));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float gpu_ms_total;

    auto bench_ham = [&](const char* label, auto halo_fn) {
        for (int it = 0; it < 5; ++it) {
            halo_fn(d_psi, d_x_ex, N, N, N, FDn, ncol, true, true, true);
            laplacian_orth_v2_gpu(d_x_ex, d_Veff, d_Hpsi, N, N, N, FDn, nx_ex, ny_ex,
                                  -0.5, 1.0, 0.0, diag_coeff, ncol);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(start);
        for (int it = 0; it < gpu_iters; ++it) {
            halo_fn(d_psi, d_x_ex, N, N, N, FDn, ncol, true, true, true);
            laplacian_orth_v2_gpu(d_x_ex, d_Veff, d_Hpsi, N, N, N, FDn, nx_ex, ny_ex,
                                  -0.5, 1.0, 0.0, diag_coeff, ncol);
        }
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_ms_total, start, stop);
        double ms = gpu_ms_total / gpu_iters;
        std::printf("    %-18s: %7.3f ms\n", label, ms);
    };

    std::printf("  %3d³ × %2d cols:\n", N, ncol);
    bench_ham("V2-memset", halo_exchange_batched_gpu);
    bench_ham("V3-nomemset", halo_exchange_batched_nomemset_gpu);

    // Correctness: compare V2 vs V3 output
    halo_exchange_batched_gpu(d_psi, d_x_ex, N, N, N, FDn, ncol, true, true, true);
    laplacian_orth_v2_gpu(d_x_ex, d_Veff, d_Hpsi, N, N, N, FDn, nx_ex, ny_ex,
                          -0.5, 1.0, 0.0, diag_coeff, ncol);
    halo_exchange_batched_nomemset_gpu(d_psi, d_x_ex, N, N, N, FDn, ncol, true, true, true);
    laplacian_orth_v2_gpu(d_x_ex, d_Veff, d_Hpsi2, N, N, N, FDn, nx_ex, ny_ex,
                          -0.5, 1.0, 0.0, diag_coeff, ncol);

    std::vector<double> h1(nd * ncol), h2(nd * ncol);
    CUDA_CHECK(cudaMemcpy(h1.data(), d_Hpsi, (size_t)nd * ncol * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h2.data(), d_Hpsi2, (size_t)nd * ncol * sizeof(double), cudaMemcpyDeviceToHost));
    double max_err = 0;
    for (size_t i = 0; i < h1.size(); ++i)
        max_err = std::max(max_err, std::abs(h1[i] - h2[i]));
    std::printf("    V3 vs V2 max err: %.2e\n", max_err);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_psi); cudaFree(d_Veff); cudaFree(d_Hpsi); cudaFree(d_Hpsi2); cudaFree(d_x_ex);
}

TEST_F(GPUKernelTest, PERF_Hamiltonian_HaloVariants) {
    std::printf("\n[PERF] Hamiltonian halo exchange comparison (V2-memset vs V3-nomemset):\n");
    bench_hamiltonian_halo_variants(26,  12, D2x, D2y, D2z);
    bench_hamiltonian_halo_variants(48,  12, D2x, D2y, D2z);
    bench_hamiltonian_halo_variants(100, 12, D2x, D2y, D2z);
    bench_hamiltonian_halo_variants(100, 30, D2x, D2y, D2z);
}

// ============================================================
// Gradient benchmark: V0 (per-col) vs V2 (template+multicol)
// ============================================================
static void bench_gradient_variants(int N, int ncol, const double* D1x, const double* D1y, const double* D1z) {
    int FDn = 6;
    int nd = N * N * N;
    int nx_ex = N + 2 * FDn, ny_ex = N + 2 * FDn, nz_ex = N + 2 * FDn;
    int nxny_ex = nx_ex * ny_ex;
    int nd_ex = nxny_ex * nz_ex;

    std::vector<double> h_psi(nd * ncol);
    for (int i = 0; i < nd * ncol; ++i) h_psi[i] = std::sin(0.01 * i);

    // Build x_ex on host
    std::vector<double> h_x_ex_multi((size_t)nd_ex * ncol);
    for (int n = 0; n < ncol; ++n) {
        std::vector<double> xn(nd);
        for (int i = 0; i < nd; ++i) xn[i] = h_psi[i + n * nd];
        cpu_ref::halo_execute(xn.data(), h_x_ex_multi.data() + n * nd_ex, N, N, N, FDn);
    }

    double *d_x_ex, *d_y, *d_y2;
    CUDA_CHECK(cudaMalloc(&d_x_ex, (size_t)nd_ex * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)nd * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y2, (size_t)nd * ncol * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x_ex, h_x_ex_multi.data(), (size_t)nd_ex * ncol * sizeof(double), cudaMemcpyHostToDevice));

    // CPU reference: gradient in 3 directions for all columns
    const double* coeffs[3] = {D1x, D1y, D1z};
    int strides[3] = {1, nx_ex, nxny_ex};
    std::vector<double> cpu_x_ex(nd_ex), cpu_y(nd);
    int cpu_iters = std::max(1, 500000 / (nd * ncol));
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < cpu_iters; ++it)
        for (int n = 0; n < ncol; ++n) {
            cpu_ref::halo_execute(h_psi.data() + n * nd, cpu_x_ex.data(), N, N, N, FDn);
            for (int dir = 0; dir < 3; ++dir)
                cpu_ref::gradient(cpu_x_ex.data(), cpu_y.data(),
                                  coeffs[dir], N, N, N, FDn, strides[dir]);
        }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / cpu_iters;

    int gpu_iters = std::max(10, 50000000 / (nd * ncol));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float gpu_ms_total;

    auto run_bench = [&](const char* label, auto fn) {
        for (int it = 0; it < 5; ++it) fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(start);
        for (int it = 0; it < gpu_iters; ++it) fn();
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_ms_total, start, stop);
        double ms = gpu_ms_total / gpu_iters;
        std::printf("    %-12s: %7.3f ms  %6.1fx vs CPU\n", label, ms, cpu_ms / ms);
    };

    std::printf("  %3d³ × %2d cols (3 dirs):\n", N, ncol);

    // V0: per-column, runtime FDn
    run_bench("V0-percol", [&]() {
        for (int dir = 0; dir < 3; ++dir)
            gradient_gpu(d_x_ex, d_y, N, N, N, FDn, nx_ex, ny_ex, dir, ncol);
    });

    // V2: template FDn + multi-column
    run_bench("V2-multicol", [&]() {
        for (int dir = 0; dir < 3; ++dir)
            gradient_v2_gpu(d_x_ex, d_y2, N, N, N, FDn, nx_ex, ny_ex, dir, ncol);
    });

    // Correctness: V2 vs V0
    for (int dir = 0; dir < 3; ++dir)
        gradient_gpu(d_x_ex, d_y, N, N, N, FDn, nx_ex, ny_ex, dir, ncol);
    for (int dir = 0; dir < 3; ++dir)
        gradient_v2_gpu(d_x_ex, d_y2, N, N, N, FDn, nx_ex, ny_ex, dir, ncol);
    // Check last direction (all write to same d_y/d_y2)
    std::vector<double> h1(nd * ncol), h2(nd * ncol);
    CUDA_CHECK(cudaMemcpy(h1.data(), d_y, (size_t)nd * ncol * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h2.data(), d_y2, (size_t)nd * ncol * sizeof(double), cudaMemcpyDeviceToHost));
    double max_err = 0;
    for (size_t i = 0; i < h1.size(); ++i)
        max_err = std::max(max_err, std::abs(h1[i] - h2[i]));
    std::printf("    V2 vs V0 max err: %.2e\n", max_err);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_x_ex); cudaFree(d_y); cudaFree(d_y2);
}

TEST_F(GPUKernelTest, PERF_Gradient_Variants) {
    std::printf("\n[PERF] Gradient kernel variants (V0 vs V2, all 3 directions):\n");
    bench_gradient_variants(26,  12, D1x, D1y, D1z);
    bench_gradient_variants(48,  12, D1x, D1y, D1z);
    bench_gradient_variants(72,  12, D1x, D1y, D1z);
    bench_gradient_variants(100, 12, D1x, D1y, D1z);
    bench_gradient_variants(100, 30, D1x, D1y, D1z);
}

// ============================================================
// NonlocalProjector benchmark: multiple scales
// ============================================================
TEST_F(GPUKernelTest, PERF_NonlocalProjector_MultiScale) {
    std::printf("\n[PERF] NonlocalProjector (gather → cuBLAS DGEMM → scatter):\n");
    // Small: BaTiO3 (5 atoms, ~20 images, 26³)
    bench_nonlocal(26*26*26,   20,  200, 9, 12);
    // Medium: Si supercell (16 atoms, ~40 images, 48³)
    bench_nonlocal(48*48*48,   40,  300, 4, 12);
    // Large: 72³ grid, 60 influence atoms
    bench_nonlocal(72*72*72,   60,  400, 9, 12);
    // Large with 30 cols (isolate column count effect)
    bench_nonlocal(72*72*72,   60,  400, 9, 30);
    // XL: 100³ grid, 100 influence atoms, 12 cols (isolate grid size effect)
    bench_nonlocal(100*100*100, 100, 500, 9, 12);
    // XL: 100³ grid, 100 influence atoms, 30 bands
    bench_nonlocal(100*100*100, 100, 500, 9, 30);
}

// ============================================================
// Laplacian optimization comparison benchmark
// ============================================================
static void bench_laplacian_variants(int N, int ncol, const double* D2x, const double* D2y, const double* D2z) {
    int FDn = 6;
    int nd = N * N * N;
    int nx_ex = N + 2 * FDn, ny_ex = N + 2 * FDn, nz_ex = N + 2 * FDn;
    int nd_ex = nx_ex * ny_ex * nz_ex;
    double diag = D2x[0] + D2y[0] + D2z[0];

    // Prepare compact psi data
    std::vector<double> h_psi(nd * ncol);
    for (int i = 0; i < nd * ncol; ++i)
        h_psi[i] = std::sin(0.01 * i);

    // Prepare extended data for V0/V2/V3
    std::vector<double> h_x_ex_multi((size_t)nd_ex * ncol);
    for (int n = 0; n < ncol; ++n) {
        std::vector<double> xn(nd);
        for (int i = 0; i < nd; ++i) xn[i] = h_psi[i + n * nd];
        cpu_ref::halo_execute(xn.data(), h_x_ex_multi.data() + n * nd_ex, N, N, N, FDn);
    }

    // Allocate GPU memory
    double *d_x_ex, *d_psi, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x_ex, (size_t)nd_ex * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_psi, (size_t)nd * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)nd * ncol * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x_ex, h_x_ex_multi.data(), (size_t)nd_ex * ncol * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_psi, h_psi.data(), (size_t)nd * ncol * sizeof(double), cudaMemcpyHostToDevice));

    int gpu_iters = std::max(10, 50000000 / (nd * ncol));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float gpu_ms_total;

    auto run_bench = [&](const char* label, auto fn) {
        // Warmup
        for (int it = 0; it < 5; ++it) fn();
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(start);
        for (int it = 0; it < gpu_iters; ++it) fn();
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_ms_total, start, stop);
        double ms = gpu_ms_total / gpu_iters;
        double bytes = ncol * ((double)nd + nd) * sizeof(double);  // read psi + write y (minimum)
        std::printf("    %-12s: %7.3f ms  eff_BW %4.0f GB/s\n", label, ms, bytes / (ms * 1e-3) / 1e9);
    };

    std::printf("  %3d³ × %2d cols:\n", N, ncol);

    // V0: baseline (per-column launch, runtime FDn)
    run_bench("V0-baseline", [&]() {
        laplacian_orth_gpu(d_x_ex, nullptr, d_y, N, N, N, FDn, nx_ex, ny_ex,
                           1.0, 0.0, 0.0, diag, ncol);
    });

    // V2: template FDn + multi-column batch
    run_bench("V2-multicol", [&]() {
        laplacian_orth_v2_gpu(d_x_ex, nullptr, d_y, N, N, N, FDn, nx_ex, ny_ex,
                              1.0, 0.0, 0.0, diag, ncol);
    });

    // V3: shared memory 2D tile + z-sweep (uses x_ex)
    run_bench("V3-smem+xex", [&]() {
        laplacian_orth_v3_gpu(d_x_ex, nullptr, d_y, N, N, N, FDn, nx_ex, ny_ex,
                              1.0, 0.0, 0.0, diag, ncol);
    });

    // V4: fused periodic (no x_ex)
    run_bench("V4-fused", [&]() {
        laplacian_orth_fused_gpu(d_psi, nullptr, d_y, N, N, N, FDn,
                                 1.0, 0.0, 0.0, diag, ncol);
    });

    // V5: fused periodic + shared memory
    run_bench("V5-fused+sm", [&]() {
        laplacian_orth_v5_gpu(d_psi, nullptr, d_y, N, N, N, FDn,
                              1.0, 0.0, 0.0, diag, ncol);
    });

    // V6: 1 column per block (low register pressure, high occupancy)
    run_bench("V6-1col/blk", [&]() {
        laplacian_orth_v6_gpu(d_x_ex, nullptr, d_y, N, N, N, FDn, nx_ex, ny_ex,
                              1.0, 0.0, 0.0, diag, ncol);
    });

    // V7: V6 + precomputed a*coeff for FMA fusion
    upload_precomputed_coefficients(D2x, D2y, D2z, 1.0, FDn);
    run_bench("V7-precomp", [&]() {
        laplacian_orth_v7_gpu(d_x_ex, nullptr, d_y, N, N, N, FDn, nx_ex, ny_ex,
                              1.0, 0.0, 0.0, diag, ncol);
    });

    // V8: Multi-column + precomputed (best of V2+V7)
    run_bench("V8-mc+pre", [&]() {
        laplacian_orth_v8_gpu(d_x_ex, nullptr, d_y, N, N, N, FDn, nx_ex, ny_ex,
                              1.0, 0.0, 0.0, diag, ncol);
    });

    // Verify correctness against V0
    {
        std::vector<double> y_ref(nd * ncol), y_test(nd * ncol);
        laplacian_orth_gpu(d_x_ex, nullptr, d_y, N, N, N, FDn, nx_ex, ny_ex,
                           1.0, 0.0, 0.0, diag, ncol);
        CUDA_CHECK(cudaMemcpy(y_ref.data(), d_y, (size_t)nd * ncol * sizeof(double), cudaMemcpyDeviceToHost));

        auto check = [&](const char* name, auto fn) {
            fn();
            CUDA_CHECK(cudaMemcpy(y_test.data(), d_y, (size_t)nd * ncol * sizeof(double), cudaMemcpyDeviceToHost));
            double max_err = 0;
            for (int i = 0; i < nd * ncol; ++i)
                max_err = std::max(max_err, std::abs(y_ref[i] - y_test[i]));
            std::printf("    %s vs V0: %.2e\n", name, max_err);
        };

        check("V6", [&]() { laplacian_orth_v6_gpu(d_x_ex, nullptr, d_y, N, N, N, FDn, nx_ex, ny_ex, 1.0, 0.0, 0.0, diag, ncol); });
        check("V7", [&]() { laplacian_orth_v7_gpu(d_x_ex, nullptr, d_y, N, N, N, FDn, nx_ex, ny_ex, 1.0, 0.0, 0.0, diag, ncol); });
        check("V8", [&]() { laplacian_orth_v8_gpu(d_x_ex, nullptr, d_y, N, N, N, FDn, nx_ex, ny_ex, 1.0, 0.0, 0.0, diag, ncol); });
    }

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_x_ex); cudaFree(d_psi); cudaFree(d_y);
}

TEST_F(GPUKernelTest, PERF_Laplacian_Variants) {
    std::printf("\n[PERF] Laplacian kernel variants comparison:\n");
    bench_laplacian_variants(26,  12, D2x, D2y, D2z);
    bench_laplacian_variants(48,  12, D2x, D2y, D2z);
    bench_laplacian_variants(72,  12, D2x, D2y, D2z);
    bench_laplacian_variants(100, 12, D2x, D2y, D2z);
    bench_laplacian_variants(100, 30, D2x, D2y, D2z);
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Check for CUDA device
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::fprintf(stderr, "No CUDA devices found. Skipping GPU tests.\n");
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("GPU: %s (compute %d.%d, %.1f GB)\n",
                prop.name, prop.major, prop.minor,
                prop.totalGlobalMem / 1073741824.0);

    return RUN_ALL_TESTS();
}

#else // !USE_CUDA
#include <cstdio>
int main() {
    std::printf("CUDA not enabled. Skipping GPU tests.\n");
    return 0;
}
#endif
