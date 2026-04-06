#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include "core/GPUContext.cuh"
#include "core/gpu_common.cuh"
#include "solvers/EigenSolver.cuh"
#include "solvers/EigenSolver.hpp"
#include "operators/Hamiltonian.hpp"

namespace lynx {
namespace gpu {

// ============================================================
// Vector kernels for Chebyshev filter
// ============================================================

// Y[i] = scale * (HX[i] - c * X[i])
__global__ void chefsi_init_kernel(
    const double* __restrict__ HX,
    const double* __restrict__ X,
    double* __restrict__ Y,
    double scale, double c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        Y[idx] = scale * (HX[idx] - c * X[idx]);
    }
}

// Xnew[i] = gamma * (HX[i] - c * Y[i]) - sigma_sigma_new * Xold[i]
__global__ void chefsi_step_kernel(
    const double* __restrict__ HX,
    const double* __restrict__ Y,
    const double* __restrict__ Xold,
    double* __restrict__ Xnew,
    double gamma, double c, double ss, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        Xnew[idx] = gamma * (HX[idx] - c * Y[idx]) - ss * Xold[idx];
    }
}

// dst = src (device-to-device copy for pointer swap)
// (Actually just swap pointers in host code — no kernel needed)

// ============================================================
// Custom A^T * A kernel for tall-thin matrices (N ≤ ~100)
// One CUDA block per C[i,j] element, shared-memory reduction
// Beats cuBLAS DGEMM by 3-16x for DFT-relevant sizes
// ============================================================
__global__ void ata_dot_kernel(
    const double* __restrict__ A,
    double* __restrict__ C,
    int M, int N, double scale)
{
    int col_i = blockIdx.x;
    int col_j = blockIdx.y;
    if (col_j < col_i) return;

    extern __shared__ double sdata[];
    double sum = 0.0;
    for (int row = threadIdx.x; row < M; row += blockDim.x)
        sum += A[row + col_i * M] * A[row + col_j * M];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double val = sdata[0] * scale;
        C[col_i * N + col_j] = val;
        if (col_i != col_j)
            C[col_j * N + col_i] = val;
    }
}

// A^T * B kernel for tall-thin matrices (Nd x N) — for Hs = X^T * HX
__global__ void atb_dot_kernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int M, int N, double scale)
{
    int col_i = blockIdx.x;
    int col_j = blockIdx.y;

    extern __shared__ double sdata[];
    double sum = 0.0;
    for (int row = threadIdx.x; row < M; row += blockDim.x)
        sum += A[row + col_i * M] * B[row + col_j * M];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        C[col_i * N + col_j] = sdata[0] * scale;
    }
}

// Symmetrize: C[i,j] = C[j,i] = 0.5*(C[i,j] + C[j,i])
__global__ void symmetrize_kernel(double* C, int N) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    if (j <= i || i >= N || j >= N) return;
    double avg = 0.5 * (C[i * N + j] + C[j * N + i]);
    C[i * N + j] = avg;
    C[j * N + i] = avg;
}

// ============================================================
// GPU Lanczos vector operations
// ============================================================

// v[i] = v[i] - a * u[i]
__global__ void axpy_neg_kernel(double* v, const double* u, double a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) v[idx] -= a * u[idx];
}

// v[i] = v[i] - a*u[i] - b*w[i]; w[i] = u[i]  (fused for Lanczos)
__global__ void lanczos_ortho_kernel(
    double* __restrict__ V_jp1,
    const double* __restrict__ V_j,
    double* __restrict__ V_jm1,
    double a, double b, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        V_jp1[idx] -= (a * V_j[idx] + b * V_jm1[idx]);
        V_jm1[idx] = V_j[idx];
    }
}

// v[i] = u[i] * scale
__global__ void scale_copy_kernel(double* v, const double* u, double scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) v[idx] = u[idx] * scale;
}

// v[i] *= scale
__global__ void scale_inplace_kernel(double* v, double scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) v[idx] *= scale;
}

// ============================================================
// Public API: GPU CheFSI
// ============================================================

// Chebyshev filter: all data on GPU, returns filtered Y in d_Y
// H->apply() is called directly (no callback).
void chebyshev_filter_gpu(
    const double* d_X,      // (Nd, Ns) input orbitals
    double* d_Y,            // (Nd, Ns) output filtered orbitals
    double* d_Xold,         // (Nd, Ns) workspace
    double* d_Xnew,         // (Nd, Ns) workspace
    double* d_HX,           // (Nd, Ns) workspace for H*psi
    const double* d_Veff,   // (Nd) effective potential
    int Nd, int Ns,
    double lambda_cutoff, double eigval_min, double eigval_max,
    int degree,
    const Hamiltonian* H,
    cudaStream_t stream)
{
    double e = (eigval_max - lambda_cutoff) / 2.0;
    double c = (eigval_max + lambda_cutoff) / 2.0;
    double sigma_1 = e / (eigval_min - c);
    double sigma = sigma_1;

    int total = Nd * Ns;
    int bs = 256;
    int grid = ceildiv(total, bs);

    // Step 1: Y = (H*X - c*X) * (sigma/e)
    H->apply(d_X, d_Veff, d_HX, Ns, 0.0);
    double scale = sigma / e;
    chefsi_init_kernel<<<grid, bs, 0, stream>>>(d_HX, d_X, d_Y, scale, c, total);

    // Xold = X
    CUDA_CHECK(cudaMemcpyAsync(d_Xold, d_X, total * sizeof(double), cudaMemcpyDeviceToDevice, stream));

    // Steps 2..degree
    for (int k = 2; k <= degree; ++k) {
        double sigma_new = 1.0 / (2.0 / sigma_1 - sigma);
        double gamma = 2.0 * sigma_new / e;
        double ss = sigma * sigma_new;

        H->apply(d_Y, d_Veff, d_HX, Ns, 0.0);
        chefsi_step_kernel<<<grid, bs, 0, stream>>>(d_HX, d_Y, d_Xold, d_Xnew, gamma, c, ss, total);

        // Rotate: Xold <- Y, Y <- Xnew (pointer swap via memcpy)
        CUDA_CHECK(cudaMemcpyAsync(d_Xold, d_Y, total * sizeof(double), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_Y, d_Xnew, total * sizeof(double), cudaMemcpyDeviceToDevice, stream));
        sigma = sigma_new;
    }
}

// Compute S = X^T * X * dV on GPU (uses custom dot kernel for N ≤ 200, cuBLAS otherwise)
void compute_ata_gpu(
    const double* d_X,  // (Nd, N) on device
    double* d_S,        // (N, N) on device
    int Nd, int N, double dV,
    cudaStream_t stream)
{
    if (N <= 200) {
        // Custom dot kernel — much faster for small N
        int bs = 256;
        dim3 grid(N, N);
        ata_dot_kernel<<<grid, bs, bs * sizeof(double), stream>>>(d_X, d_S, Nd, N, dV);
    } else {
        // Fallback to cuBLAS for very large N
        auto& ctx = GPUContext::instance();
        double beta = 0.0;
        cublasDgemm(ctx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                     N, N, Nd, &dV, d_X, Nd, d_X, Nd, &beta, d_S, N);
    }
}

// Compute Hs = X^T * HX * dV on GPU
void compute_atb_gpu(
    const double* d_X,   // (Nd, N) on device
    const double* d_HX,  // (Nd, N) on device
    double* d_Hs,        // (N, N) on device
    int Nd, int N, double dV,
    cudaStream_t stream)
{
    if (N <= 200) {
        int bs = 256;
        dim3 grid(N, N);
        atb_dot_kernel<<<grid, bs, bs * sizeof(double), stream>>>(d_X, d_HX, d_Hs, Nd, N, dV);
    } else {
        auto& ctx = GPUContext::instance();
        double beta = 0.0;
        cublasDgemm(ctx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                     N, N, Nd, &dV, d_X, Nd, d_HX, Nd, &beta, d_Hs, N);
    }
}

// Symmetrize Hs on GPU
void symmetrize_gpu(double* d_Hs, int N, cudaStream_t stream) {
    dim3 grid(N, N);
    symmetrize_kernel<<<grid, 1, 0, stream>>>(d_Hs, N);
}

// Orthogonalize via Cholesky QR on GPU:
// S = X^T * X * dV → Cholesky S = R^T*R → X = X * R^{-1}
void orthogonalize_gpu(double* d_X, double* d_S, int Nd, int N, double dV) {
    auto& ctx = GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;

    // S = X^T * X * dV
    size_t _scratch_cp = ctx.scratch_pool.checkpoint();
    compute_ata_gpu(d_X, d_S, Nd, N, dV, stream);

    // Cholesky factorization: S = R^T * R (upper triangular)
    int lwork = 0;
    cusolverDnDpotrf_bufferSize(ctx.cusolver, CUBLAS_FILL_MODE_UPPER,
                                 N, d_S, N, &lwork);

    // Use scratch pool for workspace
    double* d_work = ctx.scratch_pool.alloc<double>(lwork);
    cusolverDnDpotrf(ctx.cusolver, CUBLAS_FILL_MODE_UPPER,
                      N, d_S, N, d_work, lwork, ctx.buf.cusolver_devinfo);

    // X = X * R^{-1}  (triangular solve: X * R = X_new)
    double one = 1.0;
    cublasDtrsm(ctx.cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                Nd, N, &one, d_S, N, d_X, Nd);

    ctx.scratch_pool.restore(_scratch_cp);
}

// Project Hamiltonian + diagonalize on GPU:
// Hs = X^T * HX * dV -> symmetrize -> dsyevd -> eigvals + eigvecs
void project_and_diag_gpu(
    const double* d_X,
    double* d_HX,
    double* d_Hs,     // (N,N) -- overwritten with eigenvectors
    double* d_eigvals, // (N)
    const double* d_Veff,
    int Nd, int N, double dV,
    const Hamiltonian* H)
{
    auto& ctx = GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;

    size_t _scratch_cp = ctx.scratch_pool.checkpoint();

    // HX = H * X
    H->apply(d_X, d_Veff, d_HX, N, 0.0);

    // Hs = X^T * HX * dV
    compute_atb_gpu(d_X, d_HX, d_Hs, Nd, N, dV, stream);

    // Symmetrize
    symmetrize_gpu(d_Hs, N, stream);

    // Diagonalize with cuSOLVER dsyevd (all on GPU — no CPU LAPACK)
    int lwork = 0;
    cusolverDnDsyevd_bufferSize(ctx.cusolver, CUSOLVER_EIG_MODE_VECTOR,
                                  CUBLAS_FILL_MODE_UPPER,
                                  N, d_Hs, N, d_eigvals, &lwork);

    double* d_work = ctx.scratch_pool.alloc<double>(lwork);
    cusolverDnDsyevd(ctx.cusolver, CUSOLVER_EIG_MODE_VECTOR,
                      CUBLAS_FILL_MODE_UPPER,
                      N, d_Hs, N, d_eigvals, d_work, lwork,
                      ctx.buf.cusolver_devinfo);

    ctx.scratch_pool.restore(_scratch_cp);
    // d_Hs now contains eigenvectors (columns), d_eigvals has eigenvalues
}

// Rotate orbitals: X_new = X * Q (cuBLAS dgemm)
void rotate_orbitals_gpu(
    double* d_X,       // (Nd, N) — overwritten with result
    const double* d_Q, // (N, N) eigenvectors
    double* d_temp,    // (Nd, N) workspace
    int Nd, int N)
{
    auto& ctx = GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;
    double one = 1.0, zero = 0.0;

    // temp = X * Q
    cublasDgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                Nd, N, N, &one, d_X, Nd, d_Q, N, &zero, d_temp, Nd);

    // X = temp
    CUDA_CHECK(cudaMemcpyAsync(d_X, d_temp, (size_t)Nd * N * sizeof(double),
                           cudaMemcpyDeviceToDevice, stream));
}

// (eigensolver_solve_gpu deleted — loop logic moved to solve_resident)

// ============================================================
// Complex (k-point) CheFSI kernels and subspace operations
// ============================================================

// Y[i] = scale * (HX[i] - c * X[i])  for cuDoubleComplex
__global__ void chefsi_init_kernel_z(
    cuDoubleComplex* __restrict__ Y,
    const cuDoubleComplex* __restrict__ HX,
    const cuDoubleComplex* __restrict__ X,
    double scale, double c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // real scalars * complex: just scale both components
        cuDoubleComplex hx = HX[idx];
        cuDoubleComplex x  = X[idx];
        Y[idx] = make_cuDoubleComplex(
            scale * (hx.x - c * x.x),
            scale * (hx.y - c * x.y));
    }
}

// Xnew[i] = gamma * (HX[i] - c * Y[i]) - ss * Xold[i]
__global__ void chefsi_step_kernel_z(
    cuDoubleComplex* __restrict__ Xnew,
    const cuDoubleComplex* __restrict__ HX,
    const cuDoubleComplex* __restrict__ Y,
    const cuDoubleComplex* __restrict__ Xold,
    double gamma, double c, double ss, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        cuDoubleComplex hx   = HX[idx];
        cuDoubleComplex y    = Y[idx];
        cuDoubleComplex xold = Xold[idx];
        Xnew[idx] = make_cuDoubleComplex(
            gamma * (hx.x - c * y.x) - ss * xold.x,
            gamma * (hx.y - c * y.y) - ss * xold.y);
    }
}

// Hermitianize a small N×N complex matrix (column-major):
// C[i,j] = 0.5*(C[i,j] + conj(C[j,i])), diagonal forced real
__global__ void hermitianize_kernel_z(cuDoubleComplex* C, int N) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    if (j <= i || i >= N || j >= N) return;
    cuDoubleComplex cij = C[i + j * N];
    cuDoubleComplex cji = C[j + i * N];
    cuDoubleComplex avg = make_cuDoubleComplex(
        0.5 * (cij.x + cji.x),
        0.5 * (cij.y - cji.y));
    C[i + j * N] = avg;
    C[j + i * N] = make_cuDoubleComplex(avg.x, -avg.y);
}

// Force diagonal to be real
__global__ void force_real_diag_kernel_z(cuDoubleComplex* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i + i * N] = make_cuDoubleComplex(C[i + i * N].x, 0.0);
    }
}

// Compute density from complex wavefunctions:
// rho[i] += weight * occ[n] * |psi[i + n*Nd]|^2
__global__ void compute_density_z_kernel(
    const cuDoubleComplex* __restrict__ psi,
    const double* __restrict__ occ,
    double* __restrict__ rho,
    int Nd, int Ns, double weight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Nd) {
        double sum = 0.0;
        for (int n = 0; n < Ns; ++n) {
            cuDoubleComplex z = psi[idx + (size_t)n * Nd];
            sum += occ[n] * (z.x * z.x + z.y * z.y);
        }
        rho[idx] += weight * sum;
    }
}

// ============================================================
// Complex Chebyshev filter (host function)
// ============================================================
void chebyshev_filter_z_gpu(
    const cuDoubleComplex* d_X,
    cuDoubleComplex* d_Y,
    cuDoubleComplex* d_Xold,
    cuDoubleComplex* d_Xnew,
    cuDoubleComplex* d_HX,
    const double* d_Veff,
    int Nd, int Ns,
    double lambda_cutoff, double eigval_min, double eigval_max,
    int degree,
    const Hamiltonian* H,
    cudaStream_t stream)
{
    double e = (eigval_max - lambda_cutoff) / 2.0;
    double c = (eigval_max + lambda_cutoff) / 2.0;
    double sigma_1 = e / (eigval_min - c);
    double sigma = sigma_1;

    int total = Nd * Ns;
    int bs = 256;
    int grid = ceildiv(total, bs);

    // Step 1: Y = (H*X - c*X) * (sigma/e)
    H->apply_kpt(
        reinterpret_cast<const std::complex<double>*>(d_X),
        d_Veff,
        reinterpret_cast<std::complex<double>*>(d_HX),
        Ns, {0,0,0}, {0,0,0}, 0.0);

    double scale = sigma / e;
    chefsi_init_kernel_z<<<grid, bs, 0, stream>>>(d_Y, d_HX, d_X, scale, c, total);

    // Xold = X
    CUDA_CHECK(cudaMemcpyAsync(d_Xold, d_X, total * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice, stream));

    // Steps 2..degree
    for (int k = 2; k <= degree; ++k) {
        double sigma_new = 1.0 / (2.0 / sigma_1 - sigma);
        double gamma = 2.0 * sigma_new / e;
        double ss = sigma * sigma_new;

        H->apply_kpt(
            reinterpret_cast<const std::complex<double>*>(d_Y),
            d_Veff,
            reinterpret_cast<std::complex<double>*>(d_HX),
            Ns, {0,0,0}, {0,0,0}, 0.0);
        chefsi_step_kernel_z<<<grid, bs, 0, stream>>>(d_Xnew, d_HX, d_Y, d_Xold, gamma, c, ss, total);

        // Rotate: Xold <- Y, Y <- Xnew
        CUDA_CHECK(cudaMemcpyAsync(d_Xold, d_Y, total * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_Y, d_Xnew, total * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice, stream));
        sigma = sigma_new;
    }
}

// ============================================================
// Complex subspace operations using cuBLAS/cuSOLVER
// ============================================================

// Compute S = X^H * X * dV using cublasZgemm
void compute_ata_z_gpu(
    const cuDoubleComplex* d_X,  // (Nd, N)
    cuDoubleComplex* d_S,        // (N, N)
    int Nd, int N, double dV)
{
    auto& ctx = GPUContext::instance();
    cuDoubleComplex alpha = make_cuDoubleComplex(dV, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
    cublasZgemm(ctx.cublas, CUBLAS_OP_C, CUBLAS_OP_N,
                N, N, Nd, &alpha, d_X, Nd, d_X, Nd, &beta, d_S, N);
}

// Compute Hs = X^H * HX * dV using cublasZgemm
void compute_atb_z_gpu(
    const cuDoubleComplex* d_X,   // (Nd, N)
    const cuDoubleComplex* d_HX,  // (Nd, N)
    cuDoubleComplex* d_Hs,        // (N, N)
    int Nd, int N, double dV)
{
    auto& ctx = GPUContext::instance();
    cuDoubleComplex alpha = make_cuDoubleComplex(dV, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
    cublasZgemm(ctx.cublas, CUBLAS_OP_C, CUBLAS_OP_N,
                N, N, Nd, &alpha, d_X, Nd, d_HX, Nd, &beta, d_Hs, N);
}

// Hermitianize Hs on GPU
void hermitianize_z_gpu(cuDoubleComplex* d_Hs, int N, cudaStream_t stream) {
    dim3 grid(N, N);
    hermitianize_kernel_z<<<grid, 1, 0, stream>>>(d_Hs, N);
    int bs = 256;
    int g = ceildiv(N, bs);
    force_real_diag_kernel_z<<<g, bs, 0, stream>>>(d_Hs, N);
}

// Orthogonalize via Cholesky QR on GPU (complex):
// S = X^H * X * dV -> Cholesky S = R^H*R -> X = X * R^{-1}
void orthogonalize_z_gpu(cuDoubleComplex* d_X, cuDoubleComplex* d_S,
                          int Nd, int N, double dV) {
    auto& ctx = GPUContext::instance();

    size_t _scratch_cp = ctx.scratch_pool.checkpoint();
    // S = X^H * X * dV
    compute_ata_z_gpu(d_X, d_S, Nd, N, dV);

    // Cholesky factorization: S = R^H * R (upper triangular)
    int lwork = 0;
    cusolverDnZpotrf_bufferSize(ctx.cusolver, CUBLAS_FILL_MODE_UPPER,
                                 N, d_S, N, &lwork);

    cuDoubleComplex* d_work = ctx.scratch_pool.alloc<cuDoubleComplex>(lwork);
    cusolverDnZpotrf(ctx.cusolver, CUBLAS_FILL_MODE_UPPER,
                      N, d_S, N, d_work, lwork, ctx.buf.cusolver_devinfo);

    // X = X * R^{-1}  (triangular solve)
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    cublasZtrsm(ctx.cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                Nd, N, &one, d_S, N, d_X, Nd);

    ctx.scratch_pool.restore(_scratch_cp);
}

// Project Hamiltonian + diagonalize on GPU (complex):
// Hs = X^H * HX * dV -> hermitianize -> zheevd -> eigvals + eigvecs
void project_and_diag_z_gpu(
    const cuDoubleComplex* d_X,
    cuDoubleComplex* d_HX,
    cuDoubleComplex* d_Hs,     // (N,N) -- overwritten with eigenvectors
    double* d_eigvals,          // (N) -- real eigenvalues
    const double* d_Veff,
    int Nd, int N, double dV,
    const Hamiltonian* H)
{
    auto& ctx = GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;

    size_t _scratch_cp = ctx.scratch_pool.checkpoint();

    // HX = H * X
    H->apply_kpt(
        reinterpret_cast<const std::complex<double>*>(d_X),
        d_Veff,
        reinterpret_cast<std::complex<double>*>(d_HX),
        N, {0,0,0}, {0,0,0}, 0.0);

    // Hs = X^H * HX * dV
    compute_atb_z_gpu(d_X, d_HX, d_Hs, Nd, N, dV);

    // Hermitianize
    hermitianize_z_gpu(d_Hs, N, stream);

    // Diagonalize with cuSOLVER zheevd
    int lwork = 0;
    cusolverDnZheevd_bufferSize(ctx.cusolver, CUSOLVER_EIG_MODE_VECTOR,
                                  CUBLAS_FILL_MODE_UPPER,
                                  N, d_Hs, N, d_eigvals, &lwork);

    cuDoubleComplex* d_work = ctx.scratch_pool.alloc<cuDoubleComplex>(lwork);
    cusolverDnZheevd(ctx.cusolver, CUSOLVER_EIG_MODE_VECTOR,
                      CUBLAS_FILL_MODE_UPPER,
                      N, d_Hs, N, d_eigvals, d_work, lwork,
                      ctx.buf.cusolver_devinfo);

    ctx.scratch_pool.restore(_scratch_cp);
    // d_Hs now contains complex eigenvectors (columns), d_eigvals has real eigenvalues
}

// Rotate orbitals: X_new = X * Q (cuBLAS zgemm), complex version
void rotate_orbitals_z_gpu(
    cuDoubleComplex* d_X,       // (Nd, N) — overwritten with result
    const cuDoubleComplex* d_Q, // (N, N) eigenvectors
    cuDoubleComplex* d_temp,    // (Nd, N) workspace
    int Nd, int N)
{
    auto& ctx = GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;
    cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    // temp = X * Q
    cublasZgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                Nd, N, N, &one, d_X, Nd, d_Q, N, &zero, d_temp, Nd);

    // X = temp
    CUDA_CHECK(cudaMemcpyAsync(d_X, d_temp, (size_t)Nd * N * sizeof(cuDoubleComplex),
                           cudaMemcpyDeviceToDevice, stream));
}

// ============================================================
// (eigensolver_solve_z_gpu deleted -- loop logic moved to solve_kpt_resident)

// ============================================================
// Complex density computation wrapper
// ============================================================
void compute_density_z_gpu(const cuDoubleComplex* d_psi, const double* d_occ,
                            double* d_rho, int Nd, int Ns, double weight,
                            cudaStream_t stream) {
    int bs = 256;
    int grid = ceildiv(Nd, bs);
    compute_density_z_kernel<<<grid, bs, 0, stream>>>(d_psi, d_occ, d_rho, Nd, Ns, weight);
}

} // namespace gpu

// ============================================================
// Device-dispatching methods for EigenSolver
// ============================================================

// GPUEigenState — persistent device buffers for GPU-resident SCF
struct GPUEigenState {
    const Hamiltonian* H = nullptr;

    // Grid dimensions
    int Nd = 0;
    int Nband = 0;
    int Nband_global = 0;
    bool is_kpt = false;
    bool is_soc = false;
    double dV = 0.0;  // volume element for orthogonalization

    // Persistent device buffers (allocated once in setup_gpu, freed in cleanup_gpu).
    // These stay resident across all SCF iterations — no per-call alloc/free.
    double* d_psi = nullptr;         // Active (Nd, Nband) real wfn (points into d_psi_sk)
    double* d_Veff = nullptr;        // (Nd) effective potential
    double* d_eigvals = nullptr;     // (Nband_global) eigenvalues
    double* d_Y = nullptr;           // (Nd, Nband) CheFSI workspace
    double* d_Xold = nullptr;        // (Nd, Nband) CheFSI workspace
    double* d_Xnew = nullptr;        // (Nd, Nband) CheFSI workspace
    double* d_HX = nullptr;          // (Nd, Nband) H*psi workspace
    double* d_Hs = nullptr;          // (Nband_global^2) subspace Hamiltonian
    double* d_Ms = nullptr;          // (Nband_global^2) overlap matrix

    // Complex (k-point) persistent buffers
    // For SOC: sized as 2*Nd per band (spinor wavefunctions)
    cuDoubleComplex* d_psi_z = nullptr;  // Active complex wfn (points into d_psi_z_sk)
    cuDoubleComplex* d_Y_z = nullptr;
    cuDoubleComplex* d_Xold_z = nullptr;
    cuDoubleComplex* d_Xnew_z = nullptr;
    cuDoubleComplex* d_HX_z = nullptr;
    cuDoubleComplex* d_Hs_z = nullptr;
    cuDoubleComplex* d_Ms_z = nullptr;

    // SOC-specific
    double* d_Veff_spinor = nullptr;  // (4*Nd) [V_uu|V_dd|Re(V_ud)|Im(V_ud)]

    bool buffers_allocated = false;

    // Per-(spin,kpt) psi storage: each (spin,kpt) has its own device psi buffer.
    // d_psi / d_psi_z are aliases into these arrays for the active (spin,kpt).
    std::vector<double*> d_psi_sk;              // [Nspin_local * Nkpts] real gamma
    std::vector<cuDoubleComplex*> d_psi_z_sk;   // [Nspin_local * Nkpts] complex kpt
    int psi_Nspin_local = 0;
    int psi_Nkpts = 0;

    void allocate_buffers() {
        if (buffers_allocated) return;
        cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
        size_t psi_sz = (size_t)Nd * Nband;
        size_t sub_sz = (size_t)Nband_global * Nband_global;

        // Allocate ONE default psi buffer (for single-spin gamma case or before
        // allocate_psi_buffers is called). Per-(spin,kpt) buffers are allocated
        // separately via allocate_psi_buffers().
        CUDA_CHECK(cudaMallocAsync(&d_psi, psi_sz * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_Veff, Nd * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_eigvals, Nband_global * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_Y, psi_sz * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_Xold, psi_sz * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_Xnew, psi_sz * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_HX, psi_sz * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_Hs, sub_sz * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_Ms, sub_sz * sizeof(double), stream));

        // Allocate complex workspace if k-point or SOC
        if (is_kpt || is_soc) {
            size_t psi_sz_z = is_soc ? (size_t)(2 * Nd) * Nband : psi_sz;
            CUDA_CHECK(cudaMallocAsync(&d_psi_z, psi_sz_z * sizeof(cuDoubleComplex), stream));
            CUDA_CHECK(cudaMallocAsync(&d_Y_z, psi_sz_z * sizeof(cuDoubleComplex), stream));
            CUDA_CHECK(cudaMallocAsync(&d_Xold_z, psi_sz_z * sizeof(cuDoubleComplex), stream));
            CUDA_CHECK(cudaMallocAsync(&d_Xnew_z, psi_sz_z * sizeof(cuDoubleComplex), stream));
            CUDA_CHECK(cudaMallocAsync(&d_HX_z, psi_sz_z * sizeof(cuDoubleComplex), stream));
            CUDA_CHECK(cudaMallocAsync(&d_Hs_z, sub_sz * sizeof(cuDoubleComplex), stream));
            CUDA_CHECK(cudaMallocAsync(&d_Ms_z, sub_sz * sizeof(cuDoubleComplex), stream));
        }

        if (is_soc) {
            CUDA_CHECK(cudaMallocAsync(&d_Veff_spinor, 4 * Nd * sizeof(double), stream));
        }

        buffers_allocated = true;
    }

    void free_buffers() {
        if (!buffers_allocated) return;
        cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
        auto safe_free = [stream](auto*& p) { if (p) { cudaFreeAsync(p, stream); p = nullptr; } };

        // Free per-(spin,kpt) psi buffers (if allocated)
        if (!d_psi_sk.empty()) {
            for (auto*& p : d_psi_sk) safe_free(p);
            d_psi_sk.clear();
            d_psi = nullptr;  // Was alias into d_psi_sk — don't double-free
        }
        if (!d_psi_z_sk.empty()) {
            for (auto*& p : d_psi_z_sk) safe_free(p);
            d_psi_z_sk.clear();
            d_psi_z = nullptr;  // Was alias into d_psi_z_sk — don't double-free
        }
        psi_Nspin_local = 0;
        psi_Nkpts = 0;

        // Free default single psi buffer (only if not aliased into per-sk arrays)
        safe_free(d_psi); safe_free(d_Veff); safe_free(d_eigvals);
        safe_free(d_Y); safe_free(d_Xold); safe_free(d_Xnew);
        safe_free(d_HX); safe_free(d_Hs); safe_free(d_Ms);
        safe_free(d_psi_z); safe_free(d_Y_z); safe_free(d_Xold_z);
        safe_free(d_Xnew_z); safe_free(d_HX_z); safe_free(d_Hs_z);
        safe_free(d_Ms_z); safe_free(d_Veff_spinor);

        buffers_allocated = false;
    }
};

void EigenSolver::setup_gpu(const LynxContext& ctx, int Nband, int Nband_global,
                                   bool is_kpt, bool is_soc) {
    if (!gpu_state_raw_)
        gpu_state_raw_ = new GPUEigenState();
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);

    gs->H = H_;
    gs->Nd = ctx.domain().Nd_d();
    gs->Nband = Nband;
    gs->Nband_global = Nband_global;
    gs->is_kpt = is_kpt;
    gs->is_soc = is_soc;
    gs->dV = ctx.grid().dV();

    // Allocate persistent device buffers for GPU-resident SCF
    gs->allocate_buffers();
}

void EigenSolver::cleanup_gpu() {
    if (!gpu_state_raw_) return;
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    gs->free_buffers();
    delete gs;
    gpu_state_raw_ = nullptr;
}

EigenSolver::~EigenSolver() {
    cleanup_gpu();
#ifdef USE_SCALAPACK
    cleanup_blacs();
#endif
}

// ============================================================
// GPU sub-step class methods — thin wrappers around gpu:: helpers
// These access GPUEigenState device pointers and delegate to the
// gpu:: free functions. Algorithm logic lives in .cpp.
// ============================================================

// --- Real (gamma-point) GPU sub-steps ---

void EigenSolver::chebyshev_filter_gpu(int Nd_d, int Nband,
                                        double lambda_cutoff, double eigval_min, double eigval_max,
                                        int cheb_degree) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    gpu::chebyshev_filter_gpu(gs->d_psi, gs->d_Y, gs->d_Xold, gs->d_Xnew, gs->d_HX,
                               gs->d_Veff, Nd_d, Nband,
                               lambda_cutoff, eigval_min, eigval_max,
                               cheb_degree, gs->H, stream);
}

void EigenSolver::orthogonalize_gpu(int Nd_d, int Nband) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    gpu::orthogonalize_gpu(gs->d_Y, gs->d_Ms, Nd_d, Nband, gs->dV);
}

void EigenSolver::project_and_diag_gpu(int Nd_d, int Nband) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    gpu::project_and_diag_gpu(gs->d_Y, gs->d_HX, gs->d_Hs, gs->d_eigvals,
                               gs->d_Veff, Nd_d, Nband, gs->dV, gs->H);
}

void EigenSolver::subspace_rotation_gpu(int Nd_d, int Nband) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    gpu::rotate_orbitals_gpu(gs->d_Y, gs->d_Hs, gs->d_psi, Nd_d, Nband);
    // Copy Y back to psi
    CUDA_CHECK(cudaMemcpyAsync(gs->d_psi, gs->d_Y, (size_t)Nd_d * Nband * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream));
}

// --- Complex (k-point) GPU sub-steps ---

void EigenSolver::chebyshev_filter_kpt_gpu(int Nd_d, int Nband,
                                            double lambda_cutoff, double eigval_min, double eigval_max,
                                            int cheb_degree) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    gpu::chebyshev_filter_z_gpu(gs->d_psi_z, gs->d_Y_z, gs->d_Xold_z, gs->d_Xnew_z, gs->d_HX_z,
                                 gs->d_Veff, Nd_d, Nband,
                                 lambda_cutoff, eigval_min, eigval_max,
                                 cheb_degree, gs->H, stream);
}

void EigenSolver::orthogonalize_kpt_gpu(int Nd_d, int Nband) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    gpu::orthogonalize_z_gpu(gs->d_Y_z, gs->d_Ms_z, Nd_d, Nband, gs->dV);
}

void EigenSolver::project_and_diag_kpt_gpu(int Nd_d, int Nband) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    gpu::project_and_diag_z_gpu(gs->d_Y_z, gs->d_HX_z, gs->d_Hs_z, gs->d_eigvals,
                                 gs->d_Veff, Nd_d, Nband, gs->dV, gs->H);
}

void EigenSolver::subspace_rotation_kpt_gpu(int Nd_d, int Nband) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    gpu::rotate_orbitals_z_gpu(gs->d_Y_z, gs->d_Hs_z, gs->d_psi_z, Nd_d, Nband);
    // Copy Y_z back to psi_z
    CUDA_CHECK(cudaMemcpyAsync(gs->d_psi_z, gs->d_Y_z, (size_t)Nd_d * Nband * sizeof(cuDoubleComplex),
                               cudaMemcpyDeviceToDevice, stream));
}

// --- GPU workspace pointer accessors ---

double* EigenSolver::gpu_Y() {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    return gs ? gs->d_Y : nullptr;
}

double* EigenSolver::gpu_Hs() {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    return gs ? gs->d_Hs : nullptr;
}

// ============================================================
// GPU-resident solve: algorithm lives in .cpp (solve_resident),
// called through _gpu() sub-step methods.
// ============================================================
void EigenSolver::solve_resident(double* h_eigvals, const double* h_Veff,
                                  int Nd_d, int Nband,
                                  double lambda_cutoff, double eigval_min, double eigval_max,
                                  int cheb_degree) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;

    // Upload Veff only (psi stays resident on device)
    CUDA_CHECK(cudaMemcpyAsync(gs->d_Veff, h_Veff, Nd_d * sizeof(double),
                               cudaMemcpyHostToDevice, stream));

    // CheFSI sub-steps via _gpu() class methods
    chebyshev_filter_gpu(Nd_d, Nband, lambda_cutoff, eigval_min, eigval_max, cheb_degree);
    orthogonalize_gpu(Nd_d, Nband);
    project_and_diag_gpu(Nd_d, Nband);
    subspace_rotation_gpu(Nd_d, Nband);

    // Download only eigenvalues (tiny: Nband doubles)
    CUDA_CHECK(cudaMemcpyAsync(h_eigvals, gs->d_eigvals, Nband * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void EigenSolver::solve_kpt_resident(double* h_eigvals, const double* h_Veff,
                                      int Nd_d, int Nband,
                                      double lambda_cutoff, double eigval_min, double eigval_max,
                                      int cheb_degree) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;

    // Upload Veff only (psi_z stays resident on device)
    CUDA_CHECK(cudaMemcpyAsync(gs->d_Veff, h_Veff, Nd_d * sizeof(double),
                               cudaMemcpyHostToDevice, stream));

    // CheFSI sub-steps via _gpu() class methods
    chebyshev_filter_kpt_gpu(Nd_d, Nband, lambda_cutoff, eigval_min, eigval_max, cheb_degree);
    orthogonalize_kpt_gpu(Nd_d, Nband);
    project_and_diag_kpt_gpu(Nd_d, Nband);
    subspace_rotation_kpt_gpu(Nd_d, Nband);

    // Download only eigenvalues (tiny: Nband doubles)
    CUDA_CHECK(cudaMemcpyAsync(h_eigvals, gs->d_eigvals, Nband * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ============================================================
// GPU-resident data accessors
// ============================================================

double* EigenSolver::gpu_psi() {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    return gs ? gs->d_psi : nullptr;
}

const double* EigenSolver::gpu_psi() const {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    return gs ? gs->d_psi : nullptr;
}

double* EigenSolver::gpu_eigvals() {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    return gs ? gs->d_eigvals : nullptr;
}

double* EigenSolver::gpu_Veff() {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    return gs ? gs->d_Veff : nullptr;
}

void EigenSolver::upload_psi_to_device(const double* h_psi, int Nd, int Nband) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    CUDA_CHECK(cudaMemcpyAsync(gs->d_psi, h_psi, (size_t)Nd * Nband * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
}

void EigenSolver::upload_psi_z_to_device(const Complex* h_psi, int Nd, int Nband) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    CUDA_CHECK(cudaMemcpyAsync(gs->d_psi_z, h_psi, (size_t)Nd * Nband * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice, stream));
}

void EigenSolver::download_eigvals(double* h_eigvals, int Nband) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    CUDA_CHECK(cudaMemcpyAsync(h_eigvals, gs->d_eigvals, Nband * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void EigenSolver::download_psi(double* h_psi, int Nd, int Nband) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    CUDA_CHECK(cudaMemcpyAsync(h_psi, gs->d_psi, (size_t)Nd * Nband * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void EigenSolver::download_psi_z(Complex* h_psi, int Nd, int Nband) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    CUDA_CHECK(cudaMemcpyAsync(h_psi, gs->d_psi_z, (size_t)Nd * Nband * sizeof(cuDoubleComplex),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void EigenSolver::upload_Veff(const double* h_Veff, int Nd) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    CUDA_CHECK(cudaMemcpyAsync(gs->d_Veff, h_Veff, Nd * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
}

// ============================================================
// Per-(spin,kpt) device psi buffer management
// ============================================================

void EigenSolver::allocate_psi_buffers(int Nspin_local, int Nkpts) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;

    int nsk = Nspin_local * Nkpts;
    gs->psi_Nspin_local = Nspin_local;
    gs->psi_Nkpts = Nkpts;
    size_t psi_sz = (size_t)gs->Nd * gs->Nband;

    if (gs->is_kpt || gs->is_soc) {
        // Complex k-point buffers
        size_t psi_sz_z = gs->is_soc ? (size_t)(2 * gs->Nd) * gs->Nband : psi_sz;

        // Free the default single buffer (it was allocated in allocate_buffers)
        if (gs->d_psi_z) { cudaFreeAsync(gs->d_psi_z, stream); gs->d_psi_z = nullptr; }

        gs->d_psi_z_sk.resize(nsk, nullptr);
        for (int i = 0; i < nsk; ++i) {
            CUDA_CHECK(cudaMallocAsync(&gs->d_psi_z_sk[i],
                                       psi_sz_z * sizeof(cuDoubleComplex), stream));
        }
        // Active pointer = first buffer
        gs->d_psi_z = gs->d_psi_z_sk[0];
    } else {
        // Real gamma-point buffers
        // Free the default single buffer
        if (gs->d_psi) { cudaFreeAsync(gs->d_psi, stream); gs->d_psi = nullptr; }

        gs->d_psi_sk.resize(nsk, nullptr);
        for (int i = 0; i < nsk; ++i) {
            CUDA_CHECK(cudaMallocAsync(&gs->d_psi_sk[i],
                                       psi_sz * sizeof(double), stream));
        }
        // Active pointer = first buffer
        gs->d_psi = gs->d_psi_sk[0];
    }
}

void EigenSolver::set_active_psi(int spin, int kpt) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    int idx = spin * gs->psi_Nkpts + kpt;

    if (gs->is_kpt || gs->is_soc) {
        gs->d_psi_z = gs->d_psi_z_sk[idx];
    } else {
        gs->d_psi = gs->d_psi_sk[idx];
    }
}

double* EigenSolver::device_psi_real(int spin, int kpt) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    if (gs->d_psi_sk.empty()) return gs->d_psi;  // fallback: single buffer
    int idx = spin * gs->psi_Nkpts + kpt;
    return gs->d_psi_sk[idx];
}

const double* EigenSolver::device_psi_real(int spin, int kpt) const {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    if (gs->d_psi_sk.empty()) return gs->d_psi;
    int idx = spin * gs->psi_Nkpts + kpt;
    return gs->d_psi_sk[idx];
}

void* EigenSolver::device_psi_z(int spin, int kpt) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    if (gs->d_psi_z_sk.empty()) return gs->d_psi_z;
    int idx = spin * gs->psi_Nkpts + kpt;
    return gs->d_psi_z_sk[idx];
}

const void* EigenSolver::device_psi_z(int spin, int kpt) const {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    if (gs->d_psi_z_sk.empty()) return gs->d_psi_z;
    int idx = spin * gs->psi_Nkpts + kpt;
    return gs->d_psi_z_sk[idx];
}

void EigenSolver::randomize_psi_gpu(int Nspin_local, int spin_start, int Nkpts) {
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    auto& gctx = gpu::GPUContext::instance();
    cudaStream_t stream = gctx.compute_stream;
    curandGenerator_t curand = gctx.handles.curand;

    int nsk = Nspin_local * Nkpts;
    size_t psi_sz = (size_t)gs->Nd * gs->Nband;

    if (gs->is_kpt || gs->is_soc) {
        size_t psi_sz_z = gs->is_soc ? (size_t)(2 * gs->Nd) * gs->Nband : psi_sz;
        // Fill complex buffers: generate 2*psi_sz_z doubles (real+imag interleaved)
        for (int i = 0; i < nsk; ++i) {
            int s = i / Nkpts;
            unsigned long long seed = (unsigned long long)(spin_start + s) * 1000 + i * 13 + 1;
            curandSetPseudoRandomGeneratorSeed(curand, seed);
            curandGenerateUniformDouble(curand, reinterpret_cast<double*>(gs->d_psi_z_sk[i]),
                                        2 * psi_sz_z);
        }
    } else {
        // Fill real buffers
        for (int i = 0; i < nsk; ++i) {
            int s = i / Nkpts;
            unsigned long long seed = (unsigned long long)(spin_start + s) * 1000 + i * 13 + 1;
            curandSetPseudoRandomGeneratorSeed(curand, seed);
            curandGenerateUniformDouble(curand, gs->d_psi_sk[i], psi_sz);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace lynx

#endif // USE_CUDA
