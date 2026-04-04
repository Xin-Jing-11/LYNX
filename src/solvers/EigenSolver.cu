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
void chebyshev_filter_gpu(
    const double* d_X,      // (Nd, Ns) input orbitals
    double* d_Y,            // (Nd, Ns) output filtered orbitals
    double* d_Xold,         // (Nd, Ns) workspace
    double* d_Xnew,         // (Nd, Ns) workspace
    double* d_HX,           // (Nd, Ns) workspace for H*psi
    double* d_x_ex,         // halo workspace
    const double* d_Veff,   // (Nd) effective potential
    int Nd, int Ns,
    double lambda_cutoff, double eigval_min, double eigval_max,
    int degree,
    // Hamiltonian apply function pointer (callback):
    // apply(d_input, d_Veff, d_output, d_x_ex, Ns)
    void (*apply_H)(const double*, const double*, double*, double*, int),
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
    apply_H(d_X, d_Veff, d_HX, d_x_ex, Ns);
    double scale = sigma / e;
    chefsi_init_kernel<<<grid, bs, 0, stream>>>(d_HX, d_X, d_Y, scale, c, total);

    // Xold = X
    CUDA_CHECK(cudaMemcpyAsync(d_Xold, d_X, total * sizeof(double), cudaMemcpyDeviceToDevice, stream));

    // Steps 2..degree
    for (int k = 2; k <= degree; ++k) {
        double sigma_new = 1.0 / (2.0 / sigma_1 - sigma);
        double gamma = 2.0 * sigma_new / e;
        double ss = sigma * sigma_new;

        apply_H(d_Y, d_Veff, d_HX, d_x_ex, Ns);
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
// Hs = X^T * HX * dV → symmetrize → dsyevd → eigvals + eigvecs
void project_and_diag_gpu(
    const double* d_X,
    double* d_HX,
    double* d_Hs,     // (N,N) — overwritten with eigenvectors
    double* d_eigvals, // (N)
    double* d_x_ex,
    const double* d_Veff,
    int Nd, int N, double dV,
    void (*apply_H)(const double*, const double*, double*, double*, int))
{
    auto& ctx = GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;

    size_t _scratch_cp = ctx.scratch_pool.checkpoint();

    // HX = H * X
    apply_H(d_X, d_Veff, d_HX, d_x_ex, N);

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

// ============================================================
// Full GPU EigenSolver step (single call replaces CPU solve())
// ============================================================
void eigensolver_solve_gpu(
    double* d_psi,      // (Nd, Ns) wavefunctions — updated in place
    double* d_eigvals,   // (Ns) eigenvalues — updated
    const double* d_Veff,// (Nd) effective potential
    // Workspace (from SCFBuffers):
    double* d_Y,         // (Nd, Ns) filtered result
    double* d_Xold,      // (Nd, Ns)
    double* d_Xnew,      // (Nd, Ns)
    double* d_HX,        // (Nd, Ns) = Hpsi buffer
    double* d_x_ex,      // halo workspace
    double* d_Hs,        // (Ns, Ns) subspace Hamiltonian
    double* d_Ms,        // (Ns, Ns) overlap / temp
    int Nd, int Ns,
    double lambda_cutoff, double eigval_min, double eigval_max,
    int cheb_degree, double dV,
    void (*apply_H)(const double*, const double*, double*, double*, int))
{
    cudaStream_t stream = GPUContext::instance().compute_stream;

    // Step 1: Chebyshev filter
    chebyshev_filter_gpu(d_psi, d_Y, d_Xold, d_Xnew, d_HX, d_x_ex,
                          d_Veff, Nd, Ns,
                          lambda_cutoff, eigval_min, eigval_max,
                          cheb_degree, apply_H, stream);

    // Step 2: Orthogonalize filtered vectors (Cholesky QR)
    orthogonalize_gpu(d_Y, d_Ms, Nd, Ns, dV);

    // Step 3+4: Project Hamiltonian + diagonalize
    project_and_diag_gpu(d_Y, d_HX, d_Hs, d_eigvals, d_x_ex,
                          d_Veff, Nd, Ns, dV, apply_H);

    // Step 5: Rotate orbitals: psi = Y * Q (Q stored in Hs after dsyevd)
    rotate_orbitals_gpu(d_Y, d_Hs, d_psi, Nd, Ns);

    // Copy result back to psi
    CUDA_CHECK(cudaMemcpyAsync(d_psi, d_Y, (size_t)Nd * Ns * sizeof(double),
                           cudaMemcpyDeviceToDevice, stream));
}

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
    const cuDoubleComplex* d_X,      // (Nd, Ns) input orbitals
    cuDoubleComplex* d_Y,            // (Nd, Ns) output filtered orbitals
    cuDoubleComplex* d_Xold,         // (Nd, Ns) workspace
    cuDoubleComplex* d_Xnew,         // (Nd, Ns) workspace
    cuDoubleComplex* d_HX,           // (Nd, Ns) workspace for H*psi
    cuDoubleComplex* d_x_ex,         // halo workspace
    const double* d_Veff,            // (Nd) effective potential (always real)
    int Nd, int Ns,
    double lambda_cutoff, double eigval_min, double eigval_max,
    int degree,
    void (*apply_H_z)(const cuDoubleComplex*, const double*, cuDoubleComplex*, cuDoubleComplex*, int),
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
    apply_H_z(d_X, d_Veff, d_HX, d_x_ex, Ns);

    double scale = sigma / e;
    chefsi_init_kernel_z<<<grid, bs, 0, stream>>>(d_Y, d_HX, d_X, scale, c, total);

    // Xold = X
    CUDA_CHECK(cudaMemcpyAsync(d_Xold, d_X, total * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice, stream));

    // Steps 2..degree
    for (int k = 2; k <= degree; ++k) {
        double sigma_new = 1.0 / (2.0 / sigma_1 - sigma);
        double gamma = 2.0 * sigma_new / e;
        double ss = sigma * sigma_new;

        apply_H_z(d_Y, d_Veff, d_HX, d_x_ex, Ns);
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
    cuDoubleComplex* d_Hs,     // (N,N) — overwritten with eigenvectors
    double* d_eigvals,          // (N) — real eigenvalues
    cuDoubleComplex* d_x_ex,
    const double* d_Veff,
    int Nd, int N, double dV,
    void (*apply_H_z)(const cuDoubleComplex*, const double*, cuDoubleComplex*, cuDoubleComplex*, int))
{
    auto& ctx = GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;

    size_t _scratch_cp = ctx.scratch_pool.checkpoint();

    // HX = H * X
    apply_H_z(d_X, d_Veff, d_HX, d_x_ex, N);

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
// Full GPU Complex EigenSolver step (k-point variant)
// ============================================================
void eigensolver_solve_z_gpu(
    cuDoubleComplex* d_psi_z,    // (Nd, Ns) wavefunctions — updated in place
    double* d_eigvals,            // (Ns) eigenvalues — updated (real)
    const double* d_Veff,         // (Nd) effective potential (always real)
    // Workspace:
    cuDoubleComplex* d_Y_z,      // (Nd, Ns) filtered result
    cuDoubleComplex* d_Xold_z,   // (Nd, Ns)
    cuDoubleComplex* d_Xnew_z,   // (Nd, Ns)
    cuDoubleComplex* d_HX_z,     // (Nd, Ns) = Hpsi buffer
    cuDoubleComplex* d_x_ex_z,   // halo workspace
    cuDoubleComplex* d_Hs_z,     // (Ns, Ns) subspace Hamiltonian
    cuDoubleComplex* d_Ms_z,     // (Ns, Ns) overlap / temp
    int Nd, int Ns,
    double lambda_cutoff, double eigval_min, double eigval_max,
    int cheb_degree, double dV,
    void (*apply_H_z)(const cuDoubleComplex*, const double*, cuDoubleComplex*, cuDoubleComplex*, int))
{
    cudaStream_t stream = GPUContext::instance().compute_stream;

    // Step 1: Chebyshev filter
    chebyshev_filter_z_gpu(d_psi_z, d_Y_z, d_Xold_z, d_Xnew_z, d_HX_z, d_x_ex_z,
                            d_Veff, Nd, Ns,
                            lambda_cutoff, eigval_min, eigval_max,
                            cheb_degree, apply_H_z, stream);

    // Debug: check for NaN after CheFSI
    {
        cudaStreamSynchronize(stream);
        double y0[2];
        cudaMemcpyAsync(y0, d_Y_z, 2*sizeof(double), cudaMemcpyDeviceToHost, stream);
        if (std::isnan(y0[0]) || std::isnan(y0[1]))
            printf("[eigsolver_z] NaN after CheFSI filter! Y[0]=(%.3e,%.3e)\n", y0[0], y0[1]);
    }

    // Step 2: Orthogonalize filtered vectors (Cholesky QR)
    orthogonalize_z_gpu(d_Y_z, d_Ms_z, Nd, Ns, dV);

    // Debug: check for NaN after orthogonalization
    {
        cudaStreamSynchronize(stream);
        double y0[2];
        cudaMemcpyAsync(y0, d_Y_z, 2*sizeof(double), cudaMemcpyDeviceToHost, stream);
        if (std::isnan(y0[0]) || std::isnan(y0[1]))
            printf("[eigsolver_z] NaN after orthogonalize! Y[0]=(%.3e,%.3e)\n", y0[0], y0[1]);
    }

    // Step 3+4: Project Hamiltonian + diagonalize
    project_and_diag_z_gpu(d_Y_z, d_HX_z, d_Hs_z, d_eigvals, d_x_ex_z,
                            d_Veff, Nd, Ns, dV, apply_H_z);

    // Debug: check eigenvalues after diag
    {
        cudaStreamSynchronize(stream);
        double eig0;
        cudaMemcpyAsync(&eig0, d_eigvals, sizeof(double), cudaMemcpyDeviceToHost, stream);
        if (std::isnan(eig0))
            printf("[eigsolver_z] NaN eigenvalue after diag! eig[0]=%.3e\n", eig0);
    }

    // Step 5: Rotate orbitals: psi = Y * Q (Q stored in Hs after zheevd)
    rotate_orbitals_z_gpu(d_Y_z, d_Hs_z, d_psi_z, Nd, Ns);

    // Copy result back to psi
    CUDA_CHECK(cudaMemcpyAsync(d_psi_z, d_Y_z, (size_t)Nd * Ns * sizeof(cuDoubleComplex),
                           cudaMemcpyDeviceToDevice, stream));
}

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
    double* d_psi = nullptr;         // (Nd, Nband) real wavefunctions
    double* d_Veff = nullptr;        // (Nd) effective potential
    double* d_eigvals = nullptr;     // (Nband_global) eigenvalues
    double* d_Y = nullptr;           // (Nd, Nband) CheFSI workspace
    double* d_Xold = nullptr;        // (Nd, Nband) CheFSI workspace
    double* d_Xnew = nullptr;        // (Nd, Nband) CheFSI workspace
    double* d_HX = nullptr;          // (Nd, Nband) H*psi workspace
    double* d_Hs = nullptr;          // (Nband_global^2) subspace Hamiltonian
    double* d_Ms = nullptr;          // (Nband_global^2) overlap matrix

    // Complex (k-point) persistent buffers
    cuDoubleComplex* d_psi_z = nullptr;   // (Nd, Nband)
    cuDoubleComplex* d_Y_z = nullptr;
    cuDoubleComplex* d_Xold_z = nullptr;
    cuDoubleComplex* d_Xnew_z = nullptr;
    cuDoubleComplex* d_HX_z = nullptr;
    cuDoubleComplex* d_Hs_z = nullptr;
    cuDoubleComplex* d_Ms_z = nullptr;

    bool buffers_allocated = false;

    void allocate_buffers() {
        if (buffers_allocated) return;
        cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
        size_t psi_sz = (size_t)Nd * Nband;
        size_t sub_sz = (size_t)Nband_global * Nband_global;

        // Always allocate real buffers
        CUDA_CHECK(cudaMallocAsync(&d_psi, psi_sz * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_Veff, Nd * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_eigvals, Nband_global * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_Y, psi_sz * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_Xold, psi_sz * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_Xnew, psi_sz * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_HX, psi_sz * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_Hs, sub_sz * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_Ms, sub_sz * sizeof(double), stream));

        // Allocate complex buffers if k-point or SOC
        if (is_kpt || is_soc) {
            CUDA_CHECK(cudaMallocAsync(&d_psi_z, psi_sz * sizeof(cuDoubleComplex), stream));
            CUDA_CHECK(cudaMallocAsync(&d_Y_z, psi_sz * sizeof(cuDoubleComplex), stream));
            CUDA_CHECK(cudaMallocAsync(&d_Xold_z, psi_sz * sizeof(cuDoubleComplex), stream));
            CUDA_CHECK(cudaMallocAsync(&d_Xnew_z, psi_sz * sizeof(cuDoubleComplex), stream));
            CUDA_CHECK(cudaMallocAsync(&d_HX_z, psi_sz * sizeof(cuDoubleComplex), stream));
            CUDA_CHECK(cudaMallocAsync(&d_Hs_z, sub_sz * sizeof(cuDoubleComplex), stream));
            CUDA_CHECK(cudaMallocAsync(&d_Ms_z, sub_sz * sizeof(cuDoubleComplex), stream));
        }

        buffers_allocated = true;
    }

    void free_buffers() {
        if (!buffers_allocated) return;
        cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
        auto safe_free = [stream](auto*& p) { if (p) { cudaFreeAsync(p, stream); p = nullptr; } };

        safe_free(d_psi); safe_free(d_Veff); safe_free(d_eigvals);
        safe_free(d_Y); safe_free(d_Xold); safe_free(d_Xnew);
        safe_free(d_HX); safe_free(d_Hs); safe_free(d_Ms);
        safe_free(d_psi_z); safe_free(d_Y_z); safe_free(d_Xold_z);
        safe_free(d_Xnew_z); safe_free(d_HX_z); safe_free(d_Hs_z);
        safe_free(d_Ms_z);

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

// File-static trampoline for gpu::eigensolver_solve_gpu callback.
// Uses the same pattern as GPUSCF.cu's s_instance_ approach.
static const Hamiltonian* s_eigen_H_ptr_ = nullptr;

static void eigen_apply_H_cb(const double* psi, const double* Veff,
                               double* Hpsi, double* /*x_ex*/, int ncol)
{
    // Delegate to Hamiltonian::apply(Device::GPU) which uses its own d_x_ex workspace.
    // The x_ex parameter from the eigensolver is ignored — the Hamiltonian manages its own halo buffer.
    s_eigen_H_ptr_->apply(psi, Veff, Hpsi, ncol, Device::GPU, 0.0);
}

static void eigen_apply_H_z_cb(const cuDoubleComplex* psi, const double* Veff,
                                 cuDoubleComplex* Hpsi, cuDoubleComplex* /*x_ex*/, int ncol)
{
    // Delegate to Hamiltonian::apply_kpt(Device::GPU).
    // kpt_cart and cell_lengths are already set in the Hamiltonian's GPU state (kxLx, kyLy, kzLz).
    s_eigen_H_ptr_->apply_kpt(
        reinterpret_cast<const std::complex<double>*>(psi),
        Veff,
        reinterpret_cast<std::complex<double>*>(Hpsi),
        ncol, {0,0,0}, {0,0,0}, Device::GPU, 0.0);
}

void EigenSolver::solve(double* psi, double* eigvals, const double* Veff,
                         int Nd_d, int Nband,
                         double lambda_cutoff, double eigval_min, double eigval_max,
                         int cheb_degree, int ld, Device dev)
{
    if (dev == Device::CPU) {
        solve(psi, eigvals, Veff, Nd_d, Nband,
              lambda_cutoff, eigval_min, eigval_max, cheb_degree, ld);
        return;
    }

    // GPU path: use persistent device buffers (no per-call alloc/free).
    // psi and Veff are assumed already on device (GPU-resident SCF flow).
    // If not yet uploaded, upload them now (backward compatibility).
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    size_t psi_bytes = (size_t)Nd_d * Nband * sizeof(double);

    // Upload psi and Veff to persistent device buffers
    CUDA_CHECK(cudaMemcpyAsync(gs->d_psi, psi, psi_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(gs->d_Veff, Veff, Nd_d * sizeof(double), cudaMemcpyHostToDevice, stream));

    // Set up H*psi callback trampoline
    s_eigen_H_ptr_ = gs->H;

    gpu::eigensolver_solve_gpu(
        gs->d_psi,         // d_psi: in/out (persistent)
        gs->d_eigvals,     // d_eigvals (persistent)
        gs->d_Veff,        // d_Veff (persistent)
        gs->d_Y,           // d_Y workspace (persistent)
        gs->d_Xold,        // d_Xold workspace (persistent)
        gs->d_Xnew,        // d_Xnew workspace (persistent)
        gs->d_HX,          // d_HX workspace (persistent)
        nullptr,           // d_x_ex (ignored by callback)
        gs->d_Hs,          // d_Hs (persistent)
        gs->d_Ms,          // d_Ms (persistent)
        Nd_d, Nband,
        lambda_cutoff, eigval_min, eigval_max,
        cheb_degree, gs->dV,
        eigen_apply_H_cb);

    // Download eigenvalues and psi back to host
    CUDA_CHECK(cudaMemcpyAsync(eigvals, gs->d_eigvals, Nband * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(psi, gs->d_psi, psi_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void EigenSolver::solve_kpt(Complex* psi, double* eigvals, const double* Veff,
                              int Nd_d, int Nband,
                              double lambda_cutoff, double eigval_min, double eigval_max,
                              const Vec3& kpt_cart, const Vec3& cell_lengths,
                              int cheb_degree, int ld, Device dev)
{
    if (dev == Device::CPU) {
        solve_kpt(psi, eigvals, Veff, Nd_d, Nband,
                  lambda_cutoff, eigval_min, eigval_max,
                  kpt_cart, cell_lengths, cheb_degree, ld);
        return;
    }

    // GPU path: use persistent device buffers (no per-call alloc/free).
    auto* gs = static_cast<GPUEigenState*>(gpu_state_raw_);
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    size_t psi_bytes = (size_t)Nd_d * Nband * sizeof(cuDoubleComplex);

    // Upload psi and Veff to persistent device buffers
    CUDA_CHECK(cudaMemcpyAsync(gs->d_psi_z, psi, psi_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(gs->d_Veff, Veff, Nd_d * sizeof(double), cudaMemcpyHostToDevice, stream));

    s_eigen_H_ptr_ = gs->H;

    gpu::eigensolver_solve_z_gpu(
        gs->d_psi_z,       // d_psi_z: in/out (persistent)
        gs->d_eigvals,     // d_eigvals (persistent)
        gs->d_Veff,        // d_Veff (persistent)
        gs->d_Y_z,         // d_Y_z workspace (persistent)
        gs->d_Xold_z,      // d_Xold_z workspace (persistent)
        gs->d_Xnew_z,      // d_Xnew_z workspace (persistent)
        gs->d_HX_z,        // d_HX_z workspace (persistent)
        nullptr,           // d_x_ex_z (ignored by callback)
        gs->d_Hs_z,        // d_Hs_z (persistent)
        gs->d_Ms_z,        // d_Ms_z (persistent)
        Nd_d, Nband,
        lambda_cutoff, eigval_min, eigval_max,
        cheb_degree, gs->dV,
        eigen_apply_H_z_cb);

    // Download eigenvalues and psi back to host
    CUDA_CHECK(cudaMemcpyAsync(eigvals, gs->d_eigvals, Nband * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(psi, gs->d_psi_z, psi_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void EigenSolver::solve_spinor_kpt(Complex* psi, double* eigvals, const double* Veff_spinor,
                                     int Nd_d, int Nband,
                                     double lambda_cutoff, double eigval_min, double eigval_max,
                                     const Vec3& kpt_cart, const Vec3& cell_lengths,
                                     int cheb_degree, int ld, Device dev)
{
    if (dev == Device::CPU) {
        solve_spinor_kpt(psi, eigvals, Veff_spinor, Nd_d, Nband,
                         lambda_cutoff, eigval_min, eigval_max,
                         kpt_cart, cell_lengths, cheb_degree, ld);
        return;
    }

    // GPU spinor path: SOC uses 2*Nd_d rows per band and complex CheFSI.
    // The spinor Hamiltonian callback is different (needs Veff_spinor), so for now
    // fall back to CPU. Full spinor GPU path requires a dedicated callback.
    // TODO: Wire spinor GPU eigensolver with Hamiltonian::apply_spinor_kpt(Device::GPU).
    static bool warned = false;
    if (!warned) { fprintf(stderr, "INFO: Spinor eigensolver GPU path not yet wired, using CPU\n"); warned = true; }
    solve_spinor_kpt(psi, eigvals, Veff_spinor, Nd_d, Nband,
                     lambda_cutoff, eigval_min, eigval_max,
                     kpt_cart, cell_lengths, cheb_degree, ld);
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

} // namespace lynx

#endif // USE_CUDA
