#ifdef USE_CUDA

#include "xc/GPUExactExchange.cuh"
#include "xc/GPUExchangePoissonSolver.cuh"
#include "core/gpu_common.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdio>
#include <cmath>
#include <vector>

namespace lynx {
namespace gpu {

// ---------------------------------------------------------------------------
// Kernel: elementwise product  out[k] = a[k] * b[k]
// ---------------------------------------------------------------------------
__global__ void elementwise_mul_kernel(const double* a, const double* b,
                                        double* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

// ---------------------------------------------------------------------------
// Kernel: Xi[:,col] -= coeff * psi_vec[:] * sol[:]
// ---------------------------------------------------------------------------
__global__ void accumulate_Xi_kernel(double* Xi, int Nd, int col,
                                      const double* psi_vec, const double* sol,
                                      double coeff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Nd)
        Xi[col * Nd + idx] -= coeff * psi_vec[idx] * sol[idx];
}

// ---------------------------------------------------------------------------
// Kernel: negate array  a[i] = -a[i]
// ---------------------------------------------------------------------------
__global__ void negate_kernel(double* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] = -a[idx];
}

// Apply exact exchange operator on GPU (gamma-point, real wavefunctions).
//
// Computes:  Hx -= exx_frac * Xi * (Xi^T * X)
//
// Step 1:  Y[Nocc x ncol] = Xi^T[Nocc x Nd] * X[Nd x ncol]     (cuBLAS dgemm)
// Step 2:  Hx[Nd x ncol] -= exx_frac * Xi[Nd x Nocc] * Y        (cuBLAS dgemm)
//
// All pointers are device pointers. d_Y is a pre-allocated scratch buffer.
void apply_Vx_gpu(cublasHandle_t cublas,
                  const double* d_Xi, int Nd, int Nocc,
                  const double* d_X, int ldx, int ncol,
                  double* d_Hx, int ldhx,
                  double* d_Y,
                  double exx_frac)
{
    if (Nocc <= 0 || ncol <= 0) return;

    // Step 1: Y = Xi^T * X  (bare sum, matching SPARC convention)
    // cublasDgemm: C = alpha * op(A) * op(B) + beta * C
    //   op(A) = Xi^T  [Nocc x Nd]
    //   B = X          [Nd x ncol]  (ld = ldx)
    //   C = Y          [Nocc x ncol] (ld = Nocc)
    {
        double alpha = 1.0;
        double beta = 0.0;
        cublasStatus_t stat = cublasDgemm(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            Nocc, ncol, Nd,
            &alpha,
            d_Xi, Nd,        // A = Xi [Nd x Nocc], transposed -> [Nocc x Nd]
            d_X, ldx,        // B = X  [Nd x ncol]
            &beta,
            d_Y, Nocc);      // C = Y  [Nocc x ncol]
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "ERROR: cublasDgemm failed in apply_Vx_gpu step 1: %d\n", (int)stat);
        }
    }

    // Step 2: Hx -= exx_frac * Xi * Y
    //   A = Xi  [Nd x Nocc]
    //   B = Y   [Nocc x ncol]
    //   C = Hx  [Nd x ncol]  (ld = ldhx)
    {
        double alpha = -exx_frac;
        double beta = 1.0;
        cublasStatus_t stat = cublasDgemm(cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            Nd, ncol, Nocc,
            &alpha,
            d_Xi, Nd,         // A = Xi [Nd x Nocc]
            d_Y, Nocc,        // B = Y  [Nocc x ncol]
            &beta,
            d_Hx, ldhx);     // C = Hx [Nd x ncol], accumulated
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "ERROR: cublasDgemm failed in apply_Vx_gpu step 2: %d\n", (int)stat);
        }
    }
}

// ---------------------------------------------------------------------------
// Build ACE operator entirely on GPU (gamma-point)
// ---------------------------------------------------------------------------
void build_ACE_gpu(cublasHandle_t cublas,
                   cusolverDnHandle_t cusolver,
                   GPUExchangePoissonSolver& poisson,
                   const double* d_psi, int Nd, int Ns, int Nocc,
                   const double* occ, double dV,
                   double* d_Xi)
{
    if (Nocc <= 0) return;

    constexpr double OCC_THRESHOLD = 1e-6;
    double coeff_scale = std::sqrt(dV);
    int block = 256;
    int grid_Nd = (Nd + block - 1) / block;

    // Zero Xi
    CUDA_CHECK(cudaMemset(d_Xi, 0, (size_t)Nd * Nocc * sizeof(double)));

    // Allocate scratch: rhs [Nd] and sol [Nd]
    double* d_rhs = nullptr;
    double* d_sol = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rhs, Nd * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sol, Nd * sizeof(double)));

    // ---- Phase 1: solve_for_Xi ----
    // For each pair (i,j) with i>=j and occ[i]+occ[j] > threshold:
    //   rhs = psi_i * psi_j
    //   sol = PoissonSolve(rhs)
    //   Xi[:,i] -= occ[j] * sqrt(dV) * psi_j * sol
    //   Xi[:,j] -= occ[i] * sqrt(dV) * psi_i * sol  (if i!=j)
    for (int j = 0; j < Nocc; j++) {
        if (occ[j] < OCC_THRESHOLD) continue;
        const double* d_psi_j = d_psi + (size_t)j * Nd;

        for (int i = j; i < Nocc; i++) {
            const double* d_psi_i = d_psi + (size_t)i * Nd;

            // rhs = psi_i * psi_j
            elementwise_mul_kernel<<<grid_Nd, block>>>(d_psi_i, d_psi_j, d_rhs, Nd);

            // Poisson solve: rhs -> sol
            poisson.solve_batch(d_rhs, 1, d_sol, cublas);

            // Xi[:,i] -= occ[j] * sqrt(dV) * psi_j * sol
            double coeff_j = occ[j] * coeff_scale;
            accumulate_Xi_kernel<<<grid_Nd, block>>>(d_Xi, Nd, i, d_psi_j, d_sol, coeff_j);

            if (i != j && occ[i] > OCC_THRESHOLD) {
                double coeff_i = occ[i] * coeff_scale;
                accumulate_Xi_kernel<<<grid_Nd, block>>>(d_Xi, Nd, j, d_psi_i, d_sol, coeff_i);
            }
        }
    }

    cudaFree(d_rhs);
    cudaFree(d_sol);

    // ---- Phase 2: calculate_ACE_operator ----
    // M = sqrt(dV) * Xi^T * psi  [Nocc x Nocc]
    // M = -M
    // Cholesky: M = L * L^T
    // Xi = Xi * L^{-T}

    // Allocate M on device [Nocc x Nocc]
    double* d_M = nullptr;
    CUDA_CHECK(cudaMalloc(&d_M, (size_t)Nocc * Nocc * sizeof(double)));

    // M = sqrt(dV) * Xi^T * psi (only first Nocc columns of psi)
    {
        double alpha = std::sqrt(dV);
        double beta = 0.0;
        cublasDgemm(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            Nocc, Nocc, Nd,
            &alpha,
            d_Xi, Nd,
            d_psi, Nd,
            &beta,
            d_M, Nocc);
    }

    // M = -M
    {
        int n_total = Nocc * Nocc;
        int grid_m = (n_total + block - 1) / block;
        negate_kernel<<<grid_m, block>>>(d_M, n_total);
    }

    // Cholesky factorization: dpotrf on device
    {
        int work_size = 0;
        cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER,
                                     Nocc, d_M, Nocc, &work_size);
        double* d_work = nullptr;
        int* d_info = nullptr;
        CUDA_CHECK(cudaMalloc(&d_work, work_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

        cusolverDnDpotrf(cusolver, CUBLAS_FILL_MODE_UPPER,
                          Nocc, d_M, Nocc, d_work, work_size, d_info);

        int h_info = 0;
        CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0) {
            std::fprintf(stderr, "WARNING: cusolverDnDpotrf failed in build_ACE_gpu (info=%d)\n", h_info);
        }
        cudaFree(d_work);
        cudaFree(d_info);
    }

    // Xi = Xi * L^{-T}: solve Xi * L^T = Xi_raw (dtrsm)
    // cublasDtrsm('R', 'U', 'N', 'N', Nd, Nocc, 1.0, M, Nocc, Xi, Nd)
    {
        double alpha = 1.0;
        cublasDtrsm(cublas,
            CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            Nd, Nocc, &alpha, d_M, Nocc, d_Xi, Nd);
    }

    cudaFree(d_M);
}

// ---------------------------------------------------------------------------
// Compute exact exchange energy on GPU (gamma-point)
//
// Eexx = -exx_frac / Nspin * sum_n( occ[n] * ||Y[n,:]||^2 )
// where Y = sqrt(dV) * psi^T * Xi  [Ns x Nocc]
// ---------------------------------------------------------------------------
double compute_energy_gpu(cublasHandle_t cublas,
                          const double* d_psi, int Nd, int Ns, int Nocc,
                          const double* occ, double dV,
                          const double* d_Xi, double* d_Y,
                          double exx_frac, int Nspin)
{
    if (Nocc <= 0 || Ns <= 0) return 0.0;

    // Y[Ns x Nocc] = sqrt(dV) * psi^T[Ns x Nd] * Xi[Nd x Nocc]
    {
        double alpha = std::sqrt(dV);
        double beta = 0.0;
        cublasDgemm(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            Ns, Nocc, Nd,
            &alpha,
            d_psi, Nd,
            d_Xi, Nd,
            &beta,
            d_Y, Ns);
    }

    // Download Y to host for reduction (Ns*Nocc is small, ~15*18 = 270 doubles)
    std::vector<double> h_Y(Ns * Nocc);
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, Ns * Nocc * sizeof(double), cudaMemcpyDeviceToHost));

    // Eexx = sum_n occ[n] * sum_j Y[n + j*Ns]^2
    double Eexx = 0.0;
    for (int n = 0; n < Ns; n++) {
        if (occ[n] < 1e-6) continue;
        double row_norm2 = 0.0;
        for (int j = 0; j < Nocc; j++) {
            double v = h_Y[n + j * Ns];
            row_norm2 += v * v;
        }
        Eexx += occ[n] * row_norm2;
    }

    Eexx /= Nspin;
    Eexx *= -exx_frac;
    return Eexx;
}

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
