#ifdef USE_CUDA

#include "xc/GPUExactExchange.cuh"
#include "xc/GPUExchangePoissonSolver.cuh"
#include "core/gpu_common.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include <cstdio>
#include <cmath>
#include <vector>

namespace lynx {
namespace gpu {

// ============================================================
// Kernels (real, gamma-point)
// ============================================================

__global__ void elementwise_mul_kernel(const double* a, const double* b,
                                        double* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

__global__ void accumulate_Xi_kernel(double* Xi, int Nd, int col,
                                      const double* psi_vec, const double* sol,
                                      double coeff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Nd)
        Xi[col * Nd + idx] -= coeff * psi_vec[idx] * sol[idx];
}

__global__ void negate_kernel(double* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] = -a[idx];
}

// ============================================================
// Kernels (complex, k-point)
// ============================================================

// rhs[m] = conj(a[m]) * b[m]  (pair product for exchange)
__global__ void conj_mul_kernel(const cuDoubleComplex* a, const cuDoubleComplex* b,
                                 cuDoubleComplex* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        cuDoubleComplex av = a[idx];
        cuDoubleComplex bv = b[idx];
        // conj(a) * b = (ar - i*ai) * (br + i*bi) = (ar*br + ai*bi) + i*(ar*bi - ai*br)
        out[idx] = make_cuDoubleComplex(
            av.x * bv.x + av.y * bv.y,
            av.x * bv.y - av.y * bv.x);
    }
}

// Xi[col*Nd + m] -= coeff * psi_q[m] * sol[m]  (complex accumulation)
__global__ void accumulate_Xi_z_kernel(cuDoubleComplex* Xi, int Nd, int col,
                                        const cuDoubleComplex* psi_vec,
                                        const cuDoubleComplex* sol,
                                        double coeff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Nd) {
        cuDoubleComplex p = psi_vec[idx];
        cuDoubleComplex s = sol[idx];
        // p * s
        double re = p.x * s.x - p.y * s.y;
        double im = p.x * s.y + p.y * s.x;
        int offset = col * Nd + idx;
        Xi[offset].x -= coeff * re;
        Xi[offset].y -= coeff * im;
    }
}

// Negate complex array: a[i] = -a[i]
__global__ void negate_z_kernel(cuDoubleComplex* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx].x = -a[idx].x;
        a[idx].y = -a[idx].y;
    }
}

// Conjugate complex array in-place: a[i] = conj(a[i])
__global__ void conjugate_kernel(cuDoubleComplex* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx].y = -a[idx].y;
    }
}

// ============================================================
// Gamma-point functions (unchanged from original)
// ============================================================

void apply_Vx_gpu(cublasHandle_t cublas,
                  const double* d_Xi, int Nd, int Nocc,
                  const double* d_X, int ldx, int ncol,
                  double* d_Hx, int ldhx,
                  double* d_Y,
                  double exx_frac)
{
    if (Nocc <= 0 || ncol <= 0) return;

    // Step 1: Y = Xi^T * X
    {
        double alpha = 1.0;
        double beta = 0.0;
        cublasStatus_t stat = cublasDgemm(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            Nocc, ncol, Nd,
            &alpha,
            d_Xi, Nd,
            d_X, ldx,
            &beta,
            d_Y, Nocc);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "ERROR: cublasDgemm failed in apply_Vx_gpu step 1: %d\n", (int)stat);
        }
    }

    // Step 2: Hx -= exx_frac * Xi * Y
    {
        double alpha = -exx_frac;
        double beta = 1.0;
        cublasStatus_t stat = cublasDgemm(cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            Nd, ncol, Nocc,
            &alpha,
            d_Xi, Nd,
            d_Y, Nocc,
            &beta,
            d_Hx, ldhx);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "ERROR: cublasDgemm failed in apply_Vx_gpu step 2: %d\n", (int)stat);
        }
    }
}

void build_ACE_gpu(cublasHandle_t cublas,
                   cusolverDnHandle_t cusolver,
                   GPUExchangePoissonSolver& poisson,
                   const double* d_psi, int Nd, int Ns, int Nocc,
                   const double* occ, double dV,
                   double* d_Xi)
{
    if (Nocc <= 0) return;

    constexpr double OCC_THRESHOLD = 1e-6;
    double coeff_scale = 1.0;  // psi now in standard normalization
    int block = 256;
    int grid_Nd = (Nd + block - 1) / block;

    CUDA_CHECK(cudaMemset(d_Xi, 0, (size_t)Nd * Nocc * sizeof(double)));

    double* d_rhs = nullptr;
    double* d_sol = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rhs, Nd * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sol, Nd * sizeof(double)));

    for (int j = 0; j < Nocc; j++) {
        if (occ[j] < OCC_THRESHOLD) continue;
        const double* d_psi_j = d_psi + (size_t)j * Nd;

        for (int i = j; i < Nocc; i++) {
            const double* d_psi_i = d_psi + (size_t)i * Nd;

            elementwise_mul_kernel<<<grid_Nd, block>>>(d_psi_i, d_psi_j, d_rhs, Nd);
            poisson.solve_batch(d_rhs, 1, d_sol, cublas);

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

    // Phase 2: ACE operator — M = Xi^T * psi (alpha=1.0)
    double* d_M = nullptr;
    CUDA_CHECK(cudaMalloc(&d_M, (size_t)Nocc * Nocc * sizeof(double)));

    {
        double alpha = 1.0;
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

    {
        int n_total = Nocc * Nocc;
        int grid_m = (n_total + block - 1) / block;
        negate_kernel<<<grid_m, block>>>(d_M, n_total);
    }

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

    {
        double alpha = 1.0;
        cublasDtrsm(cublas,
            CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            Nd, Nocc, &alpha, d_M, Nocc, d_Xi, Nd);
    }

    cudaFree(d_M);
}

double compute_energy_gpu(cublasHandle_t cublas,
                          const double* d_psi, int Nd, int Ns, int Nocc,
                          const double* occ, double dV,
                          const double* d_Xi, double* d_Y,
                          double exx_frac, int Nspin)
{
    if (Nocc <= 0 || Ns <= 0) return 0.0;

    {
        double alpha = 1.0;  // psi in standard normalization
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

    std::vector<double> h_Y(Ns * Nocc);
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, Ns * Nocc * sizeof(double), cudaMemcpyDeviceToHost));

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

// ============================================================
// K-point (complex) functions
// ============================================================

// Apply Vx for k-point: Hx -= exx_frac * Xi * (Xi^H * X)
void apply_Vx_kpt_gpu(cublasHandle_t cublas,
                      const cuDoubleComplex* d_Xi, int Nd, int Nocc,
                      const cuDoubleComplex* d_X, int ldx, int ncol,
                      cuDoubleComplex* d_Hx, int ldhx,
                      cuDoubleComplex* d_Y,
                      double exx_frac)
{
    if (Nocc <= 0 || ncol <= 0) return;

    // Step 1: Y = Xi^H * X  [Nocc x ncol]
    {
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        cublasStatus_t stat = cublasZgemm(cublas,
            CUBLAS_OP_C, CUBLAS_OP_N,
            Nocc, ncol, Nd,
            &alpha,
            d_Xi, Nd,        // A = Xi [Nd x Nocc], conjugate-transposed
            d_X, ldx,        // B = X  [Nd x ncol]
            &beta,
            d_Y, Nocc);      // C = Y  [Nocc x ncol]
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "ERROR: cublasZgemm failed in apply_Vx_kpt_gpu step 1: %d\n", (int)stat);
        }
    }

    // Step 2: Hx -= exx_frac * Xi * Y
    {
        cuDoubleComplex alpha = make_cuDoubleComplex(-exx_frac, 0.0);
        cuDoubleComplex beta = make_cuDoubleComplex(1.0, 0.0);
        cublasStatus_t stat = cublasZgemm(cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            Nd, ncol, Nocc,
            &alpha,
            d_Xi, Nd,        // A = Xi [Nd x Nocc]
            d_Y, Nocc,       // B = Y  [Nocc x ncol]
            &beta,
            d_Hx, ldhx);    // C = Hx [Nd x ncol], accumulated
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "ERROR: cublasZgemm failed in apply_Vx_kpt_gpu step 2: %d\n", (int)stat);
        }
    }
}

// Accumulate Xi for one (k, q_hf) pair on GPU.
// Matches CPU solve_for_Xi_kpt inner loop for one q_hf.
void build_ACE_kpt_accumulate_gpu(cublasHandle_t cublas,
                                   GPUExchangePoissonSolver& poisson,
                                   const cuDoubleComplex* d_psi_k, int Nd, int Ns, int Nocc,
                                   const cuDoubleComplex* d_psi_q,
                                   const double* occ_q,
                                   int kpt_glob, int q_hf,
                                   double kptWts_hf, double dV,
                                   cuDoubleComplex* d_Xi)
{
    constexpr double OCC_THRESHOLD = 1e-6;
    double coeff_scale = 1.0;  // psi now in standard normalization
    int block = 256;
    int grid_Nd = (Nd + block - 1) / block;

    // Allocate scratch: rhs [Nd] and sol [Nd], both complex
    cuDoubleComplex* d_rhs = nullptr;
    cuDoubleComplex* d_sol = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rhs, Nd * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_sol, Nd * sizeof(cuDoubleComplex)));

    // Matching SPARC: j outer (occupied q-states), i inner (occupied k-states)
    for (int j = 0; j < Nocc; j++) {
        if (occ_q[j] < OCC_THRESHOLD) continue;
        const cuDoubleComplex* d_psi_qj = d_psi_q + (size_t)j * Nd;

        for (int i = 0; i < Nocc; i++) {
            const cuDoubleComplex* d_psi_ki = d_psi_k + (size_t)i * Nd;

            // rhs = conj(psi_q[j]) * psi_k[i]
            conj_mul_kernel<<<grid_Nd, block>>>(d_psi_qj, d_psi_ki, d_rhs, Nd);

            // Poisson solve (k-point Z2Z with phase factors)
            poisson.solve_batch_kpt(d_rhs, 1, d_sol, cublas, kpt_glob, q_hf);

            // Xi[k,i] -= kptWts_hf * occ_q[j] * sqrt(dV) * psi_q[j] * sol
            double coeff = kptWts_hf * occ_q[j] * coeff_scale;
            accumulate_Xi_z_kernel<<<grid_Nd, block>>>(d_Xi, Nd, i, d_psi_qj, d_sol, coeff);
        }
    }

    cudaFree(d_rhs);
    cudaFree(d_sol);
}

// Finalize ACE: Cholesky factorize M = -sqrt(dV)*Xi^H*psi, then Xi = Xi * L^{-H}
void build_ACE_kpt_finalize_gpu(cublasHandle_t cublas,
                                 cusolverDnHandle_t cusolver,
                                 const cuDoubleComplex* d_psi_k, int Nd, int Ns, int Nocc,
                                 double dV,
                                 cuDoubleComplex* d_Xi)
{
    if (Nocc <= 0) return;

    int block = 256;

    // M = Xi^H * psi  [Nocc x Nocc] — alpha=1.0
    cuDoubleComplex* d_M = nullptr;
    CUDA_CHECK(cudaMalloc(&d_M, (size_t)Nocc * Nocc * sizeof(cuDoubleComplex)));

    {
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        cublasZgemm(cublas,
            CUBLAS_OP_C, CUBLAS_OP_N,
            Nocc, Nocc, Nd,
            &alpha,
            d_Xi, Nd,
            d_psi_k, Nd,
            &beta,
            d_M, Nocc);
    }

    // M = -M
    {
        int n_total = Nocc * Nocc;
        int grid_m = (n_total + block - 1) / block;
        negate_z_kernel<<<grid_m, block>>>(d_M, n_total);
    }

    // Cholesky factorization: zpotrf on device
    {
        int work_size = 0;
        cusolverDnZpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER,
                                     Nocc, d_M, Nocc, &work_size);
        cuDoubleComplex* d_work = nullptr;
        int* d_info = nullptr;
        CUDA_CHECK(cudaMalloc(&d_work, work_size * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

        cusolverDnZpotrf(cusolver, CUBLAS_FILL_MODE_UPPER,
                          Nocc, d_M, Nocc, d_work, work_size, d_info);

        int h_info = 0;
        CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0) {
            std::fprintf(stderr, "WARNING: cusolverDnZpotrf failed in build_ACE_kpt_finalize_gpu (info=%d)\n", h_info);
        }
        cudaFree(d_work);
        cudaFree(d_info);
    }

    // Xi = Xi * L^{-H}: solve Xi * L^H = Xi_raw
    // cublasDtrsm equivalent: ztrsm('R', 'U', 'N', 'N', Nd, Nocc, 1.0, M, Nocc, Xi, Nd)
    {
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cublasZtrsm(cublas,
            CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            Nd, Nocc, &alpha, d_M, Nocc, d_Xi, Nd);
    }

    cudaFree(d_M);
}

// Compute exchange energy for one k-point.
// Returns: wk * sum_n occ[n] * ||Y_row_n||^2  (before -exx_frac/Nspin scaling)
double compute_energy_kpt_gpu(cublasHandle_t cublas,
                               const cuDoubleComplex* d_psi, int Nd, int Ns, int Nocc,
                               const double* occ, double dV, double wk,
                               const cuDoubleComplex* d_Xi, cuDoubleComplex* d_Y)
{
    if (Nocc <= 0 || Ns <= 0) return 0.0;

    // Y[Nocc x Ns] = Xi^H * psi (alpha=1.0)
    {
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        cublasZgemm(cublas,
            CUBLAS_OP_C, CUBLAS_OP_N,
            Nocc, Ns, Nd,
            &alpha,
            d_Xi, Nd,
            d_psi, Nd,
            &beta,
            d_Y, Nocc);
    }

    // Download Y to host (Nocc*Ns is small)
    std::vector<cuDoubleComplex> h_Y(Nocc * Ns);
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, Nocc * Ns * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    // Eexx_k = wk * sum_n occ[n] * sum_j |Y[j + n*Nocc]|^2
    double Eexx_k = 0.0;
    for (int n = 0; n < Ns; n++) {
        if (occ[n] < 1e-6) continue;
        double sum = 0.0;
        for (int j = 0; j < Nocc; j++) {
            cuDoubleComplex v = h_Y[j + n * Nocc];
            sum += v.x * v.x + v.y * v.y;
        }
        Eexx_k += wk * occ[n] * sum;
    }

    return Eexx_k;
}

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
