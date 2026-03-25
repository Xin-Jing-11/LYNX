#pragma once
#ifdef USE_CUDA

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>

namespace lynx {
namespace gpu {

// Forward declaration
class GPUExchangePoissonSolver;

// ============================================================
// Gamma-point (real) functions
// ============================================================

// Apply exact exchange operator on GPU (gamma-point, real wavefunctions).
// Computes: Hx -= exx_frac * Xi * (Xi^T * X)
void apply_Vx_gpu(cublasHandle_t cublas,
                  const double* d_Xi, int Nd, int Nocc,
                  const double* d_X, int ldx, int ncol,
                  double* d_Hx, int ldhx,
                  double* d_Y,
                  double exx_frac);

// Build ACE operator entirely on GPU (gamma-point, real).
void build_ACE_gpu(cublasHandle_t cublas,
                   cusolverDnHandle_t cusolver,
                   GPUExchangePoissonSolver& poisson,
                   const double* d_psi, int Nd, int Ns, int Nocc,
                   const double* occ, double dV,
                   double* d_Xi);

// Compute exact exchange energy on GPU (gamma-point, real).
double compute_energy_gpu(cublasHandle_t cublas,
                          const double* d_psi, int Nd, int Ns, int Nocc,
                          const double* occ, double dV,
                          const double* d_Xi, double* d_Y,
                          double exx_frac, int Nspin);

// ============================================================
// K-point (complex) functions
// ============================================================

// Apply exact exchange operator on GPU (k-point, complex wavefunctions).
// Computes: Hx -= exx_frac * Xi * (Xi^H * X)
//
// d_Xi:   [Nd x Nocc] complex ACE operator on device (column-major)
// d_X:    [ldx x ncol] complex input wavefunctions on device
// d_Hx:   [ldhx x ncol] complex output H*psi on device (accumulated)
// d_Y:    [Nocc x ncol] complex scratch buffer on device
void apply_Vx_kpt_gpu(cublasHandle_t cublas,
                      const cuDoubleComplex* d_Xi, int Nd, int Nocc,
                      const cuDoubleComplex* d_X, int ldx, int ncol,
                      cuDoubleComplex* d_Hx, int ldhx,
                      cuDoubleComplex* d_Y,
                      double exx_frac);

// Build ACE operator for one k-point entirely on GPU (complex).
//
// Algorithm (matching CPU solve_for_Xi_kpt + calculate_ACE_operator_kpt):
//   Phase 1 (solve_for_Xi_kpt for one (k, q_hf) pair at a time):
//     For each q_hf in HF BZ, for each pair (j occupied in q, i occupied in k):
//       rhs = conj(psi_q[j]) * psi_k[i]
//       sol = PoissonSolve_kpt(rhs, k, q_hf)
//       Xi_k[i] -= kptWts_hf * occ_q[j] * sqrt(dV) * psi_q[j] * sol
//
//   Phase 2 (calculate_ACE_operator_kpt):
//     M = sqrt(dV) * Xi^H * psi  [Nocc x Nocc]
//     M = -M
//     Cholesky: M = L * L^H   (zpotrf, upper)
//     Xi = Xi * L^{-H}        (ztrsm)
//
// d_psi_k:    [Nd x Ns] wavefunctions at k-point on device (cuDoubleComplex)
// d_psi_q:    [Nd x Ns] wavefunctions at q-point on device (may be conjugated for TR)
// occ_q:      [Ns] occupation numbers for q-point on HOST
// kpt_glob:   global index of k-point (for Kptshift_map)
// q_hf:       full HF BZ index of q-point (for Kptshift_map)
// kptWts_hf:  weight factor (1/Nkpts_full)
// d_Xi:       [Nd x Nocc] output ACE operator (accumulated across q-points)
//
// Note: This function accumulates Xi for ONE (k, q_hf) pair.
// The caller must loop over all q_hf in the HF BZ.
void build_ACE_kpt_accumulate_gpu(cublasHandle_t cublas,
                                   GPUExchangePoissonSolver& poisson,
                                   const cuDoubleComplex* d_psi_k, int Nd, int Ns, int Nocc,
                                   const cuDoubleComplex* d_psi_q,
                                   const double* occ_q,
                                   int kpt_glob, int q_hf,
                                   double kptWts_hf, double dV,
                                   cuDoubleComplex* d_Xi);

// Finalize ACE operator: Cholesky factorize and triangular solve.
// Called once per k-point after all q-points have been accumulated.
void build_ACE_kpt_finalize_gpu(cublasHandle_t cublas,
                                 cusolverDnHandle_t cusolver,
                                 const cuDoubleComplex* d_psi_k, int Nd, int Ns, int Nocc,
                                 double dV,
                                 cuDoubleComplex* d_Xi);

// Compute exact exchange energy contribution for one k-point on GPU.
//
// Returns: wk * sum_n( occ[n] * ||Y[n,:]||^2 )
// where Y = sqrt(dV) * Xi^H * psi  [Nocc x Ns]
//
// The caller accumulates across k-points and spins, then applies:
//   Eexx = -exx_frac / Nspin * sum_contributions
double compute_energy_kpt_gpu(cublasHandle_t cublas,
                               const cuDoubleComplex* d_psi, int Nd, int Ns, int Nocc,
                               const double* occ, double dV, double wk,
                               const cuDoubleComplex* d_Xi, cuDoubleComplex* d_Y);

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
