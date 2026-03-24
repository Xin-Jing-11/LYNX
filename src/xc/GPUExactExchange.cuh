#pragma once
#ifdef USE_CUDA

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace lynx {
namespace gpu {

// Forward declaration
class GPUExchangePoissonSolver;

// Apply exact exchange operator on GPU (gamma-point, real wavefunctions).
// Computes: Hx -= exx_frac * Xi * (Xi^T * X)
//
// d_Xi:   [Nd x Nocc] ACE operator on device (column-major)
// d_X:    [ldx x ncol] input wavefunctions on device
// d_Hx:   [ldhx x ncol] output H*psi on device (accumulated, not overwritten)
// d_Y:    [Nocc x ncol] scratch buffer on device
void apply_Vx_gpu(cublasHandle_t cublas,
                  const double* d_Xi, int Nd, int Nocc,
                  const double* d_X, int ldx, int ncol,
                  double* d_Hx, int ldhx,
                  double* d_Y,
                  double exx_frac);

// Build ACE operator entirely on GPU (gamma-point, real).
//
// Algorithm:
//   1. solve_for_Xi: For each pair (i,j) with i>=j:
//        rhs = psi_i * psi_j  (elementwise)
//        sol = PoissonSolve(rhs)
//        Xi[:,i] -= occ[j] * sqrt(dV) * psi_j * sol
//        Xi[:,j] -= occ[i] * sqrt(dV) * psi_i * sol  (if i!=j)
//   2. calculate_ACE_operator:
//        M = sqrt(dV) * Xi^T * psi
//        M = -M
//        Cholesky: M = L * L^T
//        Xi = Xi * L^{-T}  (triangular solve)
//
// d_psi:   [Nd x Ns] full orbitals on device (all states, gathered)
// d_Xi:    [Nd x Nocc] output ACE operator on device (must be pre-allocated + zeroed)
// occ:     [Ns] occupation numbers on HOST
// Nd, Ns, Nocc: grid size, total states, occupied states
// dV:      volume element (cell_volume / Nd)
// poisson: GPU Poisson solver (must be already set up)
void build_ACE_gpu(cublasHandle_t cublas,
                   cusolverDnHandle_t cusolver,
                   GPUExchangePoissonSolver& poisson,
                   const double* d_psi, int Nd, int Ns, int Nocc,
                   const double* occ, double dV,
                   double* d_Xi);

// Compute exact exchange energy on GPU (gamma-point, real).
//
// Eexx = -exx_frac / Nspin * sum_n( occ[n] * ||Y[n,:]||^2 )
// where Y = sqrt(dV) * psi^T * Xi  [Ns x Nocc]
//
// d_psi:  [Nd x Ns] orbitals on device
// d_Xi:   [Nd x Nocc] ACE operator on device
// occ:    [Ns] occupation numbers on HOST
// d_Y:    [Ns x Nocc] scratch on device (pre-allocated)
// Returns: Eexx (scalar on host)
double compute_energy_gpu(cublasHandle_t cublas,
                          const double* d_psi, int Nd, int Ns, int Nocc,
                          const double* occ, double dV,
                          const double* d_Xi, double* d_Y,
                          double exx_frac, int Nspin);

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
