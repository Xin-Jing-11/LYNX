#pragma once
#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>

namespace lynx {

class Hamiltonian;  // forward declaration

namespace gpu {

// --- Real (gamma-point) GPU sub-steps ---

// Chebyshev filter: H->apply() called directly (no callback)
void chebyshev_filter_gpu(
    const double* d_X, double* d_Y, double* d_Xold, double* d_Xnew,
    double* d_HX, const double* d_Veff,
    int Nd, int Ns,
    double lambda_cutoff, double eigval_min, double eigval_max,
    int degree, const Hamiltonian* H, cudaStream_t stream);

// Orthogonalize via Cholesky QR
void orthogonalize_gpu(double* d_X, double* d_S, int Nd, int N, double dV);

// Project Hamiltonian + diagonalize (dsyevd)
void project_and_diag_gpu(
    const double* d_X, double* d_HX, double* d_Hs, double* d_eigvals,
    const double* d_Veff, int Nd, int N, double dV, const Hamiltonian* H);

// Rotate orbitals: X = X * Q
void rotate_orbitals_gpu(double* d_X, const double* d_Q, double* d_temp, int Nd, int N);

// --- Complex (k-point) GPU sub-steps ---

void chebyshev_filter_z_gpu(
    const cuDoubleComplex* d_X, cuDoubleComplex* d_Y,
    cuDoubleComplex* d_Xold, cuDoubleComplex* d_Xnew,
    cuDoubleComplex* d_HX, const double* d_Veff,
    int Nd, int Ns,
    double lambda_cutoff, double eigval_min, double eigval_max,
    int degree, const Hamiltonian* H, cudaStream_t stream);

void orthogonalize_z_gpu(cuDoubleComplex* d_X, cuDoubleComplex* d_S,
                          int Nd, int N, double dV);

void project_and_diag_z_gpu(
    const cuDoubleComplex* d_X, cuDoubleComplex* d_HX,
    cuDoubleComplex* d_Hs, double* d_eigvals,
    const double* d_Veff, int Nd, int N, double dV, const Hamiltonian* H);

void rotate_orbitals_z_gpu(cuDoubleComplex* d_X, const cuDoubleComplex* d_Q,
                            cuDoubleComplex* d_temp, int Nd, int N);

// Compute electron density from complex wavefunctions
void compute_density_z_gpu(const cuDoubleComplex* d_psi, const double* d_occ,
                            double* d_rho, int Nd, int Ns, double weight,
                            cudaStream_t stream = 0);

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
