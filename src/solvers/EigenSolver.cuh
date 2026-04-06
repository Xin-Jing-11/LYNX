#pragma once
#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>

namespace lynx {

class Hamiltonian;  // forward declaration

namespace gpu {

// --- Real (gamma-point) GPU sub-steps (internal to EigenSolver.cu) ---
// These are called by EigenSolver::*_gpu() class methods.

// Chebyshev filter init: Y = (HX - c*X) * scale, Xold = X
void chefsi_init_real_gpu(
    const double* d_HX, const double* d_X,
    double* d_Y, double* d_Xold,
    double scale, double c, int total, cudaStream_t stream);

// Chebyshev filter step: Xnew = gamma*(HX - c*Y) - ss*Xold, rotate pointers
void chefsi_step_real_gpu(
    const double* d_HX, const double* d_Y,
    const double* d_Xold, double* d_Xnew,
    double* d_Y_out, double* d_Xold_out,
    double gamma, double c, double ss, int total, cudaStream_t stream);

void orthogonalize_gpu(double* d_X, double* d_S, int Nd, int N, double dV);

void project_and_diag_gpu(
    const double* d_X, double* d_HX, double* d_Hs, double* d_eigvals,
    const double* d_Veff, int Nd, int N, double dV, const Hamiltonian* H);

void rotate_orbitals_gpu(double* d_X, const double* d_Q, double* d_temp, int Nd, int N);

// --- Complex (k-point) GPU sub-steps ---

// Complex Chebyshev filter init
void chefsi_init_z_gpu(
    const cuDoubleComplex* d_HX, const cuDoubleComplex* d_X,
    cuDoubleComplex* d_Y, cuDoubleComplex* d_Xold,
    double scale, double c, int total, cudaStream_t stream);

// Complex Chebyshev filter step
void chefsi_step_z_gpu(
    const cuDoubleComplex* d_HX, const cuDoubleComplex* d_Y,
    const cuDoubleComplex* d_Xold, cuDoubleComplex* d_Xnew,
    cuDoubleComplex* d_Y_out, cuDoubleComplex* d_Xold_out,
    double gamma, double c, double ss, int total, cudaStream_t stream);

void orthogonalize_z_gpu(cuDoubleComplex* d_X, cuDoubleComplex* d_S,
                          int Nd, int N, double dV);

void project_and_diag_z_gpu(
    const cuDoubleComplex* d_X, cuDoubleComplex* d_HX,
    cuDoubleComplex* d_Hs, double* d_eigvals,
    const double* d_Veff, int Nd, int N, double dV, const Hamiltonian* H);

void rotate_orbitals_z_gpu(cuDoubleComplex* d_X, const cuDoubleComplex* d_Q,
                            cuDoubleComplex* d_temp, int Nd, int N);

// Compute electron density from complex wavefunctions (used by ElectronDensity.cu)
void compute_density_z_gpu(const cuDoubleComplex* d_psi, const double* d_occ,
                            double* d_rho, int Nd, int Ns, double weight,
                            cudaStream_t stream = 0);

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
