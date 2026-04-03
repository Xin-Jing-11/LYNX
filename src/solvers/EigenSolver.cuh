#pragma once
#ifdef USE_CUDA

#include <cuComplex.h>

namespace lynx {
namespace gpu {

// Real eigensolver: Chebyshev-filtered subspace iteration
void eigensolver_solve_gpu(
    double* d_psi, double* d_eigvals, const double* d_Veff,
    double* d_Y, double* d_Xold, double* d_Xnew,
    double* d_HX, double* d_x_ex,
    double* d_Hs, double* d_Ms,
    int Nd, int Ns,
    double lambda_cutoff, double eigval_min, double eigval_max,
    int cheb_degree, double dV,
    void (*apply_H)(const double*, const double*, double*, double*, int));

// Complex eigensolver: Chebyshev-filtered subspace iteration (k-point)
void eigensolver_solve_z_gpu(
    cuDoubleComplex* d_psi_z, double* d_eigvals, const double* d_Veff,
    cuDoubleComplex* d_Y_z, cuDoubleComplex* d_Xold_z, cuDoubleComplex* d_Xnew_z,
    cuDoubleComplex* d_HX_z, cuDoubleComplex* d_x_ex_z,
    cuDoubleComplex* d_Hs_z, cuDoubleComplex* d_Ms_z,
    int Nd, int Ns,
    double lambda_cutoff, double eigval_min, double eigval_max,
    int cheb_degree, double dV,
    void (*apply_H_z)(const cuDoubleComplex*, const double*, cuDoubleComplex*, cuDoubleComplex*, int));

// Compute electron density from complex wavefunctions
void compute_density_z_gpu(const cuDoubleComplex* d_psi, const double* d_occ,
                            double* d_rho, int Nd, int Ns, double weight);

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
