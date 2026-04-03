#pragma once
#ifdef USE_CUDA

namespace lynx {
namespace gpu {

// AAR (Anderson Acceleration with Richardson) GPU solver
int aar_gpu(
    void (*op_gpu)(const double* d_x, double* d_Ax),
    void (*precond_gpu)(const double* d_r, double* d_f),
    const double* d_b, double* d_x, int N,
    double omega, double beta, int m, int p,
    double tol, int max_iter,
    double* d_r, double* d_f, double* d_Ax,
    double* d_X_hist, double* d_F_hist,
    double* d_x_old, double* d_f_old);

// Compute electron density from real wavefunctions (gamma-point)
void compute_density_gpu(const double* d_psi, const double* d_occ, double* d_rho,
                          int Nd, int Ns, double weight);

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
