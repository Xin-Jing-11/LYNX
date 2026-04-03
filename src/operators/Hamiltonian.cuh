#pragma once
#ifdef USE_CUDA

#include <cuda_runtime.h>

namespace lynx {
namespace gpu {

// Upload FD stencil coefficients to GPU constant memory
void upload_stencil_coefficients(
    const double* D2x, const double* D2y, const double* D2z,
    const double* D1x, const double* D1y, const double* D1z,
    const double* D2xy, const double* D2xz, const double* D2yz,
    int FDn);

// Apply local Hamiltonian: Hpsi = -0.5*Lap(psi) + Veff*psi (real, gamma-point)
void hamiltonian_apply_local_gpu(
    const double* d_psi, const double* d_Veff, double* d_Hpsi,
    double* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol, double c,
    bool is_orthogonal,
    bool periodic_x, bool periodic_y, bool periodic_z,
    double diag_coeff,
    bool has_xy, bool has_xz, bool has_yz,
    cudaStream_t stream = 0);

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
