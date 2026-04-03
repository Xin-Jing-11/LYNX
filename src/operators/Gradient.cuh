#pragma once
#ifdef USE_CUDA

namespace lynx {
namespace gpu {

// Compute gradient along direction (0=x, 1=y, 2=z) using FD stencil
void gradient_gpu(
    const double* d_x_ex, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    int direction, int ncol);

void gradient_v2_gpu(
    const double* d_x_ex, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    int direction, int ncol);

void gradient_v3_gpu(
    const double* d_x_ex, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    int direction, int ncol);

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
