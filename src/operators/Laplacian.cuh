#pragma once
#ifdef USE_CUDA

namespace lynx {
namespace gpu {

// Upload precomputed D2 coefficients to GPU constant memory (scaled by a).
void upload_precomputed_coefficients(const double* D2x, const double* D2y, const double* D2z,
                                      double a, int FDn);

// Orthogonal Laplacian: y = a * Lap(x) + b * V*x + c * x + diag_coeff * x
void laplacian_orth_gpu(
    const double* d_x_ex, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c, double diag_coeff, int ncol);

void laplacian_orth_v2_gpu(
    const double* d_x_ex, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c, double diag_coeff, int ncol);

void laplacian_orth_v3_gpu(
    const double* d_x_ex, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c, double diag_coeff, int ncol);

void laplacian_orth_v5_gpu(
    const double* d_x_ex, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c, double diag_coeff, int ncol);

void laplacian_orth_v6_gpu(
    const double* d_x_ex, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c, double diag_coeff, int ncol);

void laplacian_orth_v7_gpu(
    const double* d_x_ex, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c, double diag_coeff, int ncol);

void laplacian_orth_v8_gpu(
    const double* d_x_ex, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c, double diag_coeff, int ncol);

void laplacian_orth_fused_gpu(
    const double* d_x, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    double a, double b, double c, double diag_coeff, int ncol,
    bool periodic_x, bool periodic_y, bool periodic_z);

// Non-orthogonal Laplacian with mixed derivative terms
void laplacian_nonorth_gpu(
    const double* d_x_ex, const double* d_V, double* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c, double diag_coeff,
    bool has_xy, bool has_xz, bool has_yz, int ncol);

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
