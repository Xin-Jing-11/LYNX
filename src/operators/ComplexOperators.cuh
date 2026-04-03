#pragma once
#ifdef USE_CUDA

#include <cuComplex.h>

namespace lynx {
namespace gpu {

// Complex halo exchange with Bloch phase factors
void halo_exchange_z_gpu(
    const cuDoubleComplex* d_x, cuDoubleComplex* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol,
    bool periodic_x, bool periodic_y, bool periodic_z,
    double kx_Lx, double ky_Ly, double kz_Lz);

// Complex orthogonal Laplacian
void laplacian_orth_z_gpu(
    const cuDoubleComplex* d_x_ex, const double* d_V, cuDoubleComplex* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c,
    double diag_coeff, int ncol);

// Complex non-orthogonal Laplacian
void laplacian_nonorth_z_gpu(
    const cuDoubleComplex* d_x_ex, const double* d_V, cuDoubleComplex* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    double a, double b, double c,
    double diag_coeff,
    bool has_xy, bool has_xz, bool has_yz,
    int ncol);

// Complex gradient
void gradient_z_gpu(
    const cuDoubleComplex* d_x_ex, cuDoubleComplex* d_y,
    int nx, int ny, int nz, int FDn,
    int nx_ex, int ny_ex,
    int direction, int ncol);

// Complex local Hamiltonian application
void hamiltonian_apply_local_z_gpu(
    const cuDoubleComplex* d_psi, const double* d_Veff, cuDoubleComplex* d_Hpsi,
    cuDoubleComplex* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol, double c,
    bool is_orthogonal,
    bool periodic_x, bool periodic_y, bool periodic_z,
    double diag_coeff,
    bool has_xy, bool has_xz, bool has_yz,
    double kx_Lx, double ky_Ly, double kz_Lz);

// Complex nonlocal projector (device-side index arrays)
void nonlocal_projector_apply_z_gpu(
    const cuDoubleComplex* d_psi, cuDoubleComplex* d_Hpsi,
    const double* d_Chi_flat, const int* d_gpos_flat,
    const int* d_gpos_offsets, const int* d_chi_offsets,
    const int* d_ndc_arr, const int* d_nproj_arr,
    const int* d_IP_displ, const double* d_Gamma,
    cuDoubleComplex* d_alpha,
    const double* d_bloch_fac,
    int Nd, int ncol, double dV,
    int n_atoms, int total_nproj,
    int max_ndc, int max_nproj);

// Complex nonlocal projector (host-side index arrays, separate layout)
void nonlocal_projector_apply_z_gpu(
    const cuDoubleComplex* d_psi, cuDoubleComplex* d_Hpsi,
    const double* d_Chi_flat, const int* d_gpos_flat,
    const double* d_Gamma,
    cuDoubleComplex* d_alpha,
    const double* d_bloch_fac,
    int Nd, int ncol, double dV,
    int n_atoms, int total_nproj,
    const int* h_gpos_offsets, const int* h_chi_offsets,
    const int* h_ndc_arr, const int* h_nproj_arr,
    const int* h_IP_displ,
    int max_ndc, int max_nproj);

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
