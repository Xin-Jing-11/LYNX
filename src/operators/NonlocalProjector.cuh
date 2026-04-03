#pragma once
#ifdef USE_CUDA

#include <cuda_runtime.h>

namespace lynx {
namespace gpu {

// Apply nonlocal projector: Hpsi += Vnl * psi (real, gamma-point)
// Device-metadata interface (hot path, 3 kernel launches)
void nonlocal_projector_apply_gpu(
    const double* d_psi, double* d_Hpsi,
    const double* d_Chi_flat, const int* d_gpos_flat,
    const int* d_gpos_offsets, const int* d_chi_offsets,
    const int* d_ndc_arr, const int* d_nproj_arr,
    const int* d_IP_displ, const double* d_Gamma,
    double* d_alpha,
    int Nd, int ncol, double dV,
    int n_atoms, int total_nproj,
    int max_ndc, int max_nproj,
    cudaStream_t stream = 0);

// Convenience wrapper: takes host-side metadata, uploads to device
void nonlocal_projector_apply_gpu(
    const double* d_psi, double* d_Hpsi,
    const double* d_Chi_flat, const int* d_gpos_flat,
    const double* d_Gamma, double* d_alpha,
    int Nd, int ncol, double dV,
    int n_atoms, int total_nproj,
    const int* h_gpos_offsets, const int* h_chi_offsets,
    const int* h_ndc_arr, const int* h_nproj_arr, const int* h_IP_displ,
    int max_ndc, int max_nproj,
    cudaStream_t stream = 0);

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
