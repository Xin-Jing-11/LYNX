#pragma once
#ifdef USE_CUDA

#include <cuComplex.h>

namespace lynx {
namespace gpu {

// Apply off-diagonal Veff (spin-orbit coupling) to spinor wavefunction
void spinor_offdiag_veff_gpu(
    cuDoubleComplex* d_Hpsi, const cuDoubleComplex* d_psi,
    const double* d_V_ud_re, const double* d_V_ud_im,
    int Nd_d, int ncol);

// Apply SOC nonlocal projector to spinor wavefunction
void soc_apply_z_gpu(
    const cuDoubleComplex* d_psi, cuDoubleComplex* d_Hpsi,
    const cuDoubleComplex* d_Chi_soc_flat, const int* d_gpos_flat,
    const int* d_gpos_offsets, const int* d_chi_soc_offsets,
    const int* d_ndc_arr, const int* d_nproj_soc_arr,
    const int* d_IP_displ_soc,
    const double* d_Gamma_soc,
    const int* d_proj_l, const int* d_proj_m,
    const double* d_bloch_fac,
    cuDoubleComplex* d_alpha_up, cuDoubleComplex* d_alpha_dn,
    int Nd_d, int ncol, double dV,
    int n_influence, int total_soc_nproj,
    int max_ndc_soc, int max_nproj_soc);

// Compute spinor density and magnetization from complex psi
void spinor_density_gpu(
    const cuDoubleComplex* d_psi, const double* d_occ,
    double* d_rho, double* d_mag_x, double* d_mag_y, double* d_mag_z,
    int Nd_d, int Nband, double weight);

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
