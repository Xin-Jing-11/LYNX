#pragma once
#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>

namespace lynx {
namespace gpu {

// Compute mGGA psi stress and tau*vtau dot product on GPU
void compute_mgga_stress_gpu(
    const double* d_psi, const double* d_occ,
    const double* d_vtau, const double* d_tau, const double* d_vtau_full,
    int nx, int ny, int nz, int FDn, int Nd, int Nband,
    double dV, double occfac, int tau_dot_len,
    bool is_orth, const double* uvec_inv,
    double* h_stress_mgga, double* h_tau_vtau_dot,
    cudaStream_t stream = 0);

// Compute SOC nonlocal stress on GPU (complex spinor, k-point)
void compute_soc_stress_gpu(
    const cuDoubleComplex* d_psi_spinor, const double* d_occ,
    const cuDoubleComplex* d_Chi_soc_flat, const int* d_gpos_flat,
    const int* d_gpos_offsets_soc, const int* d_chi_soc_offsets,
    const int* d_ndc_arr_soc, const int* d_nproj_soc_arr,
    const int* d_IP_displ_soc, const double* d_Gamma_soc,
    const int* d_proj_l, const int* d_proj_m,
    const double* d_bloch_fac,
    int n_influence_soc, int total_soc_nproj,
    int max_ndc_soc, int max_nproj_soc,
    int n_phys_atoms,
    const int* h_IP_displ_phys_soc,
    const double* h_atom_pos_soc,
    int nx, int ny, int nz, int FDn, int Nd_d, int Nband,
    double dV, double dx, double dy, double dz,
    int xs, int ys, int zs,
    bool is_orth, const double* uvec, const double* uvec_inv,
    double kx_Lx, double ky_Ly, double kz_Lz,
    const int* h_proj_l, const int* h_proj_m,
    const double* h_Gamma_soc,
    const std::complex<double>* h_Chi_soc_flat,
    const int* h_gpos_flat,
    const int* h_gpos_offsets_soc,
    const int* h_chi_soc_offsets,
    const int* h_ndc_arr_soc,
    const int* h_nproj_soc_arr,
    const int* h_IP_displ_soc_inf,
    const double* h_bloch_fac,
    double spn_fac, double wk,
    double* h_stress_soc, double* h_energy_soc,
    cudaStream_t stream = 0);

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
