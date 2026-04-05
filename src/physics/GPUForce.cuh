#pragma once
#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>

namespace lynx {
namespace gpu {

// Real gamma-point: nonlocal force + energy only
void compute_nonlocal_force_gpu(
    const double* d_psi, const double* d_occ,
    const double* d_Chi_flat, const int* d_gpos_flat,
    const int* d_gpos_offsets, const int* d_chi_offsets,
    const int* d_ndc_arr, const int* d_nproj_arr,
    const int* d_IP_displ, const double* d_Gamma,
    int n_influence, int total_nproj, int max_ndc, int max_nproj,
    int n_phys_atoms,
    const int* h_IP_displ_phys,
    int nx, int ny, int nz, int FDn, int Nd, int Nband,
    double dV, double occfac,
    double* h_f_nloc, double* h_energy_nl,
    cudaStream_t stream = 0);

// Real gamma-point: kinetic + nonlocal stress only
void compute_kinetic_nonlocal_stress_gpu(
    const double* d_psi, const double* d_occ,
    const double* d_Chi_flat, const int* d_gpos_flat,
    const int* d_gpos_offsets, const int* d_chi_offsets,
    const int* d_ndc_arr, const int* d_nproj_arr,
    const int* d_IP_displ, const double* d_Gamma,
    int n_influence, int total_nproj, int max_ndc, int max_nproj,
    int n_phys_atoms,
    const int* h_IP_displ_phys, const double* h_atom_pos,
    int nx, int ny, int nz, int FDn, int Nd, int Nband,
    double dV, double dx, double dy, double dz,
    int xs, int ys, int zs, double occfac,
    double* h_stress_k, double* h_stress_nl,
    cudaStream_t stream = 0);

// Complex k-point: nonlocal force + energy only
void compute_nonlocal_force_kpt_gpu(
    const cuDoubleComplex* d_psi_z, const double* d_occ,
    const double* d_Chi_flat, const int* d_gpos_flat,
    const int* d_gpos_offsets, const int* d_chi_offsets,
    const int* d_ndc_arr, const int* d_nproj_arr,
    const int* d_IP_displ, const double* d_Gamma,
    const double* d_bloch_fac,
    int n_influence, int total_nproj, int max_ndc, int max_nproj,
    int n_phys_atoms,
    const int* h_IP_displ_phys,
    int nx, int ny, int nz, int FDn, int Nd, int Nband,
    double dV,
    double kxLx, double kyLy, double kzLz,
    double spn_fac_wk,
    double* h_f_nloc, double* h_energy_nl,
    cudaStream_t stream = 0);

// Complex k-point: kinetic + nonlocal stress only
void compute_kinetic_nonlocal_stress_kpt_gpu(
    const cuDoubleComplex* d_psi_z, const double* d_occ,
    const double* d_Chi_flat, const int* d_gpos_flat,
    const int* d_gpos_offsets, const int* d_chi_offsets,
    const int* d_ndc_arr, const int* d_nproj_arr,
    const int* d_IP_displ, const double* d_Gamma,
    const double* d_bloch_fac,
    int n_influence, int total_nproj, int max_ndc, int max_nproj,
    int n_phys_atoms,
    const int* h_IP_displ_phys, const double* h_atom_pos,
    int nx, int ny, int nz, int FDn, int Nd, int Nband,
    double dV, double dx, double dy, double dz,
    int xs, int ys, int zs,
    double kxLx, double kyLy, double kzLz,
    double spn_fac_wk,
    double* h_stress_k, double* h_stress_nl,
    cudaStream_t stream = 0);

// Compute SOC nonlocal forces on GPU (complex spinor, k-point)
void compute_soc_force_gpu(
    const cuDoubleComplex* d_psi_spinor, const double* d_occ,
    const cuDoubleComplex* d_Chi_soc_flat, const int* d_gpos_flat,
    const int* d_gpos_offsets_soc, const int* d_chi_soc_offsets,
    const int* d_ndc_arr_soc, const int* d_nproj_soc_arr,
    const int* d_IP_displ_soc,
    const double* d_Gamma_soc, const int* d_proj_l, const int* d_proj_m,
    const double* d_bloch_fac,
    int n_influence_soc, int total_soc_nproj,
    int max_ndc_soc, int max_nproj_soc,
    int n_phys_atoms,
    const int* h_IP_displ_phys_soc,
    const int* h_proj_l, const int* h_proj_m,
    const double* h_Gamma_soc,
    int nx, int ny, int nz, int FDn, int Nd_d, int Nband,
    double dV, double kx_Lx, double ky_Ly, double kz_Lz,
    double spn_fac, double wk,
    double* h_f_soc,
    cudaStream_t stream = 0);

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
