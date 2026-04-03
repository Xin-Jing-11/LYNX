#pragma once
#ifdef USE_CUDA

namespace lynx {
namespace gpu {

// LDA exchange-correlation (Perdew-Wang)
void lda_pw_gpu(const double* d_rho, double* d_exc, double* d_vxc, int N);

// LDA exchange-correlation (Perdew-Zunger)
void lda_pz_gpu(const double* d_rho, double* d_exc, double* d_vxc, int N);

// LDA spin-polarized (Perdew-Wang)
void lda_pw_spin_gpu(const double* d_rho_up, const double* d_rho_dn,
                      double* d_exc, double* d_vxc_up, double* d_vxc_dn, int N);

// GGA PBE exchange-correlation
void gga_pbe_gpu(const double* d_rho, const double* d_sigma,
                  double* d_exc, double* d_vxc, double* d_v2xc, int N);

// GGA PBE spin-polarized
void gga_pbe_spin_gpu(const double* d_rho, const double* d_sigma,
                       double* d_exc, double* d_vxc, double* d_v2xc, int N);

// mGGA SCAN exchange-correlation
void mgga_scan_gpu(const double* d_rho, const double* d_sigma, const double* d_tau,
                    double* d_exc, double* d_vxc, double* d_v2xc, double* d_vtau, int N);

// mGGA SCAN spin-polarized
void mgga_scan_spin_gpu(
    const double* d_rho, const double* d_sigma, const double* d_tau,
    double* d_exc, double* d_vxc, double* d_v2xc, double* d_vtau, int N);

// mGGA via libxc (for rSCAN, r2SCAN, etc.)
void mgga_libxc_gpu(int xc_x_id, int xc_c_id,
                     const double* d_rho, const double* d_sigma, const double* d_tau,
                     double* d_exc, double* d_vxc, double* d_v2xc, double* d_vtau, int N);

// mGGA via libxc, spin-polarized
void mgga_libxc_spin_gpu(int xc_x_id, int xc_c_id,
                          const double* d_rho, const double* d_sigma, const double* d_tau,
                          double* d_exc, double* d_vxc, double* d_v2xc, double* d_vtau, int N);

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
