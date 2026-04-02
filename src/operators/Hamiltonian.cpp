#include "operators/Hamiltonian.hpp"
#include "xc/ExactExchange.hpp"
#include <vector>
#include <cstring>
#include <cmath>
#include <type_traits>
#include <omp.h>

namespace lynx {

void Hamiltonian::setup(const FDStencil& stencil,
                        const Domain& domain,
                        const FDGrid& grid,
                        const HaloExchange& halo,
                        const NonlocalProjector* vnl) {
    stencil_ = &stencil;
    domain_ = &domain;
    grid_ = &grid;
    halo_ = &halo;
    vnl_ = vnl;
}

// ---------------------------------------------------------------------------
// Real (Gamma-point) interface
// ---------------------------------------------------------------------------

void Hamiltonian::apply(const double* psi, const double* Veff, double* y,
                        int ncol, double c) const {
    apply_local(psi, Veff, y, ncol, c);
    if (vnl_ && vnl_->is_setup()) {
        vnl_->apply(psi, y, ncol, grid_->dV());
    }
    if (vtau_) apply_mgga(psi, y, ncol);
    if (exx_ && exx_->is_setup()) {
        int Nd = domain_->Nd_d();
        exx_->apply_Vx(psi, Nd, ncol, Nd, y, Nd, exx_spin_);
    }
}

void Hamiltonian::apply_local(const double* psi, const double* Veff, double* y,
                              int ncol, double c) const {
    int nd_ex = halo_->nx_ex() * halo_->ny_ex() * halo_->nz_ex();
    std::vector<double> x_ex(ncol * nd_ex, 0.0);
    halo_->execute(psi, x_ex.data(), ncol);

    if (domain_->global_grid().lattice().is_orthogonal()) {
        lap_plus_diag_orth_impl(x_ex.data(), Veff, y, ncol, c);
    } else {
        lap_plus_diag_nonorth_impl(x_ex.data(), Veff, y, ncol, c);
    }
}

// ---------------------------------------------------------------------------
// Complex (k-point) interface
// ---------------------------------------------------------------------------

void Hamiltonian::apply_kpt(const Complex* psi, const double* Veff, Complex* y,
                            int ncol, const Vec3& kpt_cart, const Vec3& cell_lengths,
                            double c) const {
    apply_local_kpt(psi, Veff, y, ncol, kpt_cart, cell_lengths, c);
    if (vnl_kpt_ && vnl_kpt_->is_setup()) {
        vnl_kpt_->apply_kpt(psi, y, ncol, grid_->dV());
    }
    if (vtau_) apply_mgga_kpt(psi, y, ncol, kpt_cart, cell_lengths);
    if (exx_ && exx_->is_setup()) {
        int Nd = domain_->Nd_d();
        exx_->apply_Vx_kpt(psi, Nd, ncol, Nd, y, Nd, exx_spin_, exx_kpt_);
    }
}

void Hamiltonian::apply_local_kpt(const Complex* psi, const double* Veff, Complex* y,
                                   int ncol, const Vec3& kpt_cart, const Vec3& cell_lengths,
                                   double c) const {
    int nd_ex = halo_->nx_ex() * halo_->ny_ex() * halo_->nz_ex();
    std::vector<Complex> x_ex(ncol * nd_ex, Complex(0.0, 0.0));
    halo_->execute_kpt(psi, x_ex.data(), ncol, kpt_cart, cell_lengths);

    if (domain_->global_grid().lattice().is_orthogonal()) {
        lap_plus_diag_orth_impl(x_ex.data(), Veff, y, ncol, c);
    } else {
        lap_plus_diag_nonorth_impl(x_ex.data(), Veff, y, ncol, c);
    }
}

// ---------------------------------------------------------------------------
// Templated stencil implementations
// ---------------------------------------------------------------------------

template<typename T>
void Hamiltonian::lap_plus_diag_orth_impl(const T* x_ex, const double* Veff,
                                           T* y, int ncol, double c) const {
    int nx = domain_->Nx_d();
    int ny = domain_->Ny_d();
    int nz = domain_->Nz_d();
    int nd = nx * ny * nz;
    int FDn = stencil_->FDn();
    int nx_ex = halo_->nx_ex();
    int ny_ex = halo_->ny_ex();
    int nxny_ex = nx_ex * ny_ex;
    int nd_ex = nxny_ex * halo_->nz_ex();

    const double* cx = stencil_->D2_coeff_x();
    const double* cy = stencil_->D2_coeff_y();
    const double* cz = stencil_->D2_coeff_z();

    int nxny = nx * ny;
    double diag_lap = cx[0] + cy[0] + cz[0];
    int total_nk = ncol * nz;
    #pragma omp parallel for schedule(static)
    for (int nk = 0; nk < total_nk; ++nk) {
        int n = nk / nz;
        int k = nk % nz;
        const T* xn = x_ex + n * nd_ex;
        T* yn = y + n * nd;

        for (int j = 0; j < ny; ++j) {
            int offset = k * nxny + j * nx;
            int offset_ex = (k + FDn) * nxny_ex + (j + FDn) * nx_ex + FDn;
            #pragma omp simd
            for (int i = 0; i < nx; ++i) {
                int idx_ex = offset_ex + i;
                int idx_loc = offset + i;

                T lap = diag_lap * xn[idx_ex];

                for (int p = 1; p <= FDn; ++p) {
                    lap += cx[p] * (xn[idx_ex + p] + xn[idx_ex - p]);
                    lap += cy[p] * (xn[idx_ex + p * nx_ex] + xn[idx_ex - p * nx_ex]);
                    lap += cz[p] * (xn[idx_ex + p * nxny_ex] + xn[idx_ex - p * nxny_ex]);
                }

                double veff_val = Veff ? Veff[idx_loc] : 0.0;
                yn[idx_loc] = -0.5 * lap + (veff_val + c) * xn[idx_ex];
            }
        }
    }
}

template<typename T>
void Hamiltonian::lap_plus_diag_nonorth_impl(const T* x_ex, const double* Veff,
                                              T* y, int ncol, double c) const {
    int nx = domain_->Nx_d();
    int ny = domain_->Ny_d();
    int nz = domain_->Nz_d();
    int nd = nx * ny * nz;
    int FDn = stencil_->FDn();
    int nx_ex = halo_->nx_ex();
    int ny_ex = halo_->ny_ex();
    int nxny_ex = nx_ex * ny_ex;
    int nd_ex = nxny_ex * halo_->nz_ex();

    const double* cx = stencil_->D2_coeff_x();
    const double* cy = stencil_->D2_coeff_y();
    const double* cz = stencil_->D2_coeff_z();
    const double* cxy = stencil_->D2_coeff_xy();
    const double* cxz = stencil_->D2_coeff_xz();
    const double* cyz = stencil_->D2_coeff_yz();

    const double* d1y = stencil_->D1_coeff_y();
    const double* d1z = stencil_->D1_coeff_z();

    bool has_xy = cxy && std::abs(cxy[1]) > 1e-30;
    bool has_xz = cxz && std::abs(cxz[1]) > 1e-30;
    bool has_yz = cyz && std::abs(cyz[1]) > 1e-30;

    int nxny = nx * ny;
    double diag_lap = cx[0] + cy[0] + cz[0];
    int total_nk = ncol * nz;
    #pragma omp parallel for schedule(static)
    for (int nk = 0; nk < total_nk; ++nk) {
        int n = nk / nz;
        int k = nk % nz;
        const T* xn = x_ex + n * nd_ex;
        T* yn = y + n * nd;

        for (int j = 0; j < ny; ++j) {
            int offset = k * nxny + j * nx;
            int offset_ex = (k + FDn) * nxny_ex + (j + FDn) * nx_ex + FDn;
            for (int i = 0; i < nx; ++i) {
                int idx = offset_ex + i;
                int loc = offset + i;

                T lap = diag_lap * xn[idx];

                for (int p = 1; p <= FDn; ++p) {
                    lap += cx[p] * (xn[idx + p] + xn[idx - p]);
                    lap += cy[p] * (xn[idx + p * nx_ex] + xn[idx - p * nx_ex]);
                    lap += cz[p] * (xn[idx + p * nxny_ex] + xn[idx - p * nxny_ex]);
                }

                if (has_xy) {
                    for (int p = 1; p <= FDn; ++p) {
                        for (int q = 1; q <= FDn; ++q) {
                            T mixed = xn[idx + p + q * nx_ex]
                                    - xn[idx + p - q * nx_ex]
                                    - xn[idx - p + q * nx_ex]
                                    + xn[idx - p - q * nx_ex];
                            lap += cxy[p] * d1y[q] * mixed;
                        }
                    }
                }
                if (has_xz) {
                    for (int p = 1; p <= FDn; ++p) {
                        for (int q = 1; q <= FDn; ++q) {
                            T mixed = xn[idx + p + q * nxny_ex]
                                    - xn[idx + p - q * nxny_ex]
                                    - xn[idx - p + q * nxny_ex]
                                    + xn[idx - p - q * nxny_ex];
                            lap += cxz[p] * d1z[q] * mixed;
                        }
                    }
                }
                if (has_yz) {
                    for (int p = 1; p <= FDn; ++p) {
                        for (int q = 1; q <= FDn; ++q) {
                            T mixed = xn[idx + p * nx_ex + q * nxny_ex]
                                    - xn[idx + p * nx_ex - q * nxny_ex]
                                    - xn[idx - p * nx_ex + q * nxny_ex]
                                    + xn[idx - p * nx_ex - q * nxny_ex];
                            lap += cyz[p] * d1z[q] * mixed;
                        }
                    }
                }

                double veff_val = Veff ? Veff[loc] : 0.0;
                yn[loc] = -0.5 * lap + (veff_val + c) * xn[idx];
            }
        }
    }
}

// Legacy real wrappers
void Hamiltonian::lap_plus_diag_orth(const double* x_ex, const double* Veff,
                                      double* y, int ncol, double c) const {
    lap_plus_diag_orth_impl(x_ex, Veff, y, ncol, c);
}

void Hamiltonian::lap_plus_diag_nonorth(const double* x_ex, const double* Veff,
                                         double* y, int ncol, double c) const {
    lap_plus_diag_nonorth_impl(x_ex, Veff, y, ncol, c);
}

// ---------------------------------------------------------------------------
// mGGA term: H_mGGA ψ = -0.5 ∇·(vtau · ∇ψ)
// ---------------------------------------------------------------------------

template<typename T>
void Hamiltonian::apply_mgga_impl(const T* psi, T* y, int ncol,
                                    const Vec3& kpt_cart, const Vec3& cell_lengths) const {
    int Nd_d = domain_->Nd_d();
    int nd_ex = halo_->nd_ex();
    bool is_orth = domain_->global_grid().lattice().is_orthogonal();
    const Mat3& lapcT = domain_->global_grid().lattice().lapc_T();
    Gradient grad(*stencil_, *domain_);

    std::vector<T> psi_ex(nd_ex);
    std::vector<T> dpsi_x(Nd_d), dpsi_y(Nd_d), dpsi_z(Nd_d);
    std::vector<T> f_ex(nd_ex), div_comp(Nd_d);

    for (int n = 0; n < ncol; ++n) {
        const T* col = psi + n * Nd_d;

        // 1. Halo exchange and compute gradient of psi
        if constexpr (std::is_same_v<T, Complex>) {
            halo_->execute_kpt(col, psi_ex.data(), 1, kpt_cart, cell_lengths);
        } else {
            halo_->execute(col, psi_ex.data(), 1);
        }
        grad.apply(psi_ex.data(), dpsi_x.data(), 0, 1);
        grad.apply(psi_ex.data(), dpsi_y.data(), 1, 1);
        grad.apply(psi_ex.data(), dpsi_z.data(), 2, 1);

        // 2. Apply metric tensor for non-orthogonal cells, then multiply by vtau
        std::vector<T> fx(Nd_d), fy(Nd_d), fz(Nd_d);
        if (is_orth) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < Nd_d; ++i) {
                fx[i] = vtau_[i] * dpsi_x[i];
                fy[i] = vtau_[i] * dpsi_y[i];
                fz[i] = vtau_[i] * dpsi_z[i];
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < Nd_d; ++i) {
                T dx = dpsi_x[i], dy = dpsi_y[i], dz = dpsi_z[i];
                fx[i] = vtau_[i] * (lapcT(0,0)*dx + lapcT(0,1)*dy + lapcT(0,2)*dz);
                fy[i] = vtau_[i] * (lapcT(1,0)*dx + lapcT(1,1)*dy + lapcT(1,2)*dz);
                fz[i] = vtau_[i] * (lapcT(2,0)*dx + lapcT(2,1)*dy + lapcT(2,2)*dz);
            }
        }

        // 3. Compute divergence: div = d(fx)/dx + d(fy)/dy + d(fz)/dz
        T* yn = y + n * Nd_d;

        if constexpr (std::is_same_v<T, Complex>) {
            halo_->execute_kpt(fx.data(), f_ex.data(), 1, kpt_cart, cell_lengths);
        } else {
            halo_->execute(fx.data(), f_ex.data(), 1);
        }
        grad.apply(f_ex.data(), div_comp.data(), 0, 1);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; ++i) yn[i] -= 0.5 * div_comp[i];

        if constexpr (std::is_same_v<T, Complex>) {
            halo_->execute_kpt(fy.data(), f_ex.data(), 1, kpt_cart, cell_lengths);
        } else {
            halo_->execute(fy.data(), f_ex.data(), 1);
        }
        grad.apply(f_ex.data(), div_comp.data(), 1, 1);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; ++i) yn[i] -= 0.5 * div_comp[i];

        if constexpr (std::is_same_v<T, Complex>) {
            halo_->execute_kpt(fz.data(), f_ex.data(), 1, kpt_cart, cell_lengths);
        } else {
            halo_->execute(fz.data(), f_ex.data(), 1);
        }
        grad.apply(f_ex.data(), div_comp.data(), 2, 1);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; ++i) yn[i] -= 0.5 * div_comp[i];
    }
}

// Explicit instantiations
template void Hamiltonian::apply_mgga_impl<double>(const double*, double*, int,
    const Vec3&, const Vec3&) const;
template void Hamiltonian::apply_mgga_impl<Complex>(const Complex*, Complex*, int,
    const Vec3&, const Vec3&) const;

void Hamiltonian::apply_mgga(const double* psi, double* y, int ncol) const {
    apply_mgga_impl<double>(psi, y, ncol);
}

void Hamiltonian::apply_mgga_kpt(const Complex* psi, Complex* y, int ncol,
                                   const Vec3& kpt_cart, const Vec3& cell_lengths) const {
    apply_mgga_impl<Complex>(psi, y, ncol, kpt_cart, cell_lengths);
}

// ---------------------------------------------------------------------------
// Spinor (SOC) interface
// ---------------------------------------------------------------------------

void Hamiltonian::apply_spinor_kpt(const Complex* psi, const double* Veff_spinor, Complex* y,
                                    int ncol, int Nd_d, const Vec3& kpt_cart, const Vec3& cell_lengths,
                                    double c) const {
    // Veff_spinor layout: [V_uu(Nd_d) | V_dd(Nd_d) | Re(V_ud)(Nd_d) | Im(V_ud)(Nd_d)]
    const double* V_uu = Veff_spinor;
    const double* V_dd = Veff_spinor + Nd_d;
    const double* V_ud_re = Veff_spinor + 2 * Nd_d;
    const double* V_ud_im = Veff_spinor + 3 * Nd_d;

    int Nd_d_spinor = 2 * Nd_d;

    for (int n = 0; n < ncol; ++n) {
        const Complex* psi_n = psi + n * Nd_d_spinor;
        const Complex* psi_up = psi_n;
        const Complex* psi_dn = psi_n + Nd_d;
        Complex* y_n = y + n * Nd_d_spinor;
        Complex* y_up = y_n;
        Complex* y_dn = y_n + Nd_d;

        // Apply kinetic + diagonal Veff to each spinor component separately
        // Reuse apply_local_kpt for each component
        apply_local_kpt(psi_up, V_uu, y_up, 1, kpt_cart, cell_lengths, c);
        apply_local_kpt(psi_dn, V_dd, y_dn, 1, kpt_cart, cell_lengths, c);

        // Apply off-diagonal Veff: y_up += V_ud * psi_dn, y_dn += V_ud* * psi_up
        for (int i = 0; i < Nd_d; ++i) {
            Complex V_ud(V_ud_re[i], V_ud_im[i]);
            y_up[i] += V_ud * psi_dn[i];
            y_dn[i] += std::conj(V_ud) * psi_up[i];
        }
    }

    // Apply scalar-relativistic Vnl per spinor component
    if (vnl_kpt_ && vnl_kpt_->is_setup()) {
        for (int n = 0; n < ncol; ++n) {
            const Complex* psi_n = psi + n * Nd_d_spinor;
            Complex* y_n = y + n * Nd_d_spinor;
            // Apply Vnl to spin-up
            vnl_kpt_->apply_kpt(psi_n, y_n, 1, grid_->dV());
            // Apply Vnl to spin-down
            vnl_kpt_->apply_kpt(psi_n + Nd_d, y_n + Nd_d, 1, grid_->dV());
        }
    }

    // Apply SOC terms
    if (vnl_kpt_ && vnl_kpt_->has_soc()) {
        vnl_kpt_->apply_soc_kpt(psi, y, ncol, Nd_d, grid_->dV());
    }
}

// Explicit template instantiations
template void Hamiltonian::lap_plus_diag_orth_impl<double>(const double*, const double*,
                                                            double*, int, double) const;
template void Hamiltonian::lap_plus_diag_orth_impl<Complex>(const Complex*, const double*,
                                                             Complex*, int, double) const;
template void Hamiltonian::lap_plus_diag_nonorth_impl<double>(const double*, const double*,
                                                               double*, int, double) const;
template void Hamiltonian::lap_plus_diag_nonorth_impl<Complex>(const Complex*, const double*,
                                                                Complex*, int, double) const;

} // namespace lynx
