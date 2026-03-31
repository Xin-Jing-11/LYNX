// XC functional evaluation using libxc.
// Builtin implementation backed up to XCFunctional_builtin.cpp.bak

#include "xc/XCFunctional.hpp"
#include "core/constants.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <xc.h>
#include <xc_funcs.h>

namespace lynx {

void XCFunctional::setup(XCType type, const Domain& domain, const FDGrid& grid,
                          const Gradient* gradient, const HaloExchange* halo) {
    type_ = type;
    domain_ = &domain;
    grid_ = &grid;
    gradient_ = gradient;
    halo_ = halo;
}

void XCFunctional::get_func_ids(int& xc_id, int& cc_id) const {
    switch (type_) {
        case XCType::LDA_PZ:     xc_id = XC_LDA_X; cc_id = XC_LDA_C_PZ;     break;
        case XCType::LDA_PW:     xc_id = XC_LDA_X; cc_id = XC_LDA_C_PW;     break;
        case XCType::GGA_PBE:    xc_id = XC_GGA_X_PBE; cc_id = XC_GGA_C_PBE; break;
        case XCType::GGA_PBEsol: xc_id = XC_GGA_X_PBE_SOL; cc_id = XC_GGA_C_PBE_SOL; break;
        case XCType::GGA_RPBE:   xc_id = XC_GGA_X_RPBE; cc_id = XC_GGA_C_PBE; break;
        case XCType::MGGA_SCAN:   xc_id = XC_MGGA_X_SCAN;   cc_id = XC_MGGA_C_SCAN;   break;
        case XCType::MGGA_RSCAN:  xc_id = XC_MGGA_X_RSCAN;  cc_id = XC_MGGA_C_RSCAN;  break;
        case XCType::MGGA_R2SCAN: xc_id = XC_MGGA_X_R2SCAN; cc_id = XC_MGGA_C_R2SCAN; break;
        case XCType::HYB_PBE0:   xc_id = XC_GGA_X_PBE; cc_id = XC_GGA_C_PBE; break;
        case XCType::HYB_HSE:    xc_id = XC_GGA_X_PBE; cc_id = XC_GGA_C_PBE; break;
        default:                  xc_id = XC_LDA_X; cc_id = XC_LDA_C_PW;              break;
    }
}

// ============================================================
// Non-spin-polarized evaluation
// ============================================================
void XCFunctional::evaluate(const double* rho, double* Vxc, double* exc, int Nd_d,
                             double* Dxcdgrho_out,
                             const double* tau_in,
                             double* vtau_out) const {
    int xc_id, cc_id;
    get_func_ids(xc_id, cc_id);

    xc_func_type func_x, func_c;
    xc_func_init(&func_x, xc_id, XC_UNPOLARIZED);
    xc_func_init(&func_c, cc_id, XC_UNPOLARIZED);

    size_t np = static_cast<size_t>(Nd_d);

    if (is_mgga() && gradient_ && halo_) {
        // mGGA (SCAN) path using hand-coded functional
        // 1. Compute gradients (same as GGA)
        int FDn = gradient_->stencil().FDn();
        int nx = domain_->Nx_d(), ny = domain_->Ny_d(), nz = domain_->Nz_d();
        int nd_ex = (nx + 2*FDn) * (ny + 2*FDn) * (nz + 2*FDn);

        std::vector<double> rho_ex(nd_ex, 0.0);
        halo_->execute(rho, rho_ex.data(), 1);

        std::vector<double> Drho_x(Nd_d), Drho_y(Nd_d), Drho_z(Nd_d);
        gradient_->apply(rho_ex.data(), Drho_x.data(), 0, 1);
        gradient_->apply(rho_ex.data(), Drho_y.data(), 1, 1);
        gradient_->apply(rho_ex.data(), Drho_z.data(), 2, 1);

        // 2. sigma = |grad rho|^2 with floor to prevent division by zero in SCAN v2
        bool is_orth = grid_->lattice().is_orthogonal();
        const Mat3& lapcT = grid_->lattice().lapc_T();

        std::vector<double> sigma(Nd_d);
        if (is_orth) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < Nd_d; i++)
                sigma[i] = Drho_x[i]*Drho_x[i] + Drho_y[i]*Drho_y[i] + Drho_z[i]*Drho_z[i];
        } else {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < Nd_d; i++) {
                double dx = Drho_x[i], dy = Drho_y[i], dz = Drho_z[i];
                sigma[i] = lapcT(0,0)*dx*dx + lapcT(1,1)*dy*dy + lapcT(2,2)*dz*dz
                         + 2.0*lapcT(0,1)*dx*dy + 2.0*lapcT(0,2)*dx*dz + 2.0*lapcT(1,2)*dy*dz;
            }
        }
        // Floor sigma to prevent division by zero in SCAN v2 = d(nε)/d|∇n| / |∇n|
        // Matches SPARC's exchangeCorrelation.c line 170: if (sigma < 1E-14) sigma = 1E-14
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; i++) {
            if (sigma[i] < 1e-14) sigma[i] = 1e-14;
        }

        // 3. Evaluate mGGA via libxc (SCAN/rSCAN/r2SCAN) — thread-parallel
        std::vector<double> zk_x(Nd_d, 0.0), vrho_x(Nd_d, 0.0), vsigma_x(Nd_d, 0.0), vlapl_x(Nd_d, 0.0), vtau_x(Nd_d, 0.0);
        std::vector<double> zk_c(Nd_d, 0.0), vrho_c(Nd_d, 0.0), vsigma_c(Nd_d, 0.0), vlapl_c(Nd_d, 0.0), vtau_c(Nd_d, 0.0);
        std::vector<double> lapl(Nd_d, 0.0);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int chunk = (Nd_d + nt - 1) / nt;
            int start = tid * chunk;
            int end = std::min(start + chunk, Nd_d);
            int np_t = end - start;
            if (np_t > 0) {
                xc_func_type mx, mc;
                xc_func_init(&mx, xc_id, XC_UNPOLARIZED);
                xc_func_init(&mc, cc_id, XC_UNPOLARIZED);
                xc_mgga_exc_vxc(&mx, np_t, &rho[start], &sigma[start], &lapl[start], &tau_in[start],
                                &zk_x[start], &vrho_x[start], &vsigma_x[start], &vlapl_x[start], &vtau_x[start]);
                xc_mgga_exc_vxc(&mc, np_t, &rho[start], &sigma[start], &lapl[start], &tau_in[start],
                                &zk_c[start], &vrho_c[start], &vsigma_c[start], &vlapl_c[start], &vtau_c[start]);
                xc_func_end(&mx);
                xc_func_end(&mc);
            }
        }

        // 4. Combine outputs: exc, Vxc, Dxcdgrho = 2*vsigma
        std::vector<double> v2xc(Nd_d);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; i++) {
            exc[i] = zk_x[i] + zk_c[i];
            Vxc[i] = vrho_x[i] + vrho_c[i];
            v2xc[i] = 2.0 * (vsigma_x[i] + vsigma_c[i]);
        }

        // 5. Output Dxcdgrho
        if (Dxcdgrho_out)
            std::memcpy(Dxcdgrho_out, v2xc.data(), Nd_d * sizeof(double));

        // 6. GGA divergence correction to Vxc (same as GGA)
        std::vector<double> fx(Nd_d), fy(Nd_d), fz(Nd_d);
        if (is_orth) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < Nd_d; i++) {
                fx[i] = Drho_x[i] * v2xc[i];
                fy[i] = Drho_y[i] * v2xc[i];
                fz[i] = Drho_z[i] * v2xc[i];
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < Nd_d; i++) {
                double dx = Drho_x[i], dy = Drho_y[i], dz = Drho_z[i];
                fx[i] = (lapcT(0,0)*dx + lapcT(0,1)*dy + lapcT(0,2)*dz) * v2xc[i];
                fy[i] = (lapcT(1,0)*dx + lapcT(1,1)*dy + lapcT(1,2)*dz) * v2xc[i];
                fz[i] = (lapcT(2,0)*dx + lapcT(2,1)*dy + lapcT(2,2)*dz) * v2xc[i];
            }
        }

        std::vector<double> f_ex(nd_ex), DDf(Nd_d);
        halo_->execute(fx.data(), f_ex.data(), 1);
        gradient_->apply(f_ex.data(), DDf.data(), 0, 1);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; i++) Vxc[i] -= DDf[i];

        halo_->execute(fy.data(), f_ex.data(), 1);
        gradient_->apply(f_ex.data(), DDf.data(), 1, 1);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; i++) Vxc[i] -= DDf[i];

        halo_->execute(fz.data(), f_ex.data(), 1);
        gradient_->apply(f_ex.data(), DDf.data(), 2, 1);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; i++) Vxc[i] -= DDf[i];

        // 7. Output vtau
        if (vtau_out) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < Nd_d; i++)
                vtau_out[i] = vtau_x[i] + vtau_c[i];
        }

    } else if (is_gga() && gradient_ && halo_) {
        // Compute gradients
        int FDn = gradient_->stencil().FDn();
        int nx = domain_->Nx_d(), ny = domain_->Ny_d(), nz = domain_->Nz_d();
        int nd_ex = (nx + 2*FDn) * (ny + 2*FDn) * (nz + 2*FDn);

        std::vector<double> rho_ex(nd_ex, 0.0);
        halo_->execute(rho, rho_ex.data(), 1);

        std::vector<double> Drho_x(Nd_d), Drho_y(Nd_d), Drho_z(Nd_d);
        gradient_->apply(rho_ex.data(), Drho_x.data(), 0, 1);
        gradient_->apply(rho_ex.data(), Drho_y.data(), 1, 1);
        gradient_->apply(rho_ex.data(), Drho_z.data(), 2, 1);

        // sigma = |∇ρ|² (with metric tensor for non-orthogonal cells)
        bool is_orth = grid_->lattice().is_orthogonal();
        const Mat3& lapcT = grid_->lattice().lapc_T();

        std::vector<double> sigma(Nd_d);
        if (is_orth) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < Nd_d; i++)
                sigma[i] = Drho_x[i]*Drho_x[i] + Drho_y[i]*Drho_y[i] + Drho_z[i]*Drho_z[i];
        } else {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < Nd_d; i++) {
                double dx = Drho_x[i], dy = Drho_y[i], dz = Drho_z[i];
                sigma[i] = lapcT(0,0)*dx*dx + lapcT(1,1)*dy*dy + lapcT(2,2)*dz*dz
                         + 2.0*lapcT(0,1)*dx*dy + 2.0*lapcT(0,2)*dx*dz + 2.0*lapcT(1,2)*dy*dz;
            }
        }

        // Evaluate exchange and correlation via libxc — thread-parallel
        std::vector<double> zk_x(Nd_d, 0.0), vrho_x(Nd_d, 0.0), vsigma_x(Nd_d, 0.0);
        std::vector<double> zk_c(Nd_d, 0.0), vrho_c(Nd_d, 0.0), vsigma_c(Nd_d, 0.0);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int chunk = (Nd_d + nt - 1) / nt;
            int start = tid * chunk;
            int end = std::min(start + chunk, Nd_d);
            int np_t = end - start;
            if (np_t > 0) {
                xc_func_type fx, fc;
                xc_func_init(&fx, xc_id, XC_UNPOLARIZED);
                xc_func_init(&fc, cc_id, XC_UNPOLARIZED);
                xc_gga_exc_vxc(&fx, np_t, &rho[start], &sigma[start],
                               &zk_x[start], &vrho_x[start], &vsigma_x[start]);
                xc_gga_exc_vxc(&fc, np_t, &rho[start], &sigma[start],
                               &zk_c[start], &vrho_c[start], &vsigma_c[start]);
                xc_func_end(&fx);
                xc_func_end(&fc);
            }
        }

        // Apply exchange scaling for hybrid functionals (1-exx_frac during Fock loop)
        double xs = exchange_scale_;

        // Combine: Dxcdgrho = 2*vsigma (our convention for -div(Dxcdgrho * ∇ρ))
        std::vector<double> v2xc(Nd_d);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; i++) {
            exc[i] = xs * zk_x[i] + zk_c[i];
            Vxc[i] = xs * vrho_x[i] + vrho_c[i];
            v2xc[i] = 2.0 * (xs * vsigma_x[i] + vsigma_c[i]);
        }

        if (Dxcdgrho_out)
            std::memcpy(Dxcdgrho_out, v2xc.data(), Nd_d * sizeof(double));

        // GGA divergence correction: Vxc += -div(v2xc * lapcT * ∇ρ)
        std::vector<double> fx(Nd_d), fy(Nd_d), fz(Nd_d);
        if (is_orth) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < Nd_d; i++) {
                fx[i] = Drho_x[i] * v2xc[i];
                fy[i] = Drho_y[i] * v2xc[i];
                fz[i] = Drho_z[i] * v2xc[i];
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < Nd_d; i++) {
                double dx = Drho_x[i], dy = Drho_y[i], dz = Drho_z[i];
                fx[i] = (lapcT(0,0)*dx + lapcT(0,1)*dy + lapcT(0,2)*dz) * v2xc[i];
                fy[i] = (lapcT(1,0)*dx + lapcT(1,1)*dy + lapcT(1,2)*dz) * v2xc[i];
                fz[i] = (lapcT(2,0)*dx + lapcT(2,1)*dy + lapcT(2,2)*dz) * v2xc[i];
            }
        }

        std::vector<double> f_ex(nd_ex), DDf(Nd_d);
        halo_->execute(fx.data(), f_ex.data(), 1);
        gradient_->apply(f_ex.data(), DDf.data(), 0, 1);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; i++) Vxc[i] -= DDf[i];

        halo_->execute(fy.data(), f_ex.data(), 1);
        gradient_->apply(f_ex.data(), DDf.data(), 1, 1);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; i++) Vxc[i] -= DDf[i];

        halo_->execute(fz.data(), f_ex.data(), 1);
        gradient_->apply(f_ex.data(), DDf.data(), 2, 1);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; i++) Vxc[i] -= DDf[i];

    } else {
        // LDA (or GGA fallback when no gradient available) — thread-parallel libxc
        int lda_x_id = xc_id, lda_c_id = cc_id;
        if (is_gga()) {
            xc_func_end(&func_x);
            xc_func_end(&func_c);
            lda_x_id = XC_LDA_X;
            lda_c_id = XC_LDA_C_PW;
            xc_func_init(&func_x, lda_x_id, XC_UNPOLARIZED);
            xc_func_init(&func_c, lda_c_id, XC_UNPOLARIZED);
        }

        std::vector<double> zk_x(Nd_d, 0.0), vrho_x(Nd_d, 0.0);
        std::vector<double> zk_c(Nd_d, 0.0), vrho_c(Nd_d, 0.0);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int chunk = (Nd_d + nt - 1) / nt;
            int start = tid * chunk;
            int end = std::min(start + chunk, Nd_d);
            int np_t = end - start;
            if (np_t > 0) {
                xc_func_type fx, fc;
                xc_func_init(&fx, lda_x_id, XC_UNPOLARIZED);
                xc_func_init(&fc, lda_c_id, XC_UNPOLARIZED);
                xc_lda_exc_vxc(&fx, np_t, &rho[start], &zk_x[start], &vrho_x[start]);
                xc_lda_exc_vxc(&fc, np_t, &rho[start], &zk_c[start], &vrho_c[start]);
                xc_func_end(&fx);
                xc_func_end(&fc);
            }
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; i++) {
            exc[i] = zk_x[i] + zk_c[i];
            Vxc[i] = vrho_x[i] + vrho_c[i];
        }
    }

    xc_func_end(&func_x);
    xc_func_end(&func_c);
}

// ============================================================
// Spin-polarized evaluation
// rho: [total(Nd_d) | up(Nd_d) | down(Nd_d)]
// Vxc: [up(Nd_d) | down(Nd_d)]
// ============================================================
void XCFunctional::evaluate_spin(const double* rho, double* Vxc, double* exc, int Nd_d,
                                  double* Dxcdgrho_out,
                                  const double* tau_in,
                                  double* vtau_out) const {
    int xc_id, cc_id;
    get_func_ids(xc_id, cc_id);

    xc_func_type func_x, func_c;
    xc_func_init(&func_x, xc_id, XC_POLARIZED);
    xc_func_init(&func_c, cc_id, XC_POLARIZED);

    size_t np = static_cast<size_t>(Nd_d);

    // Convert our [total|up|down] to libxc interleaved [up0,dn0,up1,dn1,...]
    std::vector<double> rho_libxc(2 * Nd_d);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Nd_d; i++) {
        rho_libxc[2*i]     = rho[Nd_d + i];
        rho_libxc[2*i + 1] = rho[2*Nd_d + i];
    }

    if (is_mgga() && gradient_ && halo_) {
        // mGGA (SCAN) spin-polarized path using hand-coded functional
        int FDn = gradient_->stencil().FDn();
        int nx = domain_->Nx_d(), ny = domain_->Ny_d(), nz = domain_->Nz_d();
        int nd_ex = (nx + 2*FDn) * (ny + 2*FDn) * (nz + 2*FDn);

        // Compute gradients for total, up, down
        std::vector<double> Drho_x(3 * Nd_d), Drho_y(3 * Nd_d), Drho_z(3 * Nd_d);
        std::vector<double> rho_ex(nd_ex);
        for (int col = 0; col < 3; col++) {
            halo_->execute(rho + col * Nd_d, rho_ex.data(), 1);
            gradient_->apply(rho_ex.data(), Drho_x.data() + col * Nd_d, 0, 1);
            gradient_->apply(rho_ex.data(), Drho_y.data() + col * Nd_d, 1, 1);
            gradient_->apply(rho_ex.data(), Drho_z.data() + col * Nd_d, 2, 1);
        }

        bool is_orth = grid_->lattice().is_orthogonal();
        const Mat3& lapcT = grid_->lattice().lapc_T();

        auto dot_metric = [&](int a, int b) -> double {
            if (is_orth)
                return Drho_x[a]*Drho_x[b] + Drho_y[a]*Drho_y[b] + Drho_z[a]*Drho_z[b];
            double ax = Drho_x[a], ay = Drho_y[a], az = Drho_z[a];
            double bx = Drho_x[b], by = Drho_y[b], bz = Drho_z[b];
            return lapcT(0,0)*ax*bx + lapcT(1,1)*ay*by + lapcT(2,2)*az*bz
                 + lapcT(0,1)*(ax*by + ay*bx) + lapcT(0,2)*(ax*bz + az*bx)
                 + lapcT(1,2)*(ay*bz + az*by);
        };

        // sigma layout for SPARC: [total(Nd_d) | up(Nd_d) | dn(Nd_d)]
        std::vector<double> sigma(3 * Nd_d);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; i++) {
            sigma[i] = dot_metric(i, i);             // |grad rho_total|^2
            int iu = Nd_d + i, id = 2*Nd_d + i;
            sigma[Nd_d + i] = dot_metric(iu, iu);    // |grad rho_up|^2
            sigma[2*Nd_d + i] = dot_metric(id, id);  // |grad rho_dn|^2
        }
        // Floor sigma (matching SPARC)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < 3 * Nd_d; i++) {
            if (sigma[i] < 1e-14) sigma[i] = 1e-14;
        }

        // Evaluate SCAN via libxc (spin-polarized)
        // libxc uses interleaved layout: rho[2*np], sigma[3*np], tau[2*np]
        // sigma_libxc: [σ_uu, σ_ud, σ_dd] per point
        std::vector<double> sigma_libxc(3 * Nd_d);
        std::vector<double> tau_libxc(2 * Nd_d, 0.0);
        std::vector<double> lapl_libxc(2 * Nd_d, 0.0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; i++) {
            int iu = Nd_d + i, id = 2*Nd_d + i;
            sigma_libxc[3*i]     = sigma[iu];  // σ_uu = |∇ρ_up|²
            sigma_libxc[3*i + 1] = dot_metric(iu, id);  // σ_ud = ∇ρ_up · ∇ρ_dn
            sigma_libxc[3*i + 2] = sigma[id];  // σ_dd = |∇ρ_dn|²
            if (tau_in) {
                // tau layout: [up(0:Nd_d) | dn(Nd_d:2*Nd_d) | total(2*Nd_d:3*Nd_d)]
                tau_libxc[2*i]     = tau_in[i];          // τ_up
                tau_libxc[2*i + 1] = tau_in[Nd_d + i];  // τ_dn
            }
        }

        // Evaluate SCAN via libxc (spin-polarized) — thread-parallel
        std::vector<double> zk_x(Nd_d, 0.0), vrho_x(2*Nd_d, 0.0), vsigma_x(3*Nd_d, 0.0), vlapl_x(2*Nd_d, 0.0), vtau_x(2*Nd_d, 0.0);
        std::vector<double> zk_c(Nd_d, 0.0), vrho_c(2*Nd_d, 0.0), vsigma_c(3*Nd_d, 0.0), vlapl_c(2*Nd_d, 0.0), vtau_c(2*Nd_d, 0.0);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int chunk = (Nd_d + nt - 1) / nt;
            int start = tid * chunk;
            int end = std::min(start + chunk, Nd_d);
            int np_t = end - start;
            if (np_t > 0) {
                xc_func_type mx, mc;
                xc_func_init(&mx, xc_id, XC_POLARIZED);
                xc_func_init(&mc, cc_id, XC_POLARIZED);
                xc_mgga_exc_vxc(&mx, np_t, &rho_libxc[2*start], &sigma_libxc[3*start],
                                &lapl_libxc[2*start], &tau_libxc[2*start],
                                &zk_x[start], &vrho_x[2*start], &vsigma_x[3*start],
                                &vlapl_x[2*start], &vtau_x[2*start]);
                xc_mgga_exc_vxc(&mc, np_t, &rho_libxc[2*start], &sigma_libxc[3*start],
                                &lapl_libxc[2*start], &tau_libxc[2*start],
                                &zk_c[start], &vrho_c[2*start], &vsigma_c[3*start],
                                &vlapl_c[2*start], &vtau_c[2*start]);
                xc_func_end(&mx);
                xc_func_end(&mc);
            }
        }

        // Combine: exc, Vxc (per-spin, de-interleave from libxc)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; i++) {
            exc[i] = zk_x[i] + zk_c[i];
            Vxc[i]        = vrho_x[2*i] + vrho_c[2*i];         // up
            Vxc[Nd_d + i] = vrho_x[2*i + 1] + vrho_c[2*i + 1]; // down
        }

        // Dxcdgrho: [v2c(Nd_d) | v2x_up(Nd_d) | v2x_dn(Nd_d)]
        // Used in stress: stress -= ∫ v2 * ∂ρ/∂x_α * ∂ρ/∂x_β dV
        // SPARC convention: v2c = 2*(∂(nεc)/∂σ_total), v2x_σ = 2*(∂(nεx)/∂σ_σσ)
        // For correlation: σ_total = σ_uu + 2σ_ud + σ_dd
        //   libxc: vsigma_c_uu = ∂(nεc)/∂σ_total, so v2c = 2*vsigma_c_uu
        // For exchange: acts on individual spin channels
        //   libxc: vsigma_x_uu = ∂(nεx)/∂σ_uu, so v2x_up = 2*vsigma_x_uu
        if (Dxcdgrho_out) {
            for (int i = 0; i < Nd_d; i++) {
                Dxcdgrho_out[i] = 2.0 * vsigma_c[3*i];
                Dxcdgrho_out[Nd_d + i] = 2.0 * vsigma_x[3*i];
                Dxcdgrho_out[2*Nd_d + i] = 2.0 * vsigma_x[3*i + 2];
            }
        }

        // vtau: [up(Nd_d) | dn(Nd_d)] (de-interleave from libxc)
        if (vtau_out) {
            for (int i = 0; i < Nd_d; i++) {
                vtau_out[i]        = vtau_x[2*i] + vtau_c[2*i];         // up
                vtau_out[Nd_d + i] = vtau_x[2*i + 1] + vtau_c[2*i + 1]; // dn
            }
        }

        // GGA divergence correction for spin — same chain-rule formula as GGA path:
        // ∂(ρε)/∂∇ρ_↑ = 2*vsigma_↑↑*∇ρ_↑ + vsigma_↑↓*∇ρ_↓
        // ∂(ρε)/∂∇ρ_↓ = vsigma_↑↓*∇ρ_↑ + 2*vsigma_↓↓*∇ρ_↓
        std::vector<double> fx_up(Nd_d), fy_up(Nd_d), fz_up(Nd_d);
        std::vector<double> fx_dn(Nd_d), fy_dn(Nd_d), fz_dn(Nd_d);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; i++) {
            double vs_uu = vsigma_x[3*i]   + vsigma_c[3*i];
            double vs_ud = vsigma_x[3*i+1] + vsigma_c[3*i+1];
            double vs_dd = vsigma_x[3*i+2] + vsigma_c[3*i+2];

            int iu = Nd_d + i, id = 2*Nd_d + i;

            double fu_x = 2.0*vs_uu * Drho_x[iu] + vs_ud * Drho_x[id];
            double fu_y = 2.0*vs_uu * Drho_y[iu] + vs_ud * Drho_y[id];
            double fu_z = 2.0*vs_uu * Drho_z[iu] + vs_ud * Drho_z[id];

            double fd_x = vs_ud * Drho_x[iu] + 2.0*vs_dd * Drho_x[id];
            double fd_y = vs_ud * Drho_y[iu] + 2.0*vs_dd * Drho_y[id];
            double fd_z = vs_ud * Drho_z[iu] + 2.0*vs_dd * Drho_z[id];

            if (is_orth) {
                fx_up[i] = fu_x; fy_up[i] = fu_y; fz_up[i] = fu_z;
                fx_dn[i] = fd_x; fy_dn[i] = fd_y; fz_dn[i] = fd_z;
            } else {
                fx_up[i] = lapcT(0,0)*fu_x + lapcT(0,1)*fu_y + lapcT(0,2)*fu_z;
                fy_up[i] = lapcT(1,0)*fu_x + lapcT(1,1)*fu_y + lapcT(1,2)*fu_z;
                fz_up[i] = lapcT(2,0)*fu_x + lapcT(2,1)*fu_y + lapcT(2,2)*fu_z;
                fx_dn[i] = lapcT(0,0)*fd_x + lapcT(0,1)*fd_y + lapcT(0,2)*fd_z;
                fy_dn[i] = lapcT(1,0)*fd_x + lapcT(1,1)*fd_y + lapcT(1,2)*fd_z;
                fz_dn[i] = lapcT(2,0)*fd_x + lapcT(2,1)*fd_y + lapcT(2,2)*fd_z;
            }
        }

        std::vector<double> f_ex(nd_ex), DDf(Nd_d);
        auto apply_div = [&](double* fx, double* fy, double* fz, double* Vxc_s) {
            halo_->execute(fx, f_ex.data(), 1);
            gradient_->apply(f_ex.data(), DDf.data(), 0, 1);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < Nd_d; i++) Vxc_s[i] -= DDf[i];
            halo_->execute(fy, f_ex.data(), 1);
            gradient_->apply(f_ex.data(), DDf.data(), 1, 1);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < Nd_d; i++) Vxc_s[i] -= DDf[i];
            halo_->execute(fz, f_ex.data(), 1);
            gradient_->apply(f_ex.data(), DDf.data(), 2, 1);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < Nd_d; i++) Vxc_s[i] -= DDf[i];
        };

        apply_div(fx_up.data(), fy_up.data(), fz_up.data(), Vxc);
        apply_div(fx_dn.data(), fy_dn.data(), fz_dn.data(), Vxc + Nd_d);

    } else if (is_gga() && gradient_ && halo_) {
        // Gradients for total, up, down
        int FDn = gradient_->stencil().FDn();
        int nx = domain_->Nx_d(), ny = domain_->Ny_d(), nz = domain_->Nz_d();
        int nd_ex = (nx + 2*FDn) * (ny + 2*FDn) * (nz + 2*FDn);

        std::vector<double> Drho_x(3 * Nd_d), Drho_y(3 * Nd_d), Drho_z(3 * Nd_d);
        std::vector<double> rho_ex(nd_ex);

        for (int col = 0; col < 3; col++) {
            halo_->execute(rho + col * Nd_d, rho_ex.data(), 1);
            gradient_->apply(rho_ex.data(), Drho_x.data() + col * Nd_d, 0, 1);
            gradient_->apply(rho_ex.data(), Drho_y.data() + col * Nd_d, 1, 1);
            gradient_->apply(rho_ex.data(), Drho_z.data() + col * Nd_d, 2, 1);
        }

        bool is_orth = grid_->lattice().is_orthogonal();
        const Mat3& lapcT = grid_->lattice().lapc_T();

        // Metric dot product of gradient columns a and b at point i
        auto dot_metric = [&](int a, int b) -> double {
            if (is_orth)
                return Drho_x[a]*Drho_x[b] + Drho_y[a]*Drho_y[b] + Drho_z[a]*Drho_z[b];
            double ax = Drho_x[a], ay = Drho_y[a], az = Drho_z[a];
            double bx = Drho_x[b], by = Drho_y[b], bz = Drho_z[b];
            return lapcT(0,0)*ax*bx + lapcT(1,1)*ay*by + lapcT(2,2)*az*bz
                 + lapcT(0,1)*(ax*by + ay*bx) + lapcT(0,2)*(ax*bz + az*bx)
                 + lapcT(1,2)*(ay*bz + az*by);
        };

        // libxc sigma: interleaved [σ_uu, σ_ud, σ_dd] per point
        std::vector<double> sigma_libxc(3 * Nd_d);
        for (int i = 0; i < Nd_d; i++) {
            int iu = Nd_d + i, id = 2*Nd_d + i;
            sigma_libxc[3*i]     = dot_metric(iu, iu);
            sigma_libxc[3*i + 1] = dot_metric(iu, id);
            sigma_libxc[3*i + 2] = dot_metric(id, id);
        }

        // Evaluate via libxc — thread-parallel
        std::vector<double> zk_x(Nd_d, 0.0), vrho_x(2*Nd_d, 0.0), vsigma_x(3*Nd_d, 0.0);
        std::vector<double> zk_c(Nd_d, 0.0), vrho_c(2*Nd_d, 0.0), vsigma_c(3*Nd_d, 0.0);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int chunk = (Nd_d + nt - 1) / nt;
            int start = tid * chunk;
            int end = std::min(start + chunk, Nd_d);
            int np_t = end - start;
            if (np_t > 0) {
                xc_func_type fx, fc;
                xc_func_init(&fx, xc_id, XC_POLARIZED);
                xc_func_init(&fc, cc_id, XC_POLARIZED);
                xc_gga_exc_vxc(&fx, np_t, &rho_libxc[2*start], &sigma_libxc[3*start],
                               &zk_x[start], &vrho_x[2*start], &vsigma_x[3*start]);
                xc_gga_exc_vxc(&fc, np_t, &rho_libxc[2*start], &sigma_libxc[3*start],
                               &zk_c[start], &vrho_c[2*start], &vsigma_c[3*start]);
                xc_func_end(&fx);
                xc_func_end(&fc);
            }
        }

        // Apply exchange scaling for hybrid functionals
        double xs = exchange_scale_;

        // Combine local part (libxc vrho interleaved → our [up|down])
        for (int i = 0; i < Nd_d; i++) {
            exc[i] = xs * zk_x[i] + zk_c[i];
            Vxc[i]        = xs * vrho_x[2*i] + vrho_c[2*i];
            Vxc[Nd_d + i] = xs * vrho_x[2*i + 1] + vrho_c[2*i + 1];
        }

        // GGA divergence: Vxc_s -= ∇·(∂f/∂(∇ρ_s))
        // ∂f/∂(∇ρ_up)   = 2*vsigma_uu*∇ρ_up + vsigma_ud*∇ρ_down
        // ∂f/∂(∇ρ_down) = vsigma_ud*∇ρ_up   + 2*vsigma_dd*∇ρ_down
        std::vector<double> fx_up(Nd_d), fy_up(Nd_d), fz_up(Nd_d);
        std::vector<double> fx_dn(Nd_d), fy_dn(Nd_d), fz_dn(Nd_d);

        for (int i = 0; i < Nd_d; i++) {
            double vs_uu = xs * vsigma_x[3*i]   + vsigma_c[3*i];
            double vs_ud = xs * vsigma_x[3*i+1] + vsigma_c[3*i+1];
            double vs_dd = xs * vsigma_x[3*i+2] + vsigma_c[3*i+2];

            int iu = Nd_d + i, id = 2*Nd_d + i;

            double fu_x = 2.0*vs_uu * Drho_x[iu] + vs_ud * Drho_x[id];
            double fu_y = 2.0*vs_uu * Drho_y[iu] + vs_ud * Drho_y[id];
            double fu_z = 2.0*vs_uu * Drho_z[iu] + vs_ud * Drho_z[id];

            double fd_x = vs_ud * Drho_x[iu] + 2.0*vs_dd * Drho_x[id];
            double fd_y = vs_ud * Drho_y[iu] + 2.0*vs_dd * Drho_y[id];
            double fd_z = vs_ud * Drho_z[iu] + 2.0*vs_dd * Drho_z[id];

            if (is_orth) {
                fx_up[i] = fu_x; fy_up[i] = fu_y; fz_up[i] = fu_z;
                fx_dn[i] = fd_x; fy_dn[i] = fd_y; fz_dn[i] = fd_z;
            } else {
                fx_up[i] = lapcT(0,0)*fu_x + lapcT(0,1)*fu_y + lapcT(0,2)*fu_z;
                fy_up[i] = lapcT(1,0)*fu_x + lapcT(1,1)*fu_y + lapcT(1,2)*fu_z;
                fz_up[i] = lapcT(2,0)*fu_x + lapcT(2,1)*fu_y + lapcT(2,2)*fu_z;
                fx_dn[i] = lapcT(0,0)*fd_x + lapcT(0,1)*fd_y + lapcT(0,2)*fd_z;
                fy_dn[i] = lapcT(1,0)*fd_x + lapcT(1,1)*fd_y + lapcT(1,2)*fd_z;
                fz_dn[i] = lapcT(2,0)*fd_x + lapcT(2,1)*fd_y + lapcT(2,2)*fd_z;
            }
        }

        std::vector<double> f_ex(nd_ex), DDf(Nd_d);
        auto apply_div = [&](double* fx, double* fy, double* fz, double* Vxc_s) {
            halo_->execute(fx, f_ex.data(), 1);
            gradient_->apply(f_ex.data(), DDf.data(), 0, 1);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < Nd_d; i++) Vxc_s[i] -= DDf[i];
            halo_->execute(fy, f_ex.data(), 1);
            gradient_->apply(f_ex.data(), DDf.data(), 1, 1);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < Nd_d; i++) Vxc_s[i] -= DDf[i];
            halo_->execute(fz, f_ex.data(), 1);
            gradient_->apply(f_ex.data(), DDf.data(), 2, 1);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < Nd_d; i++) Vxc_s[i] -= DDf[i];
        };

        apply_div(fx_up.data(), fy_up.data(), fz_up.data(), Vxc);
        apply_div(fx_dn.data(), fy_dn.data(), fz_dn.data(), Vxc + Nd_d);

        // Dxcdgrho in our convention: [v2c | v2x_up | v2x_down]
        if (Dxcdgrho_out) {
            for (int i = 0; i < Nd_d; i++) {
                Dxcdgrho_out[i] = 2.0 * (vsigma_c[3*i] + 2.0*vsigma_c[3*i+1] + vsigma_c[3*i+2]);
                Dxcdgrho_out[Nd_d + i] = 2.0 * xs * vsigma_x[3*i];
                Dxcdgrho_out[2*Nd_d + i] = 2.0 * xs * vsigma_x[3*i + 2];
            }
        }

    } else {
        // LDA spin (or GGA fallback) — thread-parallel libxc
        int lda_x_id = xc_id, lda_c_id = cc_id;
        if (is_gga()) {
            xc_func_end(&func_x);
            xc_func_end(&func_c);
            lda_x_id = XC_LDA_X;
            lda_c_id = XC_LDA_C_PW;
            xc_func_init(&func_x, lda_x_id, XC_POLARIZED);
            xc_func_init(&func_c, lda_c_id, XC_POLARIZED);
        }

        std::vector<double> zk_x(Nd_d, 0.0), vrho_x(2*Nd_d, 0.0);
        std::vector<double> zk_c(Nd_d, 0.0), vrho_c(2*Nd_d, 0.0);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int chunk = (Nd_d + nt - 1) / nt;
            int start = tid * chunk;
            int end = std::min(start + chunk, Nd_d);
            int np_t = end - start;
            if (np_t > 0) {
                xc_func_type fx, fc;
                xc_func_init(&fx, lda_x_id, XC_POLARIZED);
                xc_func_init(&fc, lda_c_id, XC_POLARIZED);
                xc_lda_exc_vxc(&fx, np_t, &rho_libxc[2*start], &zk_x[start], &vrho_x[2*start]);
                xc_lda_exc_vxc(&fc, np_t, &rho_libxc[2*start], &zk_c[start], &vrho_c[2*start]);
                xc_func_end(&fx);
                xc_func_end(&fc);
            }
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nd_d; i++) {
            exc[i] = zk_x[i] + zk_c[i];
            Vxc[i]        = vrho_x[2*i] + vrho_c[2*i];
            Vxc[Nd_d + i] = vrho_x[2*i + 1] + vrho_c[2*i + 1];
        }
    }

    xc_func_end(&func_x);
    xc_func_end(&func_c);
}

} // namespace lynx
