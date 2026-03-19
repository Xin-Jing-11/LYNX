// XC functional evaluation using libxc.
// Builtin implementation backed up to XCFunctional_builtin.cpp.bak

#include "xc/XCFunctional.hpp"
#include "xc/SCANFunctional.hpp"
#include "core/constants.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
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
        case XCType::MGGA_SCAN:  xc_id = XC_LDA_X; cc_id = XC_LDA_C_PW; break; // placeholder — mGGA uses separate path
        default:                 xc_id = XC_LDA_X; cc_id = XC_LDA_C_PW;     break;
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

        // 2. sigma = |grad rho|^2
        bool is_orth = grid_->lattice().is_orthogonal();
        const Mat3& lapcT = grid_->lattice().lapc_T();

        std::vector<double> sigma(Nd_d);
        if (is_orth) {
            for (int i = 0; i < Nd_d; i++)
                sigma[i] = Drho_x[i]*Drho_x[i] + Drho_y[i]*Drho_y[i] + Drho_z[i]*Drho_z[i];
        } else {
            for (int i = 0; i < Nd_d; i++) {
                double dx = Drho_x[i], dy = Drho_y[i], dz = Drho_z[i];
                sigma[i] = lapcT(0,0)*dx*dx + lapcT(1,1)*dy*dy + lapcT(2,2)*dz*dz
                         + 2.0*lapcT(0,1)*dx*dy + 2.0*lapcT(0,2)*dx*dz + 2.0*lapcT(1,2)*dy*dz;
            }
        }

        // 3. Call hand-coded SCAN exchange and correlation
        std::vector<double> ex(Nd_d), vx1(Nd_d), vx2(Nd_d), vx3(Nd_d);
        std::vector<double> ec(Nd_d), vc1(Nd_d), vc2(Nd_d), vc3(Nd_d);
        scan::scanx(Nd_d, rho, sigma.data(), tau_in, ex.data(), vx1.data(), vx2.data(), vx3.data());
        scan::scanc(Nd_d, rho, sigma.data(), tau_in, ec.data(), vc1.data(), vc2.data(), vc3.data());

        // 4. Combine outputs
        std::vector<double> v2xc(Nd_d);
        for (int i = 0; i < Nd_d; i++) {
            exc[i] = ex[i] + ec[i];
            Vxc[i] = vx1[i] + vc1[i];
            v2xc[i] = vx2[i] + vc2[i];  // SPARC's v2 matches GGA convention: 2*vsigma
        }

        // 5. Output Dxcdgrho
        if (Dxcdgrho_out)
            std::memcpy(Dxcdgrho_out, v2xc.data(), Nd_d * sizeof(double));

        // 6. GGA divergence correction to Vxc (same as GGA)
        std::vector<double> fx(Nd_d), fy(Nd_d), fz(Nd_d);
        if (is_orth) {
            for (int i = 0; i < Nd_d; i++) {
                fx[i] = Drho_x[i] * v2xc[i];
                fy[i] = Drho_y[i] * v2xc[i];
                fz[i] = Drho_z[i] * v2xc[i];
            }
        } else {
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
        for (int i = 0; i < Nd_d; i++) Vxc[i] -= DDf[i];

        halo_->execute(fy.data(), f_ex.data(), 1);
        gradient_->apply(f_ex.data(), DDf.data(), 1, 1);
        for (int i = 0; i < Nd_d; i++) Vxc[i] -= DDf[i];

        halo_->execute(fz.data(), f_ex.data(), 1);
        gradient_->apply(f_ex.data(), DDf.data(), 2, 1);
        for (int i = 0; i < Nd_d; i++) Vxc[i] -= DDf[i];

        // 7. Output vtau
        if (vtau_out) {
            for (int i = 0; i < Nd_d; i++)
                vtau_out[i] = vx3[i] + vc3[i];
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
            for (int i = 0; i < Nd_d; i++)
                sigma[i] = Drho_x[i]*Drho_x[i] + Drho_y[i]*Drho_y[i] + Drho_z[i]*Drho_z[i];
        } else {
            for (int i = 0; i < Nd_d; i++) {
                double dx = Drho_x[i], dy = Drho_y[i], dz = Drho_z[i];
                sigma[i] = lapcT(0,0)*dx*dx + lapcT(1,1)*dy*dy + lapcT(2,2)*dz*dz
                         + 2.0*lapcT(0,1)*dx*dy + 2.0*lapcT(0,2)*dx*dz + 2.0*lapcT(1,2)*dy*dz;
            }
        }

        // Evaluate exchange and correlation via libxc
        std::vector<double> zk_x(Nd_d, 0.0), vrho_x(Nd_d, 0.0), vsigma_x(Nd_d, 0.0);
        std::vector<double> zk_c(Nd_d, 0.0), vrho_c(Nd_d, 0.0), vsigma_c(Nd_d, 0.0);

        xc_gga_exc_vxc(&func_x, np, rho, sigma.data(), zk_x.data(), vrho_x.data(), vsigma_x.data());
        xc_gga_exc_vxc(&func_c, np, rho, sigma.data(), zk_c.data(), vrho_c.data(), vsigma_c.data());

        // Combine: Dxcdgrho = 2*vsigma (our convention for -div(Dxcdgrho * ∇ρ))
        std::vector<double> v2xc(Nd_d);
        for (int i = 0; i < Nd_d; i++) {
            exc[i] = zk_x[i] + zk_c[i];
            Vxc[i] = vrho_x[i] + vrho_c[i];
            v2xc[i] = 2.0 * (vsigma_x[i] + vsigma_c[i]);
        }

        if (Dxcdgrho_out)
            std::memcpy(Dxcdgrho_out, v2xc.data(), Nd_d * sizeof(double));

        // GGA divergence correction: Vxc += -div(v2xc * lapcT * ∇ρ)
        std::vector<double> fx(Nd_d), fy(Nd_d), fz(Nd_d);
        if (is_orth) {
            for (int i = 0; i < Nd_d; i++) {
                fx[i] = Drho_x[i] * v2xc[i];
                fy[i] = Drho_y[i] * v2xc[i];
                fz[i] = Drho_z[i] * v2xc[i];
            }
        } else {
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
        for (int i = 0; i < Nd_d; i++) Vxc[i] -= DDf[i];

        halo_->execute(fy.data(), f_ex.data(), 1);
        gradient_->apply(f_ex.data(), DDf.data(), 1, 1);
        for (int i = 0; i < Nd_d; i++) Vxc[i] -= DDf[i];

        halo_->execute(fz.data(), f_ex.data(), 1);
        gradient_->apply(f_ex.data(), DDf.data(), 2, 1);
        for (int i = 0; i < Nd_d; i++) Vxc[i] -= DDf[i];

    } else {
        // LDA (or GGA fallback when no gradient available)
        // For GGA without gradient, fall back to LDA_X + LDA_C_PW
        if (is_gga()) {
            xc_func_end(&func_x);
            xc_func_end(&func_c);
            xc_func_init(&func_x, XC_LDA_X, XC_UNPOLARIZED);
            xc_func_init(&func_c, XC_LDA_C_PW, XC_UNPOLARIZED);
        }

        std::vector<double> zk_x(Nd_d, 0.0), vrho_x(Nd_d, 0.0);
        std::vector<double> zk_c(Nd_d, 0.0), vrho_c(Nd_d, 0.0);

        xc_lda_exc_vxc(&func_x, np, rho, zk_x.data(), vrho_x.data());
        xc_lda_exc_vxc(&func_c, np, rho, zk_c.data(), vrho_c.data());

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
        for (int i = 0; i < Nd_d; i++) {
            sigma[i] = dot_metric(i, i);             // |grad rho_total|^2
            int iu = Nd_d + i, id = 2*Nd_d + i;
            sigma[Nd_d + i] = dot_metric(iu, iu);    // |grad rho_up|^2
            sigma[2*Nd_d + i] = dot_metric(id, id);  // |grad rho_dn|^2
        }

        // Call hand-coded SCAN exchange (spin)
        std::vector<double> ex(Nd_d), vx1(2*Nd_d), vx2(2*Nd_d), vx3(2*Nd_d);
        scan::scanx_spin(Nd_d, rho, sigma.data(), tau_in,
                         ex.data(), vx1.data(), vx2.data(), vx3.data());

        // Call hand-coded SCAN correlation (spin)
        std::vector<double> ec(Nd_d), vc1(2*Nd_d), vc2(Nd_d), vc3(Nd_d);
        scan::scanc_spin(Nd_d, rho, sigma.data(), tau_in,
                         ec.data(), vc1.data(), vc2.data(), vc3.data());

        // Combine: exc, Vxc (per-spin)
        for (int i = 0; i < Nd_d; i++) {
            exc[i] = ex[i] + ec[i];
            Vxc[i]        = vx1[i] + vc1[i];         // up
            Vxc[Nd_d + i] = vx1[Nd_d + i] + vc1[Nd_d + i]; // down
        }

        // Dxcdgrho: [vc2(Nd_d) | vx2_up(Nd_d) | vx2_dn(Nd_d)]
        if (Dxcdgrho_out) {
            for (int i = 0; i < Nd_d; i++) {
                Dxcdgrho_out[i] = vc2[i];
                Dxcdgrho_out[Nd_d + i] = vx2[i];
                Dxcdgrho_out[2*Nd_d + i] = vx2[Nd_d + i];
            }
        }

        // vtau: [up(Nd_d) | dn(Nd_d)]
        // vx3 is per-spin (2*Nd_d), vc3 is per-total (Nd_d)
        if (vtau_out) {
            for (int i = 0; i < Nd_d; i++) {
                vtau_out[i]        = vx3[i] + vc3[i];         // up
                vtau_out[Nd_d + i] = vx3[Nd_d + i] + vc3[i];  // dn (note: vc3[i] not vc3[Nd_d+i])
            }
        }

        // GGA divergence correction for spin
        // For exchange: v2x[i] is for up, v2x[Nd_d+i] is for dn (each w.r.t. own gradient)
        // For correlation: v2c[i] is for total (w.r.t. total gradient)
        // The divergence applies to per-spin potentials
        // Up: Vxc_up -= div(vx2_up * lapcT * grad_rho_up) + div(vc2 * lapcT * grad_rho_total) [correlation uses total gradient -> applies equally to up and dn]
        // Actually SPARC convention: vc2 = d(n*eps_c)/d|grad n_total| / |grad n_total|
        // The divergence term for correlation is the same for both spins since it uses total gradient

        std::vector<double> fx_up(Nd_d), fy_up(Nd_d), fz_up(Nd_d);
        std::vector<double> fx_dn(Nd_d), fy_dn(Nd_d), fz_dn(Nd_d);

        for (int i = 0; i < Nd_d; i++) {
            int iu = Nd_d + i, id = 2*Nd_d + i;

            // Exchange contribution for up (uses grad_rho_up)
            double ex_up_x, ex_up_y, ex_up_z;
            double ex_dn_x, ex_dn_y, ex_dn_z;
            // Correlation contribution (uses grad_rho_total)
            double ec_x, ec_y, ec_z;

            if (is_orth) {
                ex_up_x = vx2[i] * Drho_x[iu];
                ex_up_y = vx2[i] * Drho_y[iu];
                ex_up_z = vx2[i] * Drho_z[iu];
                ex_dn_x = vx2[Nd_d+i] * Drho_x[id];
                ex_dn_y = vx2[Nd_d+i] * Drho_y[id];
                ex_dn_z = vx2[Nd_d+i] * Drho_z[id];
                ec_x = vc2[i] * Drho_x[i];
                ec_y = vc2[i] * Drho_y[i];
                ec_z = vc2[i] * Drho_z[i];
                fx_up[i] = ex_up_x + ec_x;
                fy_up[i] = ex_up_y + ec_y;
                fz_up[i] = ex_up_z + ec_z;
                fx_dn[i] = ex_dn_x + ec_x;
                fy_dn[i] = ex_dn_y + ec_y;
                fz_dn[i] = ex_dn_z + ec_z;
            } else {
                double du_x = Drho_x[iu], du_y = Drho_y[iu], du_z = Drho_z[iu];
                double dd_x = Drho_x[id], dd_y = Drho_y[id], dd_z = Drho_z[id];
                double dt_x = Drho_x[i], dt_y = Drho_y[i], dt_z = Drho_z[i];

                // Apply metric tensor and multiply by v2
                auto metric_mul = [&](double gx, double gy, double gz, double v2,
                                      double& out_x, double& out_y, double& out_z) {
                    out_x = (lapcT(0,0)*gx + lapcT(0,1)*gy + lapcT(0,2)*gz) * v2;
                    out_y = (lapcT(1,0)*gx + lapcT(1,1)*gy + lapcT(1,2)*gz) * v2;
                    out_z = (lapcT(2,0)*gx + lapcT(2,1)*gy + lapcT(2,2)*gz) * v2;
                };

                double fxu_x, fxu_y, fxu_z, fxd_x, fxd_y, fxd_z, fc_x, fc_y, fc_z;
                metric_mul(du_x, du_y, du_z, vx2[i], fxu_x, fxu_y, fxu_z);
                metric_mul(dd_x, dd_y, dd_z, vx2[Nd_d+i], fxd_x, fxd_y, fxd_z);
                metric_mul(dt_x, dt_y, dt_z, vc2[i], fc_x, fc_y, fc_z);

                fx_up[i] = fxu_x + fc_x; fy_up[i] = fxu_y + fc_y; fz_up[i] = fxu_z + fc_z;
                fx_dn[i] = fxd_x + fc_x; fy_dn[i] = fxd_y + fc_y; fz_dn[i] = fxd_z + fc_z;
            }
        }

        std::vector<double> f_ex(nd_ex), DDf(Nd_d);
        auto apply_div = [&](double* fx, double* fy, double* fz, double* Vxc_s) {
            halo_->execute(fx, f_ex.data(), 1);
            gradient_->apply(f_ex.data(), DDf.data(), 0, 1);
            for (int i = 0; i < Nd_d; i++) Vxc_s[i] -= DDf[i];
            halo_->execute(fy, f_ex.data(), 1);
            gradient_->apply(f_ex.data(), DDf.data(), 1, 1);
            for (int i = 0; i < Nd_d; i++) Vxc_s[i] -= DDf[i];
            halo_->execute(fz, f_ex.data(), 1);
            gradient_->apply(f_ex.data(), DDf.data(), 2, 1);
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

        // Evaluate via libxc
        std::vector<double> zk_x(Nd_d, 0.0), vrho_x(2*Nd_d, 0.0), vsigma_x(3*Nd_d, 0.0);
        std::vector<double> zk_c(Nd_d, 0.0), vrho_c(2*Nd_d, 0.0), vsigma_c(3*Nd_d, 0.0);

        xc_gga_exc_vxc(&func_x, np, rho_libxc.data(), sigma_libxc.data(),
                        zk_x.data(), vrho_x.data(), vsigma_x.data());
        xc_gga_exc_vxc(&func_c, np, rho_libxc.data(), sigma_libxc.data(),
                        zk_c.data(), vrho_c.data(), vsigma_c.data());

        // Combine local part (libxc vrho interleaved → our [up|down])
        for (int i = 0; i < Nd_d; i++) {
            exc[i] = zk_x[i] + zk_c[i];
            Vxc[i]        = vrho_x[2*i] + vrho_c[2*i];
            Vxc[Nd_d + i] = vrho_x[2*i + 1] + vrho_c[2*i + 1];
        }

        // GGA divergence: Vxc_s -= ∇·(∂f/∂(∇ρ_s))
        // ∂f/∂(∇ρ_up)   = 2*vsigma_uu*∇ρ_up + vsigma_ud*∇ρ_down
        // ∂f/∂(∇ρ_down) = vsigma_ud*∇ρ_up   + 2*vsigma_dd*∇ρ_down
        std::vector<double> fx_up(Nd_d), fy_up(Nd_d), fz_up(Nd_d);
        std::vector<double> fx_dn(Nd_d), fy_dn(Nd_d), fz_dn(Nd_d);

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
            for (int i = 0; i < Nd_d; i++) Vxc_s[i] -= DDf[i];
            halo_->execute(fy, f_ex.data(), 1);
            gradient_->apply(f_ex.data(), DDf.data(), 1, 1);
            for (int i = 0; i < Nd_d; i++) Vxc_s[i] -= DDf[i];
            halo_->execute(fz, f_ex.data(), 1);
            gradient_->apply(f_ex.data(), DDf.data(), 2, 1);
            for (int i = 0; i < Nd_d; i++) Vxc_s[i] -= DDf[i];
        };

        apply_div(fx_up.data(), fy_up.data(), fz_up.data(), Vxc);
        apply_div(fx_dn.data(), fy_dn.data(), fz_dn.data(), Vxc + Nd_d);

        // Dxcdgrho in our convention: [v2c | v2x_up | v2x_down]
        if (Dxcdgrho_out) {
            for (int i = 0; i < Nd_d; i++) {
                Dxcdgrho_out[i] = 2.0 * (vsigma_c[3*i] + 2.0*vsigma_c[3*i+1] + vsigma_c[3*i+2]);
                Dxcdgrho_out[Nd_d + i] = 2.0 * vsigma_x[3*i];
                Dxcdgrho_out[2*Nd_d + i] = 2.0 * vsigma_x[3*i + 2];
            }
        }

    } else {
        // LDA spin (or GGA fallback)
        if (is_gga()) {
            xc_func_end(&func_x);
            xc_func_end(&func_c);
            xc_func_init(&func_x, XC_LDA_X, XC_POLARIZED);
            xc_func_init(&func_c, XC_LDA_C_PW, XC_POLARIZED);
        }

        std::vector<double> zk_x(Nd_d, 0.0), vrho_x(2*Nd_d, 0.0);
        std::vector<double> zk_c(Nd_d, 0.0), vrho_c(2*Nd_d, 0.0);

        xc_lda_exc_vxc(&func_x, np, rho_libxc.data(), zk_x.data(), vrho_x.data());
        xc_lda_exc_vxc(&func_c, np, rho_libxc.data(), zk_c.data(), vrho_c.data());

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
