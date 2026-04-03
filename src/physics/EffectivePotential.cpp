#include "physics/EffectivePotential.hpp"
#include "core/constants.hpp"
#include <cmath>
#include <cstring>
#include <vector>

namespace lynx {

static bool is_mgga_type(XCType t) {
    return t == XCType::MGGA_SCAN || t == XCType::MGGA_RSCAN || t == XCType::MGGA_R2SCAN;
}

void VeffArrays::allocate(int Nd_d, int Nspin, XCType xc_type, bool is_soc) {
    Veff = NDArray<double>(Nd_d * Nspin);
    Vxc = NDArray<double>(Nd_d * Nspin);
    exc = NDArray<double>(Nd_d);
    phi = NDArray<double>(Nd_d);

    if (is_soc) {
        Veff_spinor = NDArray<double>(4 * Nd_d);
    }

    if (is_mgga_type(xc_type)) {
        int vtau_size = (Nspin == 2) ? 2 * Nd_d : Nd_d;
        vtau = NDArray<double>(vtau_size);
    }
}

void EffectivePotential::setup(const LynxContext& ctx, const Hamiltonian& hamiltonian) {
    domain_ = &ctx.domain();
    grid_ = &ctx.grid();
    stencil_ = &ctx.stencil();
    laplacian_ = &ctx.laplacian();
    gradient_ = &ctx.gradient();
    hamiltonian_ = &hamiltonian;
    halo_ = &ctx.halo();
    Nspin_global_ = ctx.Nspin();
}

void EffectivePotential::solve_poisson(const double* rho, const double* rho_b,
                                        int Nd_d, double poisson_tol, double* phi) {
    std::vector<double> rhs(Nd_d);
    for (int i = 0; i < Nd_d; ++i) {
        rhs[i] = 4.0 * constants::PI * rho[i];
        if (rho_b) rhs[i] += 4.0 * constants::PI * rho_b[i];
    }

    // Ensure Poisson RHS integrates to zero (required for periodic BC)
    {
        double rhs_sum = 0;
        for (int i = 0; i < Nd_d; ++i) rhs_sum += rhs[i];
        double rhs_mean = rhs_sum / grid_->Nd();
        for (int i = 0; i < Nd_d; ++i) rhs[i] -= rhs_mean;
    }

    PoissonSolver poisson;
    poisson.setup(*laplacian_, *stencil_, *domain_, *grid_, *halo_);
    poisson.solve(rhs.data(), phi, poisson_tol);

    // Shift phi so that its integral is zero (periodic gauge)
    {
        double sum = 0.0;
        for (int i = 0; i < Nd_d; ++i) sum += phi[i];
        double mean = sum / grid_->Nd();
        for (int i = 0; i < Nd_d; ++i) phi[i] -= mean;
    }
}

void EffectivePotential::compute(const ElectronDensity& density,
                                  const double* rho_b,
                                  const double* rho_core,
                                  XCType xc_type,
                                  double exx_frac_scale,
                                  double poisson_tol,
                                  VeffArrays& arrays,
                                  const double* tau,
                                  bool tau_valid) {
    int Nd_d = domain_->Nd_d();
    int Nspin = Nspin_global_;

    // 1. XC potential and energy density
    XCFunctional xc;
    xc.setup(xc_type, *domain_, *grid_, gradient_, halo_);

    // For hybrid functionals in Fock loop: scale exchange by (1-exx_frac)
    if (exx_frac_scale > 0.0) {
        xc.set_exchange_scale(1.0 - exx_frac_scale);
    }

    // Allocate Dxcdgrho_ if GGA or mGGA
    int dxc_ncol = (Nspin == 2) ? 3 : 1;
    if ((xc.is_gga() || xc.is_mgga()) && arrays.Dxcdgrho.size() == 0) {
        arrays.Dxcdgrho = NDArray<double>(Nd_d * dxc_ncol);
    }
    double* dxc_ptr = (xc.is_gga() || xc.is_mgga()) ? arrays.Dxcdgrho.data() : nullptr;

    // For the first Veff (before tau is computed from real wavefunctions),
    // SPARC uses PBE instead of SCAN (exchangeCorrelation.c line 64-68).
    if (xc.is_mgga() && !tau_valid) {
        xc.setup(XCType::GGA_PBE, *domain_, *grid_, gradient_, halo_);
    }
    const double* tau_ptr = (xc.is_mgga() && tau_valid) ? tau : nullptr;
    double* vtau_ptr = (xc.is_mgga() && tau_valid) ? arrays.vtau.data() : nullptr;

    if (Nspin == 2) {
        // Spin-polarized: build rho_xc array [total | up | down] (3*Nd_d)
        const double* rho_up = density.rho(0).data();
        const double* rho_dn = density.rho(1).data();

        std::vector<double> rho_xc(3 * Nd_d);
        constexpr double xc_rhotol = 1e-14;
        for (int i = 0; i < Nd_d; ++i) {
            double rt = rho_up[i] + rho_dn[i];
            if (rho_core) rt += rho_core[i];
            if (rt < xc_rhotol) rt = xc_rhotol;
            rho_xc[i] = rt;

            double ru = rho_up[i];
            double rd = rho_dn[i];
            if (rho_core) {
                ru += 0.5 * rho_core[i];
                rd += 0.5 * rho_core[i];
            }
            if (ru < xc_rhotol * 0.5) ru = xc_rhotol * 0.5;
            if (rd < xc_rhotol * 0.5) rd = xc_rhotol * 0.5;
            rho_xc[Nd_d + i] = ru;
            rho_xc[2*Nd_d + i] = rd;
        }
        xc.evaluate_spin(rho_xc.data(), arrays.Vxc.data(), arrays.exc.data(), Nd_d, dxc_ptr, tau_ptr, vtau_ptr);
    } else {
        // Non-spin-polarized
        const double* rho = density.rho_total().data();
        if (rho_core) {
            std::vector<double> rho_xc(Nd_d);
            constexpr double xc_rhotol = 1e-14;
            for (int i = 0; i < Nd_d; ++i) {
                rho_xc[i] = rho[i] + rho_core[i];
                if (rho_xc[i] < xc_rhotol) rho_xc[i] = xc_rhotol;
            }
            xc.evaluate(rho_xc.data(), arrays.Vxc.data(), arrays.exc.data(), Nd_d, dxc_ptr, tau_ptr, vtau_ptr);
        } else {
            xc.evaluate(rho, arrays.Vxc.data(), arrays.exc.data(), Nd_d, dxc_ptr, tau_ptr, vtau_ptr);
        }
    }

    // 2. Solve Poisson equation
    solve_poisson(density.rho_total().data(), rho_b, Nd_d, poisson_tol, arrays.phi.data());

    // 3. Veff = Vxc + phi (per spin channel)
    const double* phi = arrays.phi.data();
    for (int s = 0; s < Nspin; ++s) {
        double* Veff = arrays.Veff.data() + s * Nd_d;
        const double* vxc = arrays.Vxc.data() + s * Nd_d;
        for (int i = 0; i < Nd_d; ++i) {
            Veff[i] = vxc[i] + phi[i];
        }
    }

    // 4. Set vtau on Hamiltonian for mGGA (only after tau has been computed)
    if (is_mgga_type(xc_type) && tau_valid) {
        const_cast<Hamiltonian*>(hamiltonian_)->set_vtau(arrays.vtau.data());
    } else if (is_mgga_type(xc_type)) {
        const_cast<Hamiltonian*>(hamiltonian_)->set_vtau(nullptr);
    }
}

void EffectivePotential::compute_spinor(const ElectronDensity& density,
                                         const double* rho_b,
                                         const double* rho_core,
                                         XCType xc_type,
                                         double poisson_tol,
                                         VeffArrays& arrays) {
    int Nd_d = domain_->Nd_d();

    // Convert noncollinear (rho, mx, my, mz) to (rho_up_xc, rho_dn_xc) for XC
    const double* rho = density.rho_total().data();
    const double* mx = density.mag_x().data();
    const double* my = density.mag_y().data();
    const double* mz = density.mag_z().data();

    // Build XC input: rho_xc = [rho_total | rho_up | rho_down] (3*Nd_d)
    std::vector<double> rho_xc(3 * Nd_d);
    std::vector<double> m_mag(Nd_d);
    constexpr double xc_rhotol = 1e-14;

    for (int i = 0; i < Nd_d; ++i) {
        double mm = std::sqrt(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]);
        m_mag[i] = mm;
        double rt = rho[i];
        if (rho_core) rt += rho_core[i];
        if (rt < xc_rhotol) rt = xc_rhotol;
        rho_xc[i] = rt;

        double ru = 0.5 * (rt + mm);
        double rd = 0.5 * (rt - mm);
        if (ru < xc_rhotol * 0.5) ru = xc_rhotol * 0.5;
        if (rd < xc_rhotol * 0.5) rd = xc_rhotol * 0.5;
        rho_xc[Nd_d + i] = ru;
        rho_xc[2*Nd_d + i] = rd;
    }

    // Evaluate spin-polarized XC
    NDArray<double> Vxc_spin(Nd_d * 2);
    XCFunctional xc;
    xc.setup(xc_type, *domain_, *grid_, gradient_, halo_);

    int dxc_ncol = xc.is_gga() ? 3 : 0;
    if (xc.is_gga() && arrays.Dxcdgrho.size() == 0) {
        arrays.Dxcdgrho = NDArray<double>(Nd_d * dxc_ncol);
    }
    double* dxc_ptr = xc.is_gga() ? arrays.Dxcdgrho.data() : nullptr;

    xc.evaluate_spin(rho_xc.data(), Vxc_spin.data(), arrays.exc.data(), Nd_d, dxc_ptr);

    // Solve Poisson
    solve_poisson(rho, rho_b, Nd_d, poisson_tol, arrays.phi.data());

    // Build spinor Veff: [V_uu | V_dd | Re(V_ud) | Im(V_ud)]
    double* V_uu = arrays.Veff_spinor.data();
    double* V_dd = arrays.Veff_spinor.data() + Nd_d;
    double* V_ud_re = arrays.Veff_spinor.data() + 2 * Nd_d;
    double* V_ud_im = arrays.Veff_spinor.data() + 3 * Nd_d;
    const double* vxc_up = Vxc_spin.data();
    const double* vxc_dn = Vxc_spin.data() + Nd_d;
    const double* phi_ptr = arrays.phi.data();

    for (int i = 0; i < Nd_d; ++i) {
        double v_avg = 0.5 * (vxc_up[i] + vxc_dn[i]);
        double v_diff = 0.5 * (vxc_up[i] - vxc_dn[i]);
        double phi_val = phi_ptr[i];

        if (m_mag[i] > xc_rhotol) {
            double inv_m = 1.0 / m_mag[i];
            V_uu[i] = v_avg + v_diff * mz[i] * inv_m + phi_val;
            V_dd[i] = v_avg - v_diff * mz[i] * inv_m + phi_val;
            V_ud_re[i] = v_diff * mx[i] * inv_m;
            V_ud_im[i] = -v_diff * my[i] * inv_m;
        } else {
            V_uu[i] = v_avg + phi_val;
            V_dd[i] = v_avg + phi_val;
            V_ud_re[i] = 0.0;
            V_ud_im[i] = 0.0;
        }
    }

    // Also store Vxc_ and Veff_ for energy calculation (using 1-component view)
    for (int i = 0; i < Nd_d; ++i) {
        arrays.Vxc.data()[i] = 0.5 * (vxc_up[i] + vxc_dn[i]);
        arrays.Veff.data()[i] = V_uu[i];
    }
}

} // namespace lynx
