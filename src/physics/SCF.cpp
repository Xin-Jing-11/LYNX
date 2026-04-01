#include "physics/SCF.hpp"
#include "xc/ExactExchange.hpp"
#include "core/constants.hpp"
#include "core/ParameterDefaults.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <mpi.h>

namespace lynx {

static bool is_mgga_type(XCType t) {
    return t == XCType::MGGA_SCAN || t == XCType::MGGA_RSCAN || t == XCType::MGGA_R2SCAN;
}

void SCF::setup(const FDGrid& grid,
                 const Domain& domain,
                 const FDStencil& stencil,
                 const Laplacian& laplacian,
                 const Gradient& gradient,
                 const Hamiltonian& hamiltonian,
                 const HaloExchange& halo,
                 const NonlocalProjector* vnl,
                 const MPIComm& bandcomm,
                 const MPIComm& kptcomm,
                 const MPIComm& spincomm,
                 const SCFParams& params,
                 int Nspin_global,
                 int Nspin_local,
                 int spin_start,
                 const KPoints* kpoints,
                 int kpt_start,
                 int Nband_global,
                 int band_start) {
    grid_ = &grid;
    domain_ = &domain;
    stencil_ = &stencil;
    laplacian_ = &laplacian;
    gradient_ = &gradient;
    hamiltonian_ = &hamiltonian;
    halo_ = &halo;
    vnl_ = vnl;
    bandcomm_ = &bandcomm;
    kptcomm_ = &kptcomm;
    spincomm_ = &spincomm;
    params_ = params;
    Nspin_global_ = Nspin_global;
    Nspin_local_ = Nspin_local;
    spin_start_ = spin_start;
    kpoints_ = kpoints;
    is_kpt_ = kpoints && !kpoints->is_gamma_only();
    kpt_start_ = kpt_start;
    Nband_global_ = Nband_global;
    band_start_ = band_start;
}

void SCF::init_density(int Nd_d, int Nelectron) {
    density_.allocate(Nd_d, Nspin_global_);
    // Uniform initial density: rho = Nelectron / Volume
    double volume = grid_->Nd() * grid_->dV();
    double rho0 = Nelectron / volume;

    if (Nspin_global_ == 1) {
        double* rho = density_.rho(0).data();
        for (int i = 0; i < Nd_d; ++i) rho[i] = rho0;
    } else {
        // Split equally between spins
        double* rho_up = density_.rho(0).data();
        double* rho_dn = density_.rho(1).data();
        for (int i = 0; i < Nd_d; ++i) {
            rho_up[i] = rho0 * 0.5;
            rho_dn[i] = rho0 * 0.5;
        }
    }

    // Update total
    double* rho_t = density_.rho_total().data();
    for (int i = 0; i < Nd_d; ++i) rho_t[i] = rho0;
}

void SCF::set_initial_density(const double* rho_init, int Nd_d,
                               const double* mag_init) {
    density_.allocate(Nd_d, Nspin_global_);
    std::memcpy(density_.rho_total().data(), rho_init, Nd_d * sizeof(double));

    if (Nspin_global_ == 1) {
        std::memcpy(density_.rho(0).data(), rho_init, Nd_d * sizeof(double));
    } else {
        double* rho_up = density_.rho(0).data();
        double* rho_dn = density_.rho(1).data();
        if (mag_init) {
            // rho_up = (rho_total + mag) / 2, rho_dn = (rho_total - mag) / 2
            for (int i = 0; i < Nd_d; ++i) {
                rho_up[i] = 0.5 * (rho_init[i] + mag_init[i]);
                rho_dn[i] = 0.5 * (rho_init[i] - mag_init[i]);
                if (rho_up[i] < 0.0) rho_up[i] = 0.0;
                if (rho_dn[i] < 0.0) rho_dn[i] = 0.0;
            }
        } else {
            // Equal split
            for (int i = 0; i < Nd_d; ++i) {
                rho_up[i] = 0.5 * rho_init[i];
                rho_dn[i] = 0.5 * rho_init[i];
            }
        }
    }
}

void SCF::compute_tau(const Wavefunction& wfn,
                       const std::vector<double>& kpt_weights,
                       int kpt_start, int band_start) {
    int Nd_d = domain_->Nd_d();
    int Nband_loc = wfn.Nband();
    int Nspin_local = wfn.Nspin();
    int Nkpts = wfn.Nkpts();

    // Zero tau
    std::memset(tau_.data(), 0, tau_.size() * sizeof(double));

    // NOTE: Unlike density which uses spin_fac (2 for non-spin, 1 for spin),
    // tau uses g_nk = occ[n] (no spin_fac) in the accumulation loop.
    // A separate factor of 0.5 is applied to spin-polarized tau AFTER accumulation
    // (see below) to get the physical KED: τ_σ = (1/2) Σ f_nσ |∇ψ_nσ|².

    // Gradient operator and halo exchange setup
    int FDn = gradient_->stencil().FDn();
    int nx = domain_->Nx_d(), ny = domain_->Ny_d(), nz = domain_->Nz_d();
    int nd_ex = halo_->nx_ex() * halo_->ny_ex() * halo_->nz_ex();
    bool is_orth = grid_->lattice().is_orthogonal();
    const Mat3& lapcT = grid_->lattice().lapc_T();
    Vec3 cell_lengths = grid_->lattice().lengths();

    for (int s = 0; s < Nspin_local; ++s) {
        int s_glob = spin_start_ + s;
        double* tau_s = tau_.data() + s_glob * Nd_d;

        for (int k = 0; k < Nkpts; ++k) {
            const auto& occ = wfn.occupations(s, k);
            double wk = kpt_weights[kpt_start + k];

            if (wfn.is_complex()) {
                // k-point (complex) path
                const auto& psi_c = wfn.psi_kpt(s, k);
                std::vector<Complex> psi_ex(nd_ex);
                std::vector<Complex> dpsi_x(Nd_d), dpsi_y(Nd_d), dpsi_z(Nd_d);
                Vec3 kpt = kpoints_->kpts_cart()[kpt_start + k];

                for (int n = 0; n < Nband_loc; ++n) {
                    double fn = occ(band_start + n);
                    if (fn < 1e-16) continue;
                    double g_nk = wk * fn;

                    const Complex* col = psi_c.col(n);
                    halo_->execute_kpt(col, psi_ex.data(), 1, kpt, cell_lengths);

                    gradient_->apply(psi_ex.data(), dpsi_x.data(), 0, 1);
                    gradient_->apply(psi_ex.data(), dpsi_y.data(), 1, 1);
                    gradient_->apply(psi_ex.data(), dpsi_z.data(), 2, 1);

                    if (is_orth) {
                        for (int i = 0; i < Nd_d; ++i) {
                            tau_s[i] += g_nk * (std::norm(dpsi_x[i]) + std::norm(dpsi_y[i]) + std::norm(dpsi_z[i]));
                        }
                    } else {
                        for (int i = 0; i < Nd_d; ++i) {
                            Complex dx = dpsi_x[i], dy = dpsi_y[i], dz = dpsi_z[i];
                            // |∇ψ|² with metric tensor: Re(conj(∇ψ) · lapcT · ∇ψ)
                            double val = lapcT(0,0) * std::norm(dx) + lapcT(1,1) * std::norm(dy) + lapcT(2,2) * std::norm(dz)
                                       + 2.0 * lapcT(0,1) * (std::conj(dx) * dy).real()
                                       + 2.0 * lapcT(0,2) * (std::conj(dx) * dz).real()
                                       + 2.0 * lapcT(1,2) * (std::conj(dy) * dz).real();
                            tau_s[i] += g_nk * val;
                        }
                    }
                }
            } else {
                // Gamma-point (real) path
                const auto& psi = wfn.psi(s, k);
                std::vector<double> psi_ex(nd_ex);
                std::vector<double> dpsi_x(Nd_d), dpsi_y(Nd_d), dpsi_z(Nd_d);

                for (int n = 0; n < Nband_loc; ++n) {
                    double fn = occ(band_start + n);
                    if (fn < 1e-16) continue;
                    double g_nk = wk * fn;

                    const double* col = psi.col(n);
                    halo_->execute(col, psi_ex.data(), 1);

                    gradient_->apply(psi_ex.data(), dpsi_x.data(), 0, 1);
                    gradient_->apply(psi_ex.data(), dpsi_y.data(), 1, 1);
                    gradient_->apply(psi_ex.data(), dpsi_z.data(), 2, 1);

                    if (is_orth) {
                        for (int i = 0; i < Nd_d; ++i) {
                            tau_s[i] += g_nk * (dpsi_x[i]*dpsi_x[i] + dpsi_y[i]*dpsi_y[i] + dpsi_z[i]*dpsi_z[i]);
                        }
                    } else {
                        for (int i = 0; i < Nd_d; ++i) {
                            double dx = dpsi_x[i], dy = dpsi_y[i], dz = dpsi_z[i];
                            double val = lapcT(0,0)*dx*dx + lapcT(1,1)*dy*dy + lapcT(2,2)*dz*dz
                                       + 2.0*lapcT(0,1)*dx*dy + 2.0*lapcT(0,2)*dx*dz + 2.0*lapcT(1,2)*dy*dz;
                            tau_s[i] += g_nk * val;
                        }
                    }
                }
            }
        }
    }

    // Allreduce over band communicator
    if (!bandcomm_->is_null() && bandcomm_->size() > 1) {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start_ + s;
            bandcomm_->allreduce_sum(tau_.data() + s_glob * Nd_d, Nd_d);
        }
    }

    // Allreduce over kpt communicator
    if (!kptcomm_->is_null() && kptcomm_->size() > 1) {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start_ + s;
            kptcomm_->allreduce_sum(tau_.data() + s_glob * Nd_d, Nd_d);
        }
    }

    // Exchange spin channels across spin communicator
    if (spincomm_ && !spincomm_->is_null() && spincomm_->size() > 1 && Nspin_global_ == 2) {
        int my_spin = spin_start_;
        int other_spin = 1 - my_spin;
        int partner = (spincomm_->rank() == 0) ? 1 : 0;
        MPI_Sendrecv(tau_.data() + my_spin * Nd_d, Nd_d, MPI_DOUBLE, partner, 0,
                     tau_.data() + other_spin * Nd_d, Nd_d, MPI_DOUBLE, partner, 0,
                     spincomm_->comm(), MPI_STATUS_IGNORE);
    }

    // NOTE: No dV division needed. LYNX wavefunctions satisfy <ψ_m|ψ_n> = Σ_i ψ*_m(i) ψ_n(i) dV = δ_mn.
    // Density is computed as ρ[i] = spin_fac * Σ_n g_nk |ψ_n(i)|² (no 1/dV).
    // Tau follows a DIFFERENT convention from density: τ_σ = (1/2) Σ_n f_n |∇ψ_n|²
    // For non-spin: τ = Σ_n f_n |∇ψ_n|² (the 1/2 and spin-degeneracy factor of 2 cancel)
    // For spin-polarized: τ_σ = (1/2) Σ_n f_nσ |∇ψ_nσ|² — need explicit 0.5 factor
    // This ensures τ_total = τ_up + τ_dn = Σ_n f_n |∇ψ_n|² = τ_nonspin
    // Reference: SPARC mGGAtauTransferTauVxc.c line 186: vscal *= 0.5 for spin

    // For spin-polarized: apply 0.5 factor then compute total = up + dn
    if (Nspin_global_ == 2) {
        double* tau_up = tau_.data();
        double* tau_dn = tau_.data() + Nd_d;
        double* tau_tot = tau_.data() + 2 * Nd_d;
        for (int i = 0; i < Nd_d; ++i) {
            tau_up[i] *= 0.5;
            tau_dn[i] *= 0.5;
            tau_tot[i] = tau_up[i] + tau_dn[i];
        }
    }
}

void SCF::compute_Veff(const double* rho, const double* rho_b, const double* Vloc) {
    int Nd_d = domain_->Nd_d();

    // 1. XC potential and energy density
    // If NLCC, add core density to valence density for XC evaluation
    XCFunctional xc;
    xc.setup(xc_type_, *domain_, *grid_, gradient_, halo_);

    // For hybrid functionals in Fock loop: scale exchange by (1-exx_frac)
    if (in_fock_loop_ && is_hybrid(xc_type_) && exx_) {
        xc.set_exchange_scale(1.0 - params_.exx_params.exx_frac);
    }

    // Allocate Dxcdgrho_ if GGA or mGGA
    int dxc_ncol = (Nspin_global_ == 2) ? 3 : 1;  // 3 columns for spin: [v2c, v2x_up, v2x_down]
    if ((xc.is_gga() || xc.is_mgga()) && Dxcdgrho_.size() == 0) {
        Dxcdgrho_ = NDArray<double>(Nd_d * dxc_ncol);
    }
    double* dxc_ptr = (xc.is_gga() || xc.is_mgga()) ? Dxcdgrho_.data() : nullptr;
    // For the first Veff (before tau is computed from real wavefunctions),
    // SPARC uses PBE instead of SCAN (exchangeCorrelation.c line 64-68).
    // This avoids evaluating SCAN with tau=0 which produces poor potentials.
    // After the first CheFSI + tau computation, switch to SCAN.
    if (xc.is_mgga() && !tau_valid_) {
        // Use PBE for the initial Veff (matching SPARC)
        xc.setup(XCType::GGA_PBE, *domain_, *grid_, gradient_, halo_);
    }
    const double* tau_ptr = (xc.is_mgga() && tau_valid_) ? tau_.data() : nullptr;
    double* vtau_ptr = (xc.is_mgga() && tau_valid_) ? vtau_.data() : nullptr;

    if (Nspin_global_ == 2) {
        // Spin-polarized: build rho_xc array [total | up | down] (3*Nd_d)
        // rho here is the total density. We need spin-resolved from density_ object.
        const double* rho_up = density_.rho(0).data();
        const double* rho_dn = density_.rho(1).data();

        std::vector<double> rho_xc(3 * Nd_d);
        constexpr double xc_rhotol = 1e-14;
        for (int i = 0; i < Nd_d; ++i) {
            double rt = rho_up[i] + rho_dn[i];
            if (rho_core_) rt += rho_core_[i];
            if (rt < xc_rhotol) rt = xc_rhotol;
            rho_xc[i] = rt;

            double ru = rho_up[i];
            double rd = rho_dn[i];
            if (rho_core_) {
                ru += 0.5 * rho_core_[i];
                rd += 0.5 * rho_core_[i];
            }
            if (ru < xc_rhotol * 0.5) ru = xc_rhotol * 0.5;
            if (rd < xc_rhotol * 0.5) rd = xc_rhotol * 0.5;
            rho_xc[Nd_d + i] = ru;
            rho_xc[2*Nd_d + i] = rd;
        }
        xc.evaluate_spin(rho_xc.data(), Vxc_.data(), exc_.data(), Nd_d, dxc_ptr, tau_ptr, vtau_ptr);
    } else {
        // Non-spin-polarized
        if (rho_core_) {
            std::vector<double> rho_xc(Nd_d);
            constexpr double xc_rhotol = 1e-14;
            for (int i = 0; i < Nd_d; ++i) {
                rho_xc[i] = rho[i] + rho_core_[i];
                if (rho_xc[i] < xc_rhotol) rho_xc[i] = xc_rhotol;
            }
            xc.evaluate(rho_xc.data(), Vxc_.data(), exc_.data(), Nd_d, dxc_ptr, tau_ptr, vtau_ptr);
        } else {
            xc.evaluate(rho, Vxc_.data(), exc_.data(), Nd_d, dxc_ptr, tau_ptr, vtau_ptr);
        }
    }

    // 2. Solve Poisson equation: -Lap(phi) = 4*pi*(rho + b)
    // Always use total density for Poisson
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
    poisson.solve(rhs.data(), phi_.data(), params_.poisson_tol);

    // Shift phi so that its integral is zero (periodic gauge)
    {
        double sum = 0.0;
        for (int i = 0; i < Nd_d; ++i) sum += phi_.data()[i];
        double mean = sum / grid_->Nd();
        for (int i = 0; i < Nd_d; ++i) phi_.data()[i] -= mean;
    }

    // 3. Veff = Vxc + phi (per spin channel)
    // For spin: Veff[0:Nd_d] = Vxc_up + phi, Veff[Nd_d:2*Nd_d] = Vxc_down + phi
    const double* phi = phi_.data();
    for (int s = 0; s < Nspin_global_; ++s) {
        double* Veff = Veff_.data() + s * Nd_d;
        const double* vxc = Vxc_.data() + s * Nd_d;
        for (int i = 0; i < Nd_d; ++i) {
            Veff[i] = vxc[i] + phi[i];
        }
    }

    // 4. Set vtau on Hamiltonian for mGGA (only after tau has been computed)
    if (is_mgga_type(xc_type_) && tau_valid_) {
        const_cast<Hamiltonian*>(hamiltonian_)->set_vtau(vtau_.data());
    } else if (is_mgga_type(xc_type_)) {
        const_cast<Hamiltonian*>(hamiltonian_)->set_vtau(nullptr);
    }
}

void SCF::compute_Veff_spinor(const ElectronDensity& density,
                                const double* rho_b, const double* Vloc) {
    int Nd_d = domain_->Nd_d();

    // Convert noncollinear (rho, mx, my, mz) to (rho_up_xc, rho_dn_xc) for XC
    const double* rho = density.rho_total().data();
    const double* mx = density.mag_x().data();
    const double* my = density.mag_y().data();
    const double* mz = density.mag_z().data();

    // Build XC input: rho_xc = [rho_total | rho_up | rho_down] (3*Nd_d)
    // rho_up = 0.5*(rho + |m|), rho_dn = 0.5*(rho - |m|)
    std::vector<double> rho_xc(3 * Nd_d);
    std::vector<double> m_mag(Nd_d);
    constexpr double xc_rhotol = 1e-14;

    for (int i = 0; i < Nd_d; ++i) {
        double mm = std::sqrt(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]);
        m_mag[i] = mm;
        double rt = rho[i];
        if (rho_core_) rt += rho_core_[i];
        if (rt < xc_rhotol) rt = xc_rhotol;
        rho_xc[i] = rt;

        double ru = 0.5 * (rt + mm);
        double rd = 0.5 * (rt - mm);
        if (rho_core_) {
            // Already added to rt above; split NLCC equally
        }
        if (ru < xc_rhotol * 0.5) ru = xc_rhotol * 0.5;
        if (rd < xc_rhotol * 0.5) rd = xc_rhotol * 0.5;
        rho_xc[Nd_d + i] = ru;
        rho_xc[2*Nd_d + i] = rd;
    }

    // Evaluate spin-polarized XC
    // Vxc_ layout: [Vxc_up(Nd_d) | Vxc_dn(Nd_d)]
    // We need Nspin_global_=2 for XC eval, even though actual Nspin=1 for noncollinear
    NDArray<double> Vxc_spin(Nd_d * 2);
    XCFunctional xc;
    xc.setup(xc_type_, *domain_, *grid_, gradient_, halo_);

    int dxc_ncol = xc.is_gga() ? 3 : 0;
    if (xc.is_gga() && Dxcdgrho_.size() == 0) {
        Dxcdgrho_ = NDArray<double>(Nd_d * dxc_ncol);
    }
    double* dxc_ptr = xc.is_gga() ? Dxcdgrho_.data() : nullptr;

    xc.evaluate_spin(rho_xc.data(), Vxc_spin.data(), exc_.data(), Nd_d, dxc_ptr);

    // Solve Poisson: same as standard
    std::vector<double> rhs(Nd_d);
    for (int i = 0; i < Nd_d; ++i) {
        rhs[i] = 4.0 * constants::PI * rho[i];
        if (rho_b) rhs[i] += 4.0 * constants::PI * rho_b[i];
    }
    {
        double rhs_sum = 0;
        for (int i = 0; i < Nd_d; ++i) rhs_sum += rhs[i];
        double rhs_mean = rhs_sum / grid_->Nd();
        for (int i = 0; i < Nd_d; ++i) rhs[i] -= rhs_mean;
    }

    PoissonSolver poisson;
    poisson.setup(*laplacian_, *stencil_, *domain_, *grid_, *halo_);
    poisson.solve(rhs.data(), phi_.data(), params_.poisson_tol);
    {
        double sum = 0.0;
        for (int i = 0; i < Nd_d; ++i) sum += phi_.data()[i];
        double mean = sum / grid_->Nd();
        for (int i = 0; i < Nd_d; ++i) phi_.data()[i] -= mean;
    }

    // Build spinor Veff: [V_uu | V_dd | Re(V_ud) | Im(V_ud)]
    // V_avg = 0.5*(Vxc_up + Vxc_dn), V_diff = 0.5*(Vxc_up - Vxc_dn)
    // V_uu = V_avg + V_diff * mz/|m| + phi
    // V_dd = V_avg - V_diff * mz/|m| + phi
    // V_ud = V_diff * (mx - i*my) / |m|
    double* V_uu = Veff_spinor_.data();
    double* V_dd = Veff_spinor_.data() + Nd_d;
    double* V_ud_re = Veff_spinor_.data() + 2 * Nd_d;
    double* V_ud_im = Veff_spinor_.data() + 3 * Nd_d;
    const double* vxc_up = Vxc_spin.data();
    const double* vxc_dn = Vxc_spin.data() + Nd_d;
    const double* phi_ptr = phi_.data();

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
            // Zero magnetization: purely diagonal
            V_uu[i] = v_avg + phi_val;
            V_dd[i] = v_avg + phi_val;
            V_ud_re[i] = 0.0;
            V_ud_im[i] = 0.0;
        }
    }

    // Also store Vxc_ and Veff_ for energy calculation (using 1-component view)
    // Vxc_ stores the average XC potential, Veff_ stores V_uu for energy purposes
    for (int i = 0; i < Nd_d; ++i) {
        Vxc_.data()[i] = 0.5 * (vxc_up[i] + vxc_dn[i]);
        Veff_.data()[i] = V_uu[i];
    }
}

double SCF::run(Wavefunction& wfn,
                 int Nelectron,
                 int Natom,
                 const double* rho_b,
                 const double* Vloc,
                 double Eself,
                 double Ec,
                 XCType xc_type,
                 const double* rho_core) {
    int Nd_d = domain_->Nd_d();
    int Nband_loc = wfn.Nband();         // local bands on this process (psi columns)
    int Nband = wfn.Nband_global();      // global band count (eigenvalue/occupation size)
    int Nspin_local = wfn.Nspin();       // spins on this process
    int Nspin = Nspin_global_;           // global spin count (1 or 2)
    int Nkpts = wfn.Nkpts();
    int rank_world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);

    xc_type_ = xc_type;
    Vloc_ = Vloc;
    rho_core_ = rho_core;

    // Detect SOC mode (before GPU dispatch)
    is_soc_ = (wfn.Nspinor() == 2);
    if (is_soc_) is_kpt_ = true;  // spinor always complex, even at Gamma

#ifdef USE_CUDA
    // GPU dispatch: gamma-point or k-point, any Nspin, orthogonal or non-orthogonal
    // Skip GPU for SOC until eigensolver divergence is fixed
    if (gpu_enabled_ && crystal_ && nloc_influence_) {
        if (rank_world == 0)
            std::printf("GPU SCF enabled — dispatching to fully GPU-resident path (Nspin=%d, kpt=%d, soc=%d)\n",
                        Nspin_global_, is_kpt_ ? 1 : 0, is_soc_ ? 1 : 0);
        return run_gpu(wfn, Nelectron, Natom, rho_b, Eself, Ec, xc_type, rho_core);
    }
#endif

    // Auto-compute default tolerances, temperature, Chebyshev degree
    ParameterDefaults::complete_params(params_, *grid_);
    if (rank_world == 0 && params_.cheb_degree > 0) {
        double h_eff = ParameterDefaults::compute_h_eff(grid_->dx(), grid_->dy(), grid_->dz());
        std::printf("Auto Chebyshev degree: %d (h_eff=%.6f)\n", params_.cheb_degree, h_eff);
    }

    // K-point weights
    std::vector<double> kpt_weights;
    Vec3 cell_lengths = grid_->lattice().lengths();
    int Nkpts_global = kpoints_ ? kpoints_->Nkpts() : Nkpts;
    if (kpoints_) {
        kpt_weights = kpoints_->normalized_weights();
    } else {
        kpt_weights.assign(Nkpts_global, 1.0 / Nkpts_global);
    }

    // Allocate work arrays
    Veff_ = NDArray<double>(Nd_d * Nspin);
    Vxc_ = NDArray<double>(Nd_d * Nspin);
    exc_ = NDArray<double>(Nd_d);
    phi_ = NDArray<double>(Nd_d);
    if (is_soc_) {
        Veff_spinor_ = NDArray<double>(4 * Nd_d);
    }
    if (is_mgga_type(xc_type_)) {
        int tau_size = (Nspin_global_ == 2) ? 3 * Nd_d : Nd_d;
        int vtau_size = (Nspin_global_ == 2) ? 2 * Nd_d : Nd_d;
        tau_ = NDArray<double>(tau_size);
        vtau_ = NDArray<double>(vtau_size);
    }

    // Initialize density if not already set
    if (is_soc_) {
        // SOC: need noncollinear density format with mag_x/y/z
        if (density_.Nd_d() == 0) {
            // No density set at all — uniform init
            density_.allocate_noncollinear(Nd_d);
            double volume = grid_->Nd() * grid_->dV();
            double rho0 = Nelectron / volume;
            double* rho = density_.rho_total().data();
            for (int i = 0; i < Nd_d; ++i) rho[i] = rho0;
            std::memcpy(density_.rho(0).data(), rho, Nd_d * sizeof(double));
            density_.mag_x().zero();
            density_.mag_y().zero();
            density_.mag_z().zero();
        } else if (density_.mag_x().size() == 0) {
            // Density was set (e.g., via set_initial_density) but not in noncollinear format
            // Convert: keep total density, zero magnetization
            NDArray<double> rho_save = density_.rho_total().clone();
            density_.allocate_noncollinear(Nd_d);
            std::memcpy(density_.rho_total().data(), rho_save.data(), Nd_d * sizeof(double));
            std::memcpy(density_.rho(0).data(), rho_save.data(), Nd_d * sizeof(double));
            density_.mag_x().zero();
            density_.mag_y().zero();
            density_.mag_z().zero();
        }
    } else {
        if (density_.Nd_d() == 0) {
            init_density(Nd_d, Nelectron);
        }
    }

    // beta = 1/(kB*T) in atomic units
    double kBT = params_.elec_temp * constants::KB;
    double beta = 1.0 / kBT;

    // Setup mixer
    Mixer mixer;
    mixer.setup(Nd_d, params_.mixing_var, params_.mixing_precond,
                params_.mixing_history, params_.mixing_param,
                laplacian_, halo_, grid_);

    // Setup eigensolver
    EigenSolver eigsolver;
    eigsolver.setup(*hamiltonian_, *halo_, *domain_, *bandcomm_, Nband);

    // Estimate spectral bounds
    std::vector<double> eigval_min(Nspin_local, 0.0), eigval_max(Nspin_local, 0.0);

    // Initial Veff from initial density
    bool use_potential_mixing = (params_.mixing_var == MixingVariable::Potential) && !is_soc_;
    if (use_potential_mixing) {
        Veff_mean_.resize(Nspin, 0.0);
    }
    if (is_soc_) {
        compute_Veff_spinor(density_, rho_b, Vloc_);
    } else {
        compute_Veff(density_.rho_total().data(), rho_b, Vloc_);
    }
    // For potential mixing: initialize Veff_mixed_ (zero-mean copy for mixer history)
    // matching SPARC's Update_mixing_hist_xk
    NDArray<double> Veff_mixed;  // zero-mean Veff for mixer (persistent across SCF loop)
    if (use_potential_mixing) {
        Veff_mixed = NDArray<double>(Nd_d * Nspin);
        std::memcpy(Veff_mixed.data(), Veff_.data(), Nd_d * Nspin * sizeof(double));
        for (int s = 0; s < Nspin; ++s) {
            double mean = 0;
            for (int i = 0; i < Nd_d; ++i) mean += Veff_mixed.data()[s*Nd_d + i];
            mean /= grid_->Nd();
            Veff_mean_[s] = mean;
            for (int i = 0; i < Nd_d; ++i) Veff_mixed.data()[s*Nd_d + i] -= mean;
        }
    }

    // Lanczos to estimate spectrum
    if (is_soc_) {
        Vec3 kpt0 = kpoints_->kpts_cart()[kpt_start_];
        // Set k-point on nonlocal projector BEFORE Lanczos (required for SOC)
        if (vnl_ && vnl_->is_setup()) {
            const_cast<NonlocalProjector*>(vnl_)->set_kpoint(kpt0);
            const_cast<Hamiltonian*>(hamiltonian_)->set_vnl_kpt(vnl_);
        }
        // For SOC: use spinor Lanczos for eigmin (needs to capture deep core states
        // from SOC splitting at ~-2500 Ha) but scalar Lanczos for eigmax (the high-energy
        // SOC states at ~+2000 Ha are not needed). The Chebyshev filter then brackets
        // the full occupied spectrum [eigmin_spinor, eigmax_scalar].
        double eigmin_spinor, eigmax_spinor;
        eigsolver.lanczos_bounds_spinor_kpt(Veff_spinor_.data(), Nd_d,
                                             kpt0, cell_lengths,
                                             eigmin_spinor, eigmax_spinor);
        double eigmin_scalar, eigmax_scalar;
        eigsolver.lanczos_bounds_kpt(Veff_.data(), Nd_d, kpt0, cell_lengths,
                                      eigmin_scalar, eigmax_scalar);
        eigval_min[0] = eigmin_spinor;   // deep core from SOC
        eigval_max[0] = eigmax_scalar;   // kinetic energy upper bound (no SOC)
    } else {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start_ + s;
            if (is_kpt_) {
                Vec3 kpt0 = kpoints_->kpts_cart()[kpt_start_];
                eigsolver.lanczos_bounds_kpt(Veff_.data() + s_glob * Nd_d, Nd_d,
                                              kpt0, cell_lengths,
                                              eigval_min[s], eigval_max[s]);
            } else {
                eigsolver.lanczos_bounds(Veff_.data() + s_glob * Nd_d, Nd_d, eigval_min[s], eigval_max[s]);
            }
        }
    }
    double lambda_cutoff;
    if (is_soc_) {
        // For SOC: the eigmin is from deep core states (-2530 Ha) but the occupied
        // valence states are near 0 Ha. Use scalar midpoint for the cutoff so it's
        // above the Fermi level, not below.
        double eigmin_scalar, eigmax_scalar;
        Vec3 kpt0 = kpoints_->kpts_cart()[kpt_start_];
        eigsolver.lanczos_bounds_kpt(Veff_.data(), Nd_d, kpt0, cell_lengths,
                                      eigmin_scalar, eigmax_scalar);
        lambda_cutoff = 0.5 * (eigmin_scalar + eigval_max[0]);
    } else {
        lambda_cutoff = 0.5 * (eigval_min[0] + eigval_max[0]);
    }
    if (rank_world == 0) {
        for (int s = 0; s < Nspin_local; ++s)
            std::printf("Lanczos bounds (spin %d): eigmin=%.6e, eigmax=%.6e\n",
                        spin_start_ + s, eigval_min[s], eigval_max[s]);
        if (params_.cheb_degree > 0)
            std::printf("Chebyshev degree: %d\n", params_.cheb_degree);
    }

    // Randomize wavefunctions — match reference LYNX exactly
    // Reference: SetRandMat(Xorb, DMndsp, Nband, -0.5, 0.5, spincomm)
    // seed = rank_in_spincomm * 100 + 1 (serial: seed=1)
    // For gamma-point, each (spin,kpt) pair gets same seed (reference fills all at once)
    // For serial: seed = 0 * 100 + 1 = 1
    int spincomm_rank = spincomm_->is_null() ? 0 : spincomm_->rank();
    int bandcomm_rank = bandcomm_->is_null() ? 0 : bandcomm_->rank();
    for (int s = 0; s < Nspin_local; ++s) {
        // Include spin index in seed so spin-up and spin-down get different
        // random wavefunctions. This breaks spin symmetry and allows magnetic
        // ground states to be found (matches SPARC's behavior with NP_SPIN_PARAL>1
        // where different spin channels have different MPI ranks/seeds).
        int s_glob = spin_start_ + s;
        unsigned rand_seed = spincomm_rank * 100 + bandcomm_rank * 10 + s_glob * 1000 + 1;
        for (int k = 0; k < Nkpts; ++k) {
            if (is_kpt_) {
                wfn.randomize_kpt(s, k, rand_seed);
            } else {
                wfn.randomize(s, k, rand_seed);
            }
        }
    }

    double E_prev = 0.0;
    converged_ = false;
    if (Natom <= 0) Natom = std::max(1, Nelectron / 4);

    // ===== SCF Loop =====
    // Reference LYNX control flow (eigSolve_CheFSI):
    //   SCFcount=0: do rhoTrigger (default 4) CheFSI passes
    //   SCFcount>0: do Nchefsi (default 1) CheFSI passes
    // Between each CheFSI pass, occupations are computed to update lambda_cutoff.
    // After all passes for a given SCFcount, density is computed, then energy, then mixing.
    for (int scf_iter = 0; scf_iter < params_.max_iter; ++scf_iter) {
        // Number of CheFSI passes: rhoTrigger for first iter, Nchefsi for rest
        int nchefsi = (scf_iter == 0) ? params_.rho_trigger : params_.nchefsi;

        if (rank_world == 0 && scf_iter == 0 && nchefsi > 1)
            std::printf("SCF iter 1: %d CheFSI passes (rhoTrigger)\n", nchefsi);

        // 1. Solve eigenvalue problem (nchefsi passes)
        for (int chefsi_pass = 0; chefsi_pass < nchefsi; ++chefsi_pass) {
            if (is_soc_) {
                // SOC: single spin channel (s=0), spinor solve
                for (int k = 0; k < Nkpts; ++k) {
                    double* eig = wfn.eigenvalues(0, k).data();
                    Complex* psi_c = wfn.psi_kpt(0, k).data();
                    int k_glob = kpt_start_ + k;
                    Vec3 kpt = kpoints_->kpts_cart()[k_glob];

                    if (vnl_ && vnl_->is_setup()) {
                        const_cast<NonlocalProjector*>(vnl_)->set_kpoint(kpt);
                        const_cast<Hamiltonian*>(hamiltonian_)->set_vnl_kpt(vnl_);
                    }

                    eigsolver.solve_spinor_kpt(psi_c, eig, Veff_spinor_.data(),
                                                Nd_d, Nband_loc,
                                                lambda_cutoff, eigval_min[0], eigval_max[0],
                                                kpt, cell_lengths,
                                                params_.cheb_degree,
                                                wfn.psi_kpt(0, k).ld());
                }
            } else {
                // Standard: spin loop
                for (int s = 0; s < Nspin_local; ++s) {
                    int s_glob = spin_start_ + s;
                    double* Veff_s = Veff_.data() + s_glob * Nd_d;
                    // Set per-spin vtau for mGGA
                    if (is_mgga_type(xc_type_) && Nspin_global_ == 2) {
                        const_cast<Hamiltonian*>(hamiltonian_)->set_vtau(vtau_.data() + s_glob * Nd_d);
                    }
                    for (int k = 0; k < Nkpts; ++k) {
                        double* eig = wfn.eigenvalues(s, k).data();

                        if (is_kpt_) {
                            Complex* psi_c = wfn.psi_kpt(s, k).data();
                            int k_glob = kpt_start_ + k;
                            Vec3 kpt = kpoints_->kpts_cart()[k_glob];

                            if (vnl_ && vnl_->is_setup()) {
                                const_cast<NonlocalProjector*>(vnl_)->set_kpoint(kpt);
                                const_cast<Hamiltonian*>(hamiltonian_)->set_vnl_kpt(vnl_);
                            }

                            eigsolver.solve_kpt(psi_c, eig, Veff_s, Nd_d, Nband_loc,
                                                lambda_cutoff, eigval_min[s], eigval_max[s],
                                                kpt, cell_lengths,
                                                params_.cheb_degree,
                                                wfn.psi_kpt(s, k).ld());
                        } else {
                            double* psi = wfn.psi(s, k).data();
                            eigsolver.solve(psi, eig, Veff_s, Nd_d, Nband_loc,
                                            lambda_cutoff, eigval_min[s], eigval_max[s],
                                            params_.cheb_degree,
                                            wfn.psi(s, k).ld());
                        }
                    }
                }
            }

            // Update spectral bounds
            {
                double eig_last_max = -1e30;
                int n_spin_loop = is_soc_ ? 1 : Nspin_local;
                for (int s = 0; s < n_spin_loop; ++s) {
                    const double* eigs = wfn.eigenvalues(s, 0).data();
                    if (scf_iter > 0) {
                        eigval_min[s] = eigs[0];
                    }
                    if (eigs[Nband - 1] > eig_last_max)
                        eig_last_max = eigs[Nband - 1];
                }
                lambda_cutoff = eig_last_max + 0.1;
            }

            // Compute occupations between CheFSI passes
            Ef_ = Occupation::compute(wfn, Nelectron, beta, params_.smearing,
                                      kpt_weights, *kptcomm_, *spincomm_, kpt_start_);

        }

        // Compute tau for mGGA after CheFSI solve
        if (is_mgga_type(xc_type_)) {
            compute_tau(wfn, kpt_weights, kpt_start_, band_start_);
            tau_valid_ = true;
        }

        // 2. Compute new electron density
        ElectronDensity rho_new;
        if (is_soc_) {
            rho_new.allocate_noncollinear(Nd_d);
            rho_new.compute_spinor(wfn, kpt_weights, grid_->dV(),
                                    *bandcomm_, *kptcomm_, kpt_start_, band_start_);
        } else {
            rho_new.allocate(Nd_d, Nspin);
            rho_new.compute(wfn, kpt_weights, grid_->dV(), *bandcomm_, *kptcomm_,
                            Nspin, spin_start_, spincomm_, kpt_start_, band_start_);
        }

        // 3. For potential mixing: compute Veff_out from rho_out, energy with Escc correction
        //    For density mixing: compute energy using rho_in
        NDArray<double> Veff_out;

        if (use_potential_mixing) {
            // Save Veff_in
            NDArray<double> Veff_in(Nd_d * Nspin);
            std::memcpy(Veff_in.data(), Veff_.data(), Nd_d * Nspin * sizeof(double));

            // Update density_ to rho_out
            std::memcpy(density_.rho_total().data(), rho_new.rho_total().data(), Nd_d * sizeof(double));
            for (int s = 0; s < Nspin; ++s)
                std::memcpy(density_.rho(s).data(), rho_new.rho(s).data(), Nd_d * sizeof(double));

            // Compute Veff_out from rho_out
            compute_Veff(density_.rho_total().data(), rho_b, Vloc_);
            Veff_out = NDArray<double>(Nd_d * Nspin);
            std::memcpy(Veff_out.data(), Veff_.data(), Nd_d * Nspin * sizeof(double));

            // Compute energy using rho_out and Veff_out (current potentials)
            energy_ = Energy::compute_all(wfn, density_, Veff_.data(), phi_.data(),
                                           exc_.data(), Vxc_.data(), rho_b,
                                           Eself, Ec, beta, params_.smearing,
                                           kpt_weights, Nd_d, grid_->dV(),
                                           rho_core, Ef_, kpt_start_,
                                           kptcomm_, spincomm_, Nspin, nullptr,
                                           is_mgga_type(xc_type_) ? tau_.data() : nullptr,
                                           is_mgga_type(xc_type_) ? vtau_.data() : nullptr);

            // Self-consistency correction: Escc = ∫ ρ_out · (Veff_out - Veff_in) dV
            // This corrects for the fact that Eband was computed with Veff_in
            // but energy uses rho_out.
            double Escc = 0.0;
            if (Nspin == 2) {
                for (int s = 0; s < Nspin; ++s) {
                    const double* rho_s = density_.rho(s).data();
                    for (int i = 0; i < Nd_d; ++i) {
                        Escc += rho_s[i] * (Veff_out.data()[s*Nd_d + i] - Veff_in.data()[s*Nd_d + i]);
                    }
                }
            } else {
                const double* rho_tot = density_.rho_total().data();
                for (int i = 0; i < Nd_d; ++i) {
                    Escc += rho_tot[i] * (Veff_out.data()[i] - Veff_in.data()[i]);
                }
            }
            Escc *= grid_->dV();
            energy_.Etotal += Escc;

            // Restore Veff_in for mixer
            std::memcpy(Veff_.data(), Veff_in.data(), Nd_d * Nspin * sizeof(double));
        } else {
            // Density mixing: energy using rho_in
            energy_ = Energy::compute_all(wfn, density_, Veff_.data(), phi_.data(),
                                           exc_.data(), Vxc_.data(), rho_b,
                                           Eself, Ec, beta, params_.smearing,
                                           kpt_weights, Nd_d, grid_->dV(),
                                           rho_core, Ef_, kpt_start_,
                                           kptcomm_, spincomm_, Nspin, nullptr,
                                           is_mgga_type(xc_type_) ? tau_.data() : nullptr,
                                           is_mgga_type(xc_type_) ? vtau_.data() : nullptr);
        }

        // 5. Evaluate SCF error
        double scf_error = 0.0;
        if (use_potential_mixing) {
            // ||Veff_out - Veff_in|| / ||Veff_out|| (using unshifted values)
            double sum_sq_out = 0.0, sum_sq_diff = 0.0;
            for (int i = 0; i < Nd_d * Nspin; ++i) {
                // Veff_out and Veff_ are both unshifted at this point
                double diff = Veff_out.data()[i] - Veff_.data()[i];
                sum_sq_out += Veff_out.data()[i] * Veff_out.data()[i];
                sum_sq_diff += diff * diff;
            }
            scf_error = (sum_sq_out > 0.0) ? std::sqrt(sum_sq_diff / sum_sq_out) : 0.0;
        } else {
            // ||rho_out - rho_in|| / ||rho_out||
            const double* rho_in = density_.rho_total().data();
            const double* rho_out = rho_new.rho_total().data();
            double sum_sq_out = 0.0, sum_sq_diff = 0.0;
            for (int i = 0; i < Nd_d; ++i) {
                double diff = rho_out[i] - rho_in[i];
                sum_sq_out += rho_out[i] * rho_out[i];
                sum_sq_diff += diff * diff;
            }
            scf_error = (sum_sq_out > 0.0) ? std::sqrt(sum_sq_diff / sum_sq_out) : 0.0;
        }

        if (rank_world == 0) {
            if (Nspin == 2) {
                // Compute magnetization
                double mag_sum = 0.0;
                const double* ru = rho_new.rho(0).data();
                const double* rd = rho_new.rho(1).data();
                for (int i = 0; i < Nd_d; ++i) mag_sum += ru[i] - rd[i];
                mag_sum *= grid_->dV();
                std::printf("SCF iter %3d: Etot = %18.10f Ha, SCF error = %10.3e, Ef = %10.5f, mag = %8.4f\n",
                            scf_iter + 1, energy_.Etotal, scf_error, Ef_, mag_sum);
            } else {
                std::printf("SCF iter %3d: Etot = %18.10f Ha, SCF error = %10.3e, Ef = %10.5f\n",
                            scf_iter + 1, energy_.Etotal, scf_error, Ef_);
            }
        }

        // Print eigenvalues at last iteration for debugging
        if (rank_world == 0 && scf_error < params_.tol && scf_iter >= params_.min_iter) {
            std::printf("\nFinal eigenvalues (Ha) and occupations:\n");
            for (int s = 0; s < Nspin_local; ++s) {
                if (Nspin > 1) std::printf("  Spin %d:\n", spin_start_ + s);
                // Print all global eigenvalues (Nband_global)
                for (int n = 0; n < Nband; ++n) {
                    std::printf("  %3d  %20.12e  %16.12f\n",
                                n+1, wfn.eigenvalues(s, 0)(n), wfn.occupations(s, 0)(n));
                }
            }
        }

        if (scf_iter >= params_.min_iter && scf_error < params_.tol) {
            converged_ = true;
            if (rank_world == 0) {
                std::printf("SCF converged after %d iterations.\n", scf_iter + 1);
            }
            break;
        }

        E_prev = energy_.Etotal;

        // 6. Mix
        if (use_potential_mixing) {
            // Potential mixing matching SPARC's exact flow:
            // Veff_mixed = zero-mean copy of the mixer's x_k (persistent, like mixing_hist_xk)
            // Veff_out = zero-mean copy of g_k (output from compute_Veff, shifted here)
            // After mixing, Veff_mixed is updated to x_kp1, and Veff_ = Veff_mixed + mean.

            // Step 1: Shift Veff_out to zero-mean, save mean for later
            for (int s = 0; s < Nspin; ++s) {
                double mean = 0;
                for (int i = 0; i < Nd_d; ++i) mean += Veff_out.data()[s*Nd_d + i];
                mean /= grid_->Nd();
                Veff_mean_[s] = mean;
                for (int i = 0; i < Nd_d; ++i) Veff_out.data()[s*Nd_d + i] -= mean;
            }

            // Step 2: Mix. Veff_mixed is x_k (zero-mean), Veff_out is g_k (zero-mean).
            // After mix(), Veff_mixed is updated to x_{k+1} (zero-mean).
            mixer.mix(Veff_mixed.data(), Veff_out.data(), Nd_d, Nspin);

            // Step 3: Veff_ = Veff_mixed + mean (for Hamiltonian)
            for (int s = 0; s < Nspin; ++s) {
                for (int i = 0; i < Nd_d; ++i) {
                    Veff_.data()[s*Nd_d + i] = Veff_mixed.data()[s*Nd_d + i] + Veff_mean_[s];
                }
            }
        } else if (is_soc_) {
            // SOC: mix packed array [rho | mx | my | mz] (4*Nd_d)
            std::vector<double> dens_in(4 * Nd_d), dens_out(4 * Nd_d);
            std::memcpy(dens_in.data(), density_.rho_total().data(), Nd_d * sizeof(double));
            std::memcpy(dens_in.data() + Nd_d, density_.mag_x().data(), Nd_d * sizeof(double));
            std::memcpy(dens_in.data() + 2*Nd_d, density_.mag_y().data(), Nd_d * sizeof(double));
            std::memcpy(dens_in.data() + 3*Nd_d, density_.mag_z().data(), Nd_d * sizeof(double));

            std::memcpy(dens_out.data(), rho_new.rho_total().data(), Nd_d * sizeof(double));
            std::memcpy(dens_out.data() + Nd_d, rho_new.mag_x().data(), Nd_d * sizeof(double));
            std::memcpy(dens_out.data() + 2*Nd_d, rho_new.mag_y().data(), Nd_d * sizeof(double));
            std::memcpy(dens_out.data() + 3*Nd_d, rho_new.mag_z().data(), Nd_d * sizeof(double));

            mixer.mix(dens_in.data(), dens_out.data(), Nd_d, 4);

            // Unpack: clamp rho, keep magnetization
            double* rho_mix = density_.rho_total().data();
            for (int i = 0; i < Nd_d; ++i) {
                rho_mix[i] = dens_in[i];
                if (rho_mix[i] < 0.0) rho_mix[i] = 0.0;
            }
            {
                double rho_sum = 0.0;
                for (int i = 0; i < Nd_d; ++i) rho_sum += rho_mix[i];
                double Ne_current = rho_sum * grid_->dV();
                if (Ne_current > 1e-10) {
                    double scale = static_cast<double>(Nelectron) / Ne_current;
                    for (int i = 0; i < Nd_d; ++i) rho_mix[i] *= scale;
                }
            }
            std::memcpy(density_.rho(0).data(), rho_mix, Nd_d * sizeof(double));
            std::memcpy(density_.mag_x().data(), dens_in.data() + Nd_d, Nd_d * sizeof(double));
            std::memcpy(density_.mag_y().data(), dens_in.data() + 2*Nd_d, Nd_d * sizeof(double));
            std::memcpy(density_.mag_z().data(), dens_in.data() + 3*Nd_d, Nd_d * sizeof(double));
        } else if (Nspin == 1) {
            mixer.mix(density_.rho_total().data(), rho_new.rho_total().data(), Nd_d);

            double* rho_mix = density_.rho_total().data();
            for (int i = 0; i < Nd_d; ++i) {
                if (rho_mix[i] < 0.0) rho_mix[i] = 0.0;
            }
            {
                double rho_sum = 0.0;
                for (int i = 0; i < Nd_d; ++i) rho_sum += rho_mix[i];
                double Ne_current = rho_sum * grid_->dV();
                if (Ne_current > 1e-10) {
                    double scale = static_cast<double>(Nelectron) / Ne_current;
                    for (int i = 0; i < Nd_d; ++i) rho_mix[i] *= scale;
                }
            }
            std::memcpy(density_.rho(0).data(), density_.rho_total().data(), Nd_d * sizeof(double));
        } else {
            // Spin-polarized: mix packed array [total | magnetization] (2*Nd_d)
            std::vector<double> dens_in(2 * Nd_d), dens_out(2 * Nd_d);

            const double* rho_up_in = density_.rho(0).data();
            const double* rho_dn_in = density_.rho(1).data();
            for (int i = 0; i < Nd_d; ++i) {
                dens_in[i] = density_.rho_total().data()[i];
                dens_in[Nd_d + i] = rho_up_in[i] - rho_dn_in[i];
            }

            const double* rho_up_out = rho_new.rho(0).data();
            const double* rho_dn_out = rho_new.rho(1).data();
            for (int i = 0; i < Nd_d; ++i) {
                dens_out[i] = rho_new.rho_total().data()[i];
                dens_out[Nd_d + i] = rho_up_out[i] - rho_dn_out[i];
            }

            mixer.mix(dens_in.data(), dens_out.data(), Nd_d, 2);

            double* rho_tot = density_.rho_total().data();
            double* rho_up = density_.rho(0).data();
            double* rho_dn = density_.rho(1).data();
            for (int i = 0; i < Nd_d; ++i) {
                rho_tot[i] = dens_in[i];
                double mag = dens_in[Nd_d + i];
                rho_up[i] = 0.5 * (rho_tot[i] + mag);
                rho_dn[i] = 0.5 * (rho_tot[i] - mag);
            }

            for (int i = 0; i < Nd_d; ++i) {
                if (rho_up[i] < 0.0) rho_up[i] = 0.0;
                if (rho_dn[i] < 0.0) rho_dn[i] = 0.0;
                rho_tot[i] = rho_up[i] + rho_dn[i];
            }
            {
                double rho_sum = 0.0;
                for (int i = 0; i < Nd_d; ++i) rho_sum += rho_tot[i];
                double Ne_current = rho_sum * grid_->dV();
                if (Ne_current > 1e-10) {
                    double scale = static_cast<double>(Nelectron) / Ne_current;
                    for (int i = 0; i < Nd_d; ++i) {
                        rho_up[i] *= scale;
                        rho_dn[i] *= scale;
                        rho_tot[i] *= scale;
                    }
                }
            }
        }

        // 7. For density mixing: recompute Veff from mixed density
        //    For potential mixing: Veff_ already contains the mixed potential
        if (!use_potential_mixing) {
            if (is_soc_) {
                compute_Veff_spinor(density_, rho_b, Vloc_);
            } else {
                compute_Veff(density_.rho_total().data(), rho_b, Vloc_);
            }
        }
    }

    if (!converged_ && rank_world == 0) {
        std::printf("WARNING: SCF did not converge within %d iterations.\n", params_.max_iter);
    }

    // ===== Outer Fock Loop for hybrid functionals =====
    if (exx_ && is_hybrid(xc_type_)) {
        if (rank_world == 0)
            std::printf("\n===== Starting outer Fock loop for hybrid functional =====\n");

        double Eexx_prev = 0.0;
        auto exx_p = params_.exx_params;
        // Match SPARC: TOL_FOCK = 0.2 * TOL_SCF when not explicitly set
        if (exx_p.tol_fock < 0.0) {
            exx_p.tol_fock = 0.2 * params_.tol;
        }
        if (rank_world == 0)
            std::printf("Fock params: maxit=%d, minit=%d, tol_fock=%.2e\n",
                        exx_p.maxit_fock, exx_p.minit_fock, exx_p.tol_fock);

        // Enter Fock phase: switch to hybrid-scaled XC
        in_fock_loop_ = true;

        // Reset mixing history and recompute Vxc/Veff with hybrid-scaled XC
        // (matching SPARC Exact_Exchange_loop: lines 95-138)
        mixer.reset();
        compute_Veff(density_.rho_total().data(), rho_b, Vloc_);

        for (int fock_iter = 0; fock_iter < exx_p.maxit_fock; fock_iter++) {
            // 1. Build ACE operator from current orbitals
            exx_->build_ACE(wfn);

            // 2. Compute exchange energy estimate
            double Eexx_est = exx_->compute_energy(wfn);
            if (rank_world == 0)
                std::printf("Fock iter %2d: building ACE, Eexx_est = %.10f Ha\n", fock_iter + 1, Eexx_est);

            // 3. Enable Vx in the Hamiltonian for inner SCF
            // SPARC applies Vx when usefock is even (during Fock inner SCF)
            // Combined with (1-exx_frac)*PBE_X in Vxc, this gives the correct hybrid Hamiltonian
            hamiltonian_->set_exx(exx_);

            // 4. Re-estimate Chebyshev filter bounds for hybrid Hamiltonian
            // Full Lanczos with exchange to capture the correct spectrum.
            // Use small Chebyshev degree to prevent catastrophic amplification
            // from the large occupied-unoccupied eigenvalue gap created by exchange.
            {
                for (int s = 0; s < Nspin_local; ++s) {
                    int s_glob = spin_start_ + s;
                    if (is_kpt_) {
                        Vec3 kpt0 = kpoints_->kpts_cart()[kpt_start_];
                        if (vnl_ && vnl_->is_setup()) {
                            const_cast<NonlocalProjector*>(vnl_)->set_kpoint(kpt0);
                            hamiltonian_->set_vnl_kpt(vnl_);
                        }
                        hamiltonian_->set_exx_context(s_glob, kpt_start_);
                        eigsolver.lanczos_bounds_kpt(Veff_.data() + s_glob * Nd_d, Nd_d,
                                                      kpt0, cell_lengths,
                                                      eigval_min[s], eigval_max[s]);
                    } else {
                        hamiltonian_->set_exx_context(s_glob, 0);
                        eigsolver.lanczos_bounds(Veff_.data() + s_glob * Nd_d, Nd_d,
                                                  eigval_min[s], eigval_max[s]);
                    }
                    eigval_max[s] *= 1.01;  // 1% buffer
                }

                // lambda_cutoff = highest previous eigenvalue + margin
                double eig_last_max = -1e30;
                for (int s = 0; s < Nspin_local; ++s) {
                    const double* eigs = wfn.eigenvalues(s, 0).data();
                    if (eigs[Nband - 1] > eig_last_max)
                        eig_last_max = eigs[Nband - 1];
                }
                lambda_cutoff = eig_last_max + 0.1;

                if (rank_world == 0) {
                    for (int s = 0; s < Nspin_local; ++s)
                        std::printf("Fock Lanczos bounds (spin %d): eigmin=%.6e, eigmax=%.6e, cutoff=%.6e\n",
                                    spin_start_ + s, eigval_min[s], eigval_max[s], lambda_cutoff);
                }
            }

            // 5. Reset mixer at start of each Fock inner SCF
            // (matches SPARC: scf_loop resets mixing_hist at lines 124-125 of electronicGroundState.c)
            mixer.reset();

            // 6. Run inner SCF with EXX enabled
            converged_ = false;
            for (int scf_iter = 0; scf_iter < params_.max_iter; ++scf_iter) {
                int nchefsi_inner = params_.nchefsi;

                // Use reduced Chebyshev degree during inner Fock SCF to prevent
                // With correct normalization, exchange shifts eigenvalues by O(0.1) Ha
                // so the full Chebyshev degree is safe to use.
                int cheb_deg_inner = params_.cheb_degree;

                for (int chefsi_pass = 0; chefsi_pass < nchefsi_inner; ++chefsi_pass) {
                    // Eigensolver (same as above, with EXX active via Hamiltonian)
                    for (int s = 0; s < Nspin_local; ++s) {
                        int s_glob = spin_start_ + s;
                        double* Veff_s = Veff_.data() + s_glob * Nd_d;
                        for (int k = 0; k < Nkpts; ++k) {
                            double* eig_inner = wfn.eigenvalues(s, k).data();
                            if (is_kpt_) {
                                Complex* psi_c = wfn.psi_kpt(s, k).data();
                                int k_glob = kpt_start_ + k;
                                Vec3 kpt = kpoints_->kpts_cart()[k_glob];
                                if (vnl_ && vnl_->is_setup()) {
                                    const_cast<NonlocalProjector*>(vnl_)->set_kpoint(kpt);
                                    hamiltonian_->set_vnl_kpt(vnl_);
                                }
                                hamiltonian_->set_exx_context(s_glob, k_glob);
                                eigsolver.solve_kpt(psi_c, eig_inner, Veff_s, Nd_d, Nband_loc,
                                                    lambda_cutoff, eigval_min[s], eigval_max[s],
                                                    kpt, cell_lengths,
                                                    cheb_deg_inner,
                                                    wfn.psi_kpt(s, k).ld());
                            } else {
                                double* psi = wfn.psi(s, k).data();
                                hamiltonian_->set_exx_context(s_glob, 0);
                                eigsolver.solve(psi, eig_inner, Veff_s, Nd_d, Nband_loc,
                                                lambda_cutoff, eigval_min[s], eigval_max[s],
                                                cheb_deg_inner,
                                                wfn.psi(s, k).ld());
                            }
                        }
                    }

                    // Update spectral bounds
                    {
                        double eig_last_max = -1e30;
                        for (int s = 0; s < Nspin_local; ++s) {
                            const double* eigs = wfn.eigenvalues(s, 0).data();
                            if (scf_iter > 0) eigval_min[s] = eigs[0];
                            if (eigs[Nband - 1] > eig_last_max)
                                eig_last_max = eigs[Nband - 1];
                        }
                        lambda_cutoff = eig_last_max + 0.1;
                    }

                    Ef_ = Occupation::compute(wfn, Nelectron, beta, params_.smearing,
                                              kpt_weights, *kptcomm_, *spincomm_, kpt_start_);
                }

                // Compute density
                ElectronDensity rho_new_fock;
                rho_new_fock.allocate(Nd_d, Nspin);
                rho_new_fock.compute(wfn, kpt_weights, grid_->dV(), *bandcomm_, *kptcomm_,
                                     Nspin, spin_start_, spincomm_, kpt_start_, band_start_);

                // Energy with Eexx correction (matching SPARC Calculate_Free_Energy usefock%2==0)
                // Standard: Etot = Eband - E2 + Exc - E3 + Ehart + Eself + Ec + Entropy
                // Hybrid:   Exc += Eexx, Etot = standard + Eexx - 2*Eexx = standard - Eexx
                // Eexx_est is FIXED during the inner SCF
                energy_ = Energy::compute_all(wfn, density_, Veff_.data(), phi_.data(),
                                               exc_.data(), Vxc_.data(), rho_b,
                                               Eself, Ec, beta, params_.smearing,
                                               kpt_weights, Nd_d, grid_->dV(),
                                               rho_core, Ef_, kpt_start_,
                                               kptcomm_, spincomm_, Nspin);
                energy_.Exc += Eexx_est;
                energy_.Etotal -= Eexx_est;

                // SCF error
                double scf_error_fock = 0.0;
                {
                    const double* rho_in = density_.rho_total().data();
                    const double* rho_out = rho_new_fock.rho_total().data();
                    double sum_sq_out = 0.0, sum_sq_diff = 0.0;
                    for (int i = 0; i < Nd_d; ++i) {
                        double diff = rho_out[i] - rho_in[i];
                        sum_sq_out += rho_out[i] * rho_out[i];
                        sum_sq_diff += diff * diff;
                    }
                    scf_error_fock = (sum_sq_out > 0.0) ? std::sqrt(sum_sq_diff / sum_sq_out) : 0.0;
                }

                if (rank_world == 0) {
                    std::printf("  Fock %d SCF %3d: Etot = %18.10f, err = %10.3e, Ef = %10.5f",
                                fock_iter + 1, scf_iter + 1, energy_.Etotal, scf_error_fock, Ef_);
                    if (scf_iter == 0)
                        std::printf("  [Eband=%.6f Exc=%.6f Eexx_est=%.6f std=%.6f]",
                                    energy_.Eband, energy_.Exc - Eexx_est, Eexx_est,
                                    energy_.Etotal + Eexx_est);
                    std::printf("\n");
                }

                // Inner Fock SCF uses TOL_SCF (1e-6), matching SPARC when usefock%2==0
                if (scf_iter >= params_.min_iter && scf_error_fock < params_.tol) {
                    converged_ = true;
                    break;
                }

                // Mix density
                if (Nspin == 1) {
                    mixer.mix(density_.rho_total().data(), rho_new_fock.rho_total().data(), Nd_d);
                    double* rho_mix = density_.rho_total().data();
                    for (int i = 0; i < Nd_d; ++i) if (rho_mix[i] < 0.0) rho_mix[i] = 0.0;
                    {
                        double rho_sum = 0.0;
                        for (int i = 0; i < Nd_d; ++i) rho_sum += rho_mix[i];
                        double Ne_current = rho_sum * grid_->dV();
                        if (Ne_current > 1e-10) {
                            double scale = static_cast<double>(Nelectron) / Ne_current;
                            for (int i = 0; i < Nd_d; ++i) rho_mix[i] *= scale;
                        }
                    }
                    std::memcpy(density_.rho(0).data(), density_.rho_total().data(), Nd_d * sizeof(double));
                } else {
                    // Spin-polarized mixing (same as main loop)
                    std::vector<double> dens_in(2 * Nd_d), dens_out(2 * Nd_d);
                    const double* rho_up_in = density_.rho(0).data();
                    const double* rho_dn_in = density_.rho(1).data();
                    for (int i = 0; i < Nd_d; ++i) {
                        dens_in[i] = density_.rho_total().data()[i];
                        dens_in[Nd_d + i] = rho_up_in[i] - rho_dn_in[i];
                    }
                    const double* rho_up_out = rho_new_fock.rho(0).data();
                    const double* rho_dn_out = rho_new_fock.rho(1).data();
                    for (int i = 0; i < Nd_d; ++i) {
                        dens_out[i] = rho_new_fock.rho_total().data()[i];
                        dens_out[Nd_d + i] = rho_up_out[i] - rho_dn_out[i];
                    }
                    mixer.mix(dens_in.data(), dens_out.data(), Nd_d, 2);
                    double* rho_tot = density_.rho_total().data();
                    double* rho_up = density_.rho(0).data();
                    double* rho_dn = density_.rho(1).data();
                    for (int i = 0; i < Nd_d; ++i) {
                        rho_tot[i] = dens_in[i];
                        double mag = dens_in[Nd_d + i];
                        rho_up[i] = 0.5 * (rho_tot[i] + mag);
                        rho_dn[i] = 0.5 * (rho_tot[i] - mag);
                        if (rho_up[i] < 0.0) rho_up[i] = 0.0;
                        if (rho_dn[i] < 0.0) rho_dn[i] = 0.0;
                        rho_tot[i] = rho_up[i] + rho_dn[i];
                    }
                    {
                        double rho_sum = 0.0;
                        for (int i = 0; i < Nd_d; ++i) rho_sum += rho_tot[i];
                        double Ne_current = rho_sum * grid_->dV();
                        if (Ne_current > 1e-10) {
                            double scale = static_cast<double>(Nelectron) / Ne_current;
                            for (int i = 0; i < Nd_d; ++i) { rho_up[i] *= scale; rho_dn[i] *= scale; rho_tot[i] *= scale; }
                        }
                    }
                }

                compute_Veff(density_.rho_total().data(), rho_b, Vloc_);
            }

            // 6. Energy correction (matching SPARC exactExchange.c:310-325)
            // Inner SCF energy already has: Exc += Eexx_est, Etot -= Eexx_est (i.e. standard - Eexx_est)
            // Now undo old Eexx and apply new Eexx
            Eexx_prev = Eexx_est;
            // Undo old: Exc -= Eexx_est, Etot += 2*Eexx_est
            energy_.Exc -= Eexx_est;
            energy_.Etotal += 2.0 * Eexx_est;
            // Recompute Eexx with post-SCF orbitals
            double Eexx_new = exx_->compute_energy(wfn);
            // Apply new: Exc += Eexx_new, Etot -= 2*Eexx_new
            energy_.Exc += Eexx_new;
            energy_.Etotal -= 2.0 * Eexx_new;
            energy_.Eexx = Eexx_new;

            // 7. Check Fock convergence
            double err_fock = std::fabs(Eexx_new - Eexx_prev) / std::max(1, Natom);
            if (rank_world == 0) {
                std::printf("Fock iter %2d: Eexx = %.10f, |dEexx|/atom = %.3e, Etot = %.10f\n",
                            fock_iter + 1, Eexx_new, err_fock, energy_.Etotal);
            }

            if (err_fock < exx_p.tol_fock && fock_iter >= exx_p.minit_fock) {
                if (rank_world == 0)
                    std::printf("Fock loop converged after %d iterations.\n", fock_iter + 1);
                break;
            }
        }

        // Disable Vx in Hamiltonian after Fock loop
        hamiltonian_->set_exx(nullptr);
    }

    return energy_.Etotal;
}

#ifdef USE_CUDA
double SCF::run_gpu(Wavefunction& wfn, int Nelectron, int Natom,
                     const double* rho_b, double Eself, double Ec,
                     XCType xc_type, const double* rho_core) {
    int Nd_d = domain_->Nd_d();
    int Nspin = Nspin_global_;
    bool is_gga = (xc_type == XCType::GGA_PBE || xc_type == XCType::GGA_PBEsol ||
                   xc_type == XCType::GGA_RPBE ||
                   xc_type == XCType::HYB_PBE0 || xc_type == XCType::HYB_HSE);

    // Allocate work arrays (needed for download_results)
    bool has_gradient = is_gga || is_mgga_type(xc_type);
    int dxc_ncol = has_gradient ? ((Nspin == 2) ? 3 : 1) : 0;
    Veff_ = NDArray<double>(Nd_d * Nspin);
    Vxc_ = NDArray<double>(Nd_d * Nspin);
    exc_ = NDArray<double>(Nd_d);
    phi_ = NDArray<double>(Nd_d);
    if (dxc_ncol > 0) Dxcdgrho_ = NDArray<double>(Nd_d * dxc_ncol);
    if (is_mgga_type(xc_type)) {
        int tau_size = (Nspin == 2) ? 3 * Nd_d : Nd_d;
        int vtau_size = (Nspin == 2) ? 2 * Nd_d : Nd_d;
        tau_ = NDArray<double>(tau_size);
        vtau_ = NDArray<double>(vtau_size);
        xc_type_ = xc_type;
    }

    // Initialize density: use existing density if set_initial_density was called
    // (preserves initial magnetization for spin-polarized systems),
    // otherwise compute from atomic superposition.
    if (density_.Nd_d() == 0) {
        density_.allocate(Nd_d, Nspin);
        if (elec_ && influence_) {
            const_cast<Electrostatics*>(elec_)->compute_atomic_density(
                *crystal_, *influence_, *domain_, *grid_,
                density_.rho_total().data(), Nelectron);
            if (Nspin == 1) {
                std::memcpy(density_.rho(0).data(), density_.rho_total().data(), Nd_d * sizeof(double));
            } else {
                // Equal split for initial spin density
                for (int i = 0; i < Nd_d; i++) {
                    density_.rho(0).data()[i] = 0.5 * density_.rho_total().data()[i];
                    density_.rho(1).data()[i] = 0.5 * density_.rho_total().data()[i];
                }
            }
        } else {
            init_density(Nd_d, Nelectron);
        }
    }

    // K-point weights
    std::vector<double> kpt_weights;
    if (kpoints_) {
        kpt_weights = kpoints_->normalized_weights();
    } else {
        kpt_weights = {1.0};
    }

    // Detect SOC mode
    bool is_soc = (wfn.Nspinor() == 2);

    // Create and run GPU SCF
    // For hybrid functionals, the initial PBE phase uses the base GGA functional.
    // The full hybrid XC type is passed separately for the Fock loop.
    XCType xc_type_gpu = is_hybrid(xc_type) ? hybrid_base_xc(xc_type) : xc_type;
    bool is_gga_gpu = (xc_type_gpu == XCType::GGA_PBE || xc_type_gpu == XCType::GGA_PBEsol ||
                       xc_type_gpu == XCType::GGA_RPBE);
    gpu_runner_ = std::make_unique<GPUSCFRunner>();
    double Etotal = gpu_runner_->run(
        wfn, params_, *grid_, *domain_, *stencil_,
        *hamiltonian_, *halo_, vnl_,
        *crystal_, *nloc_influence_, *bandcomm_,
        Nelectron, Natom,
        density_.rho_total().data(), rho_b,
        Eself, Ec, xc_type_gpu, rho_core, is_gga_gpu,
        Nspin, is_kpt_, kpoints_, kpt_weights,
        Nspin_local_, spin_start_, kpt_start_,
        density_.Nd_d() > 0 && Nspin == 2 ? density_.rho(0).data() : nullptr,
        density_.Nd_d() > 0 && Nspin == 2 ? density_.rho(1).data() : nullptr,
        is_soc,
        exx_,       // EXX operator (may be null)
        xc_type);   // full hybrid XC type

    // Download results for forces/stress
    gpu_runner_->download_results(
        phi_.data(), Vxc_.data(), exc_.data(), Veff_.data(),
        dxc_ncol > 0 ? Dxcdgrho_.data() : nullptr,
        density_.rho_total().data(), wfn);
    // Download mGGA tau/vtau if applicable
    if ((xc_type == XCType::MGGA_SCAN || xc_type == XCType::MGGA_RSCAN || xc_type == XCType::MGGA_R2SCAN) && tau_.size() > 0 && vtau_.size() > 0) {
        if (Nspin == 2) {
            // GPU layout: [up(Nd)|dn(Nd)], CPU layout: [up(Nd)|dn(Nd)|total(Nd)]
            // Download up/dn from GPU into first 2*Nd slots, compute total at end
            int gpu_tau_size = 2 * Nd_d;
            int gpu_vtau_size = 2 * Nd_d;
            gpu_runner_->download_tau_vtau(tau_.data(), vtau_.data(),
                                            gpu_tau_size, gpu_vtau_size);
            // Compute total tau = up + dn (store in 3rd slot)
            for (int i = 0; i < Nd_d; i++)
                tau_(2 * Nd_d + i) = tau_(i) + tau_(Nd_d + i);
        } else {
            gpu_runner_->download_tau_vtau(tau_.data(), vtau_.data(),
                                            (int)tau_.size(), (int)vtau_.size());
        }
    }
    // Keep spin densities in sync
    if (Nspin == 2) {
        // Download spin-resolved densities from GPU for stress computation
        gpu_runner_->download_spin_densities(density_.rho(0).data(), density_.rho(1).data(), Nd_d);
    } else {
        std::memcpy(density_.rho(0).data(), density_.rho_total().data(), Nd_d * sizeof(double));
    }

    energy_ = gpu_runner_->energy();
    converged_ = gpu_runner_->converged();
    Ef_ = gpu_runner_->fermi_energy();

    return Etotal;
}
#endif

} // namespace lynx
