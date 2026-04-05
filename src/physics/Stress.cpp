#include "physics/Stress.hpp"
#include "physics/SCF.hpp"
#include "physics/Electrostatics.hpp"
#include "xc/ExactExchange.hpp"
#include "atoms/AtomSetup.hpp"
#include "io/InputParser.hpp"

#ifdef USE_CUDA
#include "physics/GPUStress.cuh"
#endif
#include "core/constants.hpp"
#include "core/Lattice.hpp"
#include "atoms/Pseudopotential.hpp"
#include <cmath>
#include <cstring>
#include <complex>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <cassert>

namespace lynx {

static bool is_mgga_type(XCType t) {
    return t == XCType::MGGA_SCAN || t == XCType::MGGA_RSCAN || t == XCType::MGGA_R2SCAN;
}

// Transform gradient from non-Cartesian to Cartesian: ∇_cart = LatUVec^{-1} * ∇_nc
// (matching reference LYNX nonCart2Cart_grad, which uses gradT^T)
static inline void nonCart2Cart_grad(const Mat3& uvec_inv, double& x, double& y, double& z) {
    double a = x, b = y, c = z;
    x = uvec_inv(0,0)*a + uvec_inv(0,1)*b + uvec_inv(0,2)*c;
    y = uvec_inv(1,0)*a + uvec_inv(1,1)*b + uvec_inv(1,2)*c;
    z = uvec_inv(2,0)*a + uvec_inv(2,1)*b + uvec_inv(2,2)*c;
}

// Transform coordinates from non-Cartesian to Cartesian: cart = LatUVec^T * nc
// (matching reference LYNX nonCart2Cart_coord, which uses LatUVec columns)
static inline void nonCart2Cart_coord(const Mat3& uvec, double& x, double& y, double& z) {
    double a = x, b = y, c = z;
    // cart_j = Σ_i nc_i * LatUVec(i,j) = (LatUVec^T * nc)_j
    x = uvec(0,0)*a + uvec(1,0)*b + uvec(2,0)*c;
    y = uvec(0,1)*a + uvec(1,1)*b + uvec(2,1)*c;
    z = uvec(0,2)*a + uvec(1,2)*b + uvec(2,2)*c;
}

// Transform arrays of non-Cart gradients to Cartesian
static void nonCart2Cart_grad_arrays(const Mat3& uvec_inv, double* gx, double* gy, double* gz, int n) {
    for (int i = 0; i < n; ++i) {
        nonCart2Cart_grad(uvec_inv, gx[i], gy[i], gz[i]);
    }
}

// ---------------------------------------------------------------------------
// High-level entry point: extract data, handle GPU mGGA and EXX, then print.
// ---------------------------------------------------------------------------
void Stress::compute(
    const LynxContext& ctx,
    const SystemConfig& config,
    const Wavefunction& wfn,
    SCF& scf,
    const AtomSetup& atoms,
    const NonlocalProjector& vnl) {

    int Nspin_calc = (config.spin_type == SpinType::Collinear) ? 2 :
                     (config.spin_type == SpinType::NonCollinear) ? 1 : 1;
    const double* rho_up_ptr = (Nspin_calc == 2) ? scf.density().rho(0).data() : nullptr;
    const double* rho_dn_ptr = (Nspin_calc == 2) ? scf.density().rho(1).data() : nullptr;

    // GPU mGGA stress
    // TODO: Re-enable GPU mGGA stress via unified operator GPU state
    //       (psi on device accessed through Hamiltonian::gpu_state_ptr()).
    //       For now, falls back to CPU mGGA stress computation.
    const double* gpu_mgga_ptr = nullptr;
    const double* gpu_dot_ptr = nullptr;

#ifdef USE_CUDA
    // Inject GPU kinetic+nonlocal stress (psi never left GPU)
    if (scf.has_gpu_force_stress()) {
        set_gpu_kinetic_nonlocal_stress(scf.gpu_force_stress().stress_k,
                                         scf.gpu_force_stress().stress_nl);
    }
#endif

    compute_impl(ctx, wfn, atoms.crystal,
                 atoms.influence, atoms.nloc_influence, vnl,
                 scf.phi(), scf.density().rho_total().data(),
                 rho_up_ptr, rho_dn_ptr,
                 atoms.Vloc.data(),
                 atoms.elec.pseudocharge().data(),
                 atoms.elec.pseudocharge_ref().data(),
                 scf.exc(), scf.Vxc(),
                 scf.Dxcdgrho(),
                 scf.energy().Exc,
                 atoms.elec.Eself() + atoms.elec.Ec(),
                 config.xc,
                 Nspin_calc,
                 atoms.has_nlcc ? atoms.rho_core.data() : nullptr,
                 scf.vtau(), scf.tau(),
                 gpu_mgga_ptr, gpu_dot_ptr);

    // Add EXX stress for hybrid functionals
    if (is_hybrid(config.xc) && scf.exx().is_setup()) {
        auto stress_exx = scf.exx().compute_stress(wfn);
        add_to_total(stress_exx);
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    print(rank);
}

// ---------------------------------------------------------------------------
// Detailed implementation (private).
// ---------------------------------------------------------------------------
std::array<double, 6> Stress::compute_impl(
    const LynxContext& ctx,
    const Wavefunction& wfn,
    const Crystal& crystal,
    const std::vector<AtomInfluence>& influence,
    const std::vector<AtomNlocInfluence>& nloc_influence,
    const NonlocalProjector& vnl,
    const double* phi,
    const double* rho,
    const double* rho_up,
    const double* rho_dn,
    const double* Vloc,
    const double* b,
    const double* b_ref,
    const double* exc,
    const double* Vxc,
    const double* Dxcdgrho,
    double Exc,
    double Esc,
    XCType xc_type,
    int Nspin,
    const double* rho_core,
    const double* vtau,
    const double* tau,
    const double* gpu_mgga_psi_stress,
    const double* gpu_tau_vtau_dot) {
    ctx_ = &ctx;
    std::vector<double> kpt_weights = ctx.kpoints().normalized_weights();
    const auto& grid = ctx.grid();

    stress_k_.fill(0.0);
    stress_xc_.fill(0.0);
    stress_el_.fill(0.0);
    stress_nl_.fill(0.0);
    stress_total_.fill(0.0);

    // Cell measure: for 3D periodic, volume = Jacbdet * prod(L_periodic)
    const auto& lat = grid.lattice();
    Vec3 L = lat.lengths();
    double Jacbdet = lat.jacobian() / (L.x * L.y * L.z);
    cell_measure_ = Jacbdet;
    if (grid.bcx() == BCType::Periodic) cell_measure_ *= L.x;
    if (grid.bcy() == BCType::Periodic) cell_measure_ *= L.y;
    if (grid.bcz() == BCType::Periodic) cell_measure_ *= L.z;

    // 1. XC stress (pass GPU tau_vtau_dot if available to skip CPU reduction)
    compute_xc_stress(rho, rho_up, rho_dn, exc, Vxc, Dxcdgrho, Exc, xc_type, Nspin, rho_core, vtau, tau, gpu_tau_vtau_dot);

    // 1b. NLCC XC stress correction
    if (rho_core) {
        compute_xc_nlcc_stress(crystal, influence, Vxc, Nspin);
    }

    // 2. Electrostatic (local) stress
    compute_electrostatic(crystal, influence, phi, rho, Vloc, b, b_ref, Esc);

    // 3. Nonlocal + kinetic stress
    if (!gpu_kn_set_) {
        // CPU path: compute from host psi
        compute_nonlocal_kinetic(ctx, wfn, crystal, nloc_influence, vnl, kpt_weights);
    }
    // else: GPU kinetic+nonlocal stress already injected via set_gpu_kinetic_nonlocal_stress

    // 4. mGGA psi stress term: σ_ij += -occfac · Σ_n g_n · ∫ vtau · ∇_i ψ · ∇_j ψ dV
    if (gpu_mgga_psi_stress && is_mgga_type(xc_type)) {
        // Use pre-computed GPU values (already normalized by cell_measure)
        for (int i = 0; i < 6; i++)
            stress_xc_[i] += gpu_mgga_psi_stress[i];
    } else if (vtau && is_mgga_type(xc_type)) {
        const auto& gradient_m = ctx_->gradient();
        const auto& halo_m = ctx_->halo();
        const auto& bandcomm_m = ctx_->scf_bandcomm();
        const auto& kptcomm_m = ctx_->kpt_bridge();
        const auto& spincomm_m = ctx_->spin_bridge();
        const KPoints* kpoints_m = &ctx_->kpoints();
        int kpt_start_m = ctx_->kpt_start();
        int band_start_m = ctx_->band_start();

        int Nd_d = ctx_->domain().Nd_d();
        int Nband_loc = wfn.Nband();
        int Nspin_local = wfn.Nspin();
        int Nkpts = wfn.Nkpts();
        int nd_ex = halo_m.nd_ex();
        bool is_orth = grid.lattice().is_orthogonal();
        const Mat3& uvec_inv_mgga = grid.lattice().lat_uvec_inv();
        double spin_fac = (Nspin == 1 && wfn.Nspinor() == 1) ? 2.0 : 1.0;
        bool is_kpt = wfn.is_complex();

        std::array<double, 6> stress_mgga = {};

        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = (Nspin == 2) ? s : 0;  // spin index for vtau
            const double* vtau_s = vtau + s_glob * Nd_d;

            for (int k = 0; k < Nkpts; ++k) {
                const auto& occ = wfn.occupations(s, k);
                double wk = kpt_weights[kpt_start_m + k];

                for (int n = 0; n < Nband_loc; ++n) {
                    double fn = occ(band_start_m + n);
                    if (fn < 1e-16) continue;
                    double g_nk = spin_fac * wk * fn;

                    // Compute ∇ψ in 3 directions
                    if (is_kpt) {
                        const Complex* col = wfn.psi_kpt(s, k).col(n);
                        std::vector<Complex> psi_ex(nd_ex);
                        std::vector<Complex> dpsi[3];
                        for (int d = 0; d < 3; d++) dpsi[d].resize(Nd_d);

                        Vec3 kpt = kpoints_m->kpts_cart()[kpt_start_m + k];
                        Vec3 cell_lengths = grid.lattice().lengths();
                        halo_m.execute_kpt(col, psi_ex.data(), 1, kpt, cell_lengths);
                        for (int d = 0; d < 3; d++)
                            gradient_m.apply(psi_ex.data(), dpsi[d].data(), d, 1);

                        // Accumulate stress: σ_ij += -g_nk * ∫ vtau * Re(conj(∇_i ψ) * ∇_j ψ) dV
                        // Voigt: [xx=0, xy=1, xz=2, yy=3, yz=4, zz=5]
                        int voigt[3][3] = {{0,1,2},{1,3,4},{2,4,5}};
                        for (int a = 0; a < 3; a++) {
                            for (int b = a; b < 3; b++) {
                                double sum = 0;
                                if (is_orth) {
                                    for (int i = 0; i < Nd_d; i++)
                                        sum += vtau_s[i] * std::real(std::conj(dpsi[a][i]) * dpsi[b][i]);
                                } else {
                                    // Non-orth: need to transform gradients to Cartesian
                                    for (int i = 0; i < Nd_d; i++) {
                                        Complex ga = uvec_inv_mgga(a,0)*dpsi[0][i] + uvec_inv_mgga(a,1)*dpsi[1][i] + uvec_inv_mgga(a,2)*dpsi[2][i];
                                        Complex gb = uvec_inv_mgga(b,0)*dpsi[0][i] + uvec_inv_mgga(b,1)*dpsi[1][i] + uvec_inv_mgga(b,2)*dpsi[2][i];
                                        sum += vtau_s[i] * std::real(std::conj(ga) * gb);
                                    }
                                }
                                sum *= grid.dV();
                                stress_mgga[voigt[a][b]] += -g_nk * sum;
                            }
                        }
                    } else {
                        const double* col = wfn.psi(s, k).col(n);
                        std::vector<double> psi_ex(nd_ex);
                        std::vector<double> dpsi[3];
                        for (int d = 0; d < 3; d++) dpsi[d].resize(Nd_d);

                        halo_m.execute(col, psi_ex.data(), 1);
                        for (int d = 0; d < 3; d++)
                            gradient_m.apply(psi_ex.data(), dpsi[d].data(), d, 1);

                        int voigt[3][3] = {{0,1,2},{1,3,4},{2,4,5}};
                        for (int a = 0; a < 3; a++) {
                            for (int b = a; b < 3; b++) {
                                double sum = 0;
                                if (is_orth) {
                                    for (int i = 0; i < Nd_d; i++)
                                        sum += vtau_s[i] * dpsi[a][i] * dpsi[b][i];
                                } else {
                                    for (int i = 0; i < Nd_d; i++) {
                                        double ga = uvec_inv_mgga(a,0)*dpsi[0][i] + uvec_inv_mgga(a,1)*dpsi[1][i] + uvec_inv_mgga(a,2)*dpsi[2][i];
                                        double gb = uvec_inv_mgga(b,0)*dpsi[0][i] + uvec_inv_mgga(b,1)*dpsi[1][i] + uvec_inv_mgga(b,2)*dpsi[2][i];
                                        sum += vtau_s[i] * ga * gb;
                                    }
                                }
                                sum *= grid.dV();
                                stress_mgga[voigt[a][b]] += -g_nk * sum;
                            }
                        }
                    }
                }
            }
        }

        // Allreduce over band, kpt, spin communicators
        if (!bandcomm_m.is_null() && bandcomm_m.size() > 1)
            bandcomm_m.allreduce_sum(stress_mgga.data(), 6);
        if (!kptcomm_m.is_null() && kptcomm_m.size() > 1)
            kptcomm_m.allreduce_sum(stress_mgga.data(), 6);
        if (!spincomm_m.is_null() && spincomm_m.size() > 1)
            spincomm_m.allreduce_sum(stress_mgga.data(), 6);

        // Normalize by cell_measure and add to stress_xc
        for (int i = 0; i < 6; i++) {
            stress_mgga[i] /= cell_measure_;
            stress_xc_[i] += stress_mgga[i];
        }
    }

    // Assemble total
    for (int i = 0; i < 6; ++i) {
        stress_total_[i] = stress_k_[i] + stress_xc_[i] + stress_nl_[i] + stress_el_[i];
    }

    return stress_total_;
}

double Stress::pressure() const {
    return -(stress_total_[0] + stress_total_[3] + stress_total_[5]) / 3.0;
}

// ---------------------------------------------------------------------------
// XC stress (matching reference Calculate_XC_stress)
// ---------------------------------------------------------------------------
void Stress::compute_xc_stress(
    const double* rho,
    const double* rho_up,
    const double* rho_dn,
    const double* exc,
    const double* Vxc,
    const double* Dxcdgrho,
    double Exc,
    XCType xc_type,
    int Nspin,
    const double* rho_core,
    const double* vtau,
    const double* tau,
    const double* gpu_tau_vtau_dot) {

    const auto& gradient = ctx_->gradient();
    const auto& halo = ctx_->halo();
    const auto& domain = ctx_->domain();
    const auto& grid = ctx_->grid();

    int Nd_d = domain.Nd_d();
    double dV = grid.dV();
    bool is_orth = grid.lattice().is_orthogonal();

    // Compute Exc_corr = ∫ ρ·Vxc dV  (+ ∫ τ·vtau dV for mGGA)
    // For spin-polarized: ∫ (rho_up·Vxc_up + rho_dn·Vxc_dn) dV
    double Exc_corr = 0.0;
    if (Nspin == 2 && rho_up && rho_dn) {
        for (int i = 0; i < Nd_d; ++i) {
            Exc_corr += rho_up[i] * Vxc[i] + rho_dn[i] * Vxc[Nd_d + i];
        }
    } else {
        for (int i = 0; i < Nd_d; ++i) {
            Exc_corr += rho[i] * Vxc[i];
        }
    }
    Exc_corr *= dV;

    // mGGA: add ∫ τ·vtau dV to Exc_corr (matching SPARC energy.c)
    if (is_mgga_type(xc_type)) {
        if (gpu_tau_vtau_dot) {
            // Use pre-computed GPU value
            Exc_corr += *gpu_tau_vtau_dot;
        } else if (vtau && tau) {
            double Emgga = 0.0;
            if (Nspin == 2) {
                for (int i = 0; i < 2 * Nd_d; ++i) {
                    Emgga += tau[i] * vtau[i];
                }
            } else {
                for (int i = 0; i < Nd_d; ++i) {
                    Emgga += tau[i] * vtau[i];
                }
            }
            Exc_corr += Emgga * dV;
        }
    }

    // Diagonal: Exc - Exc_corr
    double diag_val = Exc - Exc_corr;
    stress_xc_[0] = stress_xc_[3] = stress_xc_[5] = diag_val;
    stress_xc_[1] = stress_xc_[2] = stress_xc_[4] = 0.0;

    // GGA/mGGA gradient correction: -∫ Dxcdgrho · ∂ρ/∂x_α · ∂ρ/∂x_β dV
    bool is_gga = (xc_type == XCType::GGA_PBE || xc_type == XCType::GGA_PBEsol || xc_type == XCType::GGA_RPBE
                   || xc_type == XCType::HYB_PBE0 || xc_type == XCType::HYB_HSE);
    bool needs_grad_stress = is_gga || is_mgga_type(xc_type);
    if (needs_grad_stress && Dxcdgrho) {
        int FDn = gradient.stencil().FDn();
        int nx = domain.Nx_d(), ny = domain.Ny_d(), nz = domain.Nz_d();
        int Nd_ex = (nx+2*FDn) * (ny+2*FDn) * (nz+2*FDn);
        constexpr double xc_rhotol = 1e-14;

        // For spin-polarized: Nspdentd=3, need gradients of [rho_total, rho_up, rho_dn]
        // Dxcdgrho layout: [v2c(Nd_d) | v2x_up(Nd_d) | v2x_dn(Nd_d)]
        // For non-spin: Nspdentd=1, single gradient of rho_total
        // Dxcdgrho layout: [v2x+v2c (Nd_d)]
        int Nspdentd = (Nspin == 2) ? 3 : 1;
        int len_tot = Nspdentd * Nd_d;

        // Build stacked density array [rho_total (+ rho_up + rho_dn for spin)]
        std::vector<double> rho_xc(len_tot);
        // Component 0: total density
        for (int i = 0; i < Nd_d; ++i) {
            rho_xc[i] = rho[i] + (rho_core ? rho_core[i] : 0.0);
            if (rho_xc[i] < xc_rhotol) rho_xc[i] = xc_rhotol;
        }
        if (Nspin == 2 && rho_up && rho_dn) {
            // Component 1: spin-up density
            for (int i = 0; i < Nd_d; ++i) {
                double ru = rho_up[i] + (rho_core ? 0.5 * rho_core[i] : 0.0);
                if (ru < xc_rhotol * 0.5) ru = xc_rhotol * 0.5;
                rho_xc[Nd_d + i] = ru;
            }
            // Component 2: spin-down density
            for (int i = 0; i < Nd_d; ++i) {
                double rd = rho_dn[i] + (rho_core ? 0.5 * rho_core[i] : 0.0);
                if (rd < xc_rhotol * 0.5) rd = xc_rhotol * 0.5;
                rho_xc[2*Nd_d + i] = rd;
            }
        }

        // Compute ∇ρ for all Nspdentd components
        std::vector<double> Drho_x(len_tot), Drho_y(len_tot), Drho_z(len_tot);
        for (int c = 0; c < Nspdentd; ++c) {
            std::vector<double> rho_ex(Nd_ex);
            halo.execute(rho_xc.data() + c * Nd_d, rho_ex.data(), 1);
            gradient.apply(rho_ex.data(), Drho_x.data() + c * Nd_d, 0);
            gradient.apply(rho_ex.data(), Drho_y.data() + c * Nd_d, 1);
            gradient.apply(rho_ex.data(), Drho_z.data() + c * Nd_d, 2);
        }

        // Transform to Cartesian for non-orth cells
        if (!is_orth) {
            nonCart2Cart_grad_arrays(grid.lattice().lat_uvec_inv(),
                                    Drho_x.data(), Drho_y.data(), Drho_z.data(), len_tot);
        }

        // GGA stress correction: -∫ Σ_c Dxcdgrho_c · ∂ρ_c/∂x_α · ∂ρ_c/∂x_β dV
        std::array<double, 6> stress_gga = {};
        for (int i = 0; i < len_tot; ++i) {
            double v2 = Dxcdgrho[i];
            stress_gga[0] += Drho_x[i] * Drho_x[i] * v2;
            stress_gga[1] += Drho_x[i] * Drho_y[i] * v2;
            stress_gga[2] += Drho_x[i] * Drho_z[i] * v2;
            stress_gga[3] += Drho_y[i] * Drho_y[i] * v2;
            stress_gga[4] += Drho_y[i] * Drho_z[i] * v2;
            stress_gga[5] += Drho_z[i] * Drho_z[i] * v2;
        }
        for (int i = 0; i < 6; ++i) stress_gga[i] *= dV;

        for (int i = 0; i < 6; ++i) {
            stress_xc_[i] -= stress_gga[i];
        }
    }

    // Normalize by cell measure
    for (int i = 0; i < 6; ++i) {
        stress_xc_[i] /= cell_measure_;
    }

}

// ---------------------------------------------------------------------------
// NLCC XC stress correction (matching reference Calculate_XC_stress_nlcc)
// ---------------------------------------------------------------------------
void Stress::compute_xc_nlcc_stress(
    const Crystal& crystal,
    const std::vector<AtomInfluence>& influence,
    const double* Vxc,
    int Nspin) {

    const auto& stencil = ctx_->stencil();
    const auto& domain = ctx_->domain();
    const auto& grid = ctx_->grid();

    int FDn = stencil.FDn();
    int order = 2 * FDn;
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    double dV = grid.dV();
    int DMnx = domain.Nx_d(), DMny = domain.Ny_d();
    int Nd_d = domain.Nd_d();
    int xs = domain.vertices().xs, ys = domain.vertices().ys, zs = domain.vertices().zs;

    const double* D1_x = stencil.D1_coeff_x();
    const double* D1_y = stencil.D1_coeff_y();
    const double* D1_z = stencil.D1_coeff_z();

    bool is_orth = grid.lattice().is_orthogonal();
    const Lattice* lattice = &grid.lattice();
    Mat3 uvec_inv, uvec;
    if (!is_orth) {
        uvec_inv = lattice->lat_uvec_inv();
        uvec = lattice->lat_uvec();
    }

    std::array<double, 6> stress_nlcc = {};

    int ntypes = crystal.n_types();
    for (int it = 0; it < ntypes; ++it) {
        const auto& psd = crystal.types()[it].psd();
        if (psd.fchrg() < 1e-10) continue;

        const auto& inf = influence[it];
        const auto& r_grid = psd.radial_grid();
        const auto& rho_c = psd.rho_c();
        const auto& rho_c_d = psd.rho_c_spline_d();
        if (rho_c.empty()) continue;

        double rchrg = r_grid.back();

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            Vec3 pos = inf.coords[iat];

            int i_s = inf.xs[iat], i_e = inf.xe[iat];
            int j_s = inf.ys[iat], j_e = inf.ye[iat];
            int k_s = inf.zs[iat], k_e = inf.ze[iat];
            int lnx = i_e - i_s + 1;
            int lny = j_e - j_s + 1;
            int lnz = k_e - k_s + 1;

            int nxp = lnx + order;
            int nyp = lny + order;
            int nzp = lnz + order;
            int nx2p = nxp + order;
            int ny2p = nyp + order;
            int nz2p = nzp + order;
            int nd_2ex = nx2p * ny2p * nz2p;

            int icor = i_s - order;
            int jcor = j_s - order;
            int kcor = k_s - order;
            double x0_shift = pos.x - dx * icor;
            double y0_shift = pos.y - dy * jcor;
            double z0_shift = pos.z - dz * kcor;

            // Compute radii and interpolate core density on double-extended grid
            std::vector<double> rhocJ(nd_2ex, 0.0);
            std::vector<int> ind_interp;
            std::vector<double> R_interp;

            int count = 0;
            for (int kk = 0; kk < nz2p; ++kk) {
                double rz = kk * dz - z0_shift;
                for (int jj = 0; jj < ny2p; ++jj) {
                    double ry = jj * dy - y0_shift;
                    for (int ii = 0; ii < nx2p; ++ii) {
                        double rx = ii * dx - x0_shift;
                        double r = is_orth ? std::sqrt(rx*rx + ry*ry + rz*rz)
                                           : lattice->metric_distance(rx, ry, rz);
                        if (r <= rchrg) {
                            ind_interp.push_back(count);
                            R_interp.push_back(r);
                        }
                        count++;
                    }
                }
            }

            // Spline interpolation
            if (!R_interp.empty()) {
                std::vector<double> vals;
                Pseudopotential::spline_interp(r_grid, rho_c, rho_c_d, R_interp, vals);
                for (size_t ii = 0; ii < ind_interp.size(); ++ii) {
                    rhocJ[ind_interp[ii]] = vals[ii];
                }
            }

            // Compute gradient of rhocJ on extended grid (rb + FDn)
            int nd_ex = nxp * nyp * nzp;
            std::vector<double> drhocJ_x(nd_ex, 0.0), drhocJ_y(nd_ex, 0.0), drhocJ_z(nd_ex, 0.0);

            for (int kp = 0; kp < nzp; ++kp) {
                int k2p = kp + FDn;
                for (int jp = 0; jp < nyp; ++jp) {
                    int j2p = jp + FDn;
                    for (int ip = 0; ip < nxp; ++ip) {
                        int i2p = ip + FDn;
                        int idx_2ex = i2p + j2p * nx2p + k2p * nx2p * ny2p;
                        int idx_ex = ip + jp * nxp + kp * nxp * nyp;

                        double gx = 0.0, gy = 0.0, gz = 0.0;
                        for (int p = 1; p <= FDn; ++p) {
                            gx += (rhocJ[idx_2ex + p] - rhocJ[idx_2ex - p]) * D1_x[p];
                            gy += (rhocJ[idx_2ex + p * nx2p] - rhocJ[idx_2ex - p * nx2p]) * D1_y[p];
                            gz += (rhocJ[idx_2ex + p * nx2p * ny2p] - rhocJ[idx_2ex - p * nx2p * ny2p]) * D1_z[p];
                        }
                        drhocJ_x[idx_ex] = gx;
                        drhocJ_y[idx_ex] = gy;
                        drhocJ_z[idx_ex] = gz;
                    }
                }
            }

            // Accumulate stress: ∇(ρ_core_J) · (x - R_J) · Vxc
            int di = i_s - xs;
            int dj = j_s - ys;
            int dk = k_s - zs;

            for (int k = 0; k < lnz; ++k) {
                int k_DM = k + dk;
                if (k_DM < 0 || k_DM >= domain.Nz_d()) continue;
                int kp = k + FDn;

                for (int j = 0; j < lny; ++j) {
                    int j_DM = j + dj;
                    if (j_DM < 0 || j_DM >= domain.Ny_d()) continue;
                    int jp = j + FDn;

                    for (int i = 0; i < lnx; ++i) {
                        int i_DM = i + di;
                        if (i_DM < 0 || i_DM >= domain.Nx_d()) continue;
                        int ip = i + FDn;

                        int idx_DM = i_DM + j_DM * DMnx + k_DM * DMnx * DMny;
                        int idx_ex = ip + jp * nxp + kp * nxp * nyp;

                        // Position difference in non-Cart coords
                        double x1 = (i_DM + xs) * dx - pos.x;
                        double x2 = (j_DM + ys) * dy - pos.y;
                        double x3 = (k_DM + zs) * dz - pos.z;

                        double Vxc_val = (Nspin == 2) ?
                            0.5 * (Vxc[idx_DM] + Vxc[Nd_d + idx_DM]) : Vxc[idx_DM];
                        double gx = drhocJ_x[idx_ex];
                        double gy = drhocJ_y[idx_ex];
                        double gz = drhocJ_z[idx_ex];

                        // Transform to Cartesian for non-orth (matching reference lines 432,437)
                        if (!is_orth) {
                            nonCart2Cart_coord(uvec, x1, x2, x3);
                            nonCart2Cart_grad(uvec_inv, gx, gy, gz);
                        }

                        stress_nlcc[0] += gx * x1 * Vxc_val;
                        stress_nlcc[1] += gx * x2 * Vxc_val;
                        stress_nlcc[2] += gx * x3 * Vxc_val;
                        stress_nlcc[3] += gy * x2 * Vxc_val;
                        stress_nlcc[4] += gy * x3 * Vxc_val;
                        stress_nlcc[5] += gz * x3 * Vxc_val;
                    }
                }
            }
        }
    }

    // Multiply by dV
    for (int i = 0; i < 6; ++i) stress_nlcc[i] *= dV;

    // Normalize by cell_measure and add to XC stress
    for (int i = 0; i < 6; ++i) {
        stress_nlcc[i] /= cell_measure_;
        stress_xc_[i] += stress_nlcc[i];
    }
}

// ---------------------------------------------------------------------------
// Electrostatic stress (matching reference Calculate_local_stress)
// ---------------------------------------------------------------------------
void Stress::compute_electrostatic(
    const Crystal& crystal,
    const std::vector<AtomInfluence>& influence,
    const double* phi,
    const double* rho,
    const double* Vloc,
    const double* b_total,
    const double* b_ref_total,
    double Esc) {

    const auto& stencil = ctx_->stencil();
    const auto& gradient = ctx_->gradient();
    const auto& halo = ctx_->halo();
    const auto& domain = ctx_->domain();
    const auto& grid = ctx_->grid();

    int FDn = stencil.FDn();
    int order = 2 * FDn;
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    double dV = grid.dV();
    int nx = domain.Nx_d(), ny = domain.Ny_d(), nz = domain.Nz_d();
    int DMnd = domain.Nd_d();
    int xs = domain.vertices().xs, ys = domain.vertices().ys, zs = domain.vertices().zs;

    bool is_orth = grid.lattice().is_orthogonal();
    const Lattice* lattice = &grid.lattice();
    Mat3 uvec_inv, uvec;
    if (!is_orth) {
        uvec_inv = lattice->lat_uvec_inv();
        uvec = lattice->lat_uvec();
    }
    double inv_4PI = 0.25 / constants::PI;

    const double* D1_x = stencil.D1_coeff_x();
    const double* D1_y = stencil.D1_coeff_y();
    const double* D1_z = stencil.D1_coeff_z();
    const double* D2_x = stencil.D2_coeff_x();
    const double* D2_y = stencil.D2_coeff_y();
    const double* D2_z = stencil.D2_coeff_z();

    // Compute ∇φ on DM domain
    int Nd_ex = (nx+2*FDn) * (ny+2*FDn) * (nz+2*FDn);
    std::vector<double> phi_ex(Nd_ex);
    halo.execute(phi, phi_ex.data(), 1);
    std::vector<double> Dphi_x(DMnd), Dphi_y(DMnd), Dphi_z(DMnd);
    gradient.apply(phi_ex.data(), Dphi_x.data(), 0);
    gradient.apply(phi_ex.data(), Dphi_y.data(), 1);
    gradient.apply(phi_ex.data(), Dphi_z.data(), 2);

    // Transform ∇φ to Cartesian for non-orth (matching reference line 693)
    if (!is_orth) {
        nonCart2Cart_grad_arrays(uvec_inv, Dphi_x.data(), Dphi_y.data(), Dphi_z.data(), DMnd);
    }

    std::array<double, 6> sel = {}, scorr = {};

    // Part 1: (1/4π)|∇φ|² + 0.5·(b-ρ)·φ on diagonal
    for (int i = 0; i < DMnd; ++i) {
        double temp1 = 0.5 * (b_total[i] - rho[i]) * phi[i];
        sel[0] += inv_4PI * Dphi_x[i] * Dphi_x[i] + temp1;
        sel[1] += inv_4PI * Dphi_x[i] * Dphi_y[i];
        sel[2] += inv_4PI * Dphi_x[i] * Dphi_z[i];
        sel[3] += inv_4PI * Dphi_y[i] * Dphi_y[i] + temp1;
        sel[4] += inv_4PI * Dphi_y[i] * Dphi_z[i];
        sel[5] += inv_4PI * Dphi_z[i] * Dphi_z[i] + temp1;
    }

    // Part 2: Per-atom contributions with doubled extended grid
    int ntypes = crystal.n_types();
    for (int it = 0; it < ntypes; ++it) {
        const auto& psd = crystal.types()[it].psd();
        const auto& inf = influence[it];
        const auto& r_grid = psd.radial_grid();
        const auto& rVloc = psd.rVloc();
        const auto& rVloc_d = psd.rVloc_spline_d();
        double Znucl = psd.Zval();
        double rchrg = r_grid.back();
        double rc_ref = 0.5;

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            Vec3 pos = inf.coords[iat];

            int i_s = inf.xs[iat], i_e = inf.xe[iat];
            int j_s = inf.ys[iat], j_e = inf.ye[iat];
            int k_s = inf.zs[iat], k_e = inf.ze[iat];
            int lnx = i_e - i_s + 1;
            int lny = j_e - j_s + 1;
            int lnz = k_e - k_s + 1;

            int nxp = lnx + order;
            int nyp = lny + order;
            int nzp = lnz + order;
            int nx2p = nxp + order;
            int ny2p = nyp + order;
            int nz2p = nzp + order;
            int nd_2ex = nx2p * ny2p * nz2p;
            int nd_ex_loc = nxp * nyp * nzp;

            int icor = i_s - order;
            int jcor = j_s - order;
            int kcor = k_s - order;
            double x0_shift = pos.x - dx * icor;
            double y0_shift = pos.y - dy * jcor;
            double z0_shift = pos.z - dz * kcor;

            std::vector<double> R(nd_2ex);
            std::vector<int> ind_interp;
            std::vector<double> R_interp;

            int count = 0;
            for (int kk = 0; kk < nz2p; ++kk) {
                double rz = kk * dz - z0_shift;
                for (int jj = 0; jj < ny2p; ++jj) {
                    double ry = jj * dy - y0_shift;
                    for (int ii = 0; ii < nx2p; ++ii) {
                        double rx = ii * dx - x0_shift;
                        R[count] = is_orth ? std::sqrt(rx*rx + ry*ry + rz*rz)
                                           : lattice->metric_distance(rx, ry, rz);
                        if (R[count] <= rchrg) {
                            ind_interp.push_back(count);
                            R_interp.push_back(R[count]);
                        }
                        count++;
                    }
                }
            }

            std::vector<double> VJ(nd_2ex), VJ_ref_arr(nd_2ex);
            for (int i = 0; i < nd_2ex; ++i) {
                if (R[i] > rchrg) VJ[i] = -Znucl / R[i];
            }
            if (!R_interp.empty()) {
                std::vector<double> VJ_interp;
                Pseudopotential::spline_interp(r_grid, rVloc, rVloc_d, R_interp, VJ_interp);
                for (size_t idx = 0; idx < ind_interp.size(); ++idx) {
                    if (R_interp[idx] < 1e-10) VJ[ind_interp[idx]] = psd.Vloc_0();
                    else VJ[ind_interp[idx]] = VJ_interp[idx] / R_interp[idx];
                }
            }
            for (int i = 0; i < nd_2ex; ++i) {
                VJ_ref_arr[i] = Electrostatics::V_ref(R[i], rc_ref, Znucl);
            }

            // bJ and bJ_ref on FDn-extended grid
            std::vector<double> bJ(nd_ex_loc, 0.0), bJ_ref(nd_ex_loc, 0.0);
            if (is_orth) {
                Electrostatics::calc_lapV(VJ.data(), bJ.data(), nxp, nyp, nzp,
                                          nx2p, ny2p, nz2p, FDn, D2_x, D2_y, D2_z, -inv_4PI);
                Electrostatics::calc_lapV(VJ_ref_arr.data(), bJ_ref.data(), nxp, nyp, nzp,
                                          nx2p, ny2p, nz2p, FDn, D2_x, D2_y, D2_z, -inv_4PI);
            } else {
                Electrostatics::calc_lapV_nonorth(VJ.data(), bJ.data(), nxp, nyp, nzp,
                                                   nx2p, ny2p, nz2p, FDn, stencil, -inv_4PI);
                Electrostatics::calc_lapV_nonorth(VJ_ref_arr.data(), bJ_ref.data(), nxp, nyp, nzp,
                                                   nx2p, ny2p, nz2p, FDn, stencil, -inv_4PI);
            }

            int dI = i_s - xs, dJ = j_s - ys, dK = k_s - zs;

            for (int kk = 0; kk < lnz; ++kk) {
                int kp = kk + FDn, kp2 = kk + order;
                int k_DM = kk + dK;
                int kshift_DM = k_DM * nx * ny;
                int kshift_p = kp * nxp * nyp;
                int kshift_2p = kp2 * nx2p * ny2p;

                for (int jj = 0; jj < lny; ++jj) {
                    int jp = jj + FDn, jp2 = jj + order;
                    int j_DM = jj + dJ;
                    int jshift_DM = kshift_DM + j_DM * nx;
                    int jshift_p = kshift_p + jp * nxp;
                    int jshift_2p = kshift_2p + jp2 * nx2p;

                    for (int ii = 0; ii < lnx; ++ii) {
                        int ip = ii + FDn, ip2 = ii + order;
                        int i_DM = ii + dI;
                        int ishift_DM = jshift_DM + i_DM;
                        int ishift_p = jshift_p + ip;
                        int ishift_2p = jshift_2p + ip2;

                        double DbJ_x=0, DbJ_y=0, DbJ_z=0;
                        double DbJr_x=0, DbJr_y=0, DbJr_z=0;
                        double DVJ_x=0, DVJ_y=0, DVJ_z=0;
                        double DVJr_x=0, DVJr_y=0, DVJr_z=0;

                        for (int p = 1; p <= FDn; ++p) {
                            DbJ_x += (bJ[ishift_p+p] - bJ[ishift_p-p]) * D1_x[p];
                            DbJ_y += (bJ[ishift_p+p*nxp] - bJ[ishift_p-p*nxp]) * D1_y[p];
                            DbJ_z += (bJ[ishift_p+p*nxp*nyp] - bJ[ishift_p-p*nxp*nyp]) * D1_z[p];
                            DbJr_x += (bJ_ref[ishift_p+p] - bJ_ref[ishift_p-p]) * D1_x[p];
                            DbJr_y += (bJ_ref[ishift_p+p*nxp] - bJ_ref[ishift_p-p*nxp]) * D1_y[p];
                            DbJr_z += (bJ_ref[ishift_p+p*nxp*nyp] - bJ_ref[ishift_p-p*nxp*nyp]) * D1_z[p];
                            DVJ_x += (VJ[ishift_2p+p] - VJ[ishift_2p-p]) * D1_x[p];
                            DVJ_y += (VJ[ishift_2p+p*nx2p] - VJ[ishift_2p-p*nx2p]) * D1_y[p];
                            DVJ_z += (VJ[ishift_2p+p*nx2p*ny2p] - VJ[ishift_2p-p*nx2p*ny2p]) * D1_z[p];
                            DVJr_x += (VJ_ref_arr[ishift_2p+p] - VJ_ref_arr[ishift_2p-p]) * D1_x[p];
                            DVJr_y += (VJ_ref_arr[ishift_2p+p*nx2p] - VJ_ref_arr[ishift_2p-p*nx2p]) * D1_y[p];
                            DVJr_z += (VJ_ref_arr[ishift_2p+p*nx2p*ny2p] - VJ_ref_arr[ishift_2p-p*nx2p*ny2p]) * D1_z[p];
                        }

                        // Position difference in non-Cart coords
                        double x1 = (i_DM + xs) * dx - pos.x;
                        double x2 = (j_DM + ys) * dy - pos.y;
                        double x3 = (k_DM + zs) * dz - pos.z;

                        // Transform to Cartesian for non-orth (matching ref lines 967-971)
                        if (!is_orth) {
                            nonCart2Cart_coord(uvec, x1, x2, x3);
                            nonCart2Cart_grad(uvec_inv, DbJ_x, DbJ_y, DbJ_z);
                            nonCart2Cart_grad(uvec_inv, DbJr_x, DbJr_y, DbJr_z);
                            nonCart2Cart_grad(uvec_inv, DVJ_x, DVJ_y, DVJ_z);
                            nonCart2Cart_grad(uvec_inv, DVJr_x, DVJr_y, DVJr_z);
                        }

                        sel[0] += DbJ_x * x1 * phi[ishift_DM];
                        sel[1] += DbJ_x * x2 * phi[ishift_DM];
                        sel[2] += DbJ_x * x3 * phi[ishift_DM];
                        sel[3] += DbJ_y * x2 * phi[ishift_DM];
                        sel[4] += DbJ_y * x3 * phi[ishift_DM];
                        sel[5] += DbJ_z * x3 * phi[ishift_DM];

                        double t1 = Vloc[ishift_DM] - VJ_ref_arr[ishift_2p];
                        double t2 = Vloc[ishift_DM];
                        double t3 = b_total[ishift_DM] + b_ref_total[ishift_DM];
                        double tx = DbJr_x*t1 + DbJ_x*t2 + (DVJr_x-DVJ_x)*t3 - DVJr_x*bJ_ref[ishift_p];
                        double ty = DbJr_y*t1 + DbJ_y*t2 + (DVJr_y-DVJ_y)*t3 - DVJr_y*bJ_ref[ishift_p];
                        double tz = DbJr_z*t1 + DbJ_z*t2 + (DVJr_z-DVJ_z)*t3 - DVJr_z*bJ_ref[ishift_p];

                        scorr[0] += tx * x1;
                        scorr[1] += tx * x2;
                        scorr[2] += tx * x3;
                        scorr[3] += ty * x2;
                        scorr[4] += ty * x3;
                        scorr[5] += tz * x3;
                    }
                }
            }
        }
    }

    for (int i = 0; i < 6; ++i) {
        stress_el_[i] = (sel[i] + 0.5 * scorr[i]) * dV;
    }

    stress_el_[0] += Esc;
    stress_el_[3] += Esc;
    stress_el_[5] += Esc;

    for (int i = 0; i < 6; ++i) stress_el_[i] /= cell_measure_;
}

// ---------------------------------------------------------------------------
// Nonlocal + kinetic stress (matching reference Calculate_nonlocal_kinetic_stress_linear)
// ---------------------------------------------------------------------------
void Stress::compute_nonlocal_kinetic(
    const LynxContext& ctx,
    const Wavefunction& wfn,
    const Crystal& crystal,
    const std::vector<AtomNlocInfluence>& nloc_influence,
    const NonlocalProjector& vnl,
    const std::vector<double>& kpt_weights) {
    compute_nonlocal_kinetic(wfn, crystal, nloc_influence, vnl,
                             ctx.gradient(), ctx.halo(), ctx.domain(), ctx.grid(),
                             kpt_weights, ctx.scf_bandcomm(), ctx.kpt_bridge(),
                             ctx.spin_bridge(), &ctx.kpoints(),
                             ctx.kpt_start(), ctx.band_start());
}

void Stress::compute_nonlocal_kinetic(
    const Wavefunction& wfn,
    const Crystal& crystal,
    const std::vector<AtomNlocInfluence>& nloc_influence,
    const NonlocalProjector& vnl,
    const Gradient& gradient,
    const HaloExchange& halo,
    const Domain& domain,
    const FDGrid& grid,
    const std::vector<double>& kpt_weights,
    const MPIComm& bandcomm,
    const MPIComm& kptcomm,
    const MPIComm& spincomm,
    const KPoints* kpoints,
    int kpt_start,
    int band_start) {

    using Complex = std::complex<double>;

    int Nspin_local = wfn.Nspin();
    int Nkpts = wfn.Nkpts();
    int Nband = wfn.Nband();  // local band count
    int Nd_d = domain.Nd_d();
    double dV = grid.dV();
    int ntypes = crystal.n_types();
    int n_atom = crystal.n_atom_total();
    int FDn = gradient.stencil().FDn();
    // Determine global Nspin from spincomm (spin_bridge)
    int Nspin_g = Nspin_local;
    if (!spincomm.is_null() && spincomm.size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &Nspin_g, 1, MPI_INT, MPI_SUM, spincomm.comm());
    }
    int Nspinor = wfn.Nspinor();
    double occfac = (Nspinor == 2) ? 1.0 : ((Nspin_g == 1) ? 2.0 : 1.0);
    double spn_fac = occfac * 2.0;
    bool is_kpt = wfn.is_complex();

    bool is_orth = grid.lattice().is_orthogonal();
    Mat3 uvec_inv, uvec;
    if (!is_orth) {
        uvec_inv = grid.lattice().lat_uvec_inv();
        uvec = grid.lattice().lat_uvec();
    }

    Vec3 cell_lengths = grid.lattice().lengths();

    int nx_d = domain.Nx_d(), ny_d = domain.Ny_d(), nz_d = domain.Nz_d();
    int Nd_ex = (nx_d+2*FDn) * (ny_d+2*FDn) * (nz_d+2*FDn);

    stress_k_.fill(0.0);
    stress_nl_.fill(0.0);

    // Build per-unique-atom projector info (same as Forces)
    std::vector<int> IP_displ(n_atom + 1, 0);
    for (int ia = 0; ia < n_atom; ++ia) {
        int it = crystal.type_indices()[ia];
        IP_displ[ia + 1] = IP_displ[ia] + crystal.types()[it].psd().nproj_per_atom();
    }
    int total_nproj = IP_displ[n_atom];

    // Build flat Gamma array
    std::vector<double> Gamma_flat(total_nproj, 0.0);
    for (int ia = 0; ia < n_atom; ++ia) {
        int it = crystal.type_indices()[ia];
        const auto& psd = crystal.types()[it].psd();
        int offset = IP_displ[ia];
        int col = 0;
        for (int l = 0; l <= psd.lmax(); ++l) {
            if (l == psd.lloc()) continue;
            for (int p = 0; p < psd.ppl()[l]; ++p) {
                double g = psd.Gamma()[l][p];
                for (int m = -l; m <= l; ++m) {
                    Gamma_flat[offset + col] = g;
                    col++;
                }
            }
        }
    }

    const auto& Chi = vnl.Chi();

    // --- SOC projector setup ---
    bool has_soc = vnl.has_soc() && (Nspinor == 2);
    const auto& Chi_soc = vnl.Chi_soc();
    const auto& soc_proj_info = vnl.soc_proj_info();

    // Build SOC per-unique-atom projector displacement and flat Gamma_soc
    std::vector<int> IP_displ_soc;
    std::vector<double> Gamma_soc_flat;
    int total_nproj_soc = 0;
    if (has_soc) {
        IP_displ_soc.resize(n_atom + 1, 0);
        for (int ia = 0; ia < n_atom; ++ia) {
            int it = crystal.type_indices()[ia];
            const auto& psd = crystal.types()[it].psd();
            int nproj_soc = 0;
            if (psd.has_soc()) {
                for (int l = 1; l <= psd.lmax(); ++l)
                    nproj_soc += psd.ppl_soc()[l] * (2 * l + 1);
            }
            IP_displ_soc[ia + 1] = IP_displ_soc[ia] + nproj_soc;
        }
        total_nproj_soc = IP_displ_soc[n_atom];

        Gamma_soc_flat.resize(total_nproj_soc, 0.0);
        for (int ia = 0; ia < n_atom; ++ia) {
            int it = crystal.type_indices()[ia];
            const auto& psd = crystal.types()[it].psd();
            if (!psd.has_soc()) continue;
            int offset = IP_displ_soc[ia];
            int col = 0;
            for (int l = 1; l <= psd.lmax(); ++l) {
                for (int p = 0; p < psd.ppl_soc()[l]; ++p) {
                    double g = psd.Gamma_soc()[l][p];
                    for (int m = -l; m <= l; ++m) {
                        Gamma_soc_flat[offset + col] = g;
                        col++;
                    }
                }
            }
        }
    }

    double energy_nl = 0.0;
    double energy_soc = 0.0;
    std::array<double, 6> sk = {}, snl = {}, snl_soc = {};

    for (int s = 0; s < Nspin_local; ++s) {
        for (int k = 0; k < Nkpts; ++k) {
            int k_glob = kpt_start + k;
            double wk = kpt_weights[k_glob];
            const auto& occ_sk = wfn.occupations(s, k);

            if (is_kpt) {
                // ===== Complex k-point path =====
                const auto& psi_sk = wfn.psi_kpt(s, k);
                Vec3 kpt_cart = kpoints->kpts_cart()[k_glob];

                for (int n = 0; n < Nband; ++n) {
                    double g_n = occ_sk(band_start + n);
                    const Complex* psi_n = psi_sk.col(n);

                    // For spinor: psi_n has layout [psi_up(Nd_d) | psi_dn(Nd_d)]
                    // Compute gradients for each spinor component and sum kinetic stress
                    if (Nspinor == 2) {
                        // --- Spinor kinetic stress ---
                        for (int sp = 0; sp < 2; ++sp) {
                            const Complex* psi_sp = psi_n + sp * Nd_d;

                            std::vector<Complex> dpsi_cart_sp[3];
                            if (is_orth) {
                                for (int dim = 0; dim < 3; ++dim) {
                                    dpsi_cart_sp[dim].resize(Nd_d);
                                    std::vector<Complex> psi_ex(Nd_ex);
                                    halo.execute_kpt(psi_sp, psi_ex.data(), 1, kpt_cart, cell_lengths);
                                    gradient.apply(psi_ex.data(), dpsi_cart_sp[dim].data(), dim);
                                }
                            } else {
                                std::vector<Complex> dpsi_nc[3];
                                for (int dim = 0; dim < 3; ++dim) {
                                    dpsi_nc[dim].resize(Nd_d);
                                    std::vector<Complex> psi_ex(Nd_ex);
                                    halo.execute_kpt(psi_sp, psi_ex.data(), 1, kpt_cart, cell_lengths);
                                    gradient.apply(psi_ex.data(), dpsi_nc[dim].data(), dim);
                                }
                                for (int dim = 0; dim < 3; ++dim) dpsi_cart_sp[dim].resize(Nd_d);
                                for (int i = 0; i < Nd_d; ++i) {
                                    Complex d0 = dpsi_nc[0][i], d1 = dpsi_nc[1][i], d2 = dpsi_nc[2][i];
                                    dpsi_cart_sp[0][i] = uvec_inv(0,0)*d0 + uvec_inv(0,1)*d1 + uvec_inv(0,2)*d2;
                                    dpsi_cart_sp[1][i] = uvec_inv(1,0)*d0 + uvec_inv(1,1)*d1 + uvec_inv(1,2)*d2;
                                    dpsi_cart_sp[2][i] = uvec_inv(2,0)*d0 + uvec_inv(2,1)*d1 + uvec_inv(2,2)*d2;
                                }
                            }

                            // Accumulate kinetic stress from this spinor component
                            // occfac=1.0 for SOC (both spins present in one wavefunction)
                            int cnt = 0;
                            for (int dim = 0; dim < 3; ++dim) {
                                for (int dim2 = dim; dim2 < 3; ++dim2) {
                                    double dot = 0.0;
                                    for (int i = 0; i < Nd_d; ++i)
                                        dot += std::real(std::conj(dpsi_cart_sp[dim][i]) * dpsi_cart_sp[dim2][i]);
                                    sk[cnt] -= 1.0 * wk * g_n * dot * dV;
                                    cnt++;
                                }
                            }
                        }
                    } else {
                        // --- Scalar-relativistic kinetic stress (original path) ---
                        std::vector<Complex> dpsi_cart[3];
                        if (is_orth) {
                            for (int dim = 0; dim < 3; ++dim) {
                                dpsi_cart[dim].resize(Nd_d);
                                std::vector<Complex> psi_ex(Nd_ex);
                                halo.execute_kpt(psi_n, psi_ex.data(), 1, kpt_cart, cell_lengths);
                                gradient.apply(psi_ex.data(), dpsi_cart[dim].data(), dim);
                            }
                        } else {
                            std::vector<Complex> dpsi_nc[3];
                            for (int dim = 0; dim < 3; ++dim) {
                                dpsi_nc[dim].resize(Nd_d);
                                std::vector<Complex> psi_ex(Nd_ex);
                                halo.execute_kpt(psi_n, psi_ex.data(), 1, kpt_cart, cell_lengths);
                                gradient.apply(psi_ex.data(), dpsi_nc[dim].data(), dim);
                            }
                            for (int dim = 0; dim < 3; ++dim) dpsi_cart[dim].resize(Nd_d);
                            for (int i = 0; i < Nd_d; ++i) {
                                Complex d0 = dpsi_nc[0][i], d1 = dpsi_nc[1][i], d2 = dpsi_nc[2][i];
                                dpsi_cart[0][i] = uvec_inv(0,0)*d0 + uvec_inv(0,1)*d1 + uvec_inv(0,2)*d2;
                                dpsi_cart[1][i] = uvec_inv(1,0)*d0 + uvec_inv(1,1)*d1 + uvec_inv(1,2)*d2;
                                dpsi_cart[2][i] = uvec_inv(2,0)*d0 + uvec_inv(2,1)*d1 + uvec_inv(2,2)*d2;
                            }
                        }

                        int cnt = 0;
                        for (int dim = 0; dim < 3; ++dim) {
                            for (int dim2 = dim; dim2 < 3; ++dim2) {
                                double dot = 0.0;
                                for (int i = 0; i < Nd_d; ++i)
                                    dot += std::real(std::conj(dpsi_cart[dim][i]) * dpsi_cart[dim2][i]);
                                sk[cnt] -= occfac * wk * g_n * dot * dV;
                                cnt++;
                            }
                        }
                    }

                    // --- Scalar-relativistic nonlocal stress ---
                    if (total_nproj > 0) {
                        // For spinor: project each spinor component separately,
                        // then sum contributions (same Gamma, same Chi)
                        // psi_n for non-spinor is Nd_d, for spinor we use psi_up and psi_dn
                        int n_spinor_comp = (Nspinor == 2) ? 2 : 1;

                        // alpha per spinor component
                        std::vector<std::vector<Complex>> alpha_sp(n_spinor_comp,
                            std::vector<Complex>(total_nproj, Complex(0.0, 0.0)));

                        for (int sp = 0; sp < n_spinor_comp; ++sp) {
                            const Complex* psi_sp = psi_n + sp * Nd_d;
                            for (int it = 0; it < ntypes; ++it) {
                                const auto& inf = nloc_influence[it];
                                int nproj = crystal.types()[it].psd().nproj_per_atom();
                                if (nproj == 0) continue;
                                for (int iat = 0; iat < inf.n_atom; ++iat) {
                                    int ndc = inf.ndc[iat];
                                    if (ndc == 0) continue;
                                    int orig_atom = inf.atom_index[iat];
                                    int offset = IP_displ[orig_atom];
                                    const auto& gpos = inf.grid_pos[iat];
                                    const auto& chi_iat = Chi[it][iat];

                                    const Vec3& shift = inf.image_shift[iat];
                                    double theta = -(kpt_cart.x * shift.x + kpt_cart.y * shift.y + kpt_cart.z * shift.z);
                                    Complex bloch_fac(std::cos(theta), std::sin(theta));
                                    Complex alpha_scale = bloch_fac * dV;

                                    for (int jp = 0; jp < nproj; ++jp) {
                                        Complex dot(0.0, 0.0);
                                        for (int ig = 0; ig < ndc; ++ig)
                                            dot += chi_iat(ig, jp) * psi_sp[gpos[ig]];
                                        alpha_sp[sp][offset + jp] += alpha_scale * dot;
                                    }
                                }
                            }
                        }

                        // Nonlocal energy: sum over spinor components
                        for (int sp = 0; sp < n_spinor_comp; ++sp) {
                            for (int ia = 0; ia < n_atom; ++ia) {
                                int off = IP_displ[ia];
                                int np = IP_displ[ia + 1] - off;
                                for (int jp = 0; jp < np; ++jp)
                                    energy_nl += wk * g_n * Gamma_flat[off + jp] * std::norm(alpha_sp[sp][off + jp]);
                            }
                        }

                        // Nonlocal stress: compute gradients per spinor component, sum beta contributions
                        // We need gradients of each spinor component for the beta inner product
                        int cnt = 0;
                        for (int dim = 0; dim < 3; ++dim) {
                            for (int dim2 = dim; dim2 < 3; ++dim2) {
                                // Accumulate beta across spinor components
                                for (int sp = 0; sp < n_spinor_comp; ++sp) {
                                    const Complex* psi_sp = psi_n + sp * Nd_d;

                                    // Compute gradient of this spinor component in direction dim
                                    std::vector<Complex> dpsi_dim(Nd_d);
                                    if (is_orth) {
                                        std::vector<Complex> psi_ex(Nd_ex);
                                        halo.execute_kpt(psi_sp, psi_ex.data(), 1, kpt_cart, cell_lengths);
                                        gradient.apply(psi_ex.data(), dpsi_dim.data(), dim);
                                    } else {
                                        std::vector<Complex> dpsi_nc[3];
                                        for (int d = 0; d < 3; ++d) {
                                            dpsi_nc[d].resize(Nd_d);
                                            std::vector<Complex> psi_ex(Nd_ex);
                                            halo.execute_kpt(psi_sp, psi_ex.data(), 1, kpt_cart, cell_lengths);
                                            gradient.apply(psi_ex.data(), dpsi_nc[d].data(), d);
                                        }
                                        // Transform to Cartesian and extract component dim
                                        for (int i = 0; i < Nd_d; ++i) {
                                            Complex d0 = dpsi_nc[0][i], d1 = dpsi_nc[1][i], d2 = dpsi_nc[2][i];
                                            dpsi_dim[i] = uvec_inv(dim,0)*d0 + uvec_inv(dim,1)*d1 + uvec_inv(dim,2)*d2;
                                        }
                                    }

                                    std::vector<Complex> beta(total_nproj, Complex(0.0, 0.0));
                                    for (int it = 0; it < ntypes; ++it) {
                                        const auto& inf = nloc_influence[it];
                                        int nproj = crystal.types()[it].psd().nproj_per_atom();
                                        if (nproj == 0) continue;
                                        for (int iat = 0; iat < inf.n_atom; ++iat) {
                                            int ndc = inf.ndc[iat];
                                            if (ndc == 0) continue;
                                            int orig_atom = inf.atom_index[iat];
                                            int offset = IP_displ[orig_atom];
                                            const auto& gpos = inf.grid_pos[iat];
                                            const auto& chi_iat = Chi[it][iat];
                                            Vec3 ap = inf.coords[iat];

                                            const Vec3& shift = inf.image_shift[iat];
                                            double theta = -(kpt_cart.x * shift.x + kpt_cart.y * shift.y + kpt_cart.z * shift.z);
                                            Complex bloch_fac(std::cos(theta), std::sin(theta));
                                            Complex beta_scale = bloch_fac * dV;

                                            for (int jp = 0; jp < nproj; ++jp) {
                                                Complex dot(0.0, 0.0);
                                                for (int ig = 0; ig < ndc; ++ig) {
                                                    int flat = gpos[ig];
                                                    int li = flat % nx_d;
                                                    int lj = (flat / nx_d) % ny_d;
                                                    int lk = flat / (nx_d * ny_d);

                                                    double r1 = (li + domain.vertices().xs) * grid.dx() - ap.x;
                                                    double r2 = (lj + domain.vertices().ys) * grid.dy() - ap.y;
                                                    double r3 = (lk + domain.vertices().zs) * grid.dz() - ap.z;
                                                    if (!is_orth) nonCart2Cart_coord(uvec, r1, r2, r3);

                                                    double xR = (dim2 == 0) ? r1 : (dim2 == 1) ? r2 : r3;
                                                    dot += chi_iat(ig, jp) * xR * dpsi_dim[gpos[ig]];
                                                }
                                                beta[offset + jp] += beta_scale * dot;
                                            }
                                        }
                                    }
                                    // Re(conj(alpha_sp) * beta)
                                    for (int ia = 0; ia < n_atom; ++ia) {
                                        int off = IP_displ[ia];
                                        int np = IP_displ[ia + 1] - off;
                                        for (int jp = 0; jp < np; ++jp)
                                            snl[cnt] -= Gamma_flat[off + jp] *
                                                        std::real(std::conj(alpha_sp[sp][off + jp]) * beta[off + jp]) *
                                                        wk * g_n;
                                    }
                                }
                                cnt++;
                            }
                        }
                    } // end scalar-relativistic nonlocal

                    // --- SOC nonlocal stress ---
                    if (has_soc && total_nproj_soc > 0) {
                        const Complex* psi_up = psi_n;
                        const Complex* psi_dn = psi_n + Nd_d;

                        // Compute SOC alpha for spin-up and spin-down
                        std::vector<Complex> alpha_soc_up(total_nproj_soc, Complex(0.0, 0.0));
                        std::vector<Complex> alpha_soc_dn(total_nproj_soc, Complex(0.0, 0.0));

                        for (int it = 0; it < ntypes; ++it) {
                            const auto& psd = crystal.types()[it].psd();
                            if (!psd.has_soc()) continue;
                            const auto& inf = nloc_influence[it];
                            int nproj_soc = static_cast<int>(soc_proj_info[it].size());
                            if (nproj_soc == 0) continue;

                            for (int iat = 0; iat < inf.n_atom; ++iat) {
                                int ndc = inf.ndc[iat];
                                if (ndc == 0) continue;
                                int orig_atom = inf.atom_index[iat];
                                int offset = IP_displ_soc[orig_atom];
                                const auto& gpos = inf.grid_pos[iat];

                                const Vec3& shift = inf.image_shift[iat];
                                double theta = -(kpt_cart.x * shift.x + kpt_cart.y * shift.y + kpt_cart.z * shift.z);
                                Complex bloch_fac(std::cos(theta), std::sin(theta));
                                Complex alpha_scale = bloch_fac * dV;

                                for (int jp = 0; jp < nproj_soc; ++jp) {
                                    Complex dot_up(0.0), dot_dn(0.0);
                                    for (int ig = 0; ig < ndc; ++ig) {
                                        Complex chi_val = std::conj(Chi_soc[it][iat](ig, jp));
                                        dot_up += chi_val * psi_up[gpos[ig]];
                                        dot_dn += chi_val * psi_dn[gpos[ig]];
                                    }
                                    alpha_soc_up[offset + jp] += alpha_scale * dot_up;
                                    alpha_soc_dn[offset + jp] += alpha_scale * dot_dn;
                                }
                            }
                        }

                        // SOC nonlocal energy for diagonal subtraction
                        // Term 1 (Lz*Sz): 0.5 * m * Gamma_soc * (|alpha_up|^2 - |alpha_dn|^2)
                        // Term 2 (L+S-): 0.5 * ladder * Gamma_soc * Re(conj(alpha_up(m)) * alpha_dn(m+1))
                        // Term 2 (L-S+): 0.5 * ladder * Gamma_soc * Re(conj(alpha_dn(m)) * alpha_up(m-1))
                        for (int ia = 0; ia < n_atom; ++ia) {
                            int it_idx = crystal.type_indices()[ia];
                            const auto& psd = crystal.types()[it_idx].psd();
                            if (!psd.has_soc()) continue;
                            int off = IP_displ_soc[ia];
                            const auto& proj_info = soc_proj_info[it_idx];
                            int nproj_soc = static_cast<int>(proj_info.size());

                            for (int jp = 0; jp < nproj_soc; ++jp) {
                                int l = proj_info[jp].l;
                                int m = proj_info[jp].m;
                                double gamma_soc = Gamma_soc_flat[off + jp];

                                // Term 1: 0.5 * m * gamma_soc * (|alpha_up|^2 - |alpha_dn|^2)
                                if (m != 0) {
                                    energy_soc += wk * g_n * 0.5 * static_cast<double>(m) * gamma_soc *
                                        (std::norm(alpha_soc_up[off + jp]) - std::norm(alpha_soc_dn[off + jp]));
                                }

                                // Term 2: L+S- (m -> m+1 for dn->up coupling)
                                if (m + 1 <= l) {
                                    double ladder = std::sqrt(static_cast<double>(l*(l+1) - m*(m+1)));
                                    int jp_shifted = jp + 1;
                                    energy_soc += wk * g_n * 0.5 * ladder * gamma_soc *
                                        std::real(std::conj(alpha_soc_up[off + jp]) * alpha_soc_dn[off + jp_shifted]);
                                }

                                // Term 2: L-S+ (m -> m-1 for up->dn coupling)
                                if (m - 1 >= -l) {
                                    double ladder = std::sqrt(static_cast<double>(l*(l+1) - m*(m-1)));
                                    int jp_shifted = jp - 1;
                                    energy_soc += wk * g_n * 0.5 * ladder * gamma_soc *
                                        std::real(std::conj(alpha_soc_dn[off + jp]) * alpha_soc_up[off + jp_shifted]);
                                }
                            }
                        }

                        // SOC nonlocal stress tensor
                        int cnt = 0;
                        for (int dim = 0; dim < 3; ++dim) {
                            for (int dim2 = dim; dim2 < 3; ++dim2) {
                                // Compute gradient of each spinor component in direction dim
                                std::vector<Complex> dpsi_up_dim(Nd_d), dpsi_dn_dim(Nd_d);
                                for (int sp = 0; sp < 2; ++sp) {
                                    const Complex* psi_sp = (sp == 0) ? psi_up : psi_dn;
                                    std::vector<Complex>& dpsi_dim = (sp == 0) ? dpsi_up_dim : dpsi_dn_dim;

                                    if (is_orth) {
                                        std::vector<Complex> psi_ex(Nd_ex);
                                        halo.execute_kpt(psi_sp, psi_ex.data(), 1, kpt_cart, cell_lengths);
                                        gradient.apply(psi_ex.data(), dpsi_dim.data(), dim);
                                    } else {
                                        std::vector<Complex> dpsi_nc[3];
                                        for (int d = 0; d < 3; ++d) {
                                            dpsi_nc[d].resize(Nd_d);
                                            std::vector<Complex> psi_ex(Nd_ex);
                                            halo.execute_kpt(psi_sp, psi_ex.data(), 1, kpt_cart, cell_lengths);
                                            gradient.apply(psi_ex.data(), dpsi_nc[d].data(), d);
                                        }
                                        for (int i = 0; i < Nd_d; ++i) {
                                            Complex d0 = dpsi_nc[0][i], d1 = dpsi_nc[1][i], d2 = dpsi_nc[2][i];
                                            dpsi_dim[i] = uvec_inv(dim,0)*d0 + uvec_inv(dim,1)*d1 + uvec_inv(dim,2)*d2;
                                        }
                                    }
                                }

                                // Compute SOC beta_up and beta_dn:
                                // beta_sp = bloch_fac * dV * Chi_soc^T * (x-R)_dim2 * dpsi_sp/dx_dim
                                std::vector<Complex> beta_soc_up(total_nproj_soc, Complex(0.0, 0.0));
                                std::vector<Complex> beta_soc_dn(total_nproj_soc, Complex(0.0, 0.0));

                                for (int it = 0; it < ntypes; ++it) {
                                    const auto& psd = crystal.types()[it].psd();
                                    if (!psd.has_soc()) continue;
                                    const auto& inf = nloc_influence[it];
                                    int nproj_soc = static_cast<int>(soc_proj_info[it].size());
                                    if (nproj_soc == 0) continue;

                                    for (int iat = 0; iat < inf.n_atom; ++iat) {
                                        int ndc = inf.ndc[iat];
                                        if (ndc == 0) continue;
                                        int orig_atom = inf.atom_index[iat];
                                        int offset = IP_displ_soc[orig_atom];
                                        const auto& gpos = inf.grid_pos[iat];
                                        Vec3 ap = inf.coords[iat];

                                        const Vec3& shift = inf.image_shift[iat];
                                        double theta = -(kpt_cart.x * shift.x + kpt_cart.y * shift.y + kpt_cart.z * shift.z);
                                        Complex bloch_fac(std::cos(theta), std::sin(theta));
                                        Complex beta_scale = bloch_fac * dV;

                                        for (int jp = 0; jp < nproj_soc; ++jp) {
                                            Complex dot_up(0.0), dot_dn(0.0);
                                            for (int ig = 0; ig < ndc; ++ig) {
                                                int flat = gpos[ig];
                                                int li = flat % nx_d;
                                                int lj = (flat / nx_d) % ny_d;
                                                int lk = flat / (nx_d * ny_d);

                                                double r1 = (li + domain.vertices().xs) * grid.dx() - ap.x;
                                                double r2 = (lj + domain.vertices().ys) * grid.dy() - ap.y;
                                                double r3 = (lk + domain.vertices().zs) * grid.dz() - ap.z;
                                                if (!is_orth) nonCart2Cart_coord(uvec, r1, r2, r3);

                                                double xR = (dim2 == 0) ? r1 : (dim2 == 1) ? r2 : r3;
                                                Complex chi_val = std::conj(Chi_soc[it][iat](ig, jp));
                                                dot_up += chi_val * xR * dpsi_up_dim[gpos[ig]];
                                                dot_dn += chi_val * xR * dpsi_dn_dim[gpos[ig]];
                                            }
                                            beta_soc_up[offset + jp] += beta_scale * dot_up;
                                            beta_soc_dn[offset + jp] += beta_scale * dot_dn;
                                        }
                                    }
                                }

                                // Accumulate SOC nonlocal stress
                                // Term 1 (Lz*Sz): 0.5*m*Gamma_soc * [Re(conj(alpha_up)*beta_up) - Re(conj(alpha_dn)*beta_dn)]
                                // Term 2 (L+S-): 0.5*ladder*Gamma_soc * Re(conj(alpha_up(m))*beta_dn(m+1))
                                // Term 2 (L-S+): 0.5*ladder*Gamma_soc * Re(conj(alpha_dn(m))*beta_up(m-1))
                                for (int ia = 0; ia < n_atom; ++ia) {
                                    int it_idx = crystal.type_indices()[ia];
                                    const auto& psd = crystal.types()[it_idx].psd();
                                    if (!psd.has_soc()) continue;
                                    int off = IP_displ_soc[ia];
                                    const auto& proj_info = soc_proj_info[it_idx];
                                    int nproj_soc = static_cast<int>(proj_info.size());

                                    for (int jp = 0; jp < nproj_soc; ++jp) {
                                        int l = proj_info[jp].l;
                                        int m = proj_info[jp].m;
                                        double gamma_soc = Gamma_soc_flat[off + jp];

                                        // Term 1: on-diagonal (Lz*Sz)
                                        if (m != 0) {
                                            double val = 0.5 * static_cast<double>(m) * gamma_soc * (
                                                std::real(std::conj(alpha_soc_up[off + jp]) * beta_soc_up[off + jp]) -
                                                std::real(std::conj(alpha_soc_dn[off + jp]) * beta_soc_dn[off + jp]));
                                            snl[cnt] -= val * wk * g_n; snl_soc[cnt] -= val * wk * g_n;
                                        }

                                        // Term 2: L+S- (raises m, dn->up coupling)
                                        if (m + 1 <= l) {
                                            double ladder = std::sqrt(static_cast<double>(l*(l+1) - m*(m+1)));
                                            int jp_shifted = jp + 1;
                                            double val = 0.5 * ladder * gamma_soc *
                                                std::real(std::conj(alpha_soc_up[off + jp]) * beta_soc_dn[off + jp_shifted]);
                                            snl[cnt] -= val * wk * g_n; snl_soc[cnt] -= val * wk * g_n;
                                        }

                                        // Term 2: L-S+ (lowers m, up->dn coupling)
                                        if (m - 1 >= -l) {
                                            double ladder = std::sqrt(static_cast<double>(l*(l+1) - m*(m-1)));
                                            int jp_shifted = jp - 1;
                                            double val = 0.5 * ladder * gamma_soc *
                                                std::real(std::conj(alpha_soc_dn[off + jp]) * beta_soc_up[off + jp_shifted]);
                                            snl[cnt] -= val * wk * g_n; snl_soc[cnt] -= val * wk * g_n;
                                        }
                                    }
                                }
                                cnt++;
                            }
                        }
                    } // end SOC nonlocal stress
                }
            } else {
                // ===== Real gamma-point path =====
                const auto& psi_sk = wfn.psi(s, k);

                for (int n = 0; n < Nband; ++n) {
                    double g_n = occ_sk(band_start + n);
                    const double* psi_n = psi_sk.col(n);

                    std::vector<double> dpsi_cart[3];
                    if (is_orth) {
                        for (int dim = 0; dim < 3; ++dim) {
                            dpsi_cart[dim].resize(Nd_d);
                            std::vector<double> psi_ex(Nd_ex);
                            halo.execute(psi_n, psi_ex.data(), 1);
                            gradient.apply(psi_ex.data(), dpsi_cart[dim].data(), dim);
                        }
                    } else {
                        std::vector<double> dpsi_nc[3];
                        for (int dim = 0; dim < 3; ++dim) {
                            dpsi_nc[dim].resize(Nd_d);
                            std::vector<double> psi_ex(Nd_ex);
                            halo.execute(psi_n, psi_ex.data(), 1);
                            gradient.apply(psi_ex.data(), dpsi_nc[dim].data(), dim);
                        }
                        for (int dim = 0; dim < 3; ++dim) dpsi_cart[dim].resize(Nd_d);
                        for (int i = 0; i < Nd_d; ++i) {
                            double d0 = dpsi_nc[0][i], d1 = dpsi_nc[1][i], d2 = dpsi_nc[2][i];
                            dpsi_cart[0][i] = uvec_inv(0,0)*d0 + uvec_inv(0,1)*d1 + uvec_inv(0,2)*d2;
                            dpsi_cart[1][i] = uvec_inv(1,0)*d0 + uvec_inv(1,1)*d1 + uvec_inv(1,2)*d2;
                            dpsi_cart[2][i] = uvec_inv(2,0)*d0 + uvec_inv(2,1)*d1 + uvec_inv(2,2)*d2;
                        }
                    }

                    // Kinetic stress
                    int cnt = 0;
                    for (int dim = 0; dim < 3; ++dim) {
                        for (int dim2 = dim; dim2 < 3; ++dim2) {
                            double dot = 0.0;
                            for (int i = 0; i < Nd_d; ++i) dot += dpsi_cart[dim][i] * dpsi_cart[dim2][i];
                            sk[cnt] -= occfac * wk * g_n * dot * dV;
                            cnt++;
                        }
                    }

                    if (total_nproj == 0) continue;

                    // Real alpha
                    std::vector<double> alpha(total_nproj, 0.0);
                    for (int it = 0; it < ntypes; ++it) {
                        const auto& inf = nloc_influence[it];
                        int nproj = crystal.types()[it].psd().nproj_per_atom();
                        if (nproj == 0) continue;
                        for (int iat = 0; iat < inf.n_atom; ++iat) {
                            int ndc = inf.ndc[iat];
                            if (ndc == 0) continue;
                            int orig_atom = inf.atom_index[iat];
                            int offset = IP_displ[orig_atom];
                            const auto& gpos = inf.grid_pos[iat];
                            const auto& chi_iat = Chi[it][iat];
                            for (int jp = 0; jp < nproj; ++jp) {
                                double dot = 0.0;
                                for (int ig = 0; ig < ndc; ++ig)
                                    dot += chi_iat(ig, jp) * psi_n[gpos[ig]];
                                alpha[offset + jp] += dot * dV;
                            }
                        }
                    }

                    // Nonlocal energy
                    for (int ia = 0; ia < n_atom; ++ia) {
                        int off = IP_displ[ia];
                        int np = IP_displ[ia + 1] - off;
                        for (int jp = 0; jp < np; ++jp)
                            energy_nl += wk * g_n * Gamma_flat[off + jp] * alpha[off + jp] * alpha[off + jp];
                    }

                    // Nonlocal stress
                    cnt = 0;
                    for (int dim = 0; dim < 3; ++dim) {
                        for (int dim2 = dim; dim2 < 3; ++dim2) {
                            std::vector<double> beta(total_nproj, 0.0);
                            for (int it = 0; it < ntypes; ++it) {
                                const auto& inf = nloc_influence[it];
                                int nproj = crystal.types()[it].psd().nproj_per_atom();
                                if (nproj == 0) continue;
                                for (int iat = 0; iat < inf.n_atom; ++iat) {
                                    int ndc = inf.ndc[iat];
                                    if (ndc == 0) continue;
                                    int orig_atom = inf.atom_index[iat];
                                    int offset = IP_displ[orig_atom];
                                    const auto& gpos = inf.grid_pos[iat];
                                    const auto& chi_iat = Chi[it][iat];
                                    Vec3 ap = inf.coords[iat];

                                    for (int jp = 0; jp < nproj; ++jp) {
                                        double dot = 0.0;
                                        for (int ig = 0; ig < ndc; ++ig) {
                                            int flat = gpos[ig];
                                            int li = flat % nx_d;
                                            int lj = (flat / nx_d) % ny_d;
                                            int lk = flat / (nx_d * ny_d);

                                            double r1 = (li + domain.vertices().xs) * grid.dx() - ap.x;
                                            double r2 = (lj + domain.vertices().ys) * grid.dy() - ap.y;
                                            double r3 = (lk + domain.vertices().zs) * grid.dz() - ap.z;
                                            if (!is_orth) nonCart2Cart_coord(uvec, r1, r2, r3);

                                            double xR = (dim2 == 0) ? r1 : (dim2 == 1) ? r2 : r3;
                                            dot += chi_iat(ig, jp) * xR * dpsi_cart[dim][gpos[ig]];
                                        }
                                        beta[offset + jp] += dot * dV;
                                    }
                                }
                            }
                            for (int ia = 0; ia < n_atom; ++ia) {
                                int off = IP_displ[ia];
                                int np = IP_displ[ia + 1] - off;
                                for (int jp = 0; jp < np; ++jp)
                                    snl[cnt] -= Gamma_flat[off + jp] * alpha[off + jp] * beta[off + jp] * wk * g_n;
                            }
                            cnt++;
                        }
                    }
                }
            }
        }
    }

    // Scale nonlocal by spn_fac
    // For SOC (Nspinor==2): spn_fac = 1.0 * 2.0 = 2.0
    for (int i = 0; i < 6; ++i) snl[i] *= spn_fac;
    for (int i = 0; i < 6; ++i) snl_soc[i] *= 2.0;  // SOC spn_fac = 2.0
    energy_nl *= occfac;
    energy_soc *= 2.0;  // SOC energy scaled by 2.0 (spn_fac for SOC)

    // Debug: print NL energy
    // Subtract E_nl + E_soc from diagonal
    double energy_diag = energy_nl + energy_soc;
    stress_nl_[0] = snl[0] - energy_diag;
    stress_nl_[1] = snl[1];
    stress_nl_[2] = snl[2];
    stress_nl_[3] = snl[3] - energy_diag;
    stress_nl_[4] = snl[4];
    stress_nl_[5] = snl[5] - energy_diag;
    for (int i = 0; i < 6; ++i) stress_k_[i] = sk[i];

    // SOC-only stress (for GPU comparison)
    stress_soc_[0] = snl_soc[0] - energy_soc;
    stress_soc_[1] = snl_soc[1];
    stress_soc_[2] = snl_soc[2];
    stress_soc_[3] = snl_soc[3] - energy_soc;
    stress_soc_[4] = snl_soc[4];
    stress_soc_[5] = snl_soc[5] - energy_soc;
    energy_soc_ = energy_soc;

    // Allreduce across spin, kpt, band
    auto allreduce6 = [](std::array<double,6>& arr, const MPIComm& comm) {
        if (!comm.is_null() && comm.size() > 1)
            MPI_Allreduce(MPI_IN_PLACE, arr.data(), 6, MPI_DOUBLE, MPI_SUM, comm.comm());
    };
    allreduce6(stress_nl_, spincomm); allreduce6(stress_k_, spincomm);
    allreduce6(stress_nl_, kptcomm);  allreduce6(stress_k_, kptcomm);
    allreduce6(stress_nl_, bandcomm); allreduce6(stress_k_, bandcomm);

    allreduce6(stress_soc_, spincomm);
    allreduce6(stress_soc_, kptcomm);
    allreduce6(stress_soc_, bandcomm);

    for (int i = 0; i < 6; ++i) {
        stress_nl_[i] /= cell_measure_;
        stress_k_[i] /= cell_measure_;
        stress_soc_[i] /= cell_measure_;
    }
}

void Stress::print(int rank) const {
    if (rank != 0) return;

    const double au_to_gpa = 29421.01569650548;
    const auto& sigma = stress_total_;
    std::printf("\nStress tensor (GPa):\n");
    std::printf("  sigma_xx = %14.6f  sigma_xy = %14.6f  sigma_xz = %14.6f\n",
                sigma[0] * au_to_gpa, sigma[1] * au_to_gpa, sigma[2] * au_to_gpa);
    std::printf("  sigma_yy = %14.6f  sigma_yz = %14.6f  sigma_zz = %14.6f\n",
                sigma[3] * au_to_gpa, sigma[4] * au_to_gpa, sigma[5] * au_to_gpa);
    std::printf("\nPressure: %.6f GPa\n",
                -(sigma[0] + sigma[3] + sigma[5]) / 3.0 * au_to_gpa);
}

} // namespace lynx
