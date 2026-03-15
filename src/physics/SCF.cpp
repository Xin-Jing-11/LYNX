#include "physics/SCF.hpp"
#include "core/constants.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <mpi.h>

namespace lynx {

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

void SCF::compute_Veff(const double* rho, const double* rho_b, const double* Vloc) {
    int Nd_d = domain_->Nd_d();

    // 1. XC potential and energy density
    // If NLCC, add core density to valence density for XC evaluation
    XCFunctional xc;
    xc.setup(xc_type_, *domain_, *grid_, gradient_, halo_);

    // Allocate Dxcdgrho_ if GGA
    int dxc_ncol = (Nspin_global_ == 2) ? 3 : 1;  // 3 columns for spin: [v2c, v2x_up, v2x_down]
    if (xc.is_gga() && Dxcdgrho_.size() == 0) {
        Dxcdgrho_ = NDArray<double>(Nd_d * dxc_ncol);
    }
    double* dxc_ptr = xc.is_gga() ? Dxcdgrho_.data() : nullptr;

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
        xc.evaluate_spin(rho_xc.data(), Vxc_.data(), exc_.data(), Nd_d, dxc_ptr);
    } else {
        // Non-spin-polarized
        if (rho_core_) {
            std::vector<double> rho_xc(Nd_d);
            constexpr double xc_rhotol = 1e-14;
            for (int i = 0; i < Nd_d; ++i) {
                rho_xc[i] = rho[i] + rho_core_[i];
                if (rho_xc[i] < xc_rhotol) rho_xc[i] = xc_rhotol;
            }
            xc.evaluate(rho_xc.data(), Vxc_.data(), exc_.data(), Nd_d, dxc_ptr);
        } else {
            xc.evaluate(rho, Vxc_.data(), exc_.data(), Nd_d, dxc_ptr);
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

    // Compute default tolerances (reference: initialization.c:2655-2681)
    if (params_.poisson_tol < 0.0)
        params_.poisson_tol = params_.tol * 0.01;

    // Auto-compute electronic temperature (reference: initialization.c:1914-1923)
    if (params_.elec_temp < 0.0) {
        double smearing_eV = (params_.smearing == SmearingType::GaussianSmearing) ? 0.2 : 0.1;
        double beta_au = constants::EH / smearing_eV;
        params_.elec_temp = 1.0 / (constants::KB * beta_au);
    }

    // Auto-compute Chebyshev degree from mesh spacing
    if (params_.cheb_degree < 0) {
        double dx = grid_->dx(), dy = grid_->dy(), dz = grid_->dz();
        double h_eff;
        if (std::abs(dx - dy) < 1e-12 && std::abs(dy - dz) < 1e-12) {
            h_eff = dx;
        } else {
            double dx2i = 1.0/(dx*dx), dy2i = 1.0/(dy*dy), dz2i = 1.0/(dz*dz);
            h_eff = std::sqrt(3.0 / (dx2i + dy2i + dz2i));
        }
        double p3 = -700.0 / 3.0, p2 = 1240.0 / 3.0, p1 = -773.0 / 3.0, p0 = 1078.0 / 15.0;
        double npl;
        if (h_eff > 0.7) {
            npl = 14.0;
        } else {
            npl = ((p3 * h_eff + p2) * h_eff + p1) * h_eff + p0;
        }
        params_.cheb_degree = static_cast<int>(std::round(npl));
        if (rank_world == 0)
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
    if (is_soc_) {
        compute_Veff_spinor(density_, rho_b, Vloc_);
    } else {
        compute_Veff(density_.rho_total().data(), rho_b, Vloc_);
    }

    // Debug: dump initial state for comparison with reference
    if (rank_world == 0) {
        const double* rho0 = density_.rho_total().data();
        double rho_sum = 0;
        for (int i = 0; i < Nd_d; ++i) rho_sum += rho0[i];
        std::printf("DUMP_INIT_RHO sum=%.15e sum*dV=%.15e rho[0]=%.15e rho[6143]=%.15e\n",
                    rho_sum, rho_sum * grid_->dV(), rho0[0], Nd_d > 6143 ? rho0[6143] : 0.0);
        std::printf("DUMP_PSEUDOCHARGE sum=%.15e b[0]=%.15e b[6143]=%.15e\n",
                    0.0, rho_b[0], Nd_d > 6143 ? rho_b[6143] : 0.0); // sum computed later
        std::printf("DUMP_INIT_PHI min=%.15e max=%.15e phi[0]=%.15e\n",
                    *std::min_element(phi_.data(), phi_.data() + Nd_d),
                    *std::max_element(phi_.data(), phi_.data() + Nd_d),
                    phi_.data()[0]);
        double vxc_sum = 0, veff_sum = 0;
        for (int i = 0; i < Nd_d; ++i) { vxc_sum += Vxc_.data()[i]; veff_sum += Veff_.data()[i]; }
        std::printf("DUMP_INIT_VXC sum=%.15e Vxc[0]=%.15e\n", vxc_sum, Vxc_.data()[0]);
        std::printf("DUMP_INIT_VEFF sum=%.15e Veff[0]=%.15e\n", veff_sum, Veff_.data()[0]);
        for (int i = 0; i < 10 && i < Nd_d; ++i) {
            std::printf("DUMP_VEFF[%d]=%.15e\n", i, Veff_.data()[i]);
        }
        for (int i = 0; i < 10 && i < Nd_d; ++i) {
            std::printf("DUMP_PHI[%d]=%.15e\n", i, phi_.data()[i]);
        }
        for (int i = 0; i < 10 && i < Nd_d; ++i) {
            std::printf("DUMP_RHO[%d]=%.15e\n", i, rho0[i]);
        }
        for (int i = 0; i < 10 && i < Nd_d; ++i) {
            std::printf("DUMP_B[%d]=%.15e\n", i, rho_b[i]);
        }
    }

    // Lanczos to estimate spectrum
    if (is_soc_) {
        Vec3 kpt0 = kpoints_->kpts_cart()[kpt_start_];
        eigsolver.lanczos_bounds_spinor_kpt(Veff_spinor_.data(), Nd_d,
                                             kpt0, cell_lengths,
                                             eigval_min[0], eigval_max[0]);
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
    double lambda_cutoff = 0.5 * (eigval_min[0] + eigval_max[0]);  // initial guess
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
    unsigned rand_seed = spincomm_rank * 100 + bandcomm_rank * 10 + 1;
    for (int s = 0; s < Nspin_local; ++s) {
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

            // Debug: dump eigenvalues for first 3 iterations
            if (rank_world == 0 && scf_iter < 3) {
                std::printf("DUMP_SCF_ITER=%d PASS=%d\n", scf_iter, chefsi_pass);
                std::printf("DUMP_EIGVALS");
                for (int n = 0; n < Nband && n < 30; ++n)
                    std::printf(" %.10e", wfn.eigenvalues(0, 0)(n));
                std::printf("\n");
                std::printf("DUMP_EFERMI=%.15e\n", Ef_);
                std::printf("DUMP_OCC");
                for (int n = 0; n < Nband && n < 30; ++n)
                    std::printf(" %.10e", wfn.occupations(0, 0)(n));
                std::printf("\n");
            }

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

        // Debug: dump new density for first 3 iterations
        if (rank_world == 0 && scf_iter < 3) {
            double rho_out_sum = 0;
            for (int i = 0; i < Nd_d; ++i) rho_out_sum += rho_new.rho_total().data()[i];
            std::printf("DUMP_RHO_OUT sum=%.15e sum*dV=%.15e\n",
                        rho_out_sum, rho_out_sum * grid_->dV());
            for (int i = 0; i < 10 && i < Nd_d; ++i)
                std::printf("DUMP_RHO_OUT[%d]=%.15e\n", i, rho_new.rho_total().data()[i]);
            std::printf("DUMP_EBAND=%.15e\n", Energy::band_energy(wfn, kpt_weights, Nspin, kpt_start_));
        }

        // 3. Compute energy BEFORE mixing, using rho_in (the density that
        //    generated the current potentials). This matches reference LYNX,
        //    which computes energy based on rho_in for faster convergence.
        energy_ = Energy::compute_all(wfn, density_, Veff_.data(), phi_.data(),
                                       exc_.data(), Vxc_.data(), rho_b,
                                       Eself, Ec, beta, params_.smearing,
                                       kpt_weights, Nd_d, grid_->dV(),
                                       rho_core, Ef_, kpt_start_,
                                       kptcomm_, spincomm_, Nspin);

        // 4. Evaluate SCF error: ||rho_out - rho_in|| / ||rho_out||
        //    Reference: Evaluate_scf_error in electronicGroundState.c
        //    var_in = mixing_hist_xk (input density), var_out = electronDens (output density)
        double scf_error = 0.0;
        {
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

        // 5. Mix density
        if (is_soc_) {
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

        // 6. Compute new Veff from mixed density
        if (is_soc_) {
            compute_Veff_spinor(density_, rho_b, Vloc_);
        } else {
            compute_Veff(density_.rho_total().data(), rho_b, Vloc_);
        }
    }

    if (!converged_ && rank_world == 0) {
        std::printf("WARNING: SCF did not converge within %d iterations.\n", params_.max_iter);
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
                   xc_type == XCType::GGA_RPBE);

    // Allocate work arrays (needed for download_results)
    int dxc_ncol = is_gga ? ((Nspin == 2) ? 3 : 1) : 0;
    Veff_ = NDArray<double>(Nd_d * Nspin);
    Vxc_ = NDArray<double>(Nd_d * Nspin);
    exc_ = NDArray<double>(Nd_d);
    phi_ = NDArray<double>(Nd_d);
    if (dxc_ncol > 0) Dxcdgrho_ = NDArray<double>(Nd_d * dxc_ncol);

    // Initialize density: use atomic superposition for better GPU SCF convergence
    if (density_.Nd_d() == 0) {
        density_.allocate(Nd_d, Nspin);
    }
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
    gpu_runner_ = std::make_unique<GPUSCFRunner>();
    double Etotal = gpu_runner_->run(
        wfn, params_, *grid_, *domain_, *stencil_,
        *hamiltonian_, *halo_, vnl_,
        *crystal_, *nloc_influence_, *bandcomm_,
        Nelectron, Natom,
        density_.rho_total().data(), rho_b,
        Eself, Ec, xc_type, rho_core, is_gga,
        Nspin, is_kpt_, kpoints_, kpt_weights,
        Nspin_local_, spin_start_, kpt_start_,
        density_.Nd_d() > 0 && Nspin == 2 ? density_.rho(0).data() : nullptr,
        density_.Nd_d() > 0 && Nspin == 2 ? density_.rho(1).data() : nullptr,
        is_soc);

    // Download results for forces/stress
    gpu_runner_->download_results(
        phi_.data(), Vxc_.data(), exc_.data(), Veff_.data(),
        dxc_ncol > 0 ? Dxcdgrho_.data() : nullptr,
        density_.rho_total().data(), wfn);
    // Keep spin densities in sync
    if (Nspin == 1) {
        std::memcpy(density_.rho(0).data(), density_.rho_total().data(), Nd_d * sizeof(double));
    }

    energy_ = gpu_runner_->energy();
    converged_ = gpu_runner_->converged();
    Ef_ = gpu_runner_->fermi_energy();

    return Etotal;
}
#endif

} // namespace lynx
