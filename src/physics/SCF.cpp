#include "physics/SCF.hpp"
#include "core/constants.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

namespace sparc {

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
                 int Nspin) {
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
    Nspin_ = Nspin;
}

void SCF::init_density(int Nd_d, int Nelectron) {
    density_.allocate(Nd_d, Nspin_);
    // Uniform initial density: rho = Nelectron / Volume
    double volume = grid_->Nd() * grid_->dV();
    double rho0 = Nelectron / volume;

    if (Nspin_ == 1) {
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
    density_.allocate(Nd_d, Nspin_);
    std::memcpy(density_.rho_total().data(), rho_init, Nd_d * sizeof(double));

    if (Nspin_ == 1) {
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
    int dxc_ncol = (Nspin_ == 2) ? 3 : 1;  // 3 columns for spin: [v2c, v2x_up, v2x_down]
    if (xc.is_gga() && Dxcdgrho_.size() == 0) {
        Dxcdgrho_ = NDArray<double>(Nd_d * dxc_ncol);
    }
    double* dxc_ptr = xc.is_gga() ? Dxcdgrho_.data() : nullptr;

    if (Nspin_ == 2) {
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

    // Debug: print RHS 2-norm
    {
        double rhs_norm2 = 0;
        for (int i = 0; i < Nd_d; ++i) rhs_norm2 += rhs[i] * rhs[i];
        std::printf("DEBUG_POISSON: rhs_2norm=%.10f\n", std::sqrt(rhs_norm2));
    }

    PoissonSolver poisson;
    poisson.setup(*laplacian_, *stencil_, *domain_, *grid_, *halo_);
    int poisson_iters = poisson.solve(rhs.data(), phi_.data(), params_.poisson_tol);

    // Debug: print phi stats
    {
        double phi_min = 1e99, phi_max = -1e99, phi_sum = 0;
        for (int i = 0; i < Nd_d; ++i) {
            phi_min = std::min(phi_min, phi_.data()[i]);
            phi_max = std::max(phi_max, phi_.data()[i]);
            phi_sum += phi_.data()[i];
        }
        std::printf("DEBUG_POISSON: iters=%d phi_min=%.10f phi_max=%.10f phi_mean=%.10e\n",
                    poisson_iters, phi_min, phi_max, phi_sum / grid_->Nd());
    }

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
    for (int s = 0; s < Nspin_; ++s) {
        double* Veff = Veff_.data() + s * Nd_d;
        const double* vxc = Vxc_.data() + s * Nd_d;
        for (int i = 0; i < Nd_d; ++i) {
            Veff[i] = vxc[i] + phi[i];
        }
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
    int Nband = wfn.Nband();
    int Nspin = wfn.Nspin();
    int Nkpts = wfn.Nkpts();

    xc_type_ = xc_type;
    Vloc_ = Vloc;
    rho_core_ = rho_core;

    // Compute default tolerances (reference: initialization.c:2655-2681)
    if (params_.poisson_tol < 0.0)
        params_.poisson_tol = params_.tol * 0.01;

    // Auto-compute electronic temperature (reference: initialization.c:1914-1923)
    // Gaussian: smearing = 0.2 eV => Beta = CONST_EH / 0.2
    // FermiDirac: smearing = 0.1 eV => Beta = CONST_EH / 0.1
    if (params_.elec_temp < 0.0) {
        double smearing_eV = (params_.smearing == SmearingType::GaussianSmearing) ? 0.2 : 0.1;
        double beta_au = constants::EH / smearing_eV;  // 1/(kB*T) in atomic units
        params_.elec_temp = 1.0 / (constants::KB * beta_au);  // T in Kelvin
    }

    // Auto-compute Chebyshev degree from mesh spacing (reference: eigenSolver.c Mesh2ChebDegree)
    if (params_.cheb_degree < 0) {
        double dx = grid_->dx(), dy = grid_->dy(), dz = grid_->dz();
        double h_eff;
        if (std::abs(dx - dy) < 1e-12 && std::abs(dy - dz) < 1e-12) {
            h_eff = dx;
        } else {
            double dx2i = 1.0/(dx*dx), dy2i = 1.0/(dy*dy), dz2i = 1.0/(dz*dz);
            h_eff = std::sqrt(3.0 / (dx2i + dy2i + dz2i));
        }
        // Cubic polynomial fit: (0.1,50), (0.2,35), (0.4,20), (0.7,14)
        double p3 = -700.0 / 3.0, p2 = 1240.0 / 3.0, p1 = -773.0 / 3.0, p0 = 1078.0 / 15.0;
        double npl;
        if (h_eff > 0.7) {
            npl = 14.0;
        } else {
            npl = ((p3 * h_eff + p2) * h_eff + p1) * h_eff + p0;
        }
        params_.cheb_degree = static_cast<int>(std::round(npl));
        if (true /* rank 0 */)
            std::printf("Auto Chebyshev degree: %d (h_eff=%.6f)\n", params_.cheb_degree, h_eff);
    }

    // kpoint weights (uniform grid)
    std::vector<double> kpt_weights(Nkpts, 1.0 / Nkpts);

    // Allocate work arrays (spin-resolved for Veff and Vxc)
    Veff_ = NDArray<double>(Nd_d * Nspin);
    Vxc_ = NDArray<double>(Nd_d * Nspin);
    exc_ = NDArray<double>(Nd_d);
    phi_ = NDArray<double>(Nd_d);

    // Initialize density if not already set
    if (density_.Nd_d() == 0) {
        init_density(Nd_d, Nelectron);
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
    eigsolver.setup(*hamiltonian_, *halo_, *domain_, *bandcomm_);

    // Estimate spectral bounds — per-spin like reference
    std::vector<double> eigval_min(Nspin, 0.0), eigval_max(Nspin, 0.0);

    // Initial Veff from initial density
    compute_Veff(density_.rho_total().data(), rho_b, Vloc_);

    // Debug: dump initial state for comparison with reference
    if (true /* rank 0 */) {
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

    // Lanczos to estimate spectrum per spin
    for (int s = 0; s < Nspin; ++s) {
        eigsolver.lanczos_bounds(Veff_.data() + s * Nd_d, Nd_d, eigval_min[s], eigval_max[s]);
    }
    double lambda_cutoff = 0.5 * (eigval_min[0] + eigval_max[0]);  // initial guess
    if (true /* rank 0 */) {
        for (int s = 0; s < Nspin; ++s)
            std::printf("Lanczos bounds (spin %d): eigmin=%.6e, eigmax=%.6e\n",
                        s, eigval_min[s], eigval_max[s]);
        if (params_.cheb_degree > 0)
            std::printf("Chebyshev degree: %d\n", params_.cheb_degree);
    }

    // Randomize wavefunctions — match reference SPARC exactly
    // Reference: SetRandMat(Xorb, DMndsp, Nband, -0.5, 0.5, spincomm)
    // seed = rank_in_spincomm * 100 + 1 (serial: seed=1)
    // For gamma-point, each (spin,kpt) pair gets same seed (reference fills all at once)
    // For serial: seed = 0 * 100 + 1 = 1
    int spincomm_rank = 0; // TODO: get from parallel context for multi-process
    unsigned rand_seed = spincomm_rank * 100 + 1;
    for (int s = 0; s < Nspin; ++s) {
        for (int k = 0; k < Nkpts; ++k) {
            wfn.randomize(s, k, rand_seed);
        }
    }

    double E_prev = 0.0;
    converged_ = false;
    if (Natom <= 0) Natom = std::max(1, Nelectron / 4);

    // ===== SCF Loop =====
    // Reference SPARC control flow (eigSolve_CheFSI):
    //   SCFcount=0: do rhoTrigger (default 4) CheFSI passes
    //   SCFcount>0: do Nchefsi (default 1) CheFSI passes
    // Between each CheFSI pass, occupations are computed to update lambda_cutoff.
    // After all passes for a given SCFcount, density is computed, then energy, then mixing.
    for (int scf_iter = 0; scf_iter < params_.max_iter; ++scf_iter) {
        // Number of CheFSI passes: rhoTrigger for first iter, Nchefsi for rest
        int nchefsi = (scf_iter == 0) ? params_.rho_trigger : params_.nchefsi;

        if (true /* rank 0 */ && scf_iter == 0 && nchefsi > 1)
            std::printf("SCF iter 1: %d CheFSI passes (rhoTrigger)\n", nchefsi);

        // 1. Solve eigenvalue problem (nchefsi passes)
        for (int chefsi_pass = 0; chefsi_pass < nchefsi; ++chefsi_pass) {
            for (int s = 0; s < Nspin; ++s) {
                // Pass spin-specific Veff and spectral bounds to eigensolver
                double* Veff_s = Veff_.data() + s * Nd_d;
                for (int k = 0; k < Nkpts; ++k) {
                    double* psi = wfn.psi(s, k).data();
                    double* eig = wfn.eigenvalues(s, k).data();

                    eigsolver.solve(psi, eig, Veff_s, Nd_d, Nband,
                                    lambda_cutoff, eigval_min[s], eigval_max[s],
                                    params_.cheb_degree);
                }
            }

            // Update spectral bounds from eigenvalues (per-spin)
            // Reference: Chebyshevfilter_constants()
            //   count==0: Lanczos (already done), lambda_cutoff = 0.5*(eigmin+eigmax)
            //   count>0:  lambda_cutoff = lambda_sorted[last] + 0.1
            //   count>=rhoTrigger: eigmin = lambda_sorted[0]
            {
                // lambda_cutoff: max of last eigenvalue across all spins + 0.1
                double eig_last_max = -1e30;
                for (int s = 0; s < Nspin; ++s) {
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
                                      kpt_weights, *kptcomm_, *spincomm_);

            // Debug: dump eigenvalues for first 3 iterations
            if (true /* rank 0 */ && scf_iter < 3) {
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
        rho_new.allocate(Nd_d, Nspin);
        rho_new.compute(wfn, kpt_weights, grid_->dV(), *bandcomm_, *kptcomm_);

        // Debug: dump new density for first 3 iterations
        if (true /* rank 0 */ && scf_iter < 3) {
            double rho_out_sum = 0;
            for (int i = 0; i < Nd_d; ++i) rho_out_sum += rho_new.rho_total().data()[i];
            std::printf("DUMP_RHO_OUT sum=%.15e sum*dV=%.15e\n",
                        rho_out_sum, rho_out_sum * grid_->dV());
            for (int i = 0; i < 10 && i < Nd_d; ++i)
                std::printf("DUMP_RHO_OUT[%d]=%.15e\n", i, rho_new.rho_total().data()[i]);
            std::printf("DUMP_EBAND=%.15e\n", Energy::band_energy(wfn, kpt_weights, Nspin));
        }

        // 3. Compute energy BEFORE mixing, using rho_in (the density that
        //    generated the current potentials). This matches reference SPARC,
        //    which computes energy based on rho_in for faster convergence.
        energy_ = Energy::compute_all(wfn, density_, Veff_.data(), phi_.data(),
                                       exc_.data(), Vxc_.data(), rho_b,
                                       Eself, Ec, beta, params_.smearing,
                                       kpt_weights, Nd_d, grid_->dV(),
                                       rho_core, Ef_);

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

        if (true /* rank 0 */) {
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
        if (true /* rank 0 */ && scf_error < params_.tol && scf_iter >= params_.min_iter) {
            std::printf("\nFinal eigenvalues (Ha) and occupations:\n");
            for (int s = 0; s < Nspin; ++s) {
                if (Nspin > 1) std::printf("  Spin %d:\n", s);
                for (int n = 0; n < Nband; ++n) {
                    std::printf("  %3d  %20.12e  %16.12f\n",
                                n+1, wfn.eigenvalues(s, 0)(n), wfn.occupations(s, 0)(n));
                }
            }
        }

        if (scf_iter >= params_.min_iter && scf_error < params_.tol) {
            converged_ = true;
            if (true /* rank 0 */) {
                std::printf("SCF converged after %d iterations.\n", scf_iter + 1);
            }
            break;
        }

        E_prev = energy_.Etotal;

        // 5. Mix density
        if (Nspin == 1) {
            mixer.mix(density_.rho_total().data(), rho_new.rho_total().data(), Nd_d);

            // Reference: clamp negative density to 0, then renormalize
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
            // Keep rho(0) in sync with rho_total
            std::memcpy(density_.rho(0).data(), density_.rho_total().data(), Nd_d * sizeof(double));
        } else {
            // Spin-polarized: mix packed array [total | magnetization] (2*Nd_d)
            // Reference SPARC: mixing variable is [rho, mag] where mag = rho_up - rho_down
            std::vector<double> dens_in(2 * Nd_d), dens_out(2 * Nd_d);

            // Pack input: [rho_total | mag]
            const double* rho_up_in = density_.rho(0).data();
            const double* rho_dn_in = density_.rho(1).data();
            for (int i = 0; i < Nd_d; ++i) {
                dens_in[i] = density_.rho_total().data()[i];
                dens_in[Nd_d + i] = rho_up_in[i] - rho_dn_in[i];
            }

            // Pack output: [rho_total | mag]
            const double* rho_up_out = rho_new.rho(0).data();
            const double* rho_dn_out = rho_new.rho(1).data();
            for (int i = 0; i < Nd_d; ++i) {
                dens_out[i] = rho_new.rho_total().data()[i];
                dens_out[Nd_d + i] = rho_up_out[i] - rho_dn_out[i];
            }

            mixer.mix(dens_in.data(), dens_out.data(), Nd_d, 2);

            // Unpack: reconstruct rho_up = 0.5*(rho + mag), rho_dn = 0.5*(rho - mag)
            double* rho_tot = density_.rho_total().data();
            double* rho_up = density_.rho(0).data();
            double* rho_dn = density_.rho(1).data();
            for (int i = 0; i < Nd_d; ++i) {
                rho_tot[i] = dens_in[i];
                double mag = dens_in[Nd_d + i];
                rho_up[i] = 0.5 * (rho_tot[i] + mag);
                rho_dn[i] = 0.5 * (rho_tot[i] - mag);
            }

            // Clamp and renormalize
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
        compute_Veff(density_.rho_total().data(), rho_b, Vloc_);
    }

    if (!converged_ && true /* rank 0 */) {
        std::printf("WARNING: SCF did not converge within %d iterations.\n", params_.max_iter);
    }

    return energy_.Etotal;
}

} // namespace sparc
