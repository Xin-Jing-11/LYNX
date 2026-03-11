#include "physics/SCF.hpp"
#include "core/constants.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace sparc {

void SCF::setup(const FDGrid& grid,
                 const Domain& domain,
                 const FDStencil& stencil,
                 const Laplacian& laplacian,
                 const Gradient& gradient,
                 const Hamiltonian& hamiltonian,
                 const HaloExchange& halo,
                 const NonlocalProjector* vnl,
                 const MPIComm& dmcomm,
                 const MPIComm& bandcomm,
                 const MPIComm& kptcomm,
                 const MPIComm& spincomm,
                 const SCFParams& params) {
    grid_ = &grid;
    domain_ = &domain;
    stencil_ = &stencil;
    laplacian_ = &laplacian;
    gradient_ = &gradient;
    hamiltonian_ = &hamiltonian;
    halo_ = &halo;
    vnl_ = vnl;
    dmcomm_ = &dmcomm;
    bandcomm_ = &bandcomm;
    kptcomm_ = &kptcomm;
    spincomm_ = &spincomm;
    params_ = params;
}

void SCF::init_density(int Nd_d, int Nelectron) {
    density_.allocate(Nd_d, 1);  // non-spin-polarized for now
    // Uniform initial density: rho = Nelectron / Volume
    double volume = grid_->Nd() * grid_->dV();
    double rho0 = Nelectron / volume;
    double* rho = density_.rho(0).data();
    for (int i = 0; i < Nd_d; ++i) {
        rho[i] = rho0;
    }
    // Update total
    double* rho_t = density_.rho_total().data();
    std::memcpy(rho_t, rho, Nd_d * sizeof(double));
}

void SCF::set_initial_density(const double* rho_init, int Nd_d) {
    density_.allocate(Nd_d, 1);
    std::memcpy(density_.rho(0).data(), rho_init, Nd_d * sizeof(double));
    std::memcpy(density_.rho_total().data(), rho_init, Nd_d * sizeof(double));
}

void SCF::compute_Veff(const double* rho, const double* rho_b, const double* Vloc) {
    int Nd_d = domain_->Nd_d();

    // 1. XC potential and energy density
    // If NLCC, add core density to valence density for XC evaluation
    XCFunctional xc;
    xc.setup(xc_type_, *domain_, *grid_, gradient_, halo_);

    if (rho_core_) {
        std::vector<double> rho_xc(Nd_d);
        constexpr double xc_rhotol = 1e-14;
        for (int i = 0; i < Nd_d; ++i) {
            rho_xc[i] = rho[i] + rho_core_[i];
            if (rho_xc[i] < xc_rhotol) rho_xc[i] = xc_rhotol;
        }
        xc.evaluate(rho_xc.data(), Vxc_.data(), exc_.data(), Nd_d);
    } else {
        xc.evaluate(rho, Vxc_.data(), exc_.data(), Nd_d);
    }

    // 2. Solve Poisson equation: -Lap(phi) = 4*pi*(rho + b)
    // where b is the pseudocharge density
    // RHS = 4*pi*(rho + b) with positive sign
    std::vector<double> rhs(Nd_d);
    for (int i = 0; i < Nd_d; ++i) {
        rhs[i] = 4.0 * constants::PI * rho[i];
        if (rho_b) rhs[i] += 4.0 * constants::PI * rho_b[i];
    }

    // Ensure Poisson RHS integrates to zero (required for periodic BC)
    {
        double rhs_sum = 0;
        for (int i = 0; i < Nd_d; ++i) rhs_sum += rhs[i];
        if (!dmcomm_->is_null() && dmcomm_->size() > 1)
            rhs_sum = dmcomm_->allreduce_sum(rhs_sum);
        double rhs_mean = rhs_sum / grid_->Nd();
        for (int i = 0; i < Nd_d; ++i) rhs[i] -= rhs_mean;

    }

    PoissonSolver poisson;
    poisson.setup(*laplacian_, *stencil_, *domain_, *grid_, *halo_, *dmcomm_);
    poisson.solve(rhs.data(), phi_.data(), params_.poisson_tol);

    // Shift phi so that its integral is zero (periodic gauge)
    {
        double sum = 0.0;
        for (int i = 0; i < Nd_d; ++i) sum += phi_.data()[i];
        if (!dmcomm_->is_null() && dmcomm_->size() > 1)
            sum = dmcomm_->allreduce_sum(sum);
        double mean = sum / grid_->Nd();
        for (int i = 0; i < Nd_d; ++i) phi_.data()[i] -= mean;
    }

    // 3. Veff = Vxc + phi
    // Note: Vloc is NOT added here. The local pseudopotential contribution
    // is captured through the pseudocharge density in the Poisson equation.
    double* Veff = Veff_.data();
    const double* vxc = Vxc_.data();
    const double* phi = phi_.data();
    for (int i = 0; i < Nd_d; ++i) {
        Veff[i] = vxc[i] + phi[i];
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
        if (dmcomm_->rank() == 0)
            std::printf("Auto Chebyshev degree: %d (h_eff=%.6f)\n", params_.cheb_degree, h_eff);
    }

    // kpoint weights (uniform grid)
    std::vector<double> kpt_weights(Nkpts, 1.0 / Nkpts);

    // Allocate work arrays
    Veff_ = NDArray<double>(Nd_d);
    Vxc_ = NDArray<double>(Nd_d);
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
                laplacian_, halo_, grid_, dmcomm_);

    // Setup eigensolver
    EigenSolver eigsolver;
    eigsolver.setup(*hamiltonian_, *halo_, *domain_, *dmcomm_, *bandcomm_);

    // Estimate spectral bounds
    double eigval_min = 0.0, eigval_max = 0.0;

    // Initial Veff from initial density
    compute_Veff(density_.rho_total().data(), rho_b, Vloc_);

    // Lanczos to estimate spectrum
    eigsolver.lanczos_bounds(Veff_.data(), Nd_d, eigval_min, eigval_max);
    double lambda_cutoff = 0.5 * (eigval_min + eigval_max);  // initial guess
    if (dmcomm_->rank() == 0) {
        std::printf("Lanczos bounds: eigmin=%.6e, eigmax=%.6e\n", eigval_min, eigval_max);
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

        if (dmcomm_->rank() == 0 && scf_iter == 0 && nchefsi > 1)
            std::printf("SCF iter 1: %d CheFSI passes (rhoTrigger)\n", nchefsi);

        // 1. Solve eigenvalue problem (nchefsi passes)
        for (int chefsi_pass = 0; chefsi_pass < nchefsi; ++chefsi_pass) {
            for (int s = 0; s < Nspin; ++s) {
                for (int k = 0; k < Nkpts; ++k) {
                    double* psi = wfn.psi(s, k).data();
                    double* eig = wfn.eigenvalues(s, k).data();

                    eigsolver.solve(psi, eig, Veff_.data(), Nd_d, Nband,
                                    lambda_cutoff, eigval_min, eigval_max,
                                    params_.cheb_degree);
                }
            }

            // Update spectral bounds from eigenvalues
            // Reference: Chebyshevfilter_constants()
            //   count==0: Lanczos (already done), lambda_cutoff = 0.5*(eigmin+eigmax)
            //   count>0:  lambda_cutoff = lambda_sorted[last] + 0.1
            //   count>=rhoTrigger: eigmin = lambda_sorted[0]
            // eigmin stays at Lanczos value during initial rhoTrigger passes
            // eigmax never changes (for default chefsibound_flag=0)
            {
                const double* eig0 = wfn.eigenvalues(0, 0).data();
                if (scf_iter > 0) {
                    eigval_min = eig0[0];
                }
                // Always update lambda_cutoff after each pass (reference: count>0)
                // For the first pass (count=0), the initial value was already set
                lambda_cutoff = eig0[Nband - 1] + 0.1;
            }

            // Compute occupations between CheFSI passes
            Ef_ = Occupation::compute(wfn, Nelectron, beta, params_.smearing,
                                      kpt_weights, *kptcomm_, *spincomm_);

        }

        // 2. Compute new electron density
        ElectronDensity rho_new;
        rho_new.allocate(Nd_d, Nspin);
        rho_new.compute(wfn, kpt_weights, grid_->dV(), *bandcomm_, *kptcomm_);

        // 3. Compute energy BEFORE mixing, using rho_in (the density that
        //    generated the current potentials). This matches reference SPARC,
        //    which computes energy based on rho_in for faster convergence.
        energy_ = Energy::compute_all(wfn, density_, Veff_.data(), phi_.data(),
                                       exc_.data(), Vxc_.data(), rho_b,
                                       Eself, Ec, beta, params_.smearing,
                                       kpt_weights, Nd_d, grid_->dV(), *dmcomm_,
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
            if (!dmcomm_->is_null() && dmcomm_->size() > 1) {
                double sbuf[2] = {sum_sq_out, sum_sq_diff};
                dmcomm_->allreduce_sum(sbuf, 2);
                sum_sq_out = sbuf[0];
                sum_sq_diff = sbuf[1];
            }
            scf_error = (sum_sq_out > 0.0) ? std::sqrt(sum_sq_diff / sum_sq_out) : 0.0;
        }

        if (dmcomm_->rank() == 0) {
            std::printf("SCF iter %3d: Etot = %18.10f Ha, SCF error = %10.3e, Ef = %10.5f\n",
                        scf_iter + 1, energy_.Etotal, scf_error, Ef_);
        }

        if (scf_iter >= params_.min_iter && scf_error < params_.tol) {
            converged_ = true;
            if (dmcomm_->rank() == 0) {
                std::printf("SCF converged after %d iterations.\n", scf_iter + 1);
            }
            break;
        }

        E_prev = energy_.Etotal;

        // 5. Mix density
        mixer.mix(density_.rho_total().data(), rho_new.rho_total().data(), Nd_d);

        // Reference: clamp negative density to 0, then renormalize
        double* rho_mix = density_.rho_total().data();
        for (int i = 0; i < Nd_d; ++i) {
            if (rho_mix[i] < 0.0) rho_mix[i] = 0.0;
        }
        {
            double rho_sum = 0.0;
            for (int i = 0; i < Nd_d; ++i) rho_sum += rho_mix[i];
            if (!dmcomm_->is_null() && dmcomm_->size() > 1)
                rho_sum = dmcomm_->allreduce_sum(rho_sum);
            double Ne_current = rho_sum * grid_->dV();
            if (Ne_current > 1e-10) {
                double scale = static_cast<double>(Nelectron) / Ne_current;
                for (int i = 0; i < Nd_d; ++i) rho_mix[i] *= scale;
            }
        }

        // 6. Compute new Veff from mixed density
        compute_Veff(density_.rho_total().data(), rho_b, Vloc_);
    }

    if (!converged_ && dmcomm_->rank() == 0) {
        std::printf("WARNING: SCF did not converge within %d iterations.\n", params_.max_iter);
    }

    return energy_.Etotal;
}

} // namespace sparc
