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

void SCF::compute_Veff(const double* rho, const double* rho_b) {
    int Nd_d = domain_->Nd_d();

    // 1. XC potential and energy density
    XCFunctional xc;
    xc.setup(XCType::GGA_PBE, *domain_, *grid_, gradient_, halo_);
    xc.evaluate(rho, Vxc_.data(), exc_.data(), Nd_d);

    // 2. Solve Poisson equation: -Lap(phi) = 4*pi*(rho + b)
    std::vector<double> rhs(Nd_d);
    for (int i = 0; i < Nd_d; ++i) {
        rhs[i] = -4.0 * constants::PI * rho[i];
        if (rho_b) rhs[i] += -4.0 * constants::PI * rho_b[i];
    }

    PoissonSolver poisson;
    poisson.setup(*laplacian_, *stencil_, *domain_, *grid_, *halo_, *dmcomm_);
    AARParams aar_params;
    aar_params.tol = params_.poisson_tol;
    poisson.set_aar_params(aar_params);
    poisson.solve(rhs.data(), phi_.data(), params_.poisson_tol);

    // 3. Veff = Vxc + phi
    double* Veff = Veff_.data();
    const double* vxc = Vxc_.data();
    const double* phi = phi_.data();
    for (int i = 0; i < Nd_d; ++i) {
        Veff[i] = vxc[i] + phi[i];
    }
}

double SCF::run(Wavefunction& wfn,
                 int Nelectron,
                 const double* rho_b,
                 double Eself,
                 double Ec) {
    int Nd_d = domain_->Nd_d();
    int Nband = wfn.Nband();
    int Nspin = wfn.Nspin();
    int Nkpts = wfn.Nkpts();

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
    compute_Veff(density_.rho_total().data(), rho_b);

    // Lanczos to estimate spectrum
    eigsolver.lanczos_bounds(Veff_.data(), Nd_d, eigval_min, eigval_max);
    double lambda_cutoff = 0.5 * (eigval_min + eigval_max);  // initial guess

    // Randomize wavefunctions
    for (int s = 0; s < Nspin; ++s) {
        for (int k = 0; k < Nkpts; ++k) {
            wfn.randomize(s, k, 42 + s * 100 + k);
        }
    }

    double E_prev = 0.0;
    converged_ = false;
    int Natom = std::max(1, Nelectron / 4);  // rough estimate

    // ===== SCF Loop =====
    for (int scf_iter = 0; scf_iter < params_.max_iter; ++scf_iter) {
        // 1. Solve eigenvalue problem for each spin/kpt
        for (int s = 0; s < Nspin; ++s) {
            for (int k = 0; k < Nkpts; ++k) {
                double* psi = wfn.psi(s, k).data();
                double* eig = wfn.eigenvalues(s, k).data();

                eigsolver.solve(psi, eig, Veff_.data(), Nd_d, Nband,
                                lambda_cutoff, eigval_min, eigval_max,
                                params_.cheb_degree);
            }
        }

        // Update lambda_cutoff from eigenvalues
        lambda_cutoff = eigsolver.lambda_cutoff();

        // 2. Compute occupations and Fermi level
        Ef_ = Occupation::compute(wfn, Nelectron, beta, params_.smearing,
                                  kpt_weights, *kptcomm_, *spincomm_);

        // 3. Compute new electron density
        ElectronDensity rho_new;
        rho_new.allocate(Nd_d, Nspin);
        rho_new.compute(wfn, kpt_weights, grid_->dV(), *bandcomm_, *kptcomm_);

        // 4. Mix density
        NDArray<double> rho_old = density_.rho_total().clone();
        mixer.mix(density_.rho_total().data(), rho_new.rho_total().data(), Nd_d);

        // Ensure density stays positive
        double* rho_mix = density_.rho_total().data();
        for (int i = 0; i < Nd_d; ++i) {
            if (rho_mix[i] < 1e-20) rho_mix[i] = 1e-20;
        }

        // 5. Compute new Veff
        compute_Veff(density_.rho_total().data(), rho_b);

        // 6. Compute energy
        energy_ = Energy::compute_all(wfn, density_, Veff_.data(), phi_.data(),
                                       exc_.data(), Vxc_.data(), rho_b,
                                       Eself, Ec, beta, params_.smearing,
                                       kpt_weights, Nd_d, grid_->dV(), *dmcomm_);

        // 7. Check convergence
        double dE = std::abs(energy_.Etotal - E_prev) / Natom;

        if (dmcomm_->rank() == 0) {
            std::printf("SCF iter %3d: Etot = %18.10f Ha, dE/atom = %10.3e, Ef = %10.5f\n",
                        scf_iter + 1, energy_.Etotal, dE, Ef_);
        }

        if (scf_iter >= params_.min_iter && dE < params_.tol) {
            converged_ = true;
            if (dmcomm_->rank() == 0) {
                std::printf("SCF converged after %d iterations.\n", scf_iter + 1);
            }
            break;
        }

        E_prev = energy_.Etotal;
    }

    if (!converged_ && dmcomm_->rank() == 0) {
        std::printf("WARNING: SCF did not converge within %d iterations.\n", params_.max_iter);
    }

    return energy_.Etotal;
}

} // namespace sparc
