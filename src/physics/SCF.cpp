#include "physics/SCF.hpp"
#include "physics/SCFInitializer.hpp"
#include "physics/HybridSCF.hpp"
#include "xc/ExactExchange.hpp"
#include "solvers/KerkerPreconditioner.hpp"
#include "core/constants.hpp"
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

SCFParams SCFParams::from_config(const SystemConfig& config) {
    SCFParams p;
    p.max_iter = config.max_scf_iter;
    p.min_iter = config.min_scf_iter;
    p.tol = config.scf_tol;
    p.mixing_var = config.mixing_var;
    p.mixing_precond = config.mixing_precond;
    p.mixing_history = config.mixing_history;
    p.mixing_param = config.mixing_param;
    p.smearing = config.smearing;
    p.elec_temp = config.elec_temp;
    p.cheb_degree = config.cheb_degree;
    p.poisson_tol = config.poisson_tol;
    p.precond_tol = config.precond_tol;
    p.rho_trigger = config.rho_trigger;
    return p;
}

void SCF::setup(const LynxContext& ctx,
                 const Hamiltonian& hamiltonian,
                 const NonlocalProjector* vnl,
                 const SCFParams& params) {
    ctx_ = &ctx;
    grid_ = &ctx.grid();
    domain_ = &ctx.domain();
    stencil_ = &ctx.stencil();
    laplacian_ = &ctx.laplacian();
    gradient_ = &ctx.gradient();
    hamiltonian_ = &hamiltonian;
    halo_ = &ctx.halo();
    vnl_ = vnl;
    bandcomm_ = &ctx.scf_bandcomm();
    kptcomm_ = &ctx.kpt_bridge();
    spincomm_ = &ctx.spin_bridge();
    params_ = params;
    Nspin_global_ = ctx.Nspin();
    Nspin_local_ = ctx.Nspin_local();
    spin_start_ = ctx.spin_start();
    kpoints_ = &ctx.kpoints();
    is_kpt_ = kpoints_ && !kpoints_->is_gamma_only();
    kpt_start_ = ctx.kpt_start();
    Nband_global_ = ctx.Nstates();
    band_start_ = ctx.band_start();

    // Setup EffectivePotential builder
    veff_builder_.setup(ctx, hamiltonian);
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
            for (int i = 0; i < Nd_d; ++i) {
                rho_up[i] = 0.5 * (rho_init[i] + mag_init[i]);
                rho_dn[i] = 0.5 * (rho_init[i] - mag_init[i]);
                if (rho_up[i] < 0.0) rho_up[i] = 0.0;
                if (rho_dn[i] < 0.0) rho_dn[i] = 0.0;
            }
        } else {
            for (int i = 0; i < Nd_d; ++i) {
                rho_up[i] = 0.5 * rho_init[i];
                rho_dn[i] = 0.5 * rho_init[i];
            }
        }
    }
}

// ===== Main SCF Algorithm =====
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
    int rank_world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);

    xc_type_ = xc_type;
    rho_core_ = rho_core;

    // Detect SOC mode
    is_soc_ = (wfn.Nspinor() == 2);
    if (is_soc_) is_kpt_ = true;

#ifdef USE_CUDA
    // GPU dispatch
    if (gpu_enabled_ && crystal_ && nloc_influence_) {
        if (rank_world == 0)
            std::printf("GPU SCF enabled — dispatching to fully GPU-resident path (Nspin=%d, kpt=%d, soc=%d)\n",
                        Nspin_global_, is_kpt_ ? 1 : 0, is_soc_ ? 1 : 0);
        return run_gpu(wfn, Nelectron, Natom, rho_b, Eself, Ec, xc_type, rho_core);
    }
#endif

    // ===== 1. Initialize =====
    // Setup preconditioner and mixer
    std::unique_ptr<KerkerPreconditioner> kerker_precond;
    if (params_.mixing_precond == MixingPrecond::Kerker) {
        kerker_precond = std::make_unique<KerkerPreconditioner>(
            laplacian_, halo_, grid_, 1.0, 0.1, params_.precond_tol);
    }
    Mixer mixer;
    mixer.setup(Nd_d, params_.mixing_var, params_.mixing_precond,
                params_.mixing_history, params_.mixing_param,
                kerker_precond.get());

    bool use_potential_mixing = (params_.mixing_var == MixingVariable::Potential) && !is_soc_;
    if (use_potential_mixing) {
        mixer.set_potential_mean_shift(grid_->Nd());
    } else {
        mixer.set_density_constraint(Nelectron, grid_->Nd(), grid_->dV());
    }

    EigenSolver eigsolver;
    eigsolver.setup(*ctx_, *hamiltonian_);

    SCFState state = SCFInitializer::initialize(
        wfn, density_, arrays_, veff_builder_, params_,
        *grid_, *domain_, *hamiltonian_, *halo_, vnl_,
        *bandcomm_, *kptcomm_, *spincomm_, eigsolver, mixer,
        Nelectron, Nspin_global_, Nspin_local_, spin_start_,
        kpoints_, kpt_start_, band_start_,
        xc_type, rho_b, rho_core, is_kpt_, is_soc_);

    state.Nelectron = Nelectron;

    // Allocate kinetic energy density for mGGA
    if (is_mgga_type(xc_type)) {
        tau_.allocate(Nd_d, Nspin_global_);
    }

    if (Natom <= 0) Natom = std::max(1, Nelectron / 4);
    converged_ = false;

    // ===== 2. SCF Iteration Loop =====
    ElectronDensity rho_new;
    for (int scf_iter = 0; scf_iter < params_.max_iter; ++scf_iter) {
        solve_eigenproblem(wfn, eigsolver, state, scf_iter);
        compute_new_density(wfn, state, rho_new);
        compute_scf_energy(wfn, rho_new, rho_b, Eself, Ec, state);
        if (check_convergence(wfn, rho_new, state, scf_iter)) break;
        mix_and_update(rho_new, mixer, rho_b, rho_core, Nelectron, state);
    }

    if (!converged_ && rank_world == 0)
        std::printf("WARNING: SCF did not converge within %d iterations.\n", params_.max_iter);

    // ===== 3. Outer Fock Loop for hybrid functionals =====
    if (exx_ && is_hybrid(xc_type_)) {
        HybridSCF hybrid;
        hybrid.run(wfn, density_, arrays_, veff_builder_,
                   energy_, Ef_, converged_,
                   exx_, hamiltonian_, vnl_,
                   eigsolver, mixer, params_,
                   *grid_, *domain_,
                   *bandcomm_, *kptcomm_, *spincomm_, kpoints_,
                   Nelectron, Natom, Nspin_global_, Nspin_local_,
                   spin_start_, kpt_start_, band_start_,
                   rho_b, rho_core, xc_type, is_kpt_,
                   Eself, Ec, state.kpt_weights, state);
    }

    return energy_.Etotal;
}

// ===== Extracted SCF sub-steps =====

void SCF::solve_eigenproblem(Wavefunction& wfn, EigenSolver& eigsolver,
                              SCFState& state, int scf_iter) {
    int Nd_d = domain_->Nd_d();
    int Nband = state.Nband;
    int Nband_loc = state.Nband_loc;
    int Nspin_local = state.Nspin_local;
    int Nkpts = state.Nkpts;
    int Nelectron = state.Nelectron;
    Vec3 cell_lengths = grid_->lattice().lengths();

    int nchefsi = (scf_iter == 0) ? params_.rho_trigger : params_.nchefsi;

    int rank_world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
    if (rank_world == 0 && scf_iter == 0 && nchefsi > 1)
        std::printf("SCF iter 1: %d CheFSI passes (rhoTrigger)\n", nchefsi);

    for (int chefsi_pass = 0; chefsi_pass < nchefsi; ++chefsi_pass) {
        if (is_soc_) {
            for (int k = 0; k < Nkpts; ++k) {
                double* eig = wfn.eigenvalues(0, k).data();
                Complex* psi_c = wfn.psi_kpt(0, k).data();
                int k_glob = kpt_start_ + k;
                Vec3 kpt = kpoints_->kpts_cart()[k_glob];

                if (vnl_ && vnl_->is_setup()) {
                    const_cast<NonlocalProjector*>(vnl_)->set_kpoint(kpt);
                    const_cast<Hamiltonian*>(hamiltonian_)->set_vnl_kpt(vnl_);
                }

                eigsolver.solve_spinor_kpt(psi_c, eig, arrays_.Veff_spinor.data(),
                                            Nd_d, Nband_loc,
                                            state.lambda_cutoff, state.eigval_min[0], state.eigval_max[0],
                                            kpt, cell_lengths,
                                            params_.cheb_degree,
                                            wfn.psi_kpt(0, k).ld());
            }
        } else {
            for (int s = 0; s < Nspin_local; ++s) {
                int s_glob = spin_start_ + s;
                double* Veff_s = arrays_.Veff.data() + s_glob * Nd_d;
                if (is_mgga_type(xc_type_) && Nspin_global_ == 2) {
                    const_cast<Hamiltonian*>(hamiltonian_)->set_vtau(arrays_.vtau.data() + s_glob * Nd_d);
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
                                            state.lambda_cutoff, state.eigval_min[s], state.eigval_max[s],
                                            kpt, cell_lengths,
                                            params_.cheb_degree,
                                            wfn.psi_kpt(s, k).ld());
                    } else {
                        double* psi = wfn.psi(s, k).data();
                        eigsolver.solve(psi, eig, Veff_s, Nd_d, Nband_loc,
                                        state.lambda_cutoff, state.eigval_min[s], state.eigval_max[s],
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
                    state.eigval_min[s] = eigs[0];
                }
                if (eigs[Nband - 1] > eig_last_max)
                    eig_last_max = eigs[Nband - 1];
            }
            state.lambda_cutoff = eig_last_max + 0.1;
        }

        // Compute occupations
        Ef_ = Occupation::compute(wfn, Nelectron, state.beta, params_.smearing,
                                  state.kpt_weights, *kptcomm_, *spincomm_, kpt_start_);
    }

    // Compute tau for mGGA after CheFSI solve
    if (is_mgga_type(xc_type_)) {
        tau_.compute(wfn, state.kpt_weights, *grid_, *domain_, *halo_, *gradient_,
                     kpoints_, *bandcomm_, *kptcomm_, spincomm_,
                     spin_start_, kpt_start_, band_start_, Nspin_global_);
    }
}

void SCF::compute_new_density(const Wavefunction& wfn, const SCFState& state,
                               ElectronDensity& rho_new) {
    int Nd_d = domain_->Nd_d();
    int Nspin = Nspin_global_;

    if (is_soc_) {
        rho_new.allocate_noncollinear(Nd_d);
        rho_new.compute_spinor(wfn, state.kpt_weights, grid_->dV(),
                                *bandcomm_, *kptcomm_, kpt_start_, band_start_);
    } else {
        rho_new.allocate(Nd_d, Nspin);
        rho_new.compute(wfn, state.kpt_weights, grid_->dV(), *bandcomm_, *kptcomm_,
                        Nspin, spin_start_, spincomm_, kpt_start_, band_start_);
    }
}

void SCF::compute_scf_energy(const Wavefunction& wfn, const ElectronDensity& rho_new,
                              const double* rho_b, double Eself, double Ec, SCFState& state) {
    int Nd_d = domain_->Nd_d();
    int Nspin = Nspin_global_;

    if (state.use_potential_mixing) {
        // Save Veff_in, update density to rho_out, compute Veff_out
        NDArray<double> Veff_in(Nd_d * Nspin);
        std::memcpy(Veff_in.data(), arrays_.Veff.data(), Nd_d * Nspin * sizeof(double));

        std::memcpy(density_.rho_total().data(), rho_new.rho_total().data(), Nd_d * sizeof(double));
        for (int s = 0; s < Nspin; ++s)
            std::memcpy(density_.rho(s).data(), rho_new.rho(s).data(), Nd_d * sizeof(double));

        veff_builder_.compute(density_, rho_b, rho_core_, xc_type_, 0.0, params_.poisson_tol, arrays_,
                              tau_.valid() ? tau_.data() : nullptr, tau_.valid());
        state.Veff_out = NDArray<double>(Nd_d * Nspin);
        std::memcpy(state.Veff_out.data(), arrays_.Veff.data(), Nd_d * Nspin * sizeof(double));

        energy_ = Energy::compute_all(wfn, density_, arrays_.Veff.data(), arrays_.phi.data(),
                                       arrays_.exc.data(), arrays_.Vxc.data(), rho_b,
                                       Eself, Ec, state.beta, params_.smearing,
                                       state.kpt_weights, Nd_d, grid_->dV(),
                                       rho_core_, Ef_, kpt_start_,
                                       kptcomm_, spincomm_, Nspin, nullptr,
                                       (is_mgga_type(xc_type_) && tau_.valid()) ? tau_.data() : nullptr,
                                       (is_mgga_type(xc_type_) && tau_.valid()) ? arrays_.vtau.data() : nullptr);

        // Self-consistency correction
        double Escc = Energy::self_consistency_correction(
            density_, state.Veff_out.data(), Veff_in.data(), Nd_d, grid_->dV(), Nspin);
        energy_.Etotal += Escc;

        // Restore Veff_in for mixer
        std::memcpy(arrays_.Veff.data(), Veff_in.data(), Nd_d * Nspin * sizeof(double));
    } else {
        energy_ = Energy::compute_all(wfn, density_, arrays_.Veff.data(), arrays_.phi.data(),
                                       arrays_.exc.data(), arrays_.Vxc.data(), rho_b,
                                       Eself, Ec, state.beta, params_.smearing,
                                       state.kpt_weights, Nd_d, grid_->dV(),
                                       rho_core_, Ef_, kpt_start_,
                                       kptcomm_, spincomm_, Nspin, nullptr,
                                       (is_mgga_type(xc_type_) && tau_.valid()) ? tau_.data() : nullptr,
                                       (is_mgga_type(xc_type_) && tau_.valid()) ? arrays_.vtau.data() : nullptr);
    }
}

bool SCF::check_convergence(const Wavefunction& wfn, const ElectronDensity& rho_new,
                             const SCFState& state, int scf_iter) {
    int Nd_d = domain_->Nd_d();
    int Nspin = Nspin_global_;
    int rank_world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);

    double scf_error = 0.0;
    if (state.use_potential_mixing) {
        double sum_sq_out = 0.0, sum_sq_diff = 0.0;
        for (int i = 0; i < Nd_d * Nspin; ++i) {
            double diff = state.Veff_out.data()[i] - arrays_.Veff.data()[i];
            sum_sq_out += state.Veff_out.data()[i] * state.Veff_out.data()[i];
            sum_sq_diff += diff * diff;
        }
        scf_error = (sum_sq_out > 0.0) ? std::sqrt(sum_sq_diff / sum_sq_out) : 0.0;
    } else {
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

    if (rank_world == 0 && scf_error < params_.tol && scf_iter >= params_.min_iter) {
        std::printf("\nFinal eigenvalues (Ha) and occupations:\n");
        for (int s = 0; s < Nspin_local_; ++s) {
            if (Nspin > 1) std::printf("  Spin %d:\n", spin_start_ + s);
            for (int n = 0; n < state.Nband; ++n) {
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
        return true;
    }
    return false;
}

void SCF::mix_and_update(const ElectronDensity& rho_new, Mixer& mixer,
                          const double* rho_b, const double* rho_core,
                          int Nelectron, SCFState& state) {
    int Nd_d = domain_->Nd_d();
    int Nspin = Nspin_global_;

    if (state.use_potential_mixing) {
        // Mixer handles mean-shift internally via set_potential_mean_shift
        mixer.mix(state.Veff_mixed.data(), state.Veff_out.data(), Nd_d, Nspin);

        // Copy mixed potential to arrays_.Veff for Hamiltonian
        std::memcpy(arrays_.Veff.data(), state.Veff_mixed.data(), Nd_d * Nspin * sizeof(double));
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

        // Mixer handles clamping+renormalization of column 0 (density constraint)
        mixer.mix(dens_in.data(), dens_out.data(), Nd_d, 4);

        // Unpack back into density structure
        std::memcpy(density_.rho_total().data(), dens_in.data(), Nd_d * sizeof(double));
        std::memcpy(density_.rho(0).data(), dens_in.data(), Nd_d * sizeof(double));
        std::memcpy(density_.mag_x().data(), dens_in.data() + Nd_d, Nd_d * sizeof(double));
        std::memcpy(density_.mag_y().data(), dens_in.data() + 2*Nd_d, Nd_d * sizeof(double));
        std::memcpy(density_.mag_z().data(), dens_in.data() + 3*Nd_d, Nd_d * sizeof(double));
    } else if (Nspin == 1) {
        // Mixer handles clamping+renormalization (density constraint)
        mixer.mix(density_.rho_total().data(), rho_new.rho_total().data(), Nd_d);
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

        // Mixer handles clamping+renormalization (density constraint for ncol==2)
        mixer.mix(dens_in.data(), dens_out.data(), Nd_d, 2);

        // Unpack [total | magnetization] back to [up | down]
        double* rho_tot = density_.rho_total().data();
        double* rho_up = density_.rho(0).data();
        double* rho_dn = density_.rho(1).data();
        for (int i = 0; i < Nd_d; ++i) {
            rho_tot[i] = dens_in[i];
            double mag = dens_in[Nd_d + i];
            rho_up[i] = 0.5 * (rho_tot[i] + mag);
            rho_dn[i] = 0.5 * (rho_tot[i] - mag);
        }
    }

    // For density mixing: recompute Veff from mixed density
    if (!state.use_potential_mixing) {
        if (is_soc_) {
            veff_builder_.compute_spinor(density_, rho_b, rho_core, xc_type_, params_.poisson_tol, arrays_);
        } else {
            veff_builder_.compute(density_, rho_b, rho_core, xc_type_, 0.0, params_.poisson_tol, arrays_,
                                  tau_.valid() ? tau_.data() : nullptr, tau_.valid());
        }
    }
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
    arrays_.Veff = NDArray<double>(Nd_d * Nspin);
    arrays_.Vxc = NDArray<double>(Nd_d * Nspin);
    arrays_.exc = NDArray<double>(Nd_d);
    arrays_.phi = NDArray<double>(Nd_d);
    if (dxc_ncol > 0) arrays_.Dxcdgrho = NDArray<double>(Nd_d * dxc_ncol);
    if (is_mgga_type(xc_type)) {
        int vtau_size = (Nspin == 2) ? 2 * Nd_d : Nd_d;
        tau_.allocate(Nd_d, Nspin);
        arrays_.vtau = NDArray<double>(vtau_size);
        xc_type_ = xc_type;
    }

    // Initialize density
    if (density_.Nd_d() == 0) {
        density_.allocate(Nd_d, Nspin);
        if (elec_ && influence_) {
            const_cast<Electrostatics*>(elec_)->compute_atomic_density(
                *crystal_, *influence_, *domain_, *grid_,
                density_.rho_total().data(), Nelectron);
            if (Nspin == 1) {
                std::memcpy(density_.rho(0).data(), density_.rho_total().data(), Nd_d * sizeof(double));
            } else {
                for (int i = 0; i < Nd_d; i++) {
                    density_.rho(0).data()[i] = 0.5 * density_.rho_total().data()[i];
                    density_.rho(1).data()[i] = 0.5 * density_.rho_total().data()[i];
                }
            }
        } else {
            // Uniform init
            double volume = grid_->Nd() * grid_->dV();
            double rho0 = Nelectron / volume;
            if (Nspin == 1) {
                double* rho = density_.rho(0).data();
                for (int i = 0; i < Nd_d; ++i) rho[i] = rho0;
            } else {
                for (int i = 0; i < Nd_d; ++i) {
                    density_.rho(0).data()[i] = rho0 * 0.5;
                    density_.rho(1).data()[i] = rho0 * 0.5;
                }
            }
            double* rho_t = density_.rho_total().data();
            for (int i = 0; i < Nd_d; ++i) rho_t[i] = rho0;
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
    XCType xc_type_gpu = is_hybrid(xc_type) ? hybrid_base_xc(xc_type) : xc_type;
    bool is_gga_gpu = (xc_type_gpu == XCType::GGA_PBE || xc_type_gpu == XCType::GGA_PBEsol ||
                       xc_type_gpu == XCType::GGA_RPBE);
    gpu_runner_ = std::make_unique<GPUSCFRunner>();
    gpu_runner_->set_context(*ctx_);
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
        exx_,
        xc_type);

    // Download results for forces/stress
    gpu_runner_->download_results(
        arrays_.phi.data(), arrays_.Vxc.data(), arrays_.exc.data(), arrays_.Veff.data(),
        dxc_ncol > 0 ? arrays_.Dxcdgrho.data() : nullptr,
        density_.rho_total().data(), wfn);
    // Download mGGA tau/vtau if applicable
    if ((xc_type == XCType::MGGA_SCAN || xc_type == XCType::MGGA_RSCAN || xc_type == XCType::MGGA_R2SCAN) && tau_.size() > 0 && arrays_.vtau.size() > 0) {
        if (Nspin == 2) {
            int gpu_tau_size = 2 * Nd_d;
            int gpu_vtau_size = 2 * Nd_d;
            gpu_runner_->download_tau_vtau(tau_.data(), arrays_.vtau.data(),
                                            gpu_tau_size, gpu_vtau_size);
            // Compute total tau = up + dn (stored at offset 2*Nd_d)
            double* tau_data = tau_.data();
            for (int i = 0; i < Nd_d; i++)
                tau_data[2 * Nd_d + i] = tau_data[i] + tau_data[Nd_d + i];
        } else {
            gpu_runner_->download_tau_vtau(tau_.data(), arrays_.vtau.data(),
                                            tau_.size(), (int)arrays_.vtau.size());
        }
        tau_.set_valid(true);
    }
    // Keep spin densities in sync
    if (Nspin == 2) {
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
