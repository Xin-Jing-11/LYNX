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

    // Setup EffectivePotential builder
    veff_builder_.setup(domain, grid, stencil, laplacian, gradient, hamiltonian, halo, Nspin_global);
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

void SCF::compute_tau(const Wavefunction& wfn,
                       const std::vector<double>& kpt_weights,
                       int kpt_start, int band_start) {
    int Nd_d = domain_->Nd_d();
    int Nband_loc = wfn.Nband();
    int Nspin_local = wfn.Nspin();
    int Nkpts = wfn.Nkpts();

    // Zero tau
    std::memset(arrays_.tau.data(), 0, arrays_.tau.size() * sizeof(double));

    // NOTE: Unlike density which uses spin_fac (2 for non-spin, 1 for spin),
    // tau uses g_nk = occ[n] (no spin_fac) in the accumulation loop.

    // Gradient operator and halo exchange setup
    int nd_ex = halo_->nx_ex() * halo_->ny_ex() * halo_->nz_ex();
    bool is_orth = grid_->lattice().is_orthogonal();
    const Mat3& lapcT = grid_->lattice().lapc_T();
    Vec3 cell_lengths = grid_->lattice().lengths();

    for (int s = 0; s < Nspin_local; ++s) {
        int s_glob = spin_start_ + s;
        double* tau_s = arrays_.tau.data() + s_glob * Nd_d;

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
            bandcomm_->allreduce_sum(arrays_.tau.data() + s_glob * Nd_d, Nd_d);
        }
    }

    // Allreduce over kpt communicator
    if (!kptcomm_->is_null() && kptcomm_->size() > 1) {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start_ + s;
            kptcomm_->allreduce_sum(arrays_.tau.data() + s_glob * Nd_d, Nd_d);
        }
    }

    // Exchange spin channels across spin communicator
    if (spincomm_ && !spincomm_->is_null() && spincomm_->size() > 1 && Nspin_global_ == 2) {
        int my_spin = spin_start_;
        int other_spin = 1 - my_spin;
        int partner = (spincomm_->rank() == 0) ? 1 : 0;
        MPI_Sendrecv(arrays_.tau.data() + my_spin * Nd_d, Nd_d, MPI_DOUBLE, partner, 0,
                     arrays_.tau.data() + other_spin * Nd_d, Nd_d, MPI_DOUBLE, partner, 0,
                     spincomm_->comm(), MPI_STATUS_IGNORE);
    }

    // For spin-polarized: apply 0.5 factor then compute total = up + dn
    if (Nspin_global_ == 2) {
        double* tau_up = arrays_.tau.data();
        double* tau_dn = arrays_.tau.data() + Nd_d;
        double* tau_tot = arrays_.tau.data() + 2 * Nd_d;
        for (int i = 0; i < Nd_d; ++i) {
            tau_up[i] *= 0.5;
            tau_dn[i] *= 0.5;
            tau_tot[i] = tau_up[i] + tau_dn[i];
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
    eigsolver.setup(*hamiltonian_, *halo_, *domain_, *bandcomm_, wfn.Nband_global());

    SCFState state = SCFInitializer::initialize(
        wfn, density_, arrays_, veff_builder_, params_,
        *grid_, *domain_, *hamiltonian_, *halo_, vnl_,
        *bandcomm_, *kptcomm_, *spincomm_, eigsolver, mixer,
        Nelectron, Nspin_global_, Nspin_local_, spin_start_,
        kpoints_, kpt_start_, band_start_,
        xc_type, rho_b, rho_core, is_kpt_, is_soc_);

    state.Nelectron = Nelectron;
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
        compute_tau(wfn, state.kpt_weights, kpt_start_, band_start_);
        arrays_.tau_valid = true;
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

        veff_builder_.compute(density_, rho_b, rho_core_, xc_type_, 0.0, params_.poisson_tol, arrays_);
        state.Veff_out = NDArray<double>(Nd_d * Nspin);
        std::memcpy(state.Veff_out.data(), arrays_.Veff.data(), Nd_d * Nspin * sizeof(double));

        energy_ = Energy::compute_all(wfn, density_, arrays_.Veff.data(), arrays_.phi.data(),
                                       arrays_.exc.data(), arrays_.Vxc.data(), rho_b,
                                       Eself, Ec, state.beta, params_.smearing,
                                       state.kpt_weights, Nd_d, grid_->dV(),
                                       rho_core_, Ef_, kpt_start_,
                                       kptcomm_, spincomm_, Nspin, nullptr,
                                       is_mgga_type(xc_type_) ? arrays_.tau.data() : nullptr,
                                       is_mgga_type(xc_type_) ? arrays_.vtau.data() : nullptr);

        // Self-consistency correction
        double Escc = 0.0;
        if (Nspin == 2) {
            for (int s = 0; s < Nspin; ++s) {
                const double* rho_s = density_.rho(s).data();
                for (int i = 0; i < Nd_d; ++i) {
                    Escc += rho_s[i] * (state.Veff_out.data()[s*Nd_d + i] - Veff_in.data()[s*Nd_d + i]);
                }
            }
        } else {
            const double* rho_tot = density_.rho_total().data();
            for (int i = 0; i < Nd_d; ++i) {
                Escc += rho_tot[i] * (state.Veff_out.data()[i] - Veff_in.data()[i]);
            }
        }
        Escc *= grid_->dV();
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
                                       is_mgga_type(xc_type_) ? arrays_.tau.data() : nullptr,
                                       is_mgga_type(xc_type_) ? arrays_.vtau.data() : nullptr);
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
            veff_builder_.compute(density_, rho_b, rho_core, xc_type_, 0.0, params_.poisson_tol, arrays_);
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
        int tau_size = (Nspin == 2) ? 3 * Nd_d : Nd_d;
        int vtau_size = (Nspin == 2) ? 2 * Nd_d : Nd_d;
        arrays_.tau = NDArray<double>(tau_size);
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
    if ((xc_type == XCType::MGGA_SCAN || xc_type == XCType::MGGA_RSCAN || xc_type == XCType::MGGA_R2SCAN) && arrays_.tau.size() > 0 && arrays_.vtau.size() > 0) {
        if (Nspin == 2) {
            int gpu_tau_size = 2 * Nd_d;
            int gpu_vtau_size = 2 * Nd_d;
            gpu_runner_->download_tau_vtau(arrays_.tau.data(), arrays_.vtau.data(),
                                            gpu_tau_size, gpu_vtau_size);
            for (int i = 0; i < Nd_d; i++)
                arrays_.tau(2 * Nd_d + i) = arrays_.tau(i) + arrays_.tau(Nd_d + i);
        } else {
            gpu_runner_->download_tau_vtau(arrays_.tau.data(), arrays_.vtau.data(),
                                            (int)arrays_.tau.size(), (int)arrays_.vtau.size());
        }
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
