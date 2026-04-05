#include "physics/SCF.hpp"
#include "physics/HybridSCF.hpp"
#include "atoms/AtomSetup.hpp"
#include "io/DensityIO.hpp"
#include "electronic/ElectronDensity.hpp"
#include "xc/ExactExchange.hpp"
#include "solvers/KerkerPreconditioner.hpp"
#include "core/constants.hpp"
#ifdef USE_CUDA
#include "core/GPUContext.cuh"
#endif
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
                 Hamiltonian& hamiltonian,
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

    // Device dispatch: GPU when available, CPU otherwise.
    // All SCF operators dispatch internally: EigenSolver, ElectronDensity,
    // EffectivePotential (XC+Poisson+combine), Mixer on GPU.
    // KineticEnergyDensity still falls back to CPU (mGGA only).
    dev_ = Device::CPU;
#ifdef USE_CUDA
    if (gpu_enabled_ && crystal_ && nloc_influence_) {
        dev_ = Device::GPU;
        if (rank_world == 0) {
            std::printf("GPU unified dispatch enabled — all SCF operators on GPU\n");
        }
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

    eigsolver_.setup(*ctx_, *hamiltonian_);

    SCFState state = SCF::initialize_scf(
        *ctx_, wfn, density_, arrays_, veff_builder_, params_,
        *hamiltonian_, vnl_, eigsolver_, mixer,
        Nelectron, xc_type, rho_b, rho_core);

    state.Nelectron = Nelectron;
    Nband_loc_ = state.Nband_loc;
    Nkpts_ = state.Nkpts;

    // Allocate kinetic energy density for mGGA
    if (is_mgga_type(xc_type)) {
        tau_.allocate(Nd_d, Nspin_global_);
    }

    if (Natom <= 0) Natom = std::max(1, Nelectron / 4);
    converged_ = false;

    // ===== 1b. GPU setup: allocate device buffers for GPU operators =====
#ifdef USE_CUDA
    if (dev_ == Device::GPU) {
        int Nband_loc = state.Nband_loc;
        int Nband_g = state.Nband;

        // Initialize shared GPU workspace pool (SCFBuffers in GPUContext).
        // Must be called before any operator setup_gpu since they use ctx.buf.
        {
            auto& gctx = gpu::GPUContext::instance();
            const auto& grid = ctx_->grid();
            const auto& stencil = ctx_->stencil();
            bool is_gga = (xc_type != XCType::LDA_PW && xc_type != XCType::LDA_PZ);
            bool band_parallel = (!ctx_->scf_bandcomm().is_null() && ctx_->scf_bandcomm().size() > 1);
            int mix_ncol = Nspin_global_;
            if (is_soc_) mix_ncol = 4;
            gctx.init_scf_buffers(
                Nd_d, grid.Nx(), grid.Ny(), grid.Nz(), stencil.FDn(),
                Nband_loc, Nband_g, Nspin_global_,
                7, params_.mixing_history, mix_ncol,
                0, 0, 0,  // nonlocal projector sizes (operators alloc their own)
                is_gga, band_parallel, is_kpt_);
        }

        // Hamiltonian: upload stencil, nonlocal projectors, allocate halo workspace
        hamiltonian_->setup_gpu(*ctx_, vnl_, *crystal_, *nloc_influence_, Nband_loc);

        // EigenSolver: allocate CheFSI workspace buffers (real + complex)
        eigsolver_.setup_gpu(*ctx_, Nband_loc, Nband_g, is_kpt_, is_soc_);

        // Allocate per-(spin,kpt) device psi buffers.
        // Each (spin,kpt) gets its own GPU buffer — psi stays resident for
        // entire SCF + forces/stress pipeline. No upload/download during SCF.
        int Nspin_local = state.Nspin_local;
        int Nkpts = state.Nkpts;
        eigsolver_.allocate_psi_buffers(Nspin_local, Nkpts);

        // Upload initial psi ONCE for ALL (spin,kpt).
        // This is the only psi H2D transfer — psi stays on GPU from here until cleanup.
        if (!is_soc_) {
            for (int s = 0; s < Nspin_local; ++s) {
                for (int k = 0; k < Nkpts; ++k) {
                    eigsolver_.set_active_psi(s, k);
                    if (is_kpt_) {
                        eigsolver_.upload_psi_z_to_device(wfn.psi_kpt(s, k).data(), Nd_d, Nband_loc);
                    } else {
                        eigsolver_.upload_psi_to_device(wfn.psi(s, k).data(), Nd_d, Nband_loc);
                    }
                }
            }
            eigsolver_.set_active_psi(0, 0);  // Reset active to first
        }

        // ElectronDensity: lightweight (no persistent device buffers)
        density_.setup_gpu(*ctx_, Nspin_global_);

        // Veff builder: allocate device potential arrays
        veff_builder_.setup_gpu(*ctx_, Nspin_global_, xc_type, rho_b, rho_core);

        // KineticEnergyDensity: allocate tau/vtau device buffers (for mGGA)
        if (is_mgga_type(xc_type)) {
            tau_.setup_gpu(*ctx_, Nspin_global_);
        }

        // Upload initial Veff to device EigenSolver buffer
        eigsolver_.upload_Veff(arrays_.Veff.data(), Nd_d);

        if (rank_world == 0)
            std::printf("GPU-resident SCF: per-(spin,kpt) psi buffers on device (stay resident)\n");
    }
#endif

    // ===== 2. SCF Iteration Loop =====
    ElectronDensity rho_new;
    for (int scf_iter = 0; scf_iter < params_.max_iter; ++scf_iter) {
        solve_eigenproblem(wfn, state, scf_iter);
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
        hybrid.run(*ctx_, wfn, density_, arrays_, veff_builder_,
                   energy_, Ef_, converged_,
                   exx_, hamiltonian_, vnl_,
                   eigsolver_, mixer, params_,
                   Nelectron, Natom,
                   rho_b, rho_core, xc_type,
                   Eself, Ec, state.kpt_weights, state);
    }

    // ===== 4. GPU force+stress + cleanup =====
#ifdef USE_CUDA
    if (dev_ == Device::GPU) {
        // Compute nonlocal force + kinetic/nonlocal stress on GPU while psi is resident.
        // Only scalar results (force array, stress tensor) come to host. Psi stays on device.
        // SOC excluded: spinor eigensolver falls back to CPU, psi stays on host.
        if (!is_soc_) {
            compute_gpu_force_stress(wfn);
        }
        // Free all GPU state — psi never touched the host.
        cleanup_gpu();
    }
#endif

    return energy_.Etotal;
}

#ifdef USE_CUDA
void SCF::cleanup_gpu() {
    if (dev_ != Device::GPU) return;
    hamiltonian_->cleanup_gpu();
    eigsolver_.cleanup_gpu();
    density_.cleanup_gpu();
    veff_builder_.cleanup_gpu();
    if (is_mgga_type(xc_type_)) {
        tau_.cleanup_gpu();
    }
}

void SCF::compute_gpu_force_stress(const Wavefunction& wfn) {
    if (dev_ != Device::GPU) return;

    int Nband_loc = Nband_loc_;
    int Nkpts = Nkpts_;
    int n_atom = crystal_->n_atom_total();

    double occfac = (Nspin_global_ == 1) ? 2.0 : 1.0;

    gpu_fs_.f_nloc.assign(3 * n_atom, 0.0);
    gpu_fs_.stress_k = {};
    gpu_fs_.stress_nl = {};
    gpu_fs_.energy_nl = 0.0;

    std::vector<double> kpt_weights = kpoints_->normalized_weights();

    for (int s = 0; s < Nspin_local_; ++s) {
        for (int k = 0; k < Nkpts; ++k) {
            double wk = kpt_weights[kpt_start_ + k];
            const double* h_occ = wfn.occupations(s, k).data();

            // Per-(spin,kpt) result buffers
            std::vector<double> h_f_tmp(3 * n_atom, 0.0);
            std::array<double, 6> h_sk_tmp = {}, h_snl_tmp = {};
            double h_enl_tmp = 0.0;

            if (is_kpt_) {
                // Set Bloch phases for this k-point
                int k_glob = kpt_start_ + k;
                Vec3 kpt = kpoints_->kpts_cart()[k_glob];
                Vec3 cell_lengths = grid_->lattice().lengths();
                hamiltonian_->set_kpoint_gpu(kpt, cell_lengths);

                double spn_fac_wk = occfac * 2.0 * wk;

                // Hamiltonian method handles occ upload to device internally
                hamiltonian_->compute_force_stress_kpt_gpu(
                    eigsolver_.device_psi_z(s, k), h_occ, Nband_loc,
                    spn_fac_wk,
                    h_f_tmp.data(), h_sk_tmp.data(), h_snl_tmp.data(), &h_enl_tmp);
            } else {
                // Gamma-point: Hamiltonian method handles occ upload internally
                hamiltonian_->compute_force_stress_gpu(
                    eigsolver_.device_psi_real(s, k), h_occ, Nband_loc,
                    occfac,
                    h_f_tmp.data(), h_sk_tmp.data(), h_snl_tmp.data(), &h_enl_tmp);
            }

            // Accumulate across (spin, kpt)
            for (int i = 0; i < 3 * n_atom; ++i)
                gpu_fs_.f_nloc[i] += h_f_tmp[i];
            for (int i = 0; i < 6; ++i) {
                gpu_fs_.stress_k[i] += h_sk_tmp[i];
                gpu_fs_.stress_nl[i] += h_snl_tmp[i];
            }
            gpu_fs_.energy_nl += h_enl_tmp;
        }
    }

    gpu_fs_.computed = true;

    int rank_world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
    if (rank_world == 0)
        std::printf("GPU force+stress computed: psi stayed on device\n");
}
#endif

// ===== Extracted SCF sub-steps =====

void SCF::solve_eigenproblem(Wavefunction& wfn,
                              SCFState& state, int scf_iter) {
    EigenSolver& eigsolver = eigsolver_;
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
                    hamiltonian_->set_vnl_kpt(vnl_);
                }
#ifdef USE_CUDA
                if (dev_ == Device::GPU)
                    hamiltonian_->set_kpoint_gpu(kpt, cell_lengths);
#endif

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
                    hamiltonian_->set_vtau(arrays_.vtau.data() + s_glob * Nd_d);
                }
                for (int k = 0; k < Nkpts; ++k) {
                    double* eig = wfn.eigenvalues(s, k).data();

                    if (is_kpt_) {
                        int k_glob = kpt_start_ + k;
                        Vec3 kpt = kpoints_->kpts_cart()[k_glob];

                        if (vnl_ && vnl_->is_setup()) {
                            const_cast<NonlocalProjector*>(vnl_)->set_kpoint(kpt);
                            hamiltonian_->set_vnl_kpt(vnl_);
                        }
#ifdef USE_CUDA
                        if (dev_ == Device::GPU) {
                            hamiltonian_->set_kpoint_gpu(kpt, cell_lengths);
                            // Activate this (spin,kpt)'s device psi buffer — no upload/download
                            eigsolver.set_active_psi(s, k);
                            eigsolver.solve_kpt_resident(eig, Veff_s, Nd_d, Nband_loc,
                                                          state.lambda_cutoff, state.eigval_min[s], state.eigval_max[s],
                                                          params_.cheb_degree);
                        } else
#endif
                        {
                            Complex* psi_c = wfn.psi_kpt(s, k).data();
                            eigsolver.solve_kpt(psi_c, eig, Veff_s, Nd_d, Nband_loc,
                                                state.lambda_cutoff, state.eigval_min[s], state.eigval_max[s],
                                                kpt, cell_lengths,
                                                params_.cheb_degree,
                                                wfn.psi_kpt(s, k).ld());
                        }
                    } else {
#ifdef USE_CUDA
                        if (dev_ == Device::GPU) {
                            // Activate this (spin,kpt)'s device psi buffer — no upload/download
                            eigsolver.set_active_psi(s, k);
                            eigsolver.solve_resident(eig, Veff_s, Nd_d, Nband_loc,
                                                      state.lambda_cutoff, state.eigval_min[s], state.eigval_max[s],
                                                      params_.cheb_degree);
                        } else
#endif
                        {
                            double* psi = wfn.psi(s, k).data();
                            eigsolver.solve(psi, eig, Veff_s, Nd_d, Nband_loc,
                                            state.lambda_cutoff, state.eigval_min[s], state.eigval_max[s],
                                            params_.cheb_degree,
                                            wfn.psi(s, k).ld());
                        }
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
        tau_.set_device(dev_);
#ifdef USE_CUDA
        if (dev_ == Device::GPU && !is_soc_) {
            // Build device psi pointer vectors — tau reads directly from device psi
            std::vector<const double*> d_psi_real_ptrs;
            std::vector<const void*> d_psi_z_ptrs;
            if (is_kpt_) {
                d_psi_z_ptrs.resize(Nspin_local * Nkpts);
                for (int s = 0; s < Nspin_local; ++s)
                    for (int k = 0; k < Nkpts; ++k)
                        d_psi_z_ptrs[s * Nkpts + k] = eigsolver.device_psi_z(s, k);
            } else {
                d_psi_real_ptrs.resize(Nspin_local * Nkpts);
                for (int s = 0; s < Nspin_local; ++s)
                    for (int k = 0; k < Nkpts; ++k)
                        d_psi_real_ptrs[s * Nkpts + k] = eigsolver.device_psi_real(s, k);
            }
            tau_.compute_gpu_from_device(*ctx_, wfn, state.kpt_weights,
                                          d_psi_real_ptrs, d_psi_z_ptrs);
            // Wire device tau → EffectivePotential for GPU mGGA XC pipeline
            veff_builder_.set_device_tau(tau_.d_tau(), tau_.d_vtau());
        } else
#endif
        {
            tau_.compute(*ctx_, wfn, state.kpt_weights);
        }
    }
}

void SCF::compute_new_density(const Wavefunction& wfn, const SCFState& state,
                               ElectronDensity& rho_new) {
    int Nd_d = domain_->Nd_d();
    int Nspin = Nspin_global_;

    if (is_soc_) {
        rho_new.allocate_noncollinear(Nd_d);
        rho_new.set_device(dev_);
        rho_new.compute_spinor(*ctx_, wfn, state.kpt_weights);
    } else {
        rho_new.allocate(Nd_d, Nspin);
        rho_new.set_device(dev_);
#ifdef USE_CUDA
        if (dev_ == Device::GPU) {
            // Build per-(spin,kpt) device psi pointer vectors.
            // Density reads directly from device — no psi H2D transfer.
            int Nspin_local = state.Nspin_local;
            int Nkpts = state.Nkpts;
            std::vector<const double*> d_psi_real_ptrs;
            std::vector<const void*> d_psi_z_ptrs;
            if (is_kpt_) {
                d_psi_z_ptrs.resize(Nspin_local * Nkpts);
                for (int s = 0; s < Nspin_local; ++s)
                    for (int k = 0; k < Nkpts; ++k)
                        d_psi_z_ptrs[s * Nkpts + k] = eigsolver_.device_psi_z(s, k);
            } else {
                d_psi_real_ptrs.resize(Nspin_local * Nkpts);
                for (int s = 0; s < Nspin_local; ++s)
                    for (int k = 0; k < Nkpts; ++k)
                        d_psi_real_ptrs[s * Nkpts + k] = eigsolver_.device_psi_real(s, k);
            }
            rho_new.compute_from_device_ptrs(*ctx_, wfn, state.kpt_weights,
                                              d_psi_real_ptrs, d_psi_z_ptrs);
        } else
#endif
        {
            rho_new.compute(*ctx_, wfn, state.kpt_weights);
        }
    }
}

void SCF::compute_scf_energy(const Wavefunction& wfn, const ElectronDensity& rho_new,
                              const double* rho_b, double Eself, double Ec, SCFState& state) {
    int Nd_d = domain_->Nd_d();
    int Nspin = Nspin_global_;

    if (state.use_potential_mixing) {
        // Save Veff_in, update density to rho_out, compute Veff_out
        DeviceArray<double> Veff_in(Nd_d * Nspin);
        std::memcpy(Veff_in.data(), arrays_.Veff.data(), Nd_d * Nspin * sizeof(double));

        std::memcpy(density_.rho_total().data(), rho_new.rho_total().data(), Nd_d * sizeof(double));
        for (int s = 0; s < Nspin; ++s)
            std::memcpy(density_.rho(s).data(), rho_new.rho(s).data(), Nd_d * sizeof(double));

        veff_builder_.set_device(dev_);
        veff_builder_.compute(density_, rho_b, rho_core_, xc_type_, 0.0, params_.poisson_tol, arrays_,
                              tau_.valid() ? tau_.data() : nullptr, tau_.valid());
        state.Veff_out = DeviceArray<double>(Nd_d * Nspin);
        std::memcpy(state.Veff_out.data(), arrays_.Veff.data(), Nd_d * Nspin * sizeof(double));

        energy_ = Energy::compute_all(*ctx_, wfn, density_, arrays_.Veff.data(), arrays_.phi.data(),
                                       arrays_.exc.data(), arrays_.Vxc.data(), rho_b,
                                       Eself, Ec, state.beta, params_.smearing,
                                       state.kpt_weights, rho_core_, Ef_,
                                       (is_mgga_type(xc_type_) && tau_.valid()) ? tau_.data() : nullptr,
                                       (is_mgga_type(xc_type_) && tau_.valid()) ? arrays_.vtau.data() : nullptr);

        // Self-consistency correction
        double Escc = Energy::self_consistency_correction(
            density_, state.Veff_out.data(), Veff_in.data(), Nd_d, grid_->dV(), Nspin);
        energy_.Etotal += Escc;

        // Restore Veff_in for mixer
        std::memcpy(arrays_.Veff.data(), Veff_in.data(), Nd_d * Nspin * sizeof(double));
    } else {
        energy_ = Energy::compute_all(*ctx_, wfn, density_, arrays_.Veff.data(), arrays_.phi.data(),
                                       arrays_.exc.data(), arrays_.Vxc.data(), rho_b,
                                       Eself, Ec, state.beta, params_.smearing,
                                       state.kpt_weights, rho_core_, Ef_,
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
        mixer.mix(density_.rho_total().data(), rho_new.rho_total().data(), Nd_d, 1);
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
        veff_builder_.set_device(dev_);
        if (is_soc_) {
            veff_builder_.compute_spinor(density_, rho_b, rho_core, xc_type_, params_.poisson_tol, arrays_);
        } else {
            veff_builder_.compute(density_, rho_b, rho_core, xc_type_, 0.0, params_.poisson_tol, arrays_,
                                  tau_.valid() ? tau_.data() : nullptr, tau_.valid());
        }
    }

    // Note: Veff is on host after recomputation. For GPU-resident SCF,
    // solve_resident() uploads Veff to device at the start of each call.
}


SCFResult SCF::run_calculation(const SystemConfig& config,
                               const LynxContext& ctx,
                               const Crystal& crystal,
                               AtomSetup& atoms,
                               Hamiltonian& hamiltonian,
                               const NonlocalProjector& vnl) {
    int rank = ctx.rank();
    int Nd_d = ctx.domain().Nd_d();

    auto scf_params = SCFParams::from_config(config);

    SCF scf;
    scf.setup(ctx, hamiltonian, &vnl, scf_params);
#ifdef USE_CUDA
    scf.set_gpu_data(crystal, atoms.nloc_influence, atoms.influence, atoms.elec);
#endif

    // Allocate wavefunctions
    Wavefunction wfn;
    wfn.allocate(Nd_d, ctx.Nband_local(), ctx.Nstates(),
                 ctx.Nspin_local(), ctx.Nkpts_local(),
                 ctx.is_kpt(), ctx.Nspinor());

    // Initialize density
    {
        bool density_loaded = false;
        if (!config.density_restart_file.empty()) {
            ElectronDensity restart_rho;
            restart_rho.allocate(Nd_d, ctx.Nspin());
            if (DensityIO::read(config.density_restart_file, restart_rho,
                                ctx.grid(), ctx.lattice())) {
                scf.set_initial_density(restart_rho.rho_total().data(), Nd_d,
                                        ctx.Nspin() == 2 ? restart_rho.mag().data() : nullptr);
                density_loaded = true;
                if (rank == 0) {
                    double Ne = restart_rho.integrate(ctx.grid().dV());
                    std::printf("Density restart: loaded from %s (Ne=%.6f)\n",
                                config.density_restart_file.c_str(), Ne);
                }
            } else if (rank == 0) {
                std::printf("WARNING: Failed to read density from %s, using atomic density\n",
                            config.density_restart_file.c_str());
            }
        }

        if (!density_loaded) {
            std::vector<double> rho_at(Nd_d, 0.0);
            atoms.elec.compute_atomic_density(crystal, atoms.influence, ctx.domain(),
                                              ctx.grid(), rho_at.data(), atoms.Nelectron);

            std::vector<double> mag_init;
            if (!ctx.is_soc() && ctx.Nspin() == 2) {
                mag_init.resize(Nd_d, 0.0);
                double total_spin = 0.0;
                for (size_t it = 0; it < config.atom_types.size(); ++it) {
                    const auto& at_in = config.atom_types[it];
                    for (size_t ia = 0; ia < at_in.coords.size(); ++ia) {
                        double atom_spin = (ia < at_in.spin.size()) ? at_in.spin[ia] : 0.0;
                        total_spin += atom_spin;
                    }
                }
                if (std::abs(total_spin) > 1e-12) {
                    double scale = total_spin / static_cast<double>(atoms.Nelectron);
                    for (int i = 0; i < Nd_d; ++i)
                        mag_init[i] = scale * rho_at[i];
                }
            }

            scf.set_initial_density(rho_at.data(), Nd_d,
                                    (!ctx.is_soc() && ctx.Nspin() == 2) ? mag_init.data() : nullptr);
            if (rank == 0) {
                double rho_max = 0, rsum = 0;
                for (int i = 0; i < Nd_d; ++i) {
                    rho_max = std::max(rho_max, rho_at[i]);
                    rsum += rho_at[i];
                }
                std::printf("Atomic density: max=%.4f, int*dV=%.6f (expected %d)\n",
                            rho_max, rsum * ctx.dV(), atoms.Nelectron);
            }
        }
    }

    // Run SCF
    if (rank == 0) std::printf("\n===== Starting SCF =====\n");

    double Etot = scf.run(wfn, atoms.Nelectron, atoms.Natom,
                          atoms.elec.pseudocharge().data(), atoms.Vloc.data(),
                          atoms.elec.Eself(), atoms.elec.Ec(), config.xc,
                          atoms.has_nlcc ? atoms.rho_core.data() : nullptr);

    if (rank == 0) {
        std::printf("\n===== SCF %s =====\n", scf.converged() ? "CONVERGED" : "NOT CONVERGED");
        const auto& E = scf.energy();
        std::printf("  Eband   = %18.10f Ha\n", E.Eband);
        std::printf("  Exc     = %18.10f Ha\n", E.Exc);
        std::printf("  Ehart   = %18.10f Ha\n", E.Ehart);
        std::printf("  Eself   = %18.10f Ha\n", E.Eself);
        std::printf("  Ec      = %18.10f Ha\n", E.Ec);
        std::printf("  Entropy = %18.10f Ha\n", E.Entropy);
        std::printf("  Etotal  = %18.10f Ha\n", E.Etotal);
        std::printf("  Eatom   = %18.10f Ha/atom\n", E.Etotal / atoms.Natom);
        std::printf("  Ef      = %18.10f Ha\n", scf.fermi_energy());
    }

    // Write converged density
    if (!config.density_output_file.empty() && rank == 0) {
        if (DensityIO::write(config.density_output_file, scf.density(),
                             ctx.grid(), ctx.lattice())) {
            std::printf("Density written to %s\n", config.density_output_file.c_str());
        } else {
            std::fprintf(stderr, "WARNING: Failed to write density to %s\n",
                         config.density_output_file.c_str());
        }
    }

    return SCFResult{std::move(wfn), std::move(scf)};
}

// ============================================================
// SCF initialization (merged from SCFInitializer)
// ============================================================

void SCF::init_density(ElectronDensity& density, int Nd_d, int Nelectron,
                       int Nspin_global, const FDGrid& grid, bool is_soc) {
    double volume = grid.Nd() * grid.dV();

    if (is_soc) {
        if (density.Nd_d() == 0) {
            density.initialize_uniform_noncollinear(Nd_d, Nelectron, volume);
        } else if (density.mag_x().size() == 0) {
            DeviceArray<double> rho_save = density.rho_total().clone();
            density.allocate_noncollinear(Nd_d);
            std::memcpy(density.rho_total().data(), rho_save.data(), Nd_d * sizeof(double));
            std::memcpy(density.rho(0).data(), rho_save.data(), Nd_d * sizeof(double));
            density.mag_x().zero();
            density.mag_y().zero();
            density.mag_z().zero();
        }
    } else {
        if (density.Nd_d() == 0) {
            density.initialize_uniform(Nd_d, Nspin_global, Nelectron, volume);
        }
    }
}

void SCF::randomize_wavefunctions(Wavefunction& wfn, int Nspin_local, int spin_start,
                                   const MPIComm& spincomm, const MPIComm& bandcomm,
                                   bool is_kpt) {
    int spincomm_rank = spincomm.is_null() ? 0 : spincomm.rank();
    int bandcomm_rank = bandcomm.is_null() ? 0 : bandcomm.rank();
    int Nkpts = wfn.Nkpts();

    for (int s = 0; s < Nspin_local; ++s) {
        int s_glob = spin_start + s;
        unsigned rand_seed = spincomm_rank * 100 + bandcomm_rank * 10 + s_glob * 1000 + 1;
        for (int k = 0; k < Nkpts; ++k) {
            if (is_kpt) {
                wfn.randomize_kpt(s, k, rand_seed);
            } else {
                wfn.randomize(s, k, rand_seed);
            }
        }
    }
}

void SCF::estimate_spectral_bounds(
    SCFState& state, EigenSolver& eigsolver,
    const double* Veff, const double* Veff_spinor,
    int Nd_d, int Nspin_local, int spin_start,
    bool is_kpt, bool is_soc, const KPoints* kpoints, int kpt_start,
    const Vec3& cell_lengths, Hamiltonian& hamiltonian,
    const NonlocalProjector* vnl, int rank_world) {

    state.eigval_min.resize(Nspin_local, 0.0);
    state.eigval_max.resize(Nspin_local, 0.0);

    if (is_soc) {
        Vec3 kpt0 = kpoints->kpts_cart()[kpt_start];
        if (vnl && vnl->is_setup()) {
            const_cast<NonlocalProjector*>(vnl)->set_kpoint(kpt0);
            hamiltonian.set_vnl_kpt(vnl);
        }
        double eigmin_spinor, eigmax_spinor;
        eigsolver.lanczos_bounds_spinor_kpt(Veff_spinor, Nd_d,
                                             kpt0, cell_lengths,
                                             eigmin_spinor, eigmax_spinor);
        double eigmin_scalar, eigmax_scalar;
        eigsolver.lanczos_bounds_kpt(Veff, Nd_d, kpt0, cell_lengths,
                                      eigmin_scalar, eigmax_scalar);
        state.eigval_min[0] = eigmin_spinor;
        state.eigval_max[0] = eigmax_scalar;

        double eigmin_s2, eigmax_s2;
        eigsolver.lanczos_bounds_kpt(Veff, Nd_d, kpt0, cell_lengths,
                                      eigmin_s2, eigmax_s2);
        state.lambda_cutoff = 0.5 * (eigmin_s2 + state.eigval_max[0]);
    } else {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start + s;
            if (is_kpt) {
                Vec3 kpt0 = kpoints->kpts_cart()[kpt_start];
                eigsolver.lanczos_bounds_kpt(Veff + s_glob * Nd_d, Nd_d,
                                              kpt0, cell_lengths,
                                              state.eigval_min[s], state.eigval_max[s]);
            } else {
                eigsolver.lanczos_bounds(Veff + s_glob * Nd_d, Nd_d,
                                          state.eigval_min[s], state.eigval_max[s]);
            }
        }
        state.lambda_cutoff = 0.5 * (state.eigval_min[0] + state.eigval_max[0]);
    }

    if (rank_world == 0) {
        for (int s = 0; s < Nspin_local; ++s)
            std::printf("Lanczos bounds (spin %d): eigmin=%.6e, eigmax=%.6e\n",
                        spin_start + s, state.eigval_min[s], state.eigval_max[s]);
    }
}

SCFState SCF::initialize_scf(
    const LynxContext& ctx,
    Wavefunction& wfn, ElectronDensity& density,
    VeffArrays& arrays, EffectivePotential& veff_builder,
    SCFParams& params, Hamiltonian& hamiltonian,
    const NonlocalProjector* vnl, EigenSolver& eigsolver, Mixer& mixer,
    int Nelectron, XCType xc_type,
    const double* rho_b, const double* rho_core) {

    const auto& grid = ctx.grid();
    const auto& domain = ctx.domain();
    int rank_world = ctx.rank();
    int Nd_d = domain.Nd_d();
    int Nspin = ctx.Nspin();
    int Nspin_local = ctx.Nspin_local();
    int spin_start = ctx.spin_start();
    int kpt_start = ctx.kpt_start();
    int band_start = ctx.band_start();
    bool is_kpt = ctx.is_kpt();
    bool is_soc = ctx.is_soc();
    const auto* kpoints = &ctx.kpoints();

    // 1. Populate state scalars
    SCFState state;
    state.Nband = wfn.Nband_global();
    state.Nband_loc = wfn.Nband();
    state.Nspin_local = wfn.Nspin();
    state.Nkpts = wfn.Nkpts();
    state.kBT = params.elec_temp * constants::KB;
    state.beta = 1.0 / state.kBT;

    // K-point weights
    int Nkpts_global = kpoints ? kpoints->Nkpts() : state.Nkpts;
    if (kpoints) {
        state.kpt_weights = kpoints->normalized_weights();
    } else {
        state.kpt_weights.assign(Nkpts_global, 1.0 / Nkpts_global);
    }

    // 2. Allocate work arrays
    arrays.allocate(Nd_d, Nspin, xc_type, is_soc);

    // 3. Initialize density
    init_density(density, Nd_d, Nelectron, Nspin, grid, is_soc);

    // 4. Setup potential mixing state
    state.use_potential_mixing = (params.mixing_var == MixingVariable::Potential) && !is_soc;
    if (state.use_potential_mixing) {
        state.Veff_mean.resize(Nspin, 0.0);
    }

    // 5. Compute initial Veff from initial density
    if (is_soc) {
        veff_builder.compute_spinor(density, rho_b, rho_core, xc_type, params.poisson_tol, arrays);
    } else {
        veff_builder.compute(density, rho_b, rho_core, xc_type, 0.0, params.poisson_tol, arrays);
    }

    // 6. For potential mixing: initialize zero-mean copy
    if (state.use_potential_mixing) {
        state.Veff_mixed = DeviceArray<double>(Nd_d * Nspin);
        std::memcpy(state.Veff_mixed.data(), arrays.Veff.data(), Nd_d * Nspin * sizeof(double));
        for (int s = 0; s < Nspin; ++s) {
            double mean = 0;
            for (int i = 0; i < Nd_d; ++i) mean += state.Veff_mixed.data()[s*Nd_d + i];
            mean /= grid.Nd();
            state.Veff_mean[s] = mean;
            for (int i = 0; i < Nd_d; ++i) state.Veff_mixed.data()[s*Nd_d + i] -= mean;
        }
    }

    // 7. Estimate spectral bounds via Lanczos
    Vec3 cell_lengths = grid.lattice().lengths();
    estimate_spectral_bounds(
        state, eigsolver,
        arrays.Veff.data(),
        is_soc ? arrays.Veff_spinor.data() : nullptr,
        Nd_d, Nspin_local, spin_start,
        is_kpt, is_soc, kpoints, kpt_start,
        cell_lengths, hamiltonian, vnl, rank_world);

    if (rank_world == 0 && params.cheb_degree > 0)
        std::printf("Chebyshev degree: %d\n", params.cheb_degree);

    // 8. Randomize wavefunctions
    randomize_wavefunctions(wfn, Nspin_local, spin_start,
                            ctx.spin_bridge(), ctx.scf_bandcomm(), is_kpt);

    return state;
}

} // namespace lynx
