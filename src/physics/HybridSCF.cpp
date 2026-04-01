#include "physics/HybridSCF.hpp"
#include "xc/ExactExchange.hpp"
#include "core/constants.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mpi.h>

namespace lynx {

static bool is_mgga_type(XCType t) {
    return t == XCType::MGGA_SCAN || t == XCType::MGGA_RSCAN || t == XCType::MGGA_R2SCAN;
}

void HybridSCF::run(Wavefunction& wfn,
                     ElectronDensity& density,
                     VeffArrays& arrays,
                     EffectivePotential& veff_builder,
                     EnergyComponents& energy,
                     double& Ef,
                     bool& converged,
                     ExactExchange* exx,
                     const Hamiltonian* hamiltonian,
                     const NonlocalProjector* vnl,
                     EigenSolver& eigsolver,
                     Mixer& mixer,
                     const SCFParams& params,
                     const FDGrid& grid,
                     const Domain& domain,
                     const MPIComm& bandcomm,
                     const MPIComm& kptcomm,
                     const MPIComm& spincomm,
                     const KPoints* kpoints,
                     int Nelectron,
                     int Natom,
                     int Nspin_global,
                     int Nspin_local,
                     int spin_start,
                     int kpt_start,
                     int band_start,
                     const double* rho_b,
                     const double* rho_core,
                     XCType xc_type,
                     bool is_kpt,
                     double Eself,
                     double Ec,
                     const std::vector<double>& kpt_weights,
                     SCFState& state) {
    int rank_world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
    int Nd_d = domain.Nd_d();
    int Nspin = Nspin_global;
    int Nband = state.Nband;
    Vec3 cell_lengths = grid.lattice().lengths();

    if (rank_world == 0)
        std::printf("\n===== Starting outer Fock loop for hybrid functional =====\n");

    double Eexx_prev = 0.0;
    auto exx_p = params.exx_params;
    // Match SPARC: TOL_FOCK = 0.2 * TOL_SCF when not explicitly set
    if (exx_p.tol_fock < 0.0) {
        exx_p.tol_fock = 0.2 * params.tol;
    }
    if (rank_world == 0)
        std::printf("Fock params: maxit=%d, minit=%d, tol_fock=%.2e\n",
                    exx_p.maxit_fock, exx_p.minit_fock, exx_p.tol_fock);

    // Recompute Vxc/Veff with hybrid-scaled XC
    double exx_frac = params.exx_params.exx_frac;
    mixer.reset();
    veff_builder.compute(density, rho_b, rho_core, xc_type, exx_frac, params.poisson_tol, arrays);

    for (int fock_iter = 0; fock_iter < exx_p.maxit_fock; fock_iter++) {
        // 1. Build ACE operator from current orbitals
        exx->build_ACE(wfn);

        // 2. Compute exchange energy estimate
        double Eexx_est = exx->compute_energy(wfn);
        if (rank_world == 0)
            std::printf("Fock iter %2d: building ACE, Eexx_est = %.10f Ha\n", fock_iter + 1, Eexx_est);

        // 3. Enable Vx in the Hamiltonian for inner SCF
        const_cast<Hamiltonian*>(hamiltonian)->set_exx(exx);

        // 4. Re-estimate Chebyshev filter bounds for hybrid Hamiltonian
        {
            for (int s = 0; s < Nspin_local; ++s) {
                int s_glob = spin_start + s;
                if (is_kpt) {
                    Vec3 kpt0 = kpoints->kpts_cart()[kpt_start];
                    if (vnl && vnl->is_setup()) {
                        const_cast<NonlocalProjector*>(vnl)->set_kpoint(kpt0);
                        const_cast<Hamiltonian*>(hamiltonian)->set_vnl_kpt(vnl);
                    }
                    const_cast<Hamiltonian*>(hamiltonian)->set_exx_context(s_glob, kpt_start);
                    eigsolver.lanczos_bounds_kpt(arrays.Veff.data() + s_glob * Nd_d, Nd_d,
                                                  kpt0, cell_lengths,
                                                  state.eigval_min[s], state.eigval_max[s]);
                } else {
                    const_cast<Hamiltonian*>(hamiltonian)->set_exx_context(s_glob, 0);
                    eigsolver.lanczos_bounds(arrays.Veff.data() + s_glob * Nd_d, Nd_d,
                                              state.eigval_min[s], state.eigval_max[s]);
                }
                state.eigval_max[s] *= 1.01;  // 1% buffer
            }

            // lambda_cutoff = highest previous eigenvalue + margin
            double eig_last_max = -1e30;
            for (int s = 0; s < Nspin_local; ++s) {
                const double* eigs = wfn.eigenvalues(s, 0).data();
                if (eigs[Nband - 1] > eig_last_max)
                    eig_last_max = eigs[Nband - 1];
            }
            state.lambda_cutoff = eig_last_max + 0.1;

            if (rank_world == 0) {
                for (int s = 0; s < Nspin_local; ++s)
                    std::printf("Fock Lanczos bounds (spin %d): eigmin=%.6e, eigmax=%.6e, cutoff=%.6e\n",
                                spin_start + s, state.eigval_min[s], state.eigval_max[s], state.lambda_cutoff);
            }
        }

        // 5. Reset mixer and run inner SCF with EXX
        mixer.reset();
        run_inner_scf(wfn, density, arrays, veff_builder, energy, Ef, converged,
                      Eexx_est, eigsolver, mixer, params, grid, domain,
                      hamiltonian, vnl, bandcomm, kptcomm, spincomm, kpoints,
                      Nelectron, Nspin_global, Nspin_local, spin_start,
                      kpt_start, band_start, rho_b, rho_core, xc_type,
                      is_kpt, Eself, Ec, kpt_weights, state);

        // 6. Energy correction (matching SPARC exactExchange.c:310-325)
        Eexx_prev = Eexx_est;
        energy.Exc -= Eexx_est;
        energy.Etotal += 2.0 * Eexx_est;
        double Eexx_new = exx->compute_energy(wfn);
        energy.Exc += Eexx_new;
        energy.Etotal -= 2.0 * Eexx_new;
        energy.Eexx = Eexx_new;

        // 7. Check Fock convergence
        double err_fock = std::fabs(Eexx_new - Eexx_prev) / std::max(1, Natom);
        if (rank_world == 0) {
            std::printf("Fock iter %2d: Eexx = %.10f, |dEexx|/atom = %.3e, Etot = %.10f\n",
                        fock_iter + 1, Eexx_new, err_fock, energy.Etotal);
        }

        if (err_fock < exx_p.tol_fock && fock_iter >= exx_p.minit_fock) {
            if (rank_world == 0)
                std::printf("Fock loop converged after %d iterations.\n", fock_iter + 1);
            break;
        }
    }

    // Disable Vx in Hamiltonian after Fock loop
    const_cast<Hamiltonian*>(hamiltonian)->set_exx(nullptr);
}

void HybridSCF::run_inner_scf(Wavefunction& wfn,
                                ElectronDensity& density,
                                VeffArrays& arrays,
                                EffectivePotential& veff_builder,
                                EnergyComponents& energy,
                                double& Ef,
                                bool& converged,
                                double Eexx_est,
                                EigenSolver& eigsolver,
                                Mixer& mixer,
                                const SCFParams& params,
                                const FDGrid& grid,
                                const Domain& domain,
                                const Hamiltonian* hamiltonian,
                                const NonlocalProjector* vnl,
                                const MPIComm& bandcomm,
                                const MPIComm& kptcomm,
                                const MPIComm& spincomm,
                                const KPoints* kpoints,
                                int Nelectron,
                                int Nspin_global,
                                int Nspin_local,
                                int spin_start,
                                int kpt_start,
                                int band_start,
                                const double* rho_b,
                                const double* rho_core,
                                XCType xc_type,
                                bool is_kpt,
                                double Eself,
                                double Ec,
                                const std::vector<double>& kpt_weights,
                                SCFState& state) {
    int rank_world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
    int Nd_d = domain.Nd_d();
    int Nspin = Nspin_global;
    int Nband = state.Nband;
    int Nband_loc = state.Nband_loc;
    int Nkpts = state.Nkpts;
    Vec3 cell_lengths = grid.lattice().lengths();

    converged = false;

    for (int scf_iter = 0; scf_iter < params.max_iter; ++scf_iter) {
        int nchefsi_inner = params.nchefsi;
        int cheb_deg_inner = params.cheb_degree;

        for (int chefsi_pass = 0; chefsi_pass < nchefsi_inner; ++chefsi_pass) {
            // Eigensolver (with EXX active via Hamiltonian)
            for (int s = 0; s < Nspin_local; ++s) {
                int s_glob = spin_start + s;
                double* Veff_s = arrays.Veff.data() + s_glob * Nd_d;
                for (int k = 0; k < Nkpts; ++k) {
                    double* eig_inner = wfn.eigenvalues(s, k).data();
                    if (is_kpt) {
                        Complex* psi_c = wfn.psi_kpt(s, k).data();
                        int k_glob = kpt_start + k;
                        Vec3 kpt = kpoints->kpts_cart()[k_glob];
                        if (vnl && vnl->is_setup()) {
                            const_cast<NonlocalProjector*>(vnl)->set_kpoint(kpt);
                            const_cast<Hamiltonian*>(hamiltonian)->set_vnl_kpt(vnl);
                        }
                        const_cast<Hamiltonian*>(hamiltonian)->set_exx_context(s_glob, k_glob);
                        eigsolver.solve_kpt(psi_c, eig_inner, Veff_s, Nd_d, Nband_loc,
                                            state.lambda_cutoff, state.eigval_min[s], state.eigval_max[s],
                                            kpt, cell_lengths,
                                            cheb_deg_inner,
                                            wfn.psi_kpt(s, k).ld());
                    } else {
                        double* psi = wfn.psi(s, k).data();
                        const_cast<Hamiltonian*>(hamiltonian)->set_exx_context(s_glob, 0);
                        eigsolver.solve(psi, eig_inner, Veff_s, Nd_d, Nband_loc,
                                        state.lambda_cutoff, state.eigval_min[s], state.eigval_max[s],
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
                    if (scf_iter > 0) state.eigval_min[s] = eigs[0];
                    if (eigs[Nband - 1] > eig_last_max)
                        eig_last_max = eigs[Nband - 1];
                }
                state.lambda_cutoff = eig_last_max + 0.1;
            }

            Ef = Occupation::compute(wfn, Nelectron, state.beta, params.smearing,
                                      kpt_weights, kptcomm, spincomm, kpt_start);
        }

        // Compute density
        ElectronDensity rho_new_fock;
        rho_new_fock.allocate(Nd_d, Nspin);
        rho_new_fock.compute(wfn, kpt_weights, grid.dV(), bandcomm, kptcomm,
                             Nspin, spin_start, &spincomm, kpt_start, band_start);

        // Energy with Eexx correction
        energy = Energy::compute_all(wfn, density, arrays.Veff.data(), arrays.phi.data(),
                                       arrays.exc.data(), arrays.Vxc.data(), rho_b,
                                       Eself, Ec, state.beta, params.smearing,
                                       kpt_weights, Nd_d, grid.dV(),
                                       rho_core, Ef, kpt_start,
                                       &kptcomm, &spincomm, Nspin);
        energy.Exc += Eexx_est;
        energy.Etotal -= Eexx_est;

        // SCF error
        double scf_error_fock = 0.0;
        {
            const double* rho_in = density.rho_total().data();
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
            std::printf("  Fock SCF %3d: Etot = %18.10f, err = %10.3e, Ef = %10.5f",
                        scf_iter + 1, energy.Etotal, scf_error_fock, Ef);
            if (scf_iter == 0)
                std::printf("  [Eband=%.6f Exc=%.6f Eexx_est=%.6f std=%.6f]",
                            energy.Eband, energy.Exc - Eexx_est, Eexx_est,
                            energy.Etotal + Eexx_est);
            std::printf("\n");
        }

        if (scf_iter >= params.min_iter && scf_error_fock < params.tol) {
            converged = true;
            break;
        }

        // Mix density
        if (Nspin == 1) {
            mixer.mix(density.rho_total().data(), rho_new_fock.rho_total().data(), Nd_d);
            double* rho_mix = density.rho_total().data();
            for (int i = 0; i < Nd_d; ++i) if (rho_mix[i] < 0.0) rho_mix[i] = 0.0;
            {
                double rho_sum = 0.0;
                for (int i = 0; i < Nd_d; ++i) rho_sum += rho_mix[i];
                double Ne_current = rho_sum * grid.dV();
                if (Ne_current > 1e-10) {
                    double scale = static_cast<double>(Nelectron) / Ne_current;
                    for (int i = 0; i < Nd_d; ++i) rho_mix[i] *= scale;
                }
            }
            std::memcpy(density.rho(0).data(), density.rho_total().data(), Nd_d * sizeof(double));
        } else {
            // Spin-polarized mixing
            std::vector<double> dens_in(2 * Nd_d), dens_out(2 * Nd_d);
            const double* rho_up_in = density.rho(0).data();
            const double* rho_dn_in = density.rho(1).data();
            for (int i = 0; i < Nd_d; ++i) {
                dens_in[i] = density.rho_total().data()[i];
                dens_in[Nd_d + i] = rho_up_in[i] - rho_dn_in[i];
            }
            const double* rho_up_out = rho_new_fock.rho(0).data();
            const double* rho_dn_out = rho_new_fock.rho(1).data();
            for (int i = 0; i < Nd_d; ++i) {
                dens_out[i] = rho_new_fock.rho_total().data()[i];
                dens_out[Nd_d + i] = rho_up_out[i] - rho_dn_out[i];
            }
            mixer.mix(dens_in.data(), dens_out.data(), Nd_d, 2);
            double* rho_tot = density.rho_total().data();
            double* rho_up = density.rho(0).data();
            double* rho_dn = density.rho(1).data();
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
                double Ne_current = rho_sum * grid.dV();
                if (Ne_current > 1e-10) {
                    double scale = static_cast<double>(Nelectron) / Ne_current;
                    for (int i = 0; i < Nd_d; ++i) { rho_up[i] *= scale; rho_dn[i] *= scale; rho_tot[i] *= scale; }
                }
            }
        }

        double exx_frac = params.exx_params.exx_frac;
        veff_builder.compute(density, rho_b, rho_core, xc_type, exx_frac, params.poisson_tol, arrays);
    }
}

} // namespace lynx
