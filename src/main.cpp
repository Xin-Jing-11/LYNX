#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include "io/InputParser.hpp"
#include "io/OutputWriter.hpp"
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include "core/Domain.hpp"
#include "operators/FDStencil.hpp"
#include "operators/Laplacian.hpp"
#include "operators/Gradient.hpp"
#include "operators/Hamiltonian.hpp"
#include "operators/NonlocalProjector.hpp"
#include "atoms/Crystal.hpp"
#include "atoms/AtomType.hpp"
#include "electronic/Wavefunction.hpp"
#include "physics/SCF.hpp"
#include "physics/Energy.hpp"
#include "physics/Forces.hpp"
#include "physics/Stress.hpp"
#include "physics/Electrostatics.hpp"
#include "parallel/Parallelization.hpp"
#include "parallel/HaloExchange.hpp"
#include "core/KPoints.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (argc < 2) {
        if (rank == 0)
            std::fprintf(stderr, "Usage: sparc <input.json>\n");
        MPI_Finalize();
        return 1;
    }

    try {
        std::string input_file = argv[1];

        // ===== Parse input =====
        auto config = sparc::InputParser::parse(input_file);
        sparc::InputParser::validate(config);

        // ===== Create lattice and grid =====
        sparc::Lattice lattice(config.latvec, config.cell_type);
        sparc::FDGrid grid(config.Nx, config.Ny, config.Nz, lattice,
                           config.bcx, config.bcy, config.bcz);
        sparc::FDStencil stencil(config.fd_order, grid, lattice);

        sparc::OutputWriter::print_summary(config, lattice, grid, rank);

        if (rank == 0 && !lattice.is_orthogonal()) {
            auto& lT = lattice.lapc_T();
            std::printf("lapcT = [[%.6f, %.6f, %.6f],\n"
                        "         [%.6f, %.6f, %.6f],\n"
                        "         [%.6f, %.6f, %.6f]]\n",
                lT(0,0), lT(0,1), lT(0,2), lT(1,0), lT(1,1), lT(1,2), lT(2,0), lT(2,1), lT(2,2));
            std::printf("dV = %.10e, jacobian = %.6f\n", grid.dV(), lattice.jacobian());
            // Dump stencil coefficients for comparison with reference
            int FDn = stencil.FDn();
            std::printf("DUMP_STENCIL FDn=%d\n", FDn);
            for (int p = 0; p <= FDn; ++p) {
                std::printf("DUMP_D2_COEFF p=%d x=%.15e y=%.15e z=%.15e\n",
                            p, stencil.D2_coeff_x()[p], stencil.D2_coeff_y()[p], stencil.D2_coeff_z()[p]);
            }
            std::printf("DUMP_D2_XY");
            for (int p = 0; p <= FDn; ++p) std::printf(" %.15e", stencil.D2_coeff_xy()[p]);
            std::printf("\n");
            std::printf("DUMP_D1_Y");
            for (int p = 0; p <= FDn; ++p) std::printf(" %.15e", stencil.D1_coeff_y()[p]);
            std::printf("\n");
        }

        // ===== Generate k-point grid =====
        sparc::KPoints kpoints;
        kpoints.generate(config.Kx, config.Ky, config.Kz, config.kpt_shift, lattice);
        int Nkpts = kpoints.Nkpts();  // symmetry-reduced count
        bool is_kpt = !kpoints.is_gamma_only();
        int Nspin = (config.spin_type == sparc::SpinType::None) ? 1 : 2;

        if (rank == 0) {
            std::printf("K-points: %dx%dx%d grid, %d full -> %d symmetry-reduced%s\n",
                        config.Kx, config.Ky, config.Kz,
                        kpoints.Nkpts_full(), Nkpts,
                        is_kpt ? " (complex)" : " (gamma-only)");
            for (int i = 0; i < Nkpts && i < 10; ++i) {
                auto& kr = kpoints.kpts_red()[i];
                std::printf("  k[%2d]: %8.4f %8.4f %8.4f  wt=%.3f\n",
                            i, kr.x, kr.y, kr.z, kpoints.weights()[i]);
            }
            if (Nkpts > 10)
                std::printf("  ... (%d more)\n", Nkpts - 10);
        }

        // Auto-determine spin parallelization if not set
        // Only auto-enable if Nspin==2, nproc>=2, and user didn't explicitly set it
        if (config.parallel.npspin <= 1 && Nspin == 2 && nproc >= 2) {
            config.parallel.npspin = 2;
        }

        // Auto-determine k-point parallelization if not set
        // Use remaining procs (after spin) for k-points
        int nproc_after_spin = nproc / config.parallel.npspin;
        if (config.parallel.npkpt <= 1 && Nkpts > 1 && nproc_after_spin > 1) {
            config.parallel.npkpt = std::min(nproc_after_spin, Nkpts);
        }

        sparc::Parallelization parallel(MPI_COMM_WORLD, config.parallel,
                                        grid, Nspin, Nkpts, config.Nstates);

        const auto& domain = parallel.domain();
        const auto& bandcomm = parallel.bandcomm();
        const auto& kptcomm = parallel.kptcomm();
        const auto& kpt_bridge = parallel.kpt_bridge();
        const auto& spincomm = parallel.spincomm();
        const auto& spin_bridge = parallel.spin_bridge();
        int Nkpts_local = parallel.Nkpts_local();
        int Nspin_local = parallel.Nspin_local();
        int kpt_start = parallel.kpt_start();
        int spin_start = parallel.spin_start();

        if (rank == 0) {
            auto& v = domain.vertices();
            std::printf("\nParallelization: %d procs, domain [%d:%d]x[%d:%d]x[%d:%d] = %d pts\n",
                        nproc, v.xs, v.xe, v.ys, v.ye, v.zs, v.ze, domain.Nd_d());
            std::printf("  npspin=%d, npkpt=%d, npband=%d\n",
                        config.parallel.npspin, config.parallel.npkpt, config.parallel.npband);
            std::printf("  This rank: spin_start=%d Nspin_local=%d kpt_start=%d Nkpts_local=%d\n",
                        spin_start, Nspin_local, kpt_start, Nkpts_local);
        }

        // ===== Load pseudopotentials and create Crystal =====
        std::vector<sparc::AtomType> atom_types;
        std::vector<sparc::Vec3> all_positions;
        std::vector<int> type_indices;

        int total_Nelectron = 0;
        for (size_t it = 0; it < config.atom_types.size(); ++it) {
            const auto& at_in = config.atom_types[it];
            int n_atoms = static_cast<int>(at_in.coords.size());

            // Load pseudopotential to get Zval
            sparc::Pseudopotential psd_tmp;
            psd_tmp.load_psp8(at_in.pseudo_file);
            double Zval = psd_tmp.Zval();
            double mass = 1.0; // placeholder — not needed for single-point

            sparc::AtomType atype(at_in.element, mass, Zval, n_atoms);
            atype.psd().load_psp8(at_in.pseudo_file);

            for (int ia = 0; ia < n_atoms; ++ia) {
                sparc::Vec3 pos = at_in.coords[ia];
                if (at_in.fractional) {
                    // For non-orthogonal: store in non-Cartesian coords (scaled fractional)
                    // For orthogonal: non-Cart = Cartesian, so frac_to_cart is fine
                    if (lattice.is_orthogonal()) {
                        pos = lattice.frac_to_cart(pos);
                    } else {
                        sparc::Vec3 L = lattice.lengths();
                        pos = {pos.x * L.x, pos.y * L.y, pos.z * L.z};
                    }
                } else {
                    // Cartesian input: convert to non-Cart for non-orth cells
                    if (!lattice.is_orthogonal()) {
                        pos = lattice.cart_to_nonCart(pos);
                    }
                }
                all_positions.push_back(pos);
                type_indices.push_back(static_cast<int>(it));
            }

            total_Nelectron += static_cast<int>(Zval) * n_atoms;
            atom_types.push_back(std::move(atype));
        }

        int Nelectron = (config.Nelectron > 0) ? config.Nelectron : total_Nelectron;
        int Natom = static_cast<int>(all_positions.size());

        sparc::Crystal crystal(std::move(atom_types), all_positions, type_indices, lattice);

        if (rank == 0) {
            std::printf("Atoms: %d total, %d electrons, %d types\n",
                        Natom, Nelectron, crystal.n_types());
        }

        // ===== Compute atom influence =====
        // Reference SPARC uses Calculate_PseudochargeCutoff to find per-type cutoffs
        // based on TOL_PSEUDOCHARGE. We add a margin to rc_max to ensure the
        // pseudocharge influence region is large enough.
        double rc_max = 0.0;
        for (int it = 0; it < crystal.n_types(); ++it) {
            const auto& psd = crystal.types()[it].psd();
            for (auto rc : psd.rc()) rc_max = std::max(rc_max, rc);
            if (!psd.radial_grid().empty())
                rc_max = std::max(rc_max, psd.radial_grid().back());
        }
        // Add margin for FD stencil and pseudocharge tail (matches reference behavior)
        // Reference uses Calculate_PseudochargeCutoff to find per-type cutoffs.
        // Adding ~8*h margin covers the pseudocharge tail sufficiently for most systems.
        // TODO: implement proper Calculate_PseudochargeCutoff for per-type adaptive cutoffs.
        double h_max = std::max({grid.dx(), grid.dy(), grid.dz()});
        rc_max += 8.0 * h_max;

        std::vector<sparc::AtomInfluence> influence;
        crystal.compute_atom_influence(domain, rc_max, influence);

        std::vector<sparc::AtomNlocInfluence> nloc_influence;
        crystal.compute_nloc_influence(domain, nloc_influence);

        if (rank == 0) {
            int total_inf = 0;
            for (auto& inf : influence) total_inf += inf.n_atom;
            std::printf("Atom influence computed (rc_max = %.4f Bohr, %d entries)\n", rc_max, total_inf);
            std::printf("Computing pseudocharge...\n");
            std::fflush(stdout);
        }

        // ===== Electrostatics: pseudocharge + Vloc =====
        sparc::Electrostatics elec;
        elec.compute_pseudocharge(crystal, influence, domain, grid, stencil);

        int Nd_d = domain.Nd_d();
        std::vector<double> Vloc(Nd_d, 0.0);
        elec.compute_Vloc(crystal, influence, domain, grid, Vloc.data());
        elec.compute_Ec(Vloc.data(), Nd_d, grid.dV());

        if (rank == 0) {
            std::printf("Pseudocharge: int(b) = %.6f (expected: %.1f)\n",
                        elec.int_b(), -static_cast<double>(Nelectron));
            std::printf("Eself = %.6f, Ec = %.6f, Esc = %.6f Ha\n",
                        elec.Eself(), elec.Ec(), elec.Eself() + elec.Ec());

            // Debug: print pseudocharge norm and stats
            double b_norm2 = 0, b_min = 1e99, b_max = -1e99;
            for (int i = 0; i < Nd_d; ++i) {
                double bi = elec.pseudocharge().data()[i];
                b_norm2 += bi * bi;
                b_min = std::min(b_min, bi);
                b_max = std::max(b_max, bi);
            }
            std::printf("DEBUG_PSCHG: b_2norm=%.10f b_min=%.10f b_max=%.10f\n",
                        std::sqrt(b_norm2), b_min, b_max);
        }

        // ===== Setup operators =====
        sparc::HaloExchange halo(domain, stencil.FDn());
        sparc::Laplacian laplacian(stencil, domain);
        sparc::Gradient gradient(stencil, domain);

        // Setup nonlocal projectors
        sparc::NonlocalProjector vnl;
        vnl.setup(crystal, nloc_influence, domain, grid);

        if (rank == 0) {
            std::printf("Nonlocal projectors: %d total\n", vnl.total_nproj());
        }

        // Setup Hamiltonian
        sparc::Hamiltonian hamiltonian;
        hamiltonian.setup(stencil, domain, grid, halo, &vnl);

        // ===== Setup SCF =====
        int Nstates = config.Nstates;
        if (Nstates <= 0) Nstates = Nelectron / 2 + 10;

        sparc::SCFParams scf_params;
        scf_params.max_iter = config.max_scf_iter;
        scf_params.min_iter = config.min_scf_iter;
        scf_params.tol = config.scf_tol;
        scf_params.mixing_var = config.mixing_var;
        scf_params.mixing_precond = config.mixing_precond;
        scf_params.mixing_history = config.mixing_history;
        scf_params.mixing_param = config.mixing_param;
        scf_params.smearing = config.smearing;
        scf_params.elec_temp = config.elec_temp;
        scf_params.cheb_degree = config.cheb_degree;
        scf_params.rho_trigger = config.rho_trigger;

        sparc::SCF scf;
        // Pass kpt_bridge/spin_bridge (not kptcomm/spincomm) for cross-group reductions.
        // kptcomm groups same-kpt processes (size=1 when npkpt=nproc),
        // kpt_bridge connects across kpt groups (size=npkpt) for Allreduce/Allgather.
        scf.setup(grid, domain, stencil, laplacian, gradient, hamiltonian,
                  halo, &vnl, bandcomm, kpt_bridge, spin_bridge, scf_params,
                  Nspin, Nspin_local, spin_start, &kpoints, kpt_start);

        // ===== Allocate wavefunctions =====
        sparc::Wavefunction wfn;
        wfn.allocate(Nd_d, Nstates, Nspin_local, Nkpts_local, is_kpt);

        // ===== Initialize density =====
        // Use atomic superposition if available
        {
            std::vector<double> rho_at(Nd_d, 0.0);
            elec.compute_atomic_density(crystal, influence, domain, grid,
                                        rho_at.data(), Nelectron);

            // Check if atomic density is reasonable (no extreme spikes)
            double rho_max = 0;
            for (int i = 0; i < Nd_d; ++i)
                rho_max = std::max(rho_max, rho_at[i]);

            // Compute initial magnetization from per-atom spin values
            // Reference: electronicGroundState.c:Calculate_elecDens
            // atom_spin[J] * rho_at_J(r) for each atom with nonzero spin
            std::vector<double> mag_init;
            if (Nspin == 2) {
                mag_init.resize(Nd_d, 0.0);
                // Simple approach: distribute initial magnetization proportional to
                // atomic density. Total mag = sum(atom_spin_J * Zval_J / total_Zval) * rho_at(r)
                double total_spin = 0.0;
                int atom_idx = 0;
                for (size_t it = 0; it < config.atom_types.size(); ++it) {
                    const auto& at_in = config.atom_types[it];
                    for (size_t ia = 0; ia < at_in.coords.size(); ++ia) {
                        double atom_spin = 0.0;
                        if (ia < at_in.spin.size()) atom_spin = at_in.spin[ia];
                        total_spin += atom_spin;
                        atom_idx++;
                    }
                }
                // Scale magnetization: mag = (total_spin / Nelectron) * rho_at
                if (std::abs(total_spin) > 1e-12) {
                    double scale = total_spin / static_cast<double>(Nelectron);
                    for (int i = 0; i < Nd_d; ++i) {
                        mag_init[i] = scale * rho_at[i];
                    }
                }
            }

            scf.set_initial_density(rho_at.data(), Nd_d,
                                    Nspin == 2 ? mag_init.data() : nullptr);
            if (rank == 0) {
                double rsum = 0, rsum2 = 0;
                for (int i = 0; i < Nd_d; ++i) { rsum += rho_at[i]; rsum2 += rho_at[i]*rho_at[i]; }
                std::printf("Atomic density: max=%.4f, int*dV=%.6f (expected %d)\n",
                            rho_max, rsum * grid.dV(), Nelectron);
                std::printf("DEBUG_RHO0: sum=%.10f norm2=%.10f rho[0]=%.10e rho[100]=%.10e rho[1000]=%.10e\n",
                            rsum, std::sqrt(rsum2), rho_at[0], rho_at[100], rho_at[1000]);
                if (Nspin == 2) {
                    double msum = 0;
                    for (int i = 0; i < Nd_d; ++i) msum += mag_init[i];
                    std::printf("Initial magnetization: int*dV=%.6f\n", msum * grid.dV());
                }
            }
        }

        // ===== Compute NLCC core density =====
        std::vector<double> rho_core(Nd_d, 0.0);
        bool has_nlcc = elec.compute_core_density(crystal, influence, domain, grid,
                                                   rho_core.data());
        // DEBUG: temporarily disable NLCC to isolate energy discrepancy
        // has_nlcc = false;
        if (rank == 0) {
            if (has_nlcc) {
                double rc_sum = 0;
                for (int i = 0; i < Nd_d; ++i) rc_sum += rho_core[i];
                std::printf("NLCC: core charge integral = %.6f\n", rc_sum * grid.dV());
            } else {
                std::printf("NLCC: not present for any atom type\n");
            }
        }

        // ===== Run SCF =====
        if (rank == 0) std::printf("\n===== Starting SCF =====\n");

        double Etot = scf.run(wfn, Nelectron, Natom,
                              elec.pseudocharge().data(), Vloc.data(),
                              elec.Eself(), elec.Ec(), config.xc,
                              has_nlcc ? rho_core.data() : nullptr);

        if (rank == 0) {
            // Debug: dump norms for comparison with reference
            {
                double norm_b = 0, norm_phi = 0, norm_rho = 0, rho_sum = 0;
                double b_min = 1e99, b_max = -1e99, phi_min = 1e99, phi_max = -1e99;
                const double* b = elec.pseudocharge().data();
                const double* phi = scf.phi();
                const double* rho = scf.density().rho_total().data();
                for (int i = 0; i < Nd_d; ++i) {
                    norm_b += b[i]*b[i];
                    b_min = std::min(b_min, b[i]); b_max = std::max(b_max, b[i]);
                    norm_phi += phi[i]*phi[i];
                    phi_min = std::min(phi_min, phi[i]); phi_max = std::max(phi_max, phi[i]);
                    norm_rho += rho[i]*rho[i]; rho_sum += rho[i];
                }
                std::printf("DEBUG_OURS: b_2norm=%.10f b_min=%.10f b_max=%.10f\n", std::sqrt(norm_b), b_min, b_max);
                std::printf("DEBUG_OURS: phi_2norm=%.10f phi_min=%.10f phi_max=%.10f\n", std::sqrt(norm_phi), phi_min, phi_max);
                std::printf("DEBUG_OURS: rho_2norm=%.10f rho_sum=%.10f\n", std::sqrt(norm_rho), rho_sum);
                std::printf("DEBUG_OURS: rho[0]=%.10e rho[100]=%.10e rho[1000]=%.10e\n", rho[0], rho[100], rho[1000]);
                std::printf("DEBUG_OURS: b[0]=%.10e b[100]=%.10e b[1000]=%.10e\n", b[0], b[100], b[1000]);
                std::printf("DEBUG_OURS: phi[0]=%.10e phi[100]=%.10e phi[1000]=%.10e\n", phi[0], phi[100], phi[1000]);
            }
            std::printf("\n===== SCF %s =====\n", scf.converged() ? "CONVERGED" : "NOT CONVERGED");
            const auto& E = scf.energy();
            std::printf("  Eband   = %18.10f Ha\n", E.Eband);
            std::printf("  Exc     = %18.10f Ha\n", E.Exc);
            std::printf("  Ehart   = %18.10f Ha\n", E.Ehart);
            std::printf("  Eself   = %18.10f Ha\n", E.Eself);
            std::printf("  Ec      = %18.10f Ha\n", E.Ec);
            std::printf("  Entropy = %18.10f Ha\n", E.Entropy);
            std::printf("  Etotal  = %18.10f Ha\n", E.Etotal);
            std::printf("  Eatom   = %18.10f Ha/atom\n", E.Etotal / Natom);
            std::printf("  Ef      = %18.10f Ha\n", scf.fermi_energy());
        }

        // ===== Forces =====
        if (config.print_forces) {
            std::vector<double> kpt_weights = kpoints.normalized_weights();
            sparc::Forces forces;
            auto f = forces.compute(wfn, crystal, influence, nloc_influence, vnl,
                                    stencil, gradient, halo, domain, grid,
                                    scf.phi(), scf.density().rho_total().data(),
                                    Vloc.data(),
                                    elec.pseudocharge().data(),
                                    elec.pseudocharge_ref().data(),
                                    scf.Vxc(),
                                    has_nlcc ? rho_core.data() : nullptr,
                                    kpt_weights, bandcomm, kpt_bridge, spin_bridge, &kpoints, kpt_start);

            if (rank == 0) {
                std::printf("\nLocal forces (Ha/Bohr):\n");
                const auto& fl = forces.local_forces();
                for (int i = 0; i < Natom; ++i) {
                    std::printf("  Atom %3d: %14.10f %14.10f %14.10f\n",
                                i + 1, fl[3*i], fl[3*i+1], fl[3*i+2]);
                }
                std::printf("\nNonlocal forces (Ha/Bohr):\n");
                const auto& fn = forces.nonlocal_forces();
                for (int i = 0; i < Natom; ++i) {
                    std::printf("  Atom %3d: %14.10f %14.10f %14.10f\n",
                                i + 1, fn[3*i], fn[3*i+1], fn[3*i+2]);
                }
                if (has_nlcc) {
                    std::printf("\nNLCC XC forces (Ha/Bohr):\n");
                    const auto& fxc = forces.xc_forces();
                    for (int i = 0; i < Natom; ++i) {
                        std::printf("  Atom %3d: %14.10f %14.10f %14.10f\n",
                                    i + 1, fxc[3*i], fxc[3*i+1], fxc[3*i+2]);
                    }
                }
                std::printf("\nTotal forces (Ha/Bohr):\n");
                for (int i = 0; i < Natom; ++i) {
                    std::printf("  Atom %3d: %14.10f %14.10f %14.10f\n",
                                i + 1, f[3*i], f[3*i+1], f[3*i+2]);
                }
            }
        }

        // ===== Stress =====
        if (config.calc_stress || config.calc_pressure) {
            std::vector<double> kpt_weights = kpoints.normalized_weights();
            sparc::Stress stress;
            int Nspin_calc = (config.spin_type == sparc::SpinType::Collinear) ? 2 : 1;
            const double* rho_up_ptr = (Nspin_calc == 2) ? scf.density().rho(0).data() : nullptr;
            const double* rho_dn_ptr = (Nspin_calc == 2) ? scf.density().rho(1).data() : nullptr;
            auto sigma = stress.compute(wfn, crystal, influence, nloc_influence, vnl,
                                        stencil, gradient, halo, domain, grid,
                                        scf.phi(), scf.density().rho_total().data(),
                                        rho_up_ptr, rho_dn_ptr,
                                        Vloc.data(),
                                        elec.pseudocharge().data(),
                                        elec.pseudocharge_ref().data(),
                                        scf.exc(), scf.Vxc(),
                                        scf.Dxcdgrho(),
                                        scf.energy().Exc,
                                        elec.Eself() + elec.Ec(),
                                        config.xc,
                                        Nspin_calc,
                                        has_nlcc ? rho_core.data() : nullptr,
                                        kpt_weights, bandcomm, kpt_bridge, spin_bridge, &kpoints, kpt_start);

            if (rank == 0) {
                std::printf("\nStress tensor (GPa):\n");
                const double au_to_gpa = 29421.01569650548;
                std::printf("  σ_xx = %14.6f  σ_xy = %14.6f  σ_xz = %14.6f\n",
                            sigma[0] * au_to_gpa, sigma[1] * au_to_gpa, sigma[2] * au_to_gpa);
                std::printf("  σ_yy = %14.6f  σ_yz = %14.6f  σ_zz = %14.6f\n",
                            sigma[3] * au_to_gpa, sigma[4] * au_to_gpa, sigma[5] * au_to_gpa);
                std::printf("\nPressure: %.6f GPa\n", stress.pressure() * au_to_gpa);
            }
        }

        if (rank == 0) std::printf("\nSPARC calculation complete.\n");

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error on rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
