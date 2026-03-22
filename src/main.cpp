#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include "io/InputParser.hpp"
#include "io/OutputWriter.hpp"
#include "io/DensityIO.hpp"
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
#include "xc/ExactExchange.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (argc < 2) {
        if (rank == 0)
            std::fprintf(stderr, "Usage: lynx <input.json>\n");
        MPI_Finalize();
        return 1;
    }

    try {
        std::string input_file = argv[1];

        // ===== Parse input =====
        auto config = lynx::InputParser::parse(input_file);
        lynx::InputParser::resolve_pseudopotentials(config);
        lynx::InputParser::validate(config);

        // ===== Create lattice and grid =====
        lynx::Lattice lattice(config.latvec, config.cell_type);
        lynx::FDGrid grid(config.Nx, config.Ny, config.Nz, lattice,
                           config.bcx, config.bcy, config.bcz);
        lynx::FDStencil stencil(config.fd_order, grid, lattice);

        lynx::OutputWriter::print_summary(config, lattice, grid, rank);

        // ===== Generate k-point grid =====
        lynx::KPoints kpoints;
        kpoints.generate(config.Kx, config.Ky, config.Kz, config.kpt_shift, lattice);
        int Nkpts = kpoints.Nkpts();  // symmetry-reduced count
        bool is_kpt = !kpoints.is_gamma_only();
        bool is_soc = (config.spin_type == lynx::SpinType::NonCollinear);
        int Nspin = is_soc ? 1 : ((config.spin_type == lynx::SpinType::None) ? 1 : 2);
        int Nspinor = is_soc ? 2 : 1;
        if (is_soc) is_kpt = true;  // SOC always complex

        if (rank == 0 && is_soc) {
            std::printf("Spin-orbit coupling (SOC) enabled: Nspin=%d, Nspinor=%d\n", Nspin, Nspinor);
        }
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
        // SOC: disable spin parallelism (Nspin=1, Nspinor=2)
        if (is_soc) {
            config.parallel.npspin = 1;
        } else if (config.parallel.npspin <= 1 && Nspin == 2 && nproc >= 2) {
            config.parallel.npspin = 2;
        }

        // Auto-determine k-point parallelization if not set
        // Use remaining procs (after spin) for k-points
        int nproc_after_spin = nproc / config.parallel.npspin;
        if (config.parallel.npkpt <= 1 && Nkpts > 1 && nproc_after_spin > 1) {
            config.parallel.npkpt = std::min(nproc_after_spin, Nkpts);
        }

        // Auto-determine band parallelization if not set
        // Use remaining procs (after spin and kpt) for bands
        int nproc_after_kpt = nproc_after_spin / std::max(1, config.parallel.npkpt);
        if (config.parallel.npband <= 1 && nproc_after_kpt > 1) {
            config.parallel.npband = nproc_after_kpt;
        }

        lynx::Parallelization parallel(MPI_COMM_WORLD, config.parallel,
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
        int Nband_local = parallel.Nband_local();
        int band_start = parallel.band_start();

        if (rank == 0) {
            auto& v = domain.vertices();
            std::printf("\nParallelization: %d procs, domain [%d:%d]x[%d:%d]x[%d:%d] = %d pts\n",
                        nproc, v.xs, v.xe, v.ys, v.ye, v.zs, v.ze, domain.Nd_d());
            std::printf("  npspin=%d, npkpt=%d, npband=%d\n",
                        config.parallel.npspin, config.parallel.npkpt, config.parallel.npband);
            std::printf("  This rank: spin_start=%d Nspin_local=%d kpt_start=%d Nkpts_local=%d band_start=%d Nband_local=%d\n",
                        spin_start, Nspin_local, kpt_start, Nkpts_local, band_start, Nband_local);
        }

        // ===== Load pseudopotentials and create Crystal =====
        std::vector<lynx::AtomType> atom_types;
        std::vector<lynx::Vec3> all_positions;
        std::vector<int> type_indices;

        int total_Nelectron = 0;
        for (size_t it = 0; it < config.atom_types.size(); ++it) {
            const auto& at_in = config.atom_types[it];
            int n_atoms = static_cast<int>(at_in.coords.size());

            // Load pseudopotential to get Zval
            lynx::Pseudopotential psd_tmp;
            psd_tmp.load_psp8(at_in.pseudo_file);
            double Zval = psd_tmp.Zval();
            double mass = 1.0; // placeholder — not needed for single-point

            lynx::AtomType atype(at_in.element, mass, Zval, n_atoms);
            atype.psd().load_psp8(at_in.pseudo_file);

            for (int ia = 0; ia < n_atoms; ++ia) {
                lynx::Vec3 pos = at_in.coords[ia];
                if (at_in.fractional) {
                    // For non-orthogonal: store in non-Cartesian coords (scaled fractional)
                    // For orthogonal: non-Cart = Cartesian, so frac_to_cart is fine
                    if (lattice.is_orthogonal()) {
                        pos = lattice.frac_to_cart(pos);
                    } else {
                        lynx::Vec3 L = lattice.lengths();
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

        lynx::Crystal crystal(std::move(atom_types), all_positions, type_indices, lattice);

        if (rank == 0) {
            std::printf("Atoms: %d total, %d electrons, %d types\n",
                        Natom, Nelectron, crystal.n_types());
        }

        // ===== Compute atom influence =====
        // Reference LYNX uses Calculate_PseudochargeCutoff to find per-type cutoffs
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

        std::vector<lynx::AtomInfluence> influence;
        crystal.compute_atom_influence(domain, rc_max, influence);

        std::vector<lynx::AtomNlocInfluence> nloc_influence;
        crystal.compute_nloc_influence(domain, nloc_influence);

        if (rank == 0) {
            int total_inf = 0;
            for (auto& inf : influence) total_inf += inf.n_atom;
            std::printf("Atom influence computed (rc_max = %.4f Bohr, %d entries)\n", rc_max, total_inf);
            std::printf("Computing pseudocharge...\n");
            std::fflush(stdout);
        }

        // ===== Electrostatics: pseudocharge + Vloc =====
        lynx::Electrostatics elec;
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
        lynx::HaloExchange halo(domain, stencil.FDn());
        lynx::Laplacian laplacian(stencil, domain);
        lynx::Gradient gradient(stencil, domain);

        // Setup nonlocal projectors
        lynx::NonlocalProjector vnl;
        vnl.setup(crystal, nloc_influence, domain, grid);

        // Setup SOC projectors if needed
        if (is_soc) {
            vnl.setup_soc(crystal, nloc_influence, domain, grid);
            if (rank == 0) {
                std::printf("SOC projectors: %s\n", vnl.has_soc() ? "enabled" : "none");
            }
        }

        if (rank == 0) {
            std::printf("Nonlocal projectors: %d total\n", vnl.total_nproj());
        }

        // Setup Hamiltonian
        lynx::Hamiltonian hamiltonian;
        hamiltonian.setup(stencil, domain, grid, halo, &vnl);

        // ===== Setup exact exchange (hybrid functionals) =====
        std::unique_ptr<lynx::ExactExchange> exx;
        if (lynx::is_hybrid(config.xc)) {
            exx = std::make_unique<lynx::ExactExchange>();
            const auto& scf_bandcomm_ref = (config.parallel.npband > 1) ? kptcomm : bandcomm;
            int Nstates_exx = config.Nstates;
            if (Nstates_exx <= 0) {
                if (is_soc) Nstates_exx = Nelectron + 20;
                else Nstates_exx = Nelectron / 2 + 10;
            }
            exx->setup(grid, lattice, &kpoints, scf_bandcomm_ref, kpt_bridge, spin_bridge,
                       config.exx_params, Nspin, Nstates_exx,
                       Nband_local, band_start,
                       config.parallel.npband, config.parallel.npkpt, kpt_start, spin_start,
                       config.Kx, config.Ky, config.Kz);
            if (rank == 0)
                std::printf("Hybrid functional: %s with exx_frac=%.4f\n",
                            config.xc == lynx::XCType::HYB_PBE0 ? "PBE0" : "HSE06",
                            config.exx_params.exx_frac);
        }

        // ===== Setup SCF =====
        int Nstates = config.Nstates;
        if (Nstates <= 0) {
            if (is_soc) {
                // SOC: spinor holds both spin components, no /2
                Nstates = Nelectron + 20;
            } else {
                Nstates = Nelectron / 2 + 10;
            }
        }

        lynx::SCFParams scf_params;
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
        scf_params.exx_params = config.exx_params;

        lynx::SCF scf;
        // For band parallelism (npband > 1):
        //   bandcomm groups processes with the same band_index (size 1 without domain decomp)
        //   kptcomm groups processes handling the same kpt (size = npband)
        //   We pass kptcomm as the "bandcomm" to SCF, so that Allreduces over bands
        //   happen across the npband processes sharing the same (spin, kpt).
        //   kpt_bridge connects across different k-points for cross-kpt reductions.
        // For serial (npband=1): bandcomm is size 1, kpt_bridge for cross-kpt.
        const auto& scf_bandcomm = (config.parallel.npband > 1) ? kptcomm : bandcomm;
        scf.setup(grid, domain, stencil, laplacian, gradient, hamiltonian,
                  halo, &vnl, scf_bandcomm, kpt_bridge, spin_bridge, scf_params,
                  Nspin, Nspin_local, spin_start, &kpoints, kpt_start,
                  Nstates, band_start);
        // Set exact exchange on SCF if hybrid
        if (exx) {
            scf.set_exx(exx.get());
        }

#ifdef USE_CUDA
        scf.set_gpu_data(crystal, nloc_influence, influence, elec);
#endif

        // ===== Allocate wavefunctions =====
        // For band parallelism: psi has Nband_local columns, but eigenvalues/occupations
        // have Nstates (Nband_global) entries.
        lynx::Wavefunction wfn;
        wfn.allocate(Nd_d, Nband_local, Nstates, Nspin_local, Nkpts_local, is_kpt, Nspinor);

        // ===== Initialize density =====
        // Try restart file first, fall back to atomic superposition
        {
            bool density_loaded = false;
            if (!config.density_restart_file.empty()) {
                // Pre-allocate density so read() can fill it
                lynx::ElectronDensity restart_rho;
                restart_rho.allocate(Nd_d, Nspin);
                if (lynx::DensityIO::read(config.density_restart_file, restart_rho, grid, lattice)) {
                    scf.set_initial_density(restart_rho.rho_total().data(), Nd_d,
                                            Nspin == 2 ? restart_rho.mag().data() : nullptr);
                    density_loaded = true;
                    if (rank == 0) {
                        double Ne = restart_rho.integrate(grid.dV());
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
                elec.compute_atomic_density(crystal, influence, domain, grid,
                                            rho_at.data(), Nelectron);

                double rho_max = 0;
                for (int i = 0; i < Nd_d; ++i)
                    rho_max = std::max(rho_max, rho_at[i]);

                std::vector<double> mag_init;
                if (!is_soc && Nspin == 2) {
                    mag_init.resize(Nd_d, 0.0);
                    double total_spin = 0.0;
                    for (size_t it = 0; it < config.atom_types.size(); ++it) {
                        const auto& at_in = config.atom_types[it];
                        for (size_t ia = 0; ia < at_in.coords.size(); ++ia) {
                            double atom_spin = 0.0;
                            if (ia < at_in.spin.size()) atom_spin = at_in.spin[ia];
                            total_spin += atom_spin;
                        }
                    }
                    if (std::abs(total_spin) > 1e-12) {
                        double scale = total_spin / static_cast<double>(Nelectron);
                        for (int i = 0; i < Nd_d; ++i)
                            mag_init[i] = scale * rho_at[i];
                    }
                }

                scf.set_initial_density(rho_at.data(), Nd_d,
                                        (!is_soc && Nspin == 2) ? mag_init.data() : nullptr);
                if (rank == 0) {
                    double rsum = 0;
                    for (int i = 0; i < Nd_d; ++i) rsum += rho_at[i];
                    std::printf("Atomic density: max=%.4f, int*dV=%.6f (expected %d)\n",
                                rho_max, rsum * grid.dV(), Nelectron);
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
            if (lynx::is_hybrid(config.xc))
                std::printf("  Eexx    = %18.10f Ha\n", E.Eexx);
            std::printf("  Ehart   = %18.10f Ha\n", E.Ehart);
            std::printf("  Eself   = %18.10f Ha\n", E.Eself);
            std::printf("  Ec      = %18.10f Ha\n", E.Ec);
            std::printf("  Entropy = %18.10f Ha\n", E.Entropy);
            std::printf("  Etotal  = %18.10f Ha\n", E.Etotal);
            std::printf("  Eatom   = %18.10f Ha/atom\n", E.Etotal / Natom);
            std::printf("  Ef      = %18.10f Ha\n", scf.fermi_energy());
        }

        // Write converged density to file if requested
        if (!config.density_output_file.empty() && rank == 0) {
            if (lynx::DensityIO::write(config.density_output_file, scf.density(), grid, lattice)) {
                std::printf("Density written to %s\n", config.density_output_file.c_str());
            } else {
                std::fprintf(stderr, "WARNING: Failed to write density to %s\n",
                             config.density_output_file.c_str());
            }
        }

        // ===== Forces =====
        if (config.print_forces) {
            std::vector<double> kpt_weights = kpoints.normalized_weights();
            lynx::Forces forces;
            auto f = forces.compute(wfn, crystal, influence, nloc_influence, vnl,
                                    stencil, gradient, halo, domain, grid,
                                    scf.phi(), scf.density().rho_total().data(),
                                    Vloc.data(),
                                    elec.pseudocharge().data(),
                                    elec.pseudocharge_ref().data(),
                                    scf.Vxc(),
                                    has_nlcc ? rho_core.data() : nullptr,
                                    kpt_weights, scf_bandcomm, kpt_bridge, spin_bridge, &kpoints, kpt_start, band_start);

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
                if (is_soc) {
                    std::printf("\nSOC forces (Ha/Bohr):\n");
                    const auto& fs = forces.soc_forces();
                    for (int i = 0; i < Natom; ++i) {
                        std::printf("  Atom %3d: %14.10f %14.10f %14.10f\n",
                                    i + 1, fs[3*i], fs[3*i+1], fs[3*i+2]);
                    }
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
            lynx::Stress stress;
            int Nspin_calc = (config.spin_type == lynx::SpinType::Collinear) ? 2 :
                             (config.spin_type == lynx::SpinType::NonCollinear) ? 1 : 1;
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
                                        kpt_weights, scf_bandcomm, kpt_bridge, spin_bridge, &kpoints, kpt_start, band_start,
                                        scf.vtau(), scf.tau());

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

        if (rank == 0) std::printf("\nLYNX calculation complete.\n");

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error on rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
