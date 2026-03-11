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

        // ===== Setup parallelization =====
        int Nkpts = config.Kx * config.Ky * config.Kz;
        int Nspin = (config.spin_type == sparc::SpinType::None) ? 1 : 2;
        sparc::Parallelization parallel(MPI_COMM_WORLD, config.parallel,
                                        grid, Nspin, Nkpts, config.Nstates);

        const auto& domain = parallel.psi_domain();
        const auto& dmcomm = parallel.dmcomm();
        const auto& bandcomm = parallel.bandcomm();
        const auto& kptcomm = parallel.kptcomm();
        const auto& spincomm = parallel.spincomm();

        if (rank == 0) {
            auto& v = domain.vertices();
            std::printf("\nParallelization: %d procs, domain [%d:%d]x[%d:%d]x[%d:%d] = %d pts\n",
                        nproc, v.xs, v.xe, v.ys, v.ye, v.zs, v.ze, domain.Nd_d());
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
                    pos = lattice.frac_to_cart(pos);
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
        // Adding ~2*h margin covers the FD stencil tail sufficiently.
        rc_max += 2.0 * std::max({grid.dx(), grid.dy(), grid.dz()});

        std::vector<sparc::AtomInfluence> influence;
        crystal.compute_atom_influence(domain, rc_max, influence);

        std::vector<sparc::AtomNlocInfluence> nloc_influence;
        crystal.compute_nloc_influence(domain, nloc_influence);

        if (rank == 0) {
            std::printf("Atom influence computed (rc_max = %.4f Bohr)\n", rc_max);
        }

        // ===== Electrostatics: pseudocharge + Vloc =====
        sparc::Electrostatics elec;
        elec.compute_pseudocharge(crystal, influence, domain, grid, stencil, dmcomm);

        int Nd_d = domain.Nd_d();
        std::vector<double> Vloc(Nd_d, 0.0);
        elec.compute_Vloc(crystal, influence, domain, grid, Vloc.data(), dmcomm);
        elec.compute_Ec(Vloc.data(), Nd_d, grid.dV(), dmcomm);

        if (rank == 0) {
            std::printf("Pseudocharge: int(b) = %.6f (expected: %.1f)\n",
                        elec.int_b(), -static_cast<double>(Nelectron));
            std::printf("Eself = %.6f, Ec = %.6f, Esc = %.6f Ha\n",
                        elec.Eself(), elec.Ec(), elec.Eself() + elec.Ec());
        }

        // ===== Setup operators =====
        sparc::HaloExchange halo(domain, stencil.FDn(), dmcomm.comm());
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
        hamiltonian.setup(stencil, domain, grid, halo, &vnl, dmcomm);

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
        scf.setup(grid, domain, stencil, laplacian, gradient, hamiltonian,
                  halo, &vnl, dmcomm, bandcomm, kptcomm, spincomm, scf_params);

        // ===== Allocate wavefunctions =====
        sparc::Wavefunction wfn;
        wfn.allocate(Nd_d, Nstates, Nspin, Nkpts);

        // ===== Initialize density =====
        // Use atomic superposition if available
        {
            std::vector<double> rho_at(Nd_d, 0.0);
            elec.compute_atomic_density(crystal, influence, domain, grid,
                                        rho_at.data(), Nelectron, dmcomm);

            // Check if atomic density is reasonable (no extreme spikes)
            double rho_max = 0;
            for (int i = 0; i < Nd_d; ++i)
                rho_max = std::max(rho_max, rho_at[i]);

            scf.set_initial_density(rho_at.data(), Nd_d);
            if (rank == 0) {
                double rsum = 0;
                for (int i = 0; i < Nd_d; ++i) rsum += rho_at[i];
                std::printf("Atomic density: max=%.4f, int*dV=%.6f (expected %d)\n",
                            rho_max, rsum * grid.dV(), Nelectron);
            }
        }

        // ===== Compute NLCC core density =====
        std::vector<double> rho_core(Nd_d, 0.0);
        bool has_nlcc = elec.compute_core_density(crystal, influence, domain, grid,
                                                   rho_core.data(), dmcomm);
        if (rank == 0) {
            if (has_nlcc) {
                double rc_sum = 0;
                for (int i = 0; i < Nd_d; ++i) rc_sum += rho_core[i];
                if (!dmcomm.is_null() && dmcomm.size() > 1)
                    rc_sum = dmcomm.allreduce_sum(rc_sum);
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
            std::vector<double> kpt_weights(Nkpts, 1.0 / Nkpts);
            sparc::Forces forces;
            auto f = forces.compute(wfn, crystal, nloc_influence, vnl,
                                    gradient, halo, domain, grid,
                                    scf.phi(), scf.density().rho_total().data(),
                                    kpt_weights, dmcomm, bandcomm, kptcomm, spincomm);

            if (rank == 0) {
                std::printf("\nAtomic forces (Ha/Bohr):\n");
                for (int i = 0; i < Natom; ++i) {
                    std::printf("  Atom %3d: %14.10f %14.10f %14.10f\n",
                                i + 1, f[3*i], f[3*i+1], f[3*i+2]);
                }
            }
        }

        // ===== Stress =====
        if (config.calc_stress || config.calc_pressure) {
            std::vector<double> kpt_weights(Nkpts, 1.0 / Nkpts);
            sparc::Stress stress;
            auto sigma = stress.compute(wfn, crystal, nloc_influence, vnl,
                                        gradient, halo, domain, grid,
                                        scf.phi(), scf.density().rho_total().data(),
                                        elec.pseudocharge().data(),
                                        scf.exc(), scf.Vxc(),
                                        scf.energy().Exc, config.xc,
                                        kpt_weights, dmcomm, bandcomm, kptcomm, spincomm);

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
