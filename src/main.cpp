#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include "io/InputParser.hpp"
#include "io/OutputWriter.hpp"
#include "io/DensityIO.hpp"
#include "core/LynxContext.hpp"
#include "core/ParameterDefaults.hpp"
#include "operators/Hamiltonian.hpp"
#include "operators/NonlocalProjector.hpp"
#include "atoms/Crystal.hpp"
#include "atoms/AtomType.hpp"
#include "electronic/Wavefunction.hpp"
#include "physics/SCF.hpp"
#include "physics/Energy.hpp"
#include "physics/Forces.hpp"
#include "physics/Stress.hpp"
#include "xc/ExactExchange.hpp"
#include "physics/Electrostatics.hpp"

// ============================================================
// Helper: load atoms, create Crystal, compute electrostatics
// ============================================================
struct AtomSetup {
    lynx::Crystal crystal;
    lynx::Electrostatics elec;
    std::vector<double> Vloc;
    std::vector<double> rho_core;
    std::vector<lynx::AtomInfluence> influence;
    std::vector<lynx::AtomNlocInfluence> nloc_influence;
    bool has_nlcc = false;
    int Nelectron = 0;
    int Natom = 0;
};

static AtomSetup setup_atoms(lynx::SystemConfig& config, const lynx::LynxContext& ctx) {
    using namespace lynx;
    const auto& lattice = ctx.lattice();
    const auto& grid = ctx.grid();
    const auto& domain = ctx.domain();
    const auto& stencil = ctx.stencil();
    int rank = ctx.rank();

    // Load pseudopotentials and create Crystal
    std::vector<AtomType> atom_types;
    std::vector<Vec3> all_positions;
    std::vector<int> type_indices;
    int total_Nelectron = 0;

    for (size_t it = 0; it < config.atom_types.size(); ++it) {
        const auto& at_in = config.atom_types[it];
        int n_atoms = static_cast<int>(at_in.coords.size());

        Pseudopotential psd_tmp;
        psd_tmp.load_psp8(at_in.pseudo_file);
        double Zval = psd_tmp.Zval();
        double mass = 1.0;

        AtomType atype(at_in.element, mass, Zval, n_atoms);
        atype.psd().load_psp8(at_in.pseudo_file);

        for (int ia = 0; ia < n_atoms; ++ia) {
            Vec3 pos = at_in.coords[ia];
            if (at_in.fractional) {
                if (lattice.is_orthogonal()) {
                    pos = lattice.frac_to_cart(pos);
                } else {
                    Vec3 L = lattice.lengths();
                    pos = {pos.x * L.x, pos.y * L.y, pos.z * L.z};
                }
            } else {
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

    // Resolve all auto-default parameters
    ParameterDefaults::update_default(config, grid, Nelectron,
                                      (ctx.Nspin() == 2), ctx.is_soc());
    if (rank == 0) {
        double h_eff = ParameterDefaults::compute_h_eff(grid.dx(), grid.dy(), grid.dz());
        std::printf("Parameters resolved: h_eff=%.6f, cheb_degree=%d, elec_temp=%.1f K, "
                    "poisson_tol=%.2e, precond_tol=%.2e, Nstates=%d\n",
                    h_eff, config.cheb_degree, config.elec_temp,
                    config.poisson_tol, config.precond_tol, config.Nstates);
    }

    Crystal crystal(std::move(atom_types), all_positions, type_indices, lattice);

    if (rank == 0) {
        std::printf("Atoms: %d total, %d electrons, %d types\n",
                    Natom, Nelectron, crystal.n_types());
    }

    // Compute atom influence
    double rc_max = 0.0;
    for (int it = 0; it < crystal.n_types(); ++it) {
        const auto& psd = crystal.types()[it].psd();
        for (auto rc : psd.rc()) rc_max = std::max(rc_max, rc);
        if (!psd.radial_grid().empty())
            rc_max = std::max(rc_max, psd.radial_grid().back());
    }
    double h_max = std::max({grid.dx(), grid.dy(), grid.dz()});
    rc_max += 8.0 * h_max;

    std::vector<AtomInfluence> influence;
    crystal.compute_atom_influence(domain, rc_max, influence);

    std::vector<AtomNlocInfluence> nloc_influence;
    crystal.compute_nloc_influence(domain, nloc_influence);

    if (rank == 0) {
        int total_inf = 0;
        for (auto& inf : influence) total_inf += inf.n_atom;
        std::printf("Atom influence computed (rc_max = %.4f Bohr, %d entries)\n", rc_max, total_inf);
        std::printf("Computing pseudocharge...\n");
        std::fflush(stdout);
    }

    // Electrostatics: pseudocharge + Vloc
    Electrostatics elec;
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
    }

    // NLCC core density
    std::vector<double> rho_core(Nd_d, 0.0);
    bool has_nlcc = elec.compute_core_density(crystal, influence, domain, grid,
                                               rho_core.data());
    if (rank == 0) {
        if (has_nlcc) {
            double rc_sum = 0;
            for (int i = 0; i < Nd_d; ++i) rc_sum += rho_core[i];
            std::printf("NLCC: core charge integral = %.6f\n", rc_sum * grid.dV());
        } else {
            std::printf("NLCC: not present for any atom type\n");
        }
    }

    return AtomSetup{std::move(crystal), std::move(elec), std::move(Vloc),
                     std::move(rho_core), std::move(influence), std::move(nloc_influence),
                     has_nlcc, Nelectron, Natom};
}

// ============================================================
// Helper: setup operators (Hamiltonian, NonlocalProjector)
// ============================================================
struct OperatorSetup {
    lynx::Hamiltonian hamiltonian;
    lynx::NonlocalProjector vnl;
};

static OperatorSetup setup_operators(const lynx::SystemConfig& config,
                                     const lynx::LynxContext& ctx,
                                     const lynx::Crystal& crystal,
                                     const std::vector<lynx::AtomNlocInfluence>& nloc_influence) {
    using namespace lynx;
    const auto& domain = ctx.domain();
    const auto& grid = ctx.grid();
    const auto& stencil = ctx.stencil();
    const auto& halo = ctx.halo();
    int rank = ctx.rank();

    NonlocalProjector vnl;
    vnl.setup(crystal, nloc_influence, domain, grid);
    if (ctx.is_soc()) {
        vnl.setup_soc(crystal, nloc_influence, domain, grid);
        if (rank == 0)
            std::printf("SOC projectors: %s\n", vnl.has_soc() ? "enabled" : "none");
    }
    if (rank == 0)
        std::printf("Nonlocal projectors: %d total\n", vnl.total_nproj());

    Hamiltonian hamiltonian;
    hamiltonian.setup(stencil, domain, grid, halo, &vnl);

    return OperatorSetup{std::move(hamiltonian), std::move(vnl)};
}

// ============================================================
// Helper: build SCFParams from config
// ============================================================
static lynx::SCFParams make_scf_params(const lynx::SystemConfig& config) {
    lynx::SCFParams p;
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

// ============================================================
// Helper: initialize density and run SCF
// ============================================================
struct SCFResult {
    lynx::Wavefunction wfn;
    lynx::SCF scf;
};

static SCFResult run_scf(const lynx::SystemConfig& config,
                          const lynx::LynxContext& ctx,
                          const lynx::Crystal& crystal,
                          AtomSetup& atoms,
                          const lynx::Hamiltonian& hamiltonian,
                          const lynx::NonlocalProjector& vnl) {
    using namespace lynx;
    int rank = ctx.rank();
    int Nd_d = ctx.domain().Nd_d();

    auto scf_params = make_scf_params(config);

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
// Helper: compute and print forces
// ============================================================
static void compute_and_print_forces(const lynx::SystemConfig& config,
                                      const lynx::LynxContext& ctx,
                                      const lynx::Wavefunction& wfn,
                                      const lynx::SCF& scf,
                                      const lynx::Crystal& crystal,
                                      const AtomSetup& atoms,
                                      const lynx::NonlocalProjector& vnl) {
    using namespace lynx;
    int rank = ctx.rank();

    Forces forces;
    auto f = forces.compute(ctx, wfn, crystal,
                            atoms.influence, atoms.nloc_influence, vnl,
                            scf.phi(), scf.density().rho_total().data(),
                            atoms.Vloc.data(),
                            atoms.elec.pseudocharge().data(),
                            atoms.elec.pseudocharge_ref().data(),
                            scf.Vxc(),
                            atoms.has_nlcc ? atoms.rho_core.data() : nullptr);

    if (rank == 0) {
        int Natom = atoms.Natom;
        std::printf("\nLocal forces (Ha/Bohr):\n");
        const auto& fl = forces.local_forces();
        for (int i = 0; i < Natom; ++i)
            std::printf("  Atom %3d: %14.10f %14.10f %14.10f\n",
                        i + 1, fl[3*i], fl[3*i+1], fl[3*i+2]);

        std::printf("\nNonlocal forces (Ha/Bohr):\n");
        const auto& fn = forces.nonlocal_forces();
        for (int i = 0; i < Natom; ++i)
            std::printf("  Atom %3d: %14.10f %14.10f %14.10f\n",
                        i + 1, fn[3*i], fn[3*i+1], fn[3*i+2]);

        if (ctx.is_soc()) {
            std::printf("\nSOC forces (Ha/Bohr):\n");
            const auto& fs = forces.soc_forces();
            for (int i = 0; i < Natom; ++i)
                std::printf("  Atom %3d: %14.10f %14.10f %14.10f\n",
                            i + 1, fs[3*i], fs[3*i+1], fs[3*i+2]);
        }

        if (atoms.has_nlcc) {
            std::printf("\nNLCC XC forces (Ha/Bohr):\n");
            const auto& fxc = forces.xc_forces();
            for (int i = 0; i < Natom; ++i)
                std::printf("  Atom %3d: %14.10f %14.10f %14.10f\n",
                            i + 1, fxc[3*i], fxc[3*i+1], fxc[3*i+2]);
        }

        std::printf("\nTotal forces (Ha/Bohr):\n");
        for (int i = 0; i < Natom; ++i)
            std::printf("  Atom %3d: %14.10f %14.10f %14.10f\n",
                        i + 1, f[3*i], f[3*i+1], f[3*i+2]);
    }
}

// ============================================================
// Helper: compute and print stress
// ============================================================
static void compute_and_print_stress(const lynx::SystemConfig& config,
                                      const lynx::LynxContext& ctx,
                                      const lynx::Wavefunction& wfn,
                                      lynx::SCF& scf,
                                      const lynx::Crystal& crystal,
                                      const AtomSetup& atoms,
                                      const lynx::NonlocalProjector& vnl) {
    using namespace lynx;
    int rank = ctx.rank();

    int Nspin_calc = (config.spin_type == SpinType::Collinear) ? 2 :
                     (config.spin_type == SpinType::NonCollinear) ? 1 : 1;
    const double* rho_up_ptr = (Nspin_calc == 2) ? scf.density().rho(0).data() : nullptr;
    const double* rho_dn_ptr = (Nspin_calc == 2) ? scf.density().rho(1).data() : nullptr;

    // GPU mGGA stress
    const double* gpu_mgga_ptr = nullptr;
    const double* gpu_dot_ptr = nullptr;
    std::array<double, 6> gpu_mgga_stress = {};
    double gpu_tau_vtau_dot = 0.0;
#ifdef USE_CUDA
    {
        bool is_mgga = (config.xc == XCType::MGGA_SCAN ||
                        config.xc == XCType::MGGA_RSCAN ||
                        config.xc == XCType::MGGA_R2SCAN);
        if (is_mgga && scf.gpu_runner() && !ctx.is_kpt()) {
            scf.gpu_runner()->compute_mgga_stress(
                wfn, ctx.domain(), ctx.grid(), Nspin_calc,
                gpu_mgga_stress.data(), &gpu_tau_vtau_dot);
            gpu_mgga_ptr = gpu_mgga_stress.data();
            gpu_dot_ptr = &gpu_tau_vtau_dot;
        }
    }
#endif

    Stress stress;
    auto sigma = stress.compute(ctx, wfn, crystal,
                                atoms.influence, atoms.nloc_influence, vnl,
                                scf.phi(), scf.density().rho_total().data(),
                                rho_up_ptr, rho_dn_ptr,
                                atoms.Vloc.data(),
                                atoms.elec.pseudocharge().data(),
                                atoms.elec.pseudocharge_ref().data(),
                                scf.exc(), scf.Vxc(),
                                scf.Dxcdgrho(),
                                scf.energy().Exc,
                                atoms.elec.Eself() + atoms.elec.Ec(),
                                config.xc,
                                Nspin_calc,
                                atoms.has_nlcc ? atoms.rho_core.data() : nullptr,
                                scf.vtau(), scf.tau(),
                                gpu_mgga_ptr, gpu_dot_ptr);

    // Add EXX stress for hybrid functionals
    if (is_hybrid(config.xc) && scf.exx().is_setup()) {
        auto stress_exx = scf.exx().compute_stress(wfn, ctx.gradient(), ctx.halo(), ctx.domain());
        for (int i = 0; i < 6; i++)
            sigma[i] += stress_exx[i];
    }

    if (rank == 0) {
        const double au_to_gpa = 29421.01569650548;
        std::printf("\nStress tensor (GPa):\n");
        std::printf("  sigma_xx = %14.6f  sigma_xy = %14.6f  sigma_xz = %14.6f\n",
                    sigma[0] * au_to_gpa, sigma[1] * au_to_gpa, sigma[2] * au_to_gpa);
        std::printf("  sigma_yy = %14.6f  sigma_yz = %14.6f  sigma_zz = %14.6f\n",
                    sigma[3] * au_to_gpa, sigma[4] * au_to_gpa, sigma[5] * au_to_gpa);
        std::printf("\nPressure: %.6f GPa\n",
                    -(sigma[0] + sigma[3] + sigma[5]) / 3.0 * au_to_gpa);
    }
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0)
            std::fprintf(stderr, "Usage: lynx <input.json>\n");
        MPI_Finalize();
        return 1;
    }

    try {
        // 1. Parse input
        auto config = lynx::InputParser::parse(argv[1]);
        lynx::InputParser::resolve_pseudopotentials(config);
        lynx::InputParser::validate(config);

        // 2. Initialize infrastructure (grid, parallel, operators)
        auto& ctx = lynx::LynxContext::instance();
        ctx.initialize(config, MPI_COMM_WORLD);

        // 3. Setup atoms and electrostatics
        auto atoms = setup_atoms(config, ctx);
        ctx.set_atom_info(atoms.Natom, atoms.Nelectron);

        // 4. Setup operators (Hamiltonian, NonlocalProjector)
        auto ops = setup_operators(config, ctx, atoms.crystal, atoms.nloc_influence);

        // 5. Run SCF
        auto [wfn, scf] = run_scf(config, ctx, atoms.crystal, atoms,
                                   ops.hamiltonian, ops.vnl);

        // 6. Post-SCF: forces
        if (config.print_forces)
            compute_and_print_forces(config, ctx, wfn, scf, atoms.crystal, atoms, ops.vnl);

        // 7. Post-SCF: stress
        if (config.calc_stress || config.calc_pressure)
            compute_and_print_stress(config, ctx, wfn, scf, atoms.crystal, atoms, ops.vnl);

        if (rank == 0) std::printf("\nLYNX calculation complete.\n");

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error on rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Reset LynxContext before MPI_Finalize to free MPI communicators
    // while MPI is still active (the singleton destructor runs too late).
    lynx::LynxContext::instance().reset();
    MPI_Finalize();
    return 0;
}
