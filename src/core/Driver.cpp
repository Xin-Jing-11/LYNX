#include "core/Driver.hpp"
#include "core/ParameterDefaults.hpp"
#include "io/DensityIO.hpp"
#include "electronic/ElectronDensity.hpp"

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace lynx {

AtomSetup Driver::setup_atoms(SystemConfig& config, const LynxContext& ctx) {
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

OperatorSetup Driver::setup_operators(const SystemConfig& config,
                                       const LynxContext& ctx,
                                       const Crystal& crystal,
                                       const std::vector<AtomNlocInfluence>& nloc_influence) {
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

SCFResult Driver::run_scf(const SystemConfig& config,
                            const LynxContext& ctx,
                            const Crystal& crystal,
                            AtomSetup& atoms,
                            const Hamiltonian& hamiltonian,
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

} // namespace lynx
