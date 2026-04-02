#include <mpi.h>
#include <cstdio>
#include <string>

#include "io/InputParser.hpp"
#include "core/LynxContext.hpp"
#include "core/Driver.hpp"
#include "physics/Forces.hpp"
#include "physics/Stress.hpp"

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
        auto atoms = lynx::Driver::setup_atoms(config, ctx);
        ctx.set_atom_info(atoms.Natom, atoms.Nelectron);

        // 4. Setup operators (Hamiltonian, NonlocalProjector)
        auto ops = lynx::Driver::setup_operators(config, ctx, atoms.crystal, atoms.nloc_influence);

        // 5. Run SCF
        auto [wfn, scf] = lynx::Driver::run_scf(config, ctx, atoms.crystal, atoms,
                                                  ops.hamiltonian, ops.vnl);

        // 6. Post-SCF: forces
        if (config.print_forces)
            lynx::Forces::compute_and_print(config, ctx, wfn, scf, atoms.crystal, atoms, ops.vnl);

        // 7. Post-SCF: stress
        if (config.calc_stress || config.calc_pressure)
            lynx::Stress::compute_and_print(config, ctx, wfn, scf, atoms.crystal, atoms, ops.vnl);

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
