#include <mpi.h>
#include <cstdio>
#include <string>

#include "io/InputParser.hpp"
#include "core/LynxContext.hpp"
#include "atoms/Crystal.hpp"
#include "atoms/AtomSetup.hpp"
#include "operators/NonlocalProjector.hpp"
#include "operators/Hamiltonian.hpp"
#include "physics/SCF.hpp"
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
        auto atoms = lynx::Crystal::setup(config, ctx);
        ctx.set_atom_info(atoms.Natom, atoms.Nelectron);

        // Idle ranks (no spin/kpt/band work) skip computation
        if (ctx.is_active()) {
            // 4. Setup operators (NonlocalProjector, Hamiltonian)
            auto vnl = lynx::NonlocalProjector::create(ctx, atoms.crystal, atoms.nloc_influence);
            lynx::Hamiltonian hamiltonian;
            hamiltonian.setup(ctx.stencil(), ctx.domain(), ctx.grid(), ctx.halo(), &vnl);

            // 5. Run SCF
            auto [wfn, scf] = lynx::SCF::run_calculation(config, ctx, atoms.crystal, atoms,
                                                            hamiltonian, vnl);

            // 6. Forces
            lynx::Forces forces;
            forces.compute(ctx, config, wfn, scf, atoms, vnl);

            // 7. Stress
            lynx::Stress stress;
            stress.compute(ctx, config, wfn, scf, atoms, vnl);
        }

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
