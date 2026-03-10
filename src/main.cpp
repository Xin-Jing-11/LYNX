#include <mpi.h>
#include <cstdio>
#include <string>
#include "io/InputParser.hpp"
#include "io/OutputWriter.hpp"
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include "operators/FDStencil.hpp"
#include "parallel/Parallelization.hpp"

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

        // Parse input
        auto config = sparc::InputParser::parse(input_file);
        sparc::InputParser::validate(config);

        // Create lattice
        sparc::Lattice lattice(config.latvec, config.cell_type);

        // Create grid
        sparc::FDGrid grid(config.Nx, config.Ny, config.Nz, lattice,
                           config.bcx, config.bcy, config.bcz);

        // Create FD stencil
        sparc::FDStencil stencil(config.fd_order, grid, lattice);

        // Print summary
        sparc::OutputWriter::print_summary(config, lattice, grid, rank);

        if (rank == 0) {
            std::printf("\nFD stencil order: %d (half-width: %d)\n",
                        stencil.order(), stencil.FDn());
            std::printf("Max eigenvalue of -0.5*Lap: %.6f\n",
                        stencil.max_eigval_half_lap());
        }

        // Create parallelization
        int Nkpts = config.Kx * config.Ky * config.Kz;
        int Nspin = (config.spin_type == sparc::SpinType::None) ? 1 : 2;
        sparc::Parallelization parallel(MPI_COMM_WORLD, config.parallel,
                                        grid, Nspin, Nkpts, config.Nstates);

        if (rank == 0) {
            std::printf("\nParallelization:\n");
            std::printf("  Total processes: %d\n", nproc);
            std::printf("  Spin comms: %d\n", config.parallel.npspin);
            std::printf("  K-point comms: %d\n", config.parallel.npkpt);
            std::printf("  Band comms: %d\n", config.parallel.npband);
            auto& dom = parallel.psi_domain();
            auto& v = dom.vertices();
            std::printf("  Psi domain (rank 0): [%d:%d] x [%d:%d] x [%d:%d] = %d pts\n",
                        v.xs, v.xe, v.ys, v.ye, v.zs, v.ze, dom.Nd_d());
        }

        if (rank == 0)
            std::printf("\nSPARC initialization complete.\n");

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error on rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
