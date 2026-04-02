#include "io/OutputWriter.hpp"
#include "core/constants.hpp"
#include <cstdio>

namespace lynx {

void OutputWriter::print_summary(const SystemConfig& config, const Lattice& lattice,
                                 const FDGrid& grid, int world_rank) {
    if (world_rank != 0) return;

    std::printf("====================================================\n");
    std::printf("              LYNX C++\n");
    std::printf("====================================================\n\n");

    std::printf("Lattice vectors (Bohr):\n");
    for (int i = 0; i < 3; ++i)
        std::printf("  %12.6f %12.6f %12.6f\n",
                    lattice.latvec()(i, 0), lattice.latvec()(i, 1), lattice.latvec()(i, 2));

    Vec3 L = lattice.lengths();
    std::printf("\nLattice lengths: %.6f  %.6f  %.6f Bohr\n", L.x, L.y, L.z);
    std::printf("Cell volume:     %.6f Bohr^3\n", lattice.jacobian());
    std::printf("Cell type:       %s\n",
                lattice.is_orthogonal() ? "Orthogonal" : "Non-orthogonal");

    std::printf("\nGrid: %d x %d x %d = %d\n", grid.Nx(), grid.Ny(), grid.Nz(), grid.Nd());
    std::printf("Mesh spacing: dx=%.6f  dy=%.6f  dz=%.6f Bohr\n",
                grid.dx(), grid.dy(), grid.dz());
    std::printf("FD order: %d\n", config.fd_order);

    int total_atoms = 0;
    for (const auto& at : config.atom_types)
        total_atoms += static_cast<int>(at.coords.size());
    std::printf("\nAtom types: %zu, Total atoms: %d\n",
                config.atom_types.size(), total_atoms);
    for (const auto& at : config.atom_types)
        std::printf("  %s: %zu atoms\n", at.element.c_str(), at.coords.size());

    std::printf("\nNstates: %d\n", config.Nstates);
    std::printf("K-points: %d x %d x %d\n", config.Kx, config.Ky, config.Kz);
    std::printf("SCF tolerance: %.1e\n", config.scf_tol);
    std::printf("Max SCF iterations: %d\n", config.max_scf_iter);
    std::printf("====================================================\n");
}

} // namespace lynx
