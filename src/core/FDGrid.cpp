#include "core/FDGrid.hpp"
#include <stdexcept>

namespace sparc {

FDGrid::FDGrid(int Nx, int Ny, int Nz, const Lattice& lattice,
               BCType bcx, BCType bcy, BCType bcz)
    : Nx_(Nx), Ny_(Ny), Nz_(Nz),
      bcx_(bcx), bcy_(bcy), bcz_(bcz),
      lattice_(&lattice) {

    if (Nx <= 0 || Ny <= 0 || Nz <= 0)
        throw std::invalid_argument("Grid dimensions must be positive");

    Nd_ = Nx * Ny * Nz;

    // Mesh spacing = lattice_length / N for periodic, lattice_length / (N-1) for Dirichlet
    Vec3 L = lattice.lengths();
    int nx_intervals = (bcx == BCType::Periodic) ? Nx : Nx - 1;
    int ny_intervals = (bcy == BCType::Periodic) ? Ny : Ny - 1;
    int nz_intervals = (bcz == BCType::Periodic) ? Nz : Nz - 1;

    if (nx_intervals <= 0 || ny_intervals <= 0 || nz_intervals <= 0)
        throw std::invalid_argument("Grid too small for boundary conditions");

    dx_ = L.x / nx_intervals;
    dy_ = L.y / ny_intervals;
    dz_ = L.z / nz_intervals;

    dV_ = dx_ * dy_ * dz_ * lattice.jacobian()
        / (L.x * L.y * L.z);  // normalize by product of lengths
    // For orthogonal cells: jacobian = Lx*Ly*Lz, so dV = dx*dy*dz
    // For non-orthogonal: need jacobian / product of lengths correction
    dV_ = lattice.jacobian() / (nx_intervals * ny_intervals * nz_intervals);
}

} // namespace sparc
