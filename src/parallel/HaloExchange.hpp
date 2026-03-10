#pragma once

#include "MPIComm.hpp"
#include "CartTopology.hpp"
#include "core/Domain.hpp"
#include "core/NDArray.hpp"
#include <vector>

namespace sparc {

// Manages ghost/halo zone exchange for finite-difference stencil operations.
// For a domain of size (nx, ny, nz) with stencil half-width FDn,
// the extended domain has size (nx+2*FDn, ny+2*FDn, nz+2*FDn).
class HaloExchange {
public:
    HaloExchange() = default;

    // Setup for a given domain, stencil half-width, and Cartesian communicator
    HaloExchange(const Domain& domain, int FDn, MPI_Comm cart_comm);

    // Fill ghost zones of x_ex from local data x and neighbor data.
    // x: local array of size nx*ny*nz (per column)
    // x_ex: extended array of size (nx+2*FDn)*(ny+2*FDn)*(nz+2*FDn) (per column)
    // ncol: number of columns (e.g., number of bands)
    void execute(const double* x, double* x_ex, int ncol) const;

    // Dimensions of extended domain
    int nx_ex() const { return nx_ex_; }
    int ny_ex() const { return ny_ex_; }
    int nz_ex() const { return nz_ex_; }
    int nd_ex() const { return nx_ex_ * ny_ex_ * nz_ex_; }

    int FDn() const { return FDn_; }

private:
    int nx_ = 0, ny_ = 0, nz_ = 0;
    int nx_ex_ = 0, ny_ex_ = 0, nz_ex_ = 0;
    int FDn_ = 0;
    MPI_Comm cart_comm_ = MPI_COMM_NULL;
    int dims_[3] = {1, 1, 1};
    bool periods_[3] = {true, true, true};

    // Neighbor ranks: 6 faces (±x, ±y, ±z)
    int rank_xm_ = MPI_PROC_NULL, rank_xp_ = MPI_PROC_NULL;
    int rank_ym_ = MPI_PROC_NULL, rank_yp_ = MPI_PROC_NULL;
    int rank_zm_ = MPI_PROC_NULL, rank_zp_ = MPI_PROC_NULL;

    // Copy local data into extended array interior
    void copy_to_interior(const double* x, double* x_ex, int ncol) const;

    // Exchange face halos with 6 neighbors
    void exchange_faces(double* x_ex, int ncol) const;

    // For serial (nproc=1 periodic): wrap boundaries
    void apply_periodic_bc(double* x_ex, int ncol) const;
};

} // namespace sparc
