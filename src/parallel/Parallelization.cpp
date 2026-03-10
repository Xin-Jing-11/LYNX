#include "parallel/Parallelization.hpp"
#include <algorithm>
#include <stdexcept>

namespace sparc {

int Parallelization::block_size(int N, int np, int rank) {
    return N / np + ((rank < N % np) ? 1 : 0);
}

int Parallelization::block_start(int N, int np, int rank) {
    return N / np * rank + ((rank < N % np) ? rank : (N % np));
}

Parallelization::Parallelization(MPI_Comm world, const ParallelParams& params,
                                 const FDGrid& grid, int Nspin, int Nkpts, int Nstates) {
    world_ = MPIComm(world, false);
    int nproc = world_.size();
    int my_rank = world_.rank();

    // Validate
    if (params.npspin * params.npkpt * params.npband > nproc)
        throw std::runtime_error("npspin * npkpt * npband exceeds available processes");

    // --- Spin communicator ---
    int size_spincomm = nproc / params.npspin;
    if (my_rank < params.npspin * size_spincomm) {
        spin_index_ = my_rank / size_spincomm;
        Nspin_local_ = block_size(Nspin, params.npspin, spin_index_);
    } else {
        spin_index_ = -1;
        Nspin_local_ = 0;
    }
    spincomm_ = world_.split(spin_index_ >= 0 ? spin_index_ : MPI_UNDEFINED, my_rank);

    // Spin bridge: all procs with same rank within their spincomm
    int rank_in_spin = spincomm_.is_null() ? -1 : spincomm_.rank();
    spin_bridge_ = world_.split(rank_in_spin >= 0 ? rank_in_spin : MPI_UNDEFINED, my_rank);

    // --- K-point communicator ---
    if (!spincomm_.is_null()) {
        int spin_size = spincomm_.size();
        int spin_rank = spincomm_.rank();
        int size_kptcomm = spin_size / params.npkpt;

        if (spin_rank < params.npkpt * size_kptcomm) {
            kpt_index_ = spin_rank / size_kptcomm;
            Nkpts_local_ = block_size(Nkpts, params.npkpt, kpt_index_);
        } else {
            kpt_index_ = -1;
            Nkpts_local_ = 0;
        }
        kptcomm_ = spincomm_.split(kpt_index_ >= 0 ? kpt_index_ : MPI_UNDEFINED, spin_rank);

        // K-point bridge
        int rank_in_kpt = kptcomm_.is_null() ? -1 : kptcomm_.rank();
        kpt_bridge_ = spincomm_.split(rank_in_kpt >= 0 ? rank_in_kpt : MPI_UNDEFINED, spin_rank);
    }

    // --- Band communicator ---
    if (!kptcomm_.is_null()) {
        int kpt_size = kptcomm_.size();
        int kpt_rank = kptcomm_.rank();
        int size_bandcomm = kpt_size / params.npband;

        if (kpt_rank < params.npband * size_bandcomm) {
            // Row-wise assignment: band_index = rank % npband
            band_index_ = kpt_rank % params.npband;
            Nband_local_ = block_size(Nstates, params.npband, band_index_);
            band_start_ = block_start(Nstates, params.npband, band_index_);
            band_end_ = band_start_ + Nband_local_ - 1;
        } else {
            band_index_ = -1;
            Nband_local_ = 0;
        }
        bandcomm_ = kptcomm_.split(band_index_ >= 0 ? band_index_ : MPI_UNDEFINED, kpt_rank);
    }

    // --- Domain communicator ---
    setup_domain(grid, params);
}

void Parallelization::setup_domain(const FDGrid& grid, const ParallelParams& params) {
    if (bandcomm_.is_null()) return;

    int band_size = bandcomm_.size();
    int band_rank = bandcomm_.rank();

    // Determine domain decomposition dimensions
    int dims[3];
    if (params.npNdx > 0 && params.npNdy > 0 && params.npNdz > 0) {
        dims[0] = params.npNdx;
        dims[1] = params.npNdy;
        dims[2] = params.npNdz;
    } else {
        int fd_half = 6; // default FD half-order for min constraint
        auto_cart_dims(band_size, grid.Nx(), grid.Ny(), grid.Nz(), fd_half, dims);
    }

    int cart_size = dims[0] * dims[1] * dims[2];
    bool periods[3] = {
        grid.bcx() == BCType::Periodic,
        grid.bcy() == BCType::Periodic,
        grid.bcz() == BCType::Periodic
    };

    // Create Cartesian topology for psi domain
    if (band_rank < cart_size) {
        int mpi_dims[3] = {dims[0], dims[1], dims[2]};
        int mpi_periods[3] = {periods[0] ? 1 : 0, periods[1] ? 1 : 0, periods[2] ? 1 : 0};
        MPI_Comm cart;
        // First split to get only participating processes
        MPI_Comm sub;
        MPI_Comm_split(bandcomm_.comm(), 0, band_rank, &sub);
        MPI_Cart_create(sub, 3, mpi_dims, mpi_periods, 1, &cart);
        MPI_Comm_free(&sub);

        if (cart != MPI_COMM_NULL) {
            dmcomm_ = MPIComm(cart, true);
            int coord[3];
            MPI_Cart_coords(cart, dmcomm_.rank(), 3, coord);

            DomainVertices verts;
            verts.xs = block_start(grid.Nx(), dims[0], coord[0]);
            int nx_d = block_size(grid.Nx(), dims[0], coord[0]);
            verts.xe = verts.xs + nx_d - 1;

            verts.ys = block_start(grid.Ny(), dims[1], coord[1]);
            int ny_d = block_size(grid.Ny(), dims[1], coord[1]);
            verts.ye = verts.ys + ny_d - 1;

            verts.zs = block_start(grid.Nz(), dims[2], coord[2]);
            int nz_d = block_size(grid.Nz(), dims[2], coord[2]);
            verts.ze = verts.zs + nz_d - 1;

            psi_domain_ = Domain(grid, verts);
        }
    } else {
        MPI_Comm sub;
        MPI_Comm_split(bandcomm_.comm(), MPI_UNDEFINED, band_rank, &sub);
    }

    // Phi domain: use the full kptcomm (or the same as psi for now)
    // For simplicity in Phase 1, phi domain mirrors psi domain
    // Full implementation would use params.npNd*_phi
    phi_domain_ = psi_domain_;
    // dmcomm_phi_ would be set up similarly with possibly different dims
}

} // namespace sparc
