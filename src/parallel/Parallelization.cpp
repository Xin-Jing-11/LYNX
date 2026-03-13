#include "parallel/Parallelization.hpp"
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

    if (params.npspin * params.npkpt * params.npband > nproc)
        throw std::runtime_error("npspin * npkpt * npband exceeds available processes");

    // --- Spin communicator ---
    int size_spincomm = nproc / params.npspin;
    if (my_rank < params.npspin * size_spincomm) {
        spin_index_ = my_rank / size_spincomm;
        Nspin_local_ = block_size(Nspin, params.npspin, spin_index_);
        spin_start_ = block_start(Nspin, params.npspin, spin_index_);
    } else {
        spin_index_ = -1;
        Nspin_local_ = 0;
    }
    spincomm_ = world_.split(spin_index_ >= 0 ? spin_index_ : MPI_UNDEFINED, my_rank);

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
            kpt_start_ = block_start(Nkpts, params.npkpt, kpt_index_);
        } else {
            kpt_index_ = -1;
            Nkpts_local_ = 0;
        }
        kptcomm_ = spincomm_.split(kpt_index_ >= 0 ? kpt_index_ : MPI_UNDEFINED, spin_rank);

        int rank_in_kpt = kptcomm_.is_null() ? -1 : kptcomm_.rank();
        kpt_bridge_ = spincomm_.split(rank_in_kpt >= 0 ? rank_in_kpt : MPI_UNDEFINED, spin_rank);
    }

    // --- Band communicator ---
    if (!kptcomm_.is_null()) {
        int kpt_size = kptcomm_.size();
        int kpt_rank = kptcomm_.rank();

        if (kpt_rank < params.npband) {
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

    // --- Domain: always full grid (no domain decomposition) ---
    DomainVertices verts;
    verts.xs = 0;
    verts.xe = grid.Nx() - 1;
    verts.ys = 0;
    verts.ye = grid.Ny() - 1;
    verts.zs = 0;
    verts.ze = grid.Nz() - 1;
    domain_ = Domain(grid, verts);
}

} // namespace sparc
