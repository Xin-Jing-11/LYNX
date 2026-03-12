#pragma once

#include "MPIComm.hpp"
#include "core/FDGrid.hpp"
#include "core/Domain.hpp"

namespace sparc {

struct ParallelParams {
    int npspin = 1;
    int npkpt = 1;
    int npband = 1;
};

class Parallelization {
public:
    Parallelization(MPI_Comm world, const ParallelParams& params,
                    const FDGrid& grid, int Nspin, int Nkpts, int Nstates);

    const MPIComm& world() const { return world_; }
    const MPIComm& spincomm() const { return spincomm_; }
    const MPIComm& kptcomm() const { return kptcomm_; }
    const MPIComm& bandcomm() const { return bandcomm_; }

    const MPIComm& spin_bridge() const { return spin_bridge_; }
    const MPIComm& kpt_bridge() const { return kpt_bridge_; }

    int spin_index() const { return spin_index_; }
    int kpt_index() const { return kpt_index_; }
    int band_index() const { return band_index_; }

    int Nspin_local() const { return Nspin_local_; }
    int Nkpts_local() const { return Nkpts_local_; }
    int Nband_local() const { return Nband_local_; }
    int band_start() const { return band_start_; }
    int band_end() const { return band_end_; }

    const Domain& domain() const { return domain_; }

    static int block_size(int N, int np, int rank);
    static int block_start(int N, int np, int rank);

private:
    MPIComm world_;
    MPIComm spincomm_, spin_bridge_;
    MPIComm kptcomm_, kpt_bridge_;
    MPIComm bandcomm_;

    int spin_index_ = -1, kpt_index_ = -1, band_index_ = -1;
    int Nspin_local_ = 0, Nkpts_local_ = 0, Nband_local_ = 0;
    int band_start_ = 0, band_end_ = 0;

    Domain domain_;
};

} // namespace sparc
