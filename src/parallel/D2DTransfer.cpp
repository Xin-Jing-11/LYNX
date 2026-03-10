#include "parallel/D2DTransfer.hpp"

namespace sparc {

void D2DTransfer::setup(const Domain& /*src_domain*/, const Domain& /*tgt_domain*/,
                        const MPIComm& /*src_comm*/, const MPIComm& /*tgt_comm*/,
                        const MPIComm& /*parent_comm*/) {
    // TODO: compute overlap regions and build send/recv lists
    is_setup_ = true;
}

void D2DTransfer::execute(const double* /*src_data*/, double* /*tgt_data*/) const {
    // TODO: perform MPI send/recv for data redistribution
}

} // namespace sparc
