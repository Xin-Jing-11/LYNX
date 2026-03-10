#pragma once

#include "MPIComm.hpp"
#include "core/Domain.hpp"

namespace sparc {

// Domain-to-domain data redistribution
// Transfers data between two different domain decompositions (e.g., psi domain ↔ phi domain)
class D2DTransfer {
public:
    D2DTransfer() = default;

    // Set up transfer between source and target domain decompositions
    // Both must share a common parent communicator
    void setup(const Domain& src_domain, const Domain& tgt_domain,
               const MPIComm& src_comm, const MPIComm& tgt_comm,
               const MPIComm& parent_comm);

    // Transfer data from source layout to target layout
    void execute(const double* src_data, double* tgt_data) const;

    bool is_setup() const { return is_setup_; }

private:
    bool is_setup_ = false;
    // Placeholder for send/recv metadata — will be filled in Phase 2+
};

} // namespace sparc
