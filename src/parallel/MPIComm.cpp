#include "parallel/MPIComm.hpp"
#include <stdexcept>
#include <utility>

namespace lynx {

MPIComm::MPIComm(MPI_Comm comm, bool owned)
    : comm_(comm), owned_(owned) {}

MPIComm::~MPIComm() {
    if (owned_ && comm_ != MPI_COMM_NULL &&
        comm_ != MPI_COMM_WORLD && comm_ != MPI_COMM_SELF) {
        MPI_Comm_free(&comm_);
    }
}

MPIComm::MPIComm(MPIComm&& o) noexcept
    : comm_(o.comm_), owned_(o.owned_) {
    o.comm_ = MPI_COMM_NULL;
    o.owned_ = false;
}

MPIComm& MPIComm::operator=(MPIComm&& o) noexcept {
    if (this != &o) {
        if (owned_ && comm_ != MPI_COMM_NULL &&
            comm_ != MPI_COMM_WORLD && comm_ != MPI_COMM_SELF) {
            MPI_Comm_free(&comm_);
        }
        comm_ = o.comm_;
        owned_ = o.owned_;
        o.comm_ = MPI_COMM_NULL;
        o.owned_ = false;
    }
    return *this;
}

int MPIComm::rank() const {
    if (comm_ == MPI_COMM_NULL) return -1;
    int r;
    MPI_Comm_rank(comm_, &r);
    return r;
}

int MPIComm::size() const {
    if (comm_ == MPI_COMM_NULL) return 0;
    int s;
    MPI_Comm_size(comm_, &s);
    return s;
}

void MPIComm::barrier() const {
    if (comm_ != MPI_COMM_NULL)
        MPI_Barrier(comm_);
}

double MPIComm::allreduce_sum(double val) const {
    double result;
    MPI_Allreduce(&val, &result, 1, MPI_DOUBLE, MPI_SUM, comm_);
    return result;
}

void MPIComm::allreduce_sum(double* buf, int count) const {
    MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_DOUBLE, MPI_SUM, comm_);
}

void MPIComm::bcast(double* buf, int count, int root) const {
    MPI_Bcast(buf, count, MPI_DOUBLE, root, comm_);
}

void MPIComm::bcast(int* buf, int count, int root) const {
    MPI_Bcast(buf, count, MPI_INT, root, comm_);
}

MPIComm MPIComm::split(int color, int key) const {
    MPI_Comm new_comm;
    MPI_Comm_split(comm_, color, key, &new_comm);
    return MPIComm(new_comm, true);
}

} // namespace lynx
