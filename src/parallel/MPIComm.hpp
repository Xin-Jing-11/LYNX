#pragma once

#include <mpi.h>

namespace lynx {

class MPIComm {
public:
    MPIComm() = default;
    explicit MPIComm(MPI_Comm comm, bool owned = false);
    ~MPIComm();

    MPIComm(MPIComm&& o) noexcept;
    MPIComm& operator=(MPIComm&& o) noexcept;
    MPIComm(const MPIComm&) = delete;
    MPIComm& operator=(const MPIComm&) = delete;

    MPI_Comm comm() const { return comm_; }
    int rank() const;
    int size() const;
    bool is_null() const { return comm_ == MPI_COMM_NULL; }

    void barrier() const;
    double allreduce_sum(double val) const;
    void allreduce_sum(double* buf, int count) const;
    void bcast(double* buf, int count, int root = 0) const;
    void bcast(int* buf, int count, int root = 0) const;

    MPIComm split(int color, int key) const;

private:
    MPI_Comm comm_ = MPI_COMM_NULL;
    bool owned_ = false;
};

} // namespace lynx
