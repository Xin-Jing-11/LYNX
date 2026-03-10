#pragma once

#include "MPIComm.hpp"
#include <array>

namespace sparc {

class CartTopology {
public:
    CartTopology() = default;

    // Create a 3D Cartesian topology from an existing communicator
    // dims[3] = number of procs in each direction
    // periods[3] = whether each direction is periodic
    CartTopology(const MPIComm& parent, const int dims[3], const bool periods[3]);

    const MPIComm& comm() const { return cart_comm_; }
    bool is_null() const { return cart_comm_.is_null(); }

    // Get coordinates of this process
    std::array<int, 3> coords() const { return coords_; }

    // Get dims
    std::array<int, 3> dims() const { return dims_; }

    // Shift: returns (source, dest) for a shift in given direction by disp
    std::pair<int, int> shift(int direction, int disp) const;

private:
    MPIComm cart_comm_;
    std::array<int, 3> dims_ = {0, 0, 0};
    std::array<int, 3> coords_ = {0, 0, 0};
};

// Utility: auto-determine dims for nproc processes on grid [Nx, Ny, Nz]
// Tries to minimize surface area (communication) subject to min_per_proc constraint
void auto_cart_dims(int nproc, int Nx, int Ny, int Nz,
                    int min_per_proc, int dims[3]);

} // namespace sparc
