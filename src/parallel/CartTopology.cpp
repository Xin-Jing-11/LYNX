#include "parallel/CartTopology.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sparc {

CartTopology::CartTopology(const MPIComm& parent, const int dims[3], const bool periods[3]) {
    if (parent.is_null()) return;

    dims_ = {dims[0], dims[1], dims[2]};

    int cart_size = dims[0] * dims[1] * dims[2];
    int my_rank = parent.rank();

    if (my_rank < cart_size) {
        int mpi_dims[3] = {dims[0], dims[1], dims[2]};
        int mpi_periods[3] = {periods[0] ? 1 : 0, periods[1] ? 1 : 0, periods[2] ? 1 : 0};
        MPI_Comm new_comm;
        // Split: participating processes get color 0, others get MPI_UNDEFINED
        MPI_Comm temp;
        MPI_Comm_split(parent.comm(), 0, my_rank, &temp);
        MPI_Cart_create(temp, 3, mpi_dims, mpi_periods, 1, &new_comm);
        MPI_Comm_free(&temp);

        if (new_comm != MPI_COMM_NULL) {
            cart_comm_ = MPIComm(new_comm, true);
            int c[3];
            MPI_Cart_coords(new_comm, cart_comm_.rank(), 3, c);
            coords_ = {c[0], c[1], c[2]};
        }
    } else {
        // This process is not part of the Cartesian topology
        MPI_Comm temp;
        MPI_Comm_split(parent.comm(), MPI_UNDEFINED, my_rank, &temp);
        // temp will be MPI_COMM_NULL
    }
}

std::pair<int, int> CartTopology::shift(int direction, int disp) const {
    if (cart_comm_.is_null()) return {MPI_PROC_NULL, MPI_PROC_NULL};
    int src, dest;
    MPI_Cart_shift(cart_comm_.comm(), direction, disp, &src, &dest);
    return {src, dest};
}

void auto_cart_dims(int nproc, int Nx, int Ny, int Nz,
                    int min_per_proc, int dims[3]) {
    // Find all factorizations nproc = px * py * pz
    // Minimize surface area: 2*(Nx*Ny/px/py + Ny*Nz/py/pz + Nx*Nz/px/pz)
    // Subject to: Nx/px >= min_per_proc, etc.
    double best_cost = std::numeric_limits<double>::max();
    dims[0] = dims[1] = dims[2] = 1;

    // Collect all divisors of nproc
    std::vector<int> divisors;
    for (int i = 1; i * i <= nproc; ++i) {
        if (nproc % i == 0) {
            divisors.push_back(i);
            if (i != nproc / i)
                divisors.push_back(nproc / i);
        }
    }
    std::sort(divisors.begin(), divisors.end());

    for (int px : divisors) {
        if (Nx / px < min_per_proc) continue;
        int rem = nproc / px;
        for (int py : divisors) {
            if (rem % py != 0) continue;
            int pz = rem / py;
            if (Ny / py < min_per_proc) continue;
            if (Nz / pz < min_per_proc) continue;

            // Cost: approximate communication surface area
            double cost = static_cast<double>(Ny) * Nz / (py * pz)
                        + static_cast<double>(Nx) * Nz / (px * pz)
                        + static_cast<double>(Nx) * Ny / (px * py);
            if (cost < best_cost) {
                best_cost = cost;
                dims[0] = px;
                dims[1] = py;
                dims[2] = pz;
            }
        }
    }
}

} // namespace sparc
