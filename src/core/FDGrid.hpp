#pragma once

#include "types.hpp"
#include "Lattice.hpp"

namespace sparc {

class FDGrid {
public:
    FDGrid() = default;
    FDGrid(int Nx, int Ny, int Nz, const Lattice& lattice,
           BCType bcx, BCType bcy, BCType bcz);

    int Nx() const { return Nx_; }
    int Ny() const { return Ny_; }
    int Nz() const { return Nz_; }
    int Nd() const { return Nd_; }
    double dx() const { return dx_; }
    double dy() const { return dy_; }
    double dz() const { return dz_; }
    double dV() const { return dV_; }
    BCType bcx() const { return bcx_; }
    BCType bcy() const { return bcy_; }
    BCType bcz() const { return bcz_; }
    const Lattice& lattice() const { return *lattice_; }

private:
    int Nx_ = 0, Ny_ = 0, Nz_ = 0, Nd_ = 0;
    double dx_ = 0, dy_ = 0, dz_ = 0, dV_ = 0;
    BCType bcx_ = BCType::Periodic;
    BCType bcy_ = BCType::Periodic;
    BCType bcz_ = BCType::Periodic;
    const Lattice* lattice_ = nullptr;
};

} // namespace sparc
