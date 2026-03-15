#pragma once

#include "FDGrid.hpp"

namespace lynx {

struct DomainVertices {
    int xs = 0, xe = 0;  // x start/end (inclusive)
    int ys = 0, ye = 0;
    int zs = 0, ze = 0;
};

class Domain {
public:
    Domain() = default;
    Domain(const FDGrid& global_grid, const DomainVertices& verts);

    const DomainVertices& vertices() const { return verts_; }
    int Nx_d() const { return Nx_d_; }
    int Ny_d() const { return Ny_d_; }
    int Nz_d() const { return Nz_d_; }
    int Nd_d() const { return Nd_d_; }
    const FDGrid& global_grid() const { return *grid_; }

    int local_to_global(int i, int j, int k) const;

    int flat_index(int i, int j, int k) const {
        return i + j * Nx_d_ + k * Nx_d_ * Ny_d_;
    }

private:
    const FDGrid* grid_ = nullptr;
    DomainVertices verts_{};
    int Nx_d_ = 0, Ny_d_ = 0, Nz_d_ = 0, Nd_d_ = 0;
};

} // namespace lynx
