#include "core/Domain.hpp"

namespace lynx {

Domain::Domain(const FDGrid& global_grid, const DomainVertices& verts)
    : grid_(&global_grid), verts_(verts) {
    Nx_d_ = verts.xe - verts.xs + 1;
    Ny_d_ = verts.ye - verts.ys + 1;
    Nz_d_ = verts.ze - verts.zs + 1;
    Nd_d_ = Nx_d_ * Ny_d_ * Nz_d_;
}

int Domain::local_to_global(int i, int j, int k) const {
    int gi = verts_.xs + i;
    int gj = verts_.ys + j;
    int gk = verts_.zs + k;
    return gi + gj * grid_->Nx() + gk * grid_->Nx() * grid_->Ny();
}

} // namespace lynx
