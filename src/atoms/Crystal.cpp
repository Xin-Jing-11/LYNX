#include "atoms/Crystal.hpp"
#include <cmath>
#include <algorithm>

namespace sparc {

Crystal::Crystal(std::vector<AtomType> types, std::vector<Vec3> positions,
                 std::vector<int> type_indices, const Lattice& lattice)
    : types_(std::move(types)), positions_(std::move(positions)),
      type_indices_(std::move(type_indices)), lattice_(&lattice) {}

double Crystal::total_Zval() const {
    double total = 0.0;
    for (int i = 0; i < n_atom_total(); ++i)
        total += types_[type_indices_[i]].Zval();
    return total;
}

void Crystal::wrap_positions() {
    if (!lattice_) return;
    for (auto& pos : positions_) {
        Vec3 frac = lattice_->cart_to_frac(pos);
        // Wrap to [0, 1)
        frac.x -= std::floor(frac.x);
        frac.y -= std::floor(frac.y);
        frac.z -= std::floor(frac.z);
        pos = lattice_->frac_to_cart(frac);
    }
}

void Crystal::compute_atom_influence(const Domain& domain, double rc_max,
                                     std::vector<AtomInfluence>& influence) const {
    influence.resize(n_types());
    if (!lattice_) return;

    const auto& grid = domain.global_grid();
    Vec3 L = lattice_->lengths();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();

    for (int it = 0; it < n_types(); ++it) {
        auto& inf = influence[it];
        inf = AtomInfluence{};

        for (int ia = 0; ia < n_atom_total(); ++ia) {
            if (type_indices_[ia] != it) continue;

            // Check all periodic images within rc_max of the domain
            Vec3 pos = positions_[ia];
            int n_images_x = (grid.bcx() == BCType::Periodic) ? static_cast<int>(std::ceil(rc_max / L.x)) : 0;
            int n_images_y = (grid.bcy() == BCType::Periodic) ? static_cast<int>(std::ceil(rc_max / L.y)) : 0;
            int n_images_z = (grid.bcz() == BCType::Periodic) ? static_cast<int>(std::ceil(rc_max / L.z)) : 0;

            for (int iz = -n_images_z; iz <= n_images_z; ++iz) {
                for (int iy = -n_images_y; iy <= n_images_y; ++iy) {
                    for (int ix = -n_images_x; ix <= n_images_x; ++ix) {
                        Vec3 img;
                        img.x = pos.x + ix * L.x;
                        img.y = pos.y + iy * L.y;
                        img.z = pos.z + iz * L.z;

                        // Overlap region in grid indices
                        int xs_g = static_cast<int>(std::ceil((img.x - rc_max) / dx));
                        int xe_g = static_cast<int>(std::floor((img.x + rc_max) / dx));
                        int ys_g = static_cast<int>(std::ceil((img.y - rc_max) / dy));
                        int ye_g = static_cast<int>(std::floor((img.y + rc_max) / dy));
                        int zs_g = static_cast<int>(std::ceil((img.z - rc_max) / dz));
                        int ze_g = static_cast<int>(std::floor((img.z + rc_max) / dz));

                        // Intersect with local domain
                        int xs_l = std::max(xs_g, domain.vertices().xs);
                        int xe_l = std::min(xe_g, domain.vertices().xe);
                        int ys_l = std::max(ys_g, domain.vertices().ys);
                        int ye_l = std::min(ye_g, domain.vertices().ye);
                        int zs_l = std::max(zs_g, domain.vertices().zs);
                        int ze_l = std::min(ze_g, domain.vertices().ze);

                        if (xs_l <= xe_l && ys_l <= ye_l && zs_l <= ze_l) {
                            inf.n_atom++;
                            inf.coords.push_back(img);
                            inf.atom_index.push_back(ia);
                            inf.xs.push_back(xs_l - domain.vertices().xs);
                            inf.xe.push_back(xe_l - domain.vertices().xs);
                            inf.ys.push_back(ys_l - domain.vertices().ys);
                            inf.ye.push_back(ye_l - domain.vertices().ys);
                            inf.zs.push_back(zs_l - domain.vertices().zs);
                            inf.ze.push_back(ze_l - domain.vertices().zs);
                        }
                    }
                }
            }
        }
    }
}

void Crystal::compute_nloc_influence(const Domain& domain,
                                     std::vector<AtomNlocInfluence>& nloc_influence) const {
    nloc_influence.resize(n_types());
    if (!lattice_) return;

    const auto& grid = domain.global_grid();
    Vec3 L = lattice_->lengths();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();

    for (int it = 0; it < n_types(); ++it) {
        auto& inf = nloc_influence[it];
        inf = AtomNlocInfluence{};

        // Max rc for this type across all l channels
        double rc_max = 0.0;
        for (double r : types_[it].psd().rc())
            rc_max = std::max(rc_max, r);
        if (rc_max < 1e-14) continue;

        for (int ia = 0; ia < n_atom_total(); ++ia) {
            if (type_indices_[ia] != it) continue;
            Vec3 pos = positions_[ia];

            int n_images_x = (grid.bcx() == BCType::Periodic) ? static_cast<int>(std::ceil(rc_max / L.x)) : 0;
            int n_images_y = (grid.bcy() == BCType::Periodic) ? static_cast<int>(std::ceil(rc_max / L.y)) : 0;
            int n_images_z = (grid.bcz() == BCType::Periodic) ? static_cast<int>(std::ceil(rc_max / L.z)) : 0;

            for (int iz = -n_images_z; iz <= n_images_z; ++iz) {
                for (int iy = -n_images_y; iy <= n_images_y; ++iy) {
                    for (int ix = -n_images_x; ix <= n_images_x; ++ix) {
                        Vec3 img;
                        img.x = pos.x + ix * L.x;
                        img.y = pos.y + iy * L.y;
                        img.z = pos.z + iz * L.z;

                        // Find grid points within rc_max sphere
                        int xs_g = static_cast<int>(std::ceil((img.x - rc_max) / dx));
                        int xe_g = static_cast<int>(std::floor((img.x + rc_max) / dx));
                        int ys_g = static_cast<int>(std::ceil((img.y - rc_max) / dy));
                        int ye_g = static_cast<int>(std::floor((img.y + rc_max) / dy));
                        int zs_g = static_cast<int>(std::ceil((img.z - rc_max) / dz));
                        int ze_g = static_cast<int>(std::floor((img.z + rc_max) / dz));

                        int xs_l = std::max(xs_g, domain.vertices().xs);
                        int xe_l = std::min(xe_g, domain.vertices().xe);
                        int ys_l = std::max(ys_g, domain.vertices().ys);
                        int ye_l = std::min(ye_g, domain.vertices().ye);
                        int zs_l = std::max(zs_g, domain.vertices().zs);
                        int ze_l = std::min(ze_g, domain.vertices().ze);

                        if (xs_l > xe_l || ys_l > ye_l || zs_l > ze_l) continue;

                        // Collect grid points within the sphere
                        std::vector<int> gpos;
                        int nx_d = domain.Nx_d();
                        int ny_d = domain.Ny_d();

                        for (int k = zs_l; k <= ze_l; ++k) {
                            double zr = k * dz - img.z;
                            for (int j = ys_l; j <= ye_l; ++j) {
                                double yr = j * dy - img.y;
                                for (int i = xs_l; i <= xe_l; ++i) {
                                    double xr = i * dx - img.x;
                                    double r2 = xr * xr + yr * yr + zr * zr;
                                    if (r2 <= rc_max * rc_max) {
                                        // Local index in the domain
                                        int li = i - domain.vertices().xs;
                                        int lj = j - domain.vertices().ys;
                                        int lk = k - domain.vertices().zs;
                                        gpos.push_back(li + lj * nx_d + lk * nx_d * ny_d);
                                    }
                                }
                            }
                        }

                        if (!gpos.empty()) {
                            inf.n_atom++;
                            inf.coords.push_back(img);
                            inf.atom_index.push_back(ia);
                            inf.xs.push_back(xs_l); inf.xe.push_back(xe_l);
                            inf.ys.push_back(ys_l); inf.ye.push_back(ye_l);
                            inf.zs.push_back(zs_l); inf.ze.push_back(ze_l);
                            inf.ndc.push_back(static_cast<int>(gpos.size()));
                            inf.grid_pos.push_back(std::move(gpos));
                        }
                    }
                }
            }
        }
    }
}

} // namespace sparc
