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
    Vec3 L = lattice_->lengths();
    for (auto& pos : positions_) {
        // Positions are in non-Cart coords for non-orth, Cartesian for orth
        // In both cases, wrap to [0, L_i)
        if (lattice_->is_orthogonal()) {
            Vec3 frac = lattice_->cart_to_frac(pos);
            frac.x -= std::floor(frac.x);
            frac.y -= std::floor(frac.y);
            frac.z -= std::floor(frac.z);
            pos = lattice_->frac_to_cart(frac);
        } else {
            // Non-Cart coords: range [0, L_i)
            pos.x -= std::floor(pos.x / L.x) * L.x;
            pos.y -= std::floor(pos.y / L.y) * L.y;
            pos.z -= std::floor(pos.z / L.z) * L.z;
        }
    }
}

void Crystal::compute_atom_influence(const Domain& domain, double rc_max,
                                     std::vector<AtomInfluence>& influence) const {
    influence.resize(n_types());
    if (!lattice_) return;

    const auto& grid = domain.global_grid();
    Vec3 L = lattice_->lengths();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    bool is_orth = lattice_->is_orthogonal();

    for (int it = 0; it < n_types(); ++it) {
        auto& inf = influence[it];
        inf = AtomInfluence{};

        // For non-orth: extent factors for bounding box in non-Cart coords
        Vec3 ext_inf = is_orth ? Vec3{1.0, 1.0, 1.0} : lattice_->nonCart_sphere_extent();

        for (int ia = 0; ia < n_atom_total(); ++ia) {
            if (type_indices_[ia] != it) continue;

            // Positions are in non-Cart coords (same coordinate system as grid)
            Vec3 pos = positions_[ia];
            // For non-orth, Cartesian sphere extends further in non-Cart coords
            double ext_x = rc_max * ext_inf.x;
            double ext_y = rc_max * ext_inf.y;
            double ext_z = rc_max * ext_inf.z;
            int n_images_x = (grid.bcx() == BCType::Periodic) ? static_cast<int>(std::ceil(ext_x / L.x)) : 0;
            int n_images_y = (grid.bcy() == BCType::Periodic) ? static_cast<int>(std::ceil(ext_y / L.y)) : 0;
            int n_images_z = (grid.bcz() == BCType::Periodic) ? static_cast<int>(std::ceil(ext_z / L.z)) : 0;

            for (int iz = -n_images_z; iz <= n_images_z; ++iz) {
                for (int iy = -n_images_y; iy <= n_images_y; ++iy) {
                    for (int ix = -n_images_x; ix <= n_images_x; ++ix) {
                        Vec3 img;
                        img.x = pos.x + ix * L.x;
                        img.y = pos.y + iy * L.y;
                        img.z = pos.z + iz * L.z;

                        // For non-orth: need to compute bounding box in grid indices
                        // The rc_max sphere in Cartesian corresponds to a parallelepiped
                        // in non-Cart coords. Use a conservative estimate.
                        int xs_g, xe_g, ys_g, ye_g, zs_g, ze_g;
                        if (is_orth) {
                            xs_g = static_cast<int>(std::ceil((img.x - rc_max) / dx));
                            xe_g = static_cast<int>(std::floor((img.x + rc_max) / dx));
                            ys_g = static_cast<int>(std::ceil((img.y - rc_max) / dy));
                            ye_g = static_cast<int>(std::floor((img.y + rc_max) / dy));
                            zs_g = static_cast<int>(std::ceil((img.z - rc_max) / dz));
                            ze_g = static_cast<int>(std::floor((img.z + rc_max) / dz));
                        } else {
                            // Non-orth: a Cartesian sphere of radius rc_max extends
                            // differently in non-Cart coords. Use metric correction:
                            // extent_i = rc_max * ||row_i(LatUVec^{-1})|| / d_i
                            Vec3 ext = lattice_->nonCart_sphere_extent();
                            double rc_x = rc_max * ext.x / dx;
                            double rc_y = rc_max * ext.y / dy;
                            double rc_z = rc_max * ext.z / dz;
                            xs_g = static_cast<int>(std::ceil(img.x / dx - rc_x));
                            xe_g = static_cast<int>(std::floor(img.x / dx + rc_x));
                            ys_g = static_cast<int>(std::ceil(img.y / dy - rc_y));
                            ye_g = static_cast<int>(std::floor(img.y / dy + rc_y));
                            zs_g = static_cast<int>(std::ceil(img.z / dz - rc_z));
                            ze_g = static_cast<int>(std::floor(img.z / dz + rc_z));
                        }

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
    bool is_orth = lattice_->is_orthogonal();

    for (int it = 0; it < n_types(); ++it) {
        auto& inf = nloc_influence[it];
        inf = AtomNlocInfluence{};

        // Max rc for this type across all l channels
        double rc_max = 0.0;
        for (double r : types_[it].psd().rc())
            rc_max = std::max(rc_max, r);
        if (rc_max < 1e-14) continue;

        // For non-orth: extent factors for bounding box in non-Cart coords
        Vec3 ext = is_orth ? Vec3{1.0, 1.0, 1.0} : lattice_->nonCart_sphere_extent();

        for (int ia = 0; ia < n_atom_total(); ++ia) {
            if (type_indices_[ia] != it) continue;
            Vec3 pos = positions_[ia];

            // For non-orth cells, the Cartesian sphere extends further in non-Cart coords
            double ext_x = rc_max * ext.x;
            double ext_y = rc_max * ext.y;
            double ext_z = rc_max * ext.z;
            int n_images_x = (grid.bcx() == BCType::Periodic) ? static_cast<int>(std::ceil(ext_x / L.x)) : 0;
            int n_images_y = (grid.bcy() == BCType::Periodic) ? static_cast<int>(std::ceil(ext_y / L.y)) : 0;
            int n_images_z = (grid.bcz() == BCType::Periodic) ? static_cast<int>(std::ceil(ext_z / L.z)) : 0;

            for (int iz = -n_images_z; iz <= n_images_z; ++iz) {
                for (int iy = -n_images_y; iy <= n_images_y; ++iy) {
                    for (int ix = -n_images_x; ix <= n_images_x; ++ix) {
                        Vec3 img;
                        img.x = pos.x + ix * L.x;
                        img.y = pos.y + iy * L.y;
                        img.z = pos.z + iz * L.z;

                        // Bounding box in grid indices
                        int xs_g, xe_g, ys_g, ye_g, zs_g, ze_g;
                        if (is_orth) {
                            xs_g = static_cast<int>(std::ceil((img.x - rc_max) / dx));
                            xe_g = static_cast<int>(std::floor((img.x + rc_max) / dx));
                            ys_g = static_cast<int>(std::ceil((img.y - rc_max) / dy));
                            ye_g = static_cast<int>(std::floor((img.y + rc_max) / dy));
                            zs_g = static_cast<int>(std::ceil((img.z - rc_max) / dz));
                            ze_g = static_cast<int>(std::floor((img.z + rc_max) / dz));
                        } else {
                            Vec3 ext = lattice_->nonCart_sphere_extent();
                            double rc_x = rc_max * ext.x / dx;
                            double rc_y = rc_max * ext.y / dy;
                            double rc_z = rc_max * ext.z / dz;
                            xs_g = static_cast<int>(std::ceil(img.x / dx - rc_x));
                            xe_g = static_cast<int>(std::floor(img.x / dx + rc_x));
                            ys_g = static_cast<int>(std::ceil(img.y / dy - rc_y));
                            ye_g = static_cast<int>(std::floor(img.y / dy + rc_y));
                            zs_g = static_cast<int>(std::ceil(img.z / dz - rc_z));
                            ze_g = static_cast<int>(std::floor(img.z / dz + rc_z));
                        }

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
                                    // Distance: for orth, Euclidean; for non-orth, use metric
                                    double r2;
                                    if (is_orth) {
                                        r2 = xr * xr + yr * yr + zr * zr;
                                    } else {
                                        double d = lattice_->metric_distance(xr, yr, zr);
                                        r2 = d * d;
                                    }
                                    if (r2 <= rc_max * rc_max) {
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
