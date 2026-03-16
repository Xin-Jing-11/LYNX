#include "physics/Electrostatics.hpp"
#include "core/constants.hpp"
#include "core/Lattice.hpp"
#include "atoms/Pseudopotential.hpp"
#include "operators/FDStencil.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <chrono>

namespace lynx {

// Reference potential: smooth version of -Z/r
// Uses a polynomial inside rc to avoid the 1/r singularity.
// Outside rc: V_ref(r) = -Znucl / r
// Inside rc: smooth polynomial matching value and derivatives at rc
double Electrostatics::V_ref(double r, double rc, double Znucl) {
    if (r > rc) {
        return -Znucl / r;
    }
    // Smooth polynomial: V_ref(r) = Z * [r²*(r³*(a6 + r*a7) + a3) + a0]
    // where coefficients ensure matching at r=rc
    double rc2 = rc * rc, rc3 = rc2 * rc, rc6 = rc3 * rc3;
    double rc7 = rc6 * rc, rc8 = rc7 * rc;
    double r2 = r * r, r3 = r2 * r, r5 = r3 * r2, r6 = r5 * r;
    double a0 = 2.4 / rc;
    double a3 = -2.8 / rc3;
    double a5 = 5.6 / rc6;
    double a6 = -6.0 / rc7;
    double a7 = 1.8 / rc8;

    return -Znucl * (r2 * (r3 * (a5 + r * (a6 + r * a7)) + a3) + a0);
}

// Helper: compute distance from non-Cartesian coordinate differences
static double compute_distance(double rx, double ry, double rz, bool is_orth, const Lattice* lattice) {
    if (is_orth) {
        return std::sqrt(rx * rx + ry * ry + rz * rz);
    } else {
        return lattice->metric_distance(rx, ry, rz);
    }
}

// Compute Laplacian of V on a grid of size nxp×nyp×nzp,
// storing result for the inner nx×ny×nz region.
// The input V is on the extended grid (nxp×nyp×nzp),
// the output lapV is on the inner grid (nx×ny×nz).
// coef: prefactor (typically -1/(4π))
void Electrostatics::calc_lapV(
    const double* V,
    double* lapV,
    int nx, int ny, int nz,
    int nxp, int nyp, int nzp,
    int FDn,
    const double* D2_x, const double* D2_y, const double* D2_z,
    double coef) {

    double w2_diag = (D2_x[0] + D2_y[0] + D2_z[0]) * coef;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx_ex = (i + FDn) + (j + FDn) * nxp + (k + FDn) * nxp * nyp;
                double val = w2_diag * V[idx_ex];

                for (int p = 1; p <= FDn; ++p) {
                    val += coef * D2_x[p] * (V[idx_ex + p] + V[idx_ex - p]);
                    val += coef * D2_y[p] * (V[idx_ex + p * nxp] + V[idx_ex - p * nxp]);
                    val += coef * D2_z[p] * (V[idx_ex + p * nxp * nyp] + V[idx_ex - p * nxp * nyp]);
                }

                int idx_out = i + j * nx + k * nx * ny;
                lapV[idx_out] = val;
            }
        }
    }
}

// Non-orthogonal version with mixed derivatives
void Electrostatics::calc_lapV_nonorth(
    const double* V, double* lapV,
    int nx, int ny, int nz,
    int nxp, int nyp, int nzp,
    int FDn,
    const FDStencil& stencil,
    double coef) {

    const double* cx = stencil.D2_coeff_x();
    const double* cy = stencil.D2_coeff_y();
    const double* cz = stencil.D2_coeff_z();
    const double* cxy = stencil.D2_coeff_xy();
    const double* cxz = stencil.D2_coeff_xz();
    const double* cyz = stencil.D2_coeff_yz();
    const double* d1y = stencil.D1_coeff_y();
    const double* d1z = stencil.D1_coeff_z();

    bool has_xy = cxy && std::abs(cxy[1]) > 1e-30;
    bool has_xz = cxz && std::abs(cxz[1]) > 1e-30;
    bool has_yz = cyz && std::abs(cyz[1]) > 1e-30;
    double w2_diag = (cx[0] + cy[0] + cz[0]) * coef;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = (i + FDn) + (j + FDn) * nxp + (k + FDn) * nxp * nyp;
                double val = w2_diag * V[idx];

                for (int p = 1; p <= FDn; ++p) {
                    val += coef * cx[p] * (V[idx + p] + V[idx - p]);
                    val += coef * cy[p] * (V[idx + p * nxp] + V[idx - p * nxp]);
                    val += coef * cz[p] * (V[idx + p * nxp * nyp] + V[idx - p * nxp * nyp]);
                }
                if (has_xy) {
                    for (int p = 1; p <= FDn; ++p)
                        for (int q = 1; q <= FDn; ++q)
                            val += coef * cxy[p] * d1y[q] *
                                   (V[idx+p+q*nxp] - V[idx+p-q*nxp] - V[idx-p+q*nxp] + V[idx-p-q*nxp]);
                }
                if (has_xz) {
                    for (int p = 1; p <= FDn; ++p)
                        for (int q = 1; q <= FDn; ++q)
                            val += coef * cxz[p] * d1z[q] *
                                   (V[idx+p+q*nxp*nyp] - V[idx+p-q*nxp*nyp] - V[idx-p+q*nxp*nyp] + V[idx-p-q*nxp*nyp]);
                }
                if (has_yz) {
                    for (int p = 1; p <= FDn; ++p)
                        for (int q = 1; q <= FDn; ++q)
                            val += coef * cyz[p] * d1z[q] *
                                   (V[idx+p*nxp+q*nxp*nyp] - V[idx+p*nxp-q*nxp*nyp] - V[idx-p*nxp+q*nxp*nyp] + V[idx-p*nxp-q*nxp*nyp]);
                }

                lapV[i + j * nx + k * nx * ny] = val;
            }
        }
    }
}

void Electrostatics::compute_pseudocharge(
    const Crystal& crystal,
    const std::vector<AtomInfluence>& influence,
    const Domain& domain,
    const FDGrid& grid,
    const FDStencil& stencil) {

    int Nd_d = domain.Nd_d();
    int FDn = stencil.FDn();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    double dV = grid.dV();
    int xs = domain.vertices().xs, ys = domain.vertices().ys, zs = domain.vertices().zs;
    int nx = domain.Nx_d(), ny = domain.Ny_d(), nz = domain.Nz_d();

    double inv_4PI = -0.25 / constants::PI;  // -1/(4π) for b = -1/(4π) * Lap(V)
    bool is_orth = grid.lattice().is_orthogonal();
    const Lattice* lattice = &grid.lattice();

    // FD stencil coefficients (already scaled by 1/dx², etc.)
    const double* D2_x = stencil.D2_coeff_x();
    const double* D2_y = stencil.D2_coeff_y();
    const double* D2_z = stencil.D2_coeff_z();

    // Initialize pseudocharge arrays
    b_ = NDArray<double>(Nd_d);
    b_ref_ = NDArray<double>(Nd_d);
    b_.zero();
    b_ref_.zero();

    Eself_ = 0.0;
    Ec_ = 0.0;
    int_b_ = 0.0;
    auto t0 = std::chrono::steady_clock::now();

    int ntypes = crystal.n_types();
    for (int it = 0; it < ntypes; ++it) {
        const auto& psd = crystal.types()[it].psd();
        const auto& inf = influence[it];
        const auto& r_grid = psd.radial_grid();
        const auto& rVloc = psd.rVloc();
        const auto& rVloc_d = psd.rVloc_spline_d();
        double Znucl = psd.Zval();

        // Reference cutoff radius (must match LYNX's REFERENCE_CUTOFF = 0.5 Bohr)
        double rc_ref = 0.5;
        std::printf("  pschg: type=%d, %d images, Znucl=%.1f\n", it, inf.n_atom, Znucl);
        std::fflush(stdout);

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            Vec3 pos = inf.coords[iat];

            // Determine the overlap box (in global grid coords, extended by FDn)
            int i_s = inf.xs[iat], i_e = inf.xe[iat];
            int j_s = inf.ys[iat], j_e = inf.ye[iat];
            int k_s = inf.zs[iat], k_e = inf.ze[iat];

            // Extended grid dimensions for Laplacian computation
            // Non-orthogonal cells: mixed derivatives need 2*FDn halo
            int nx_loc = i_e - i_s + 1;
            int ny_loc = j_e - j_s + 1;
            int nz_loc = k_e - k_s + 1;
            int halo = is_orth ? FDn : 2 * FDn;
            int nxp = nx_loc + 2 * halo;
            int nyp = ny_loc + 2 * halo;
            int nzp = nz_loc + 2 * halo;
            int ndp = nxp * nyp * nzp;
            if (iat % 10 == 0) {
                std::printf("    img %d/%d: box=[%d:%d,%d:%d,%d:%d] ext=%dx%dx%d=%d\n",
                    iat, inf.n_atom, i_s, i_e, j_s, j_e, k_s, k_e, nxp, nyp, nzp, ndp);
                std::fflush(stdout);
            }

            // Compute V_J and V_ref on the extended grid
            std::vector<double> VJ(ndp, 0.0);
            std::vector<double> VJ_ref(ndp, 0.0);

            for (int kk = 0; kk < nzp; ++kk) {
                int gk = (k_s - halo + kk);
                double rz = gk * dz - pos.z;
                for (int jj = 0; jj < nyp; ++jj) {
                    int gj = (j_s - halo + jj);
                    double ry = gj * dy - pos.y;
                    for (int ii = 0; ii < nxp; ++ii) {
                        int gi = (i_s - halo + ii);
                        double rx = gi * dx - pos.x;
                        double r = compute_distance(rx, ry, rz, is_orth, lattice);

                        int idx = ii + jj * nxp + kk * nxp * nyp;

                        // Local pseudopotential
                        if (r < 1e-10) {
                            VJ[idx] = psd.Vloc_0();
                        } else if (r < r_grid.back()) {
                            VJ[idx] = Pseudopotential::spline_interp_single(
                                r_grid, rVloc, rVloc_d, r) / r;
                        } else {
                            VJ[idx] = -Znucl / r;
                        }

                        // Reference potential
                        VJ_ref[idx] = V_ref(r, rc_ref, Znucl);
                    }
                }
            }

            // Compute b_J = -1/(4π) * Lap(V_J) on the inner grid
            std::vector<double> bJ(nx_loc * ny_loc * nz_loc, 0.0);
            std::vector<double> bJ_ref(nx_loc * ny_loc * nz_loc, 0.0);

            if (is_orth) {
                calc_lapV(VJ.data(), bJ.data(), nx_loc, ny_loc, nz_loc,
                          nxp, nyp, nzp, FDn, D2_x, D2_y, D2_z, inv_4PI);
                calc_lapV(VJ_ref.data(), bJ_ref.data(), nx_loc, ny_loc, nz_loc,
                          nxp, nyp, nzp, FDn, D2_x, D2_y, D2_z, inv_4PI);
            } else {
                calc_lapV_nonorth(VJ.data(), bJ.data(), nx_loc, ny_loc, nz_loc,
                                  nxp, nyp, nzp, FDn, stencil, inv_4PI);
                calc_lapV_nonorth(VJ_ref.data(), bJ_ref.data(), nx_loc, ny_loc, nz_loc,
                                  nxp, nyp, nzp, FDn, stencil, inv_4PI);
            }

            // Accumulate into global arrays and compute Eself
            for (int kk = 0; kk < nz_loc; ++kk) {
                int lk = (k_s + kk) - zs;
                if (lk < 0 || lk >= nz) continue;
                for (int jj = 0; jj < ny_loc; ++jj) {
                    int lj = (j_s + jj) - ys;
                    if (lj < 0 || lj >= ny) continue;
                    for (int ii = 0; ii < nx_loc; ++ii) {
                        int li = (i_s + ii) - xs;
                        if (li < 0 || li >= nx) continue;

                        int idx_dm = li + lj * nx + lk * nx * ny;
                        int idx_loc = ii + jj * nx_loc + kk * nx_loc * ny_loc;

                        b_.data()[idx_dm] += bJ[idx_loc];
                        b_ref_.data()[idx_dm] += bJ_ref[idx_loc];
                    }
                }
            }

            // Eself contribution: -∫ b_J_ref * V_J_ref dV
            // (integral over the local grid around this atom)
            double eself_atom = 0.0;
            for (int kk = 0; kk < nz_loc; ++kk) {
                int gk = k_s + kk;
                for (int jj = 0; jj < ny_loc; ++jj) {
                    int gj = j_s + jj;
                    for (int ii = 0; ii < nx_loc; ++ii) {
                        int gi = i_s + ii;
                        int idx_loc = ii + jj * nx_loc + kk * nx_loc * ny_loc;
                        int idx_ex = (ii + FDn) + (jj + FDn) * nxp + (kk + FDn) * nxp * nyp;

                        eself_atom -= bJ_ref[idx_loc] * VJ_ref[idx_ex];
                    }
                }
            }
            Eself_ += eself_atom;  // accumulate without dV yet; will multiply by 0.5*dV at the end
        }
    }

    {
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  pschg loop done in %.1f ms\n", ms);
        std::fflush(stdout);
    }

    // Integral of pseudocharge (should equal -total_Z)
    for (int i = 0; i < Nd_d; ++i) {
        int_b_ += b_.data()[i];
    }
    int_b_ *= dV;

    // Eself = -0.5 * dV * Σ(bJ_ref * VJ_ref), matching reference LYNX
    Eself_ *= dV * 0.5;
    Eself_Ec_ = Eself_;  // Ec will be added later via compute_Ec
}

void Electrostatics::compute_Vloc(
    const Crystal& crystal,
    const std::vector<AtomInfluence>& influence,
    const Domain& domain,
    const FDGrid& grid,
    double* Vloc) {

    int Nd_d = domain.Nd_d();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    int xs = domain.vertices().xs, ys = domain.vertices().ys, zs = domain.vertices().zs;
    int nx = domain.Nx_d(), ny = domain.Ny_d(), nz = domain.Nz_d();
    bool is_orth = grid.lattice().is_orthogonal();
    const Lattice* lattice = &grid.lattice();

    std::memset(Vloc, 0, Nd_d * sizeof(double));

    int ntypes = crystal.n_types();
    for (int it = 0; it < ntypes; ++it) {
        const auto& psd = crystal.types()[it].psd();
        const auto& inf = influence[it];
        const auto& r_grid = psd.radial_grid();
        const auto& rVloc = psd.rVloc();
        const auto& rVloc_d = psd.rVloc_spline_d();
        double Znucl = psd.Zval();

        // Reference cutoff radius (must match LYNX's REFERENCE_CUTOFF = 0.5 Bohr)
        double rc_ref = 0.5;

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            Vec3 pos = inf.coords[iat];

            int i_s = inf.xs[iat], i_e = inf.xe[iat];
            int j_s = inf.ys[iat], j_e = inf.ye[iat];
            int k_s = inf.zs[iat], k_e = inf.ze[iat];

            for (int gk = k_s; gk <= k_e; ++gk) {
                int lk = gk - zs;
                if (lk < 0 || lk >= nz) continue;
                double rz = gk * dz - pos.z;

                for (int gj = j_s; gj <= j_e; ++gj) {
                    int lj = gj - ys;
                    if (lj < 0 || lj >= ny) continue;
                    double ry = gj * dy - pos.y;

                    for (int gi = i_s; gi <= i_e; ++gi) {
                        int li = gi - xs;
                        if (li < 0 || li >= nx) continue;
                        double rx = gi * dx - pos.x;

                        double r = compute_distance(rx, ry, rz, is_orth, lattice);
                        int idx = li + lj * nx + lk * nx * ny;

                        // V_loc contribution = V_J(r) - V_ref(r)
                        // (only the short-range part; long-range is in φ)
                        double vj;
                        if (r < 1e-10) {
                            vj = psd.Vloc_0();
                        } else if (r < r_grid.back()) {
                            vj = Pseudopotential::spline_interp_single(
                                r_grid, rVloc, rVloc_d, r) / r;
                        } else {
                            vj = -Znucl / r;
                        }

                        double vref = V_ref(r, rc_ref, Znucl);
                        Vloc[idx] += vref - vj;  // Vc = V_ref - V_J (matches reference LYNX sign convention)
                    }
                }
            }
        }
    }
}

void Electrostatics::compute_Ec(const double* Vloc, int Nd_d, double dV) {
    // Ec = 0.5 * ∫(b + b_ref) * Vc * dV
    // where Vc = Vloc = V_J - V_ref (the correction potential)
    Ec_ = 0.0;
    for (int i = 0; i < Nd_d; ++i) {
        Ec_ += (b_.data()[i] + b_ref_.data()[i]) * Vloc[i];
    }
    Ec_ *= dV * 0.5;

    Eself_Ec_ = Eself_ + Ec_;
}

void Electrostatics::compute_atomic_density(
    const Crystal& crystal,
    const std::vector<AtomInfluence>& influence,
    const Domain& domain,
    const FDGrid& grid,
    double* rho_at,
    int Nelectron) {

    int Nd_d = domain.Nd_d();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    double dV = grid.dV();
    int xs = domain.vertices().xs, ys = domain.vertices().ys, zs = domain.vertices().zs;
    int nx = domain.Nx_d(), ny = domain.Ny_d(), nz = domain.Nz_d();
    bool is_orth = grid.lattice().is_orthogonal();
    const Lattice* lattice = &grid.lattice();

    std::memset(rho_at, 0, Nd_d * sizeof(double));

    int ntypes = crystal.n_types();
    for (int it = 0; it < ntypes; ++it) {
        const auto& psd = crystal.types()[it].psd();
        const auto& inf = influence[it];
        const auto& r_grid = psd.radial_grid();
        const auto& rho_iso = psd.rho_iso_atom();
        const auto& rho_iso_d = psd.rho_iso_atom_spline_d();

        if (rho_iso.empty()) continue;

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            Vec3 pos = inf.coords[iat];

            int i_s = inf.xs[iat], i_e = inf.xe[iat];
            int j_s = inf.ys[iat], j_e = inf.ye[iat];
            int k_s = inf.zs[iat], k_e = inf.ze[iat];

            for (int gk = k_s; gk <= k_e; ++gk) {
                int lk = gk - zs;
                if (lk < 0 || lk >= nz) continue;
                double rz = gk * dz - pos.z;

                for (int gj = j_s; gj <= j_e; ++gj) {
                    int lj = gj - ys;
                    if (lj < 0 || lj >= ny) continue;
                    double ry = gj * dy - pos.y;

                    for (int gi = i_s; gi <= i_e; ++gi) {
                        int li = gi - xs;
                        if (li < 0 || li >= nx) continue;
                        double rx = gi * dx - pos.x;

                        double r = compute_distance(rx, ry, rz, is_orth, lattice);
                        int idx = li + lj * nx + lk * nx * ny;

                        // Interpolate isolated atom density
                        // rho_iso stores ρ_v(r)/(4π), use directly (no r² division)
                        double rho_val = 0.0;
                        if (r < r_grid.back()) {
                            rho_val = Pseudopotential::spline_interp_single(
                                r_grid, rho_iso, rho_iso_d, r);
                        }
                        rho_at[idx] += rho_val;
                    }
                }
            }
        }
    }

    // Rescale so integral = Nelectron
    double sum = 0.0;
    for (int i = 0; i < Nd_d; ++i) sum += rho_at[i];
    double int_rho = sum * dV;
    if (int_rho > 1e-10) {
        double scale = Nelectron / int_rho;
        for (int i = 0; i < Nd_d; ++i) rho_at[i] *= scale;
    }
}

bool Electrostatics::compute_core_density(
    const Crystal& crystal,
    const std::vector<AtomInfluence>& influence,
    const Domain& domain,
    const FDGrid& grid,
    double* rho_core) {

    int Nd_d = domain.Nd_d();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    int xs = domain.vertices().xs, ys = domain.vertices().ys, zs = domain.vertices().zs;
    int nx = domain.Nx_d(), ny = domain.Ny_d(), nz = domain.Nz_d();
    bool is_orth = grid.lattice().is_orthogonal();
    const Lattice* lattice = &grid.lattice();

    std::memset(rho_core, 0, Nd_d * sizeof(double));

    bool has_nlcc = false;
    int ntypes = crystal.n_types();
    for (int it = 0; it < ntypes; ++it) {
        const auto& psd = crystal.types()[it].psd();
        if (psd.fchrg() < 1e-10) continue;
        has_nlcc = true;

        const auto& inf = influence[it];
        const auto& r_grid = psd.radial_grid();
        const auto& rho_c = psd.rho_c();
        const auto& rho_c_d = psd.rho_c_spline_d();

        if (rho_c.empty()) continue;

        // Determine cutoff: last radial grid point where rho_c is nonzero
        double rchrg = r_grid.back();

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            Vec3 pos = inf.coords[iat];

            int i_s = inf.xs[iat], i_e = inf.xe[iat];
            int j_s = inf.ys[iat], j_e = inf.ye[iat];
            int k_s = inf.zs[iat], k_e = inf.ze[iat];

            for (int gk = k_s; gk <= k_e; ++gk) {
                int lk = gk - zs;
                if (lk < 0 || lk >= nz) continue;
                double rz = gk * dz - pos.z;

                for (int gj = j_s; gj <= j_e; ++gj) {
                    int lj = gj - ys;
                    if (lj < 0 || lj >= ny) continue;
                    double ry = gj * dy - pos.y;

                    for (int gi = i_s; gi <= i_e; ++gi) {
                        int li = gi - xs;
                        if (li < 0 || li >= nx) continue;
                        double rx = gi * dx - pos.x;

                        double r = compute_distance(rx, ry, rz, is_orth, lattice);
                        int idx = li + lj * nx + lk * nx * ny;

                        if (r < rchrg) {
                            rho_core[idx] += Pseudopotential::spline_interp_single(
                                r_grid, rho_c, rho_c_d, r);
                        }
                    }
                }
            }
        }
    }

    return has_nlcc;
}

} // namespace lynx
