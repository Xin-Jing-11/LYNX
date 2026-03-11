#include "physics/Electrostatics.hpp"
#include "core/constants.hpp"
#include "atoms/Pseudopotential.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <mpi.h>

namespace sparc {

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

void Electrostatics::compute_pseudocharge(
    const Crystal& crystal,
    const std::vector<AtomInfluence>& influence,
    const Domain& domain,
    const FDGrid& grid,
    const FDStencil& stencil,
    const MPIComm& dmcomm) {

    int Nd_d = domain.Nd_d();
    int FDn = stencil.FDn();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    double dV = grid.dV();
    int xs = domain.vertices().xs, ys = domain.vertices().ys, zs = domain.vertices().zs;
    int nx = domain.Nx_d(), ny = domain.Ny_d(), nz = domain.Nz_d();

    double inv_4PI = -0.25 / constants::PI;  // -1/(4π) for b = -1/(4π) * Lap(V)

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

    int ntypes = crystal.n_types();
    for (int it = 0; it < ntypes; ++it) {
        const auto& psd = crystal.types()[it].psd();
        const auto& inf = influence[it];
        const auto& r_grid = psd.radial_grid();
        const auto& rVloc = psd.rVloc();
        const auto& rVloc_d = psd.rVloc_spline_d();
        double Znucl = psd.Zval();

        // Reference cutoff radius (must match SPARC's REFERENCE_CUTOFF = 0.5 Bohr)
        double rc_ref = 0.5;

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            Vec3 pos = inf.coords[iat];

            // Determine the overlap box (in global grid coords, extended by FDn)
            int i_s = inf.xs[iat], i_e = inf.xe[iat];
            int j_s = inf.ys[iat], j_e = inf.ye[iat];
            int k_s = inf.zs[iat], k_e = inf.ze[iat];

            // Extended grid dimensions for Laplacian computation
            int nx_loc = i_e - i_s + 1;
            int ny_loc = j_e - j_s + 1;
            int nz_loc = k_e - k_s + 1;
            int nxp = nx_loc + 2 * FDn;
            int nyp = ny_loc + 2 * FDn;
            int nzp = nz_loc + 2 * FDn;
            int ndp = nxp * nyp * nzp;

            // Compute V_J and V_ref on the extended grid
            std::vector<double> VJ(ndp, 0.0);
            std::vector<double> VJ_ref(ndp, 0.0);

            for (int kk = 0; kk < nzp; ++kk) {
                int gk = (k_s - FDn + kk);
                double rz = gk * dz - pos.z;
                for (int jj = 0; jj < nyp; ++jj) {
                    int gj = (j_s - FDn + jj);
                    double ry = gj * dy - pos.y;
                    for (int ii = 0; ii < nxp; ++ii) {
                        int gi = (i_s - FDn + ii);
                        double rx = gi * dx - pos.x;
                        double r = std::sqrt(rx * rx + ry * ry + rz * rz);

                        int idx = ii + jj * nxp + kk * nxp * nyp;

                        // Local pseudopotential
                        if (r < 1e-10) {
                            VJ[idx] = psd.Vloc_0();
                        } else if (r < r_grid.back()) {
                            std::vector<double> rpts = {r};
                            std::vector<double> vals;
                            Pseudopotential::spline_interp(r_grid, rVloc, rVloc_d, rpts, vals);
                            VJ[idx] = vals[0] / r;
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

            calc_lapV(VJ.data(), bJ.data(), nx_loc, ny_loc, nz_loc,
                      nxp, nyp, nzp, FDn, D2_x, D2_y, D2_z, inv_4PI);
            calc_lapV(VJ_ref.data(), bJ_ref.data(), nx_loc, ny_loc, nz_loc,
                      nxp, nyp, nzp, FDn, D2_x, D2_y, D2_z, inv_4PI);

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

    // Compute Ec: 0.5 * ∫ (b + b_ref) * Vc dV
    // where Vc is the correction potential (difference between atomic and reference)
    // For now, compute the combined Eself + Ec
    // Ec = 0.5 * ∫ (b + b_ref) * Vc dV (requires Vc which needs Poisson solve)
    // We'll compute Ec later when we have Vc

    // Integral of pseudocharge (should equal -total_Z)
    for (int i = 0; i < Nd_d; ++i) {
        int_b_ += b_.data()[i];
    }
    int_b_ *= dV;

    // MPI reduction
    if (!dmcomm.is_null() && dmcomm.size() > 1) {
        Eself_ = dmcomm.allreduce_sum(Eself_);
        int_b_ = dmcomm.allreduce_sum(int_b_);
    }

    // Eself = -0.5 * dV * Σ(bJ_ref * VJ_ref), matching reference SPARC
    Eself_ *= dV * 0.5;
    Eself_Ec_ = Eself_;  // Ec will be added later via compute_Ec
}

void Electrostatics::compute_Vloc(
    const Crystal& crystal,
    const std::vector<AtomInfluence>& influence,
    const Domain& domain,
    const FDGrid& grid,
    double* Vloc,
    const MPIComm& dmcomm) {

    int Nd_d = domain.Nd_d();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    int xs = domain.vertices().xs, ys = domain.vertices().ys, zs = domain.vertices().zs;
    int nx = domain.Nx_d(), ny = domain.Ny_d(), nz = domain.Nz_d();

    std::memset(Vloc, 0, Nd_d * sizeof(double));

    int ntypes = crystal.n_types();
    for (int it = 0; it < ntypes; ++it) {
        const auto& psd = crystal.types()[it].psd();
        const auto& inf = influence[it];
        const auto& r_grid = psd.radial_grid();
        const auto& rVloc = psd.rVloc();
        const auto& rVloc_d = psd.rVloc_spline_d();
        double Znucl = psd.Zval();

        // Reference cutoff radius (must match SPARC's REFERENCE_CUTOFF = 0.5 Bohr)
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

                        double r = std::sqrt(rx * rx + ry * ry + rz * rz);
                        int idx = li + lj * nx + lk * nx * ny;

                        // V_loc contribution = V_J(r) - V_ref(r)
                        // (only the short-range part; long-range is in φ)
                        double vj;
                        if (r < 1e-10) {
                            vj = psd.Vloc_0();
                        } else if (r < r_grid.back()) {
                            std::vector<double> rpts = {r};
                            std::vector<double> vals;
                            Pseudopotential::spline_interp(r_grid, rVloc, rVloc_d, rpts, vals);
                            vj = vals[0] / r;
                        } else {
                            vj = -Znucl / r;
                        }

                        double vref = V_ref(r, rc_ref, Znucl);
                        Vloc[idx] += vref - vj;  // Vc = V_ref - V_J (matches reference SPARC sign convention)
                    }
                }
            }
        }
    }
}

void Electrostatics::compute_Ec(const double* Vloc, int Nd_d, double dV, const MPIComm& dmcomm) {
    // Ec = 0.5 * ∫(b + b_ref) * Vc * dV
    // where Vc = Vloc = V_J - V_ref (the correction potential)
    Ec_ = 0.0;
    for (int i = 0; i < Nd_d; ++i) {
        Ec_ += (b_.data()[i] + b_ref_.data()[i]) * Vloc[i];
    }
    Ec_ *= dV * 0.5;

    if (!dmcomm.is_null() && dmcomm.size() > 1) {
        Ec_ = dmcomm.allreduce_sum(Ec_);
    }

    Eself_Ec_ = Eself_ + Ec_;
}

void Electrostatics::compute_atomic_density(
    const Crystal& crystal,
    const std::vector<AtomInfluence>& influence,
    const Domain& domain,
    const FDGrid& grid,
    double* rho_at,
    int Nelectron,
    const MPIComm& dmcomm) {

    int Nd_d = domain.Nd_d();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    double dV = grid.dV();
    int xs = domain.vertices().xs, ys = domain.vertices().ys, zs = domain.vertices().zs;
    int nx = domain.Nx_d(), ny = domain.Ny_d(), nz = domain.Nz_d();

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

                        double r = std::sqrt(rx * rx + ry * ry + rz * rz);
                        int idx = li + lj * nx + lk * nx * ny;

                        // Interpolate isolated atom density
                        // rho_iso stores ρ_v(r)/(4π), use directly (no r² division)
                        double rho_val = 0.0;
                        if (r < r_grid.back()) {
                            std::vector<double> rpts = {r};
                            std::vector<double> vals;
                            Pseudopotential::spline_interp(r_grid, rho_iso, rho_iso_d, rpts, vals);
                            rho_val = vals[0];
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
    if (!dmcomm.is_null() && dmcomm.size() > 1) {
        sum = dmcomm.allreduce_sum(sum);
    }
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
    double* rho_core,
    const MPIComm& dmcomm) {

    int Nd_d = domain.Nd_d();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    int xs = domain.vertices().xs, ys = domain.vertices().ys, zs = domain.vertices().zs;
    int nx = domain.Nx_d(), ny = domain.Ny_d(), nz = domain.Nz_d();

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

                        double r = std::sqrt(rx * rx + ry * ry + rz * rz);
                        int idx = li + lj * nx + lk * nx * ny;

                        if (r < rchrg) {
                            std::vector<double> rpts = {r};
                            std::vector<double> vals;
                            Pseudopotential::spline_interp(r_grid, rho_c, rho_c_d, rpts, vals);
                            rho_core[idx] += vals[0];
                        }
                    }
                }
            }
        }
    }

    return has_nlcc;
}

} // namespace sparc
