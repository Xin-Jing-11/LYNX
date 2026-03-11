#include "physics/Forces.hpp"
#include "core/constants.hpp"
#include "atoms/Pseudopotential.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <mpi.h>

namespace sparc {

std::vector<double> Forces::compute(
    const Wavefunction& wfn,
    const Crystal& crystal,
    const std::vector<AtomNlocInfluence>& nloc_influence,
    const NonlocalProjector& vnl,
    const Gradient& gradient,
    const HaloExchange& halo,
    const Domain& domain,
    const FDGrid& grid,
    const double* phi,
    const double* rho,
    const std::vector<double>& kpt_weights,
    const MPIComm& dmcomm,
    const MPIComm& bandcomm,
    const MPIComm& kptcomm,
    const MPIComm& spincomm) {

    int n_atom = crystal.n_atom_total();
    f_local_.assign(3 * n_atom, 0.0);
    f_nloc_.assign(3 * n_atom, 0.0);
    f_total_.assign(3 * n_atom, 0.0);

    // Local force: Hellmann-Feynman with ρ and ∇V_loc
    compute_local(crystal, domain, grid, rho, dmcomm);

    // Nonlocal force from KB projectors
    compute_nonlocal(wfn, crystal, nloc_influence, gradient, halo,
                     domain, grid, kpt_weights, dmcomm, bandcomm, kptcomm, spincomm);

    // Sum contributions
    for (int i = 0; i < 3 * n_atom; ++i) {
        f_total_[i] = f_local_[i] + f_nloc_[i];
    }

    // Symmetrize: ensure total force sums to zero
    symmetrize(f_total_, n_atom);

    return f_total_;
}

// ---------------------------------------------------------------------------
// Local force: Hellmann-Feynman theorem
//   F_J = -∫ ρ(r) ∇_r V_J^loc(|r - R_J|) dV
//
// V_J^loc(s) = rVloc(s) / s, where rVloc is the tabulated pseudopotential.
// ∇V = (dV/ds) * (r - R_J)/s
// dV/ds = [d(rVloc)/ds · s - rVloc(s)] / s²
//       = [d(rVloc)/ds - V(s)] / s
// ---------------------------------------------------------------------------
void Forces::compute_local(
    const Crystal& crystal,
    const Domain& domain,
    const FDGrid& grid,
    const double* rho,
    const MPIComm& dmcomm) {

    int n_atom = crystal.n_atom_total();
    f_local_.assign(3 * n_atom, 0.0);

    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    double dV = grid.dV();
    int nx = domain.Nx_d(), ny = domain.Ny_d(), nz = domain.Nz_d();
    int xs = domain.vertices().xs;
    int ys = domain.vertices().ys;
    int zs = domain.vertices().zs;

    int atom_count = 0;
    for (int it = 0; it < crystal.n_types(); ++it) {
        const auto& psd = crystal.types()[it].psd();
        const auto& r_grid = psd.radial_grid();
        const auto& rVloc = psd.rVloc();
        const auto& rVloc_d = psd.rVloc_spline_d();

        // Max cutoff for this atom type
        double rc_max = 0.0;
        for (auto rc : psd.rc()) rc_max = std::max(rc_max, rc);
        // Use a slightly larger cutoff to include the full local potential tail
        // rVloc -> -Zval asymptotically, so Vloc = -Zval/r. We only need the
        // difference from Coulomb which decays within rc.
        double rc_local = r_grid.back();  // extent of radial grid
        if (rc_local > 20.0) rc_local = 20.0;

        int rb_x = (int)std::ceil(rc_local / dx) + 1;
        int rb_y = (int)std::ceil(rc_local / dy) + 1;
        int rb_z = (int)std::ceil(rc_local / dz) + 1;

        for (int ia = 0; ia < crystal.types()[it].n_atoms(); ++ia) {
            Vec3 pos = crystal.positions()[atom_count];
            double fx = 0.0, fy = 0.0, fz = 0.0;

            // Global grid index nearest to atom
            int gi0 = (int)std::round(pos.x / dx);
            int gj0 = (int)std::round(pos.y / dy);
            int gk0 = (int)std::round(pos.z / dz);

            // Scan grid points within rc_local
            for (int dk = -rb_z; dk <= rb_z; ++dk) {
                int gk = gk0 + dk;
                int lk = gk - zs;
                if (lk < 0 || lk >= nz) continue;

                for (int dj = -rb_y; dj <= rb_y; ++dj) {
                    int gj = gj0 + dj;
                    int lj = gj - ys;
                    if (lj < 0 || lj >= ny) continue;

                    for (int di = -rb_x; di <= rb_x; ++di) {
                        int gi = gi0 + di;
                        int li = gi - xs;
                        if (li < 0 || li >= nx) continue;

                        double rx = gi * dx - pos.x;
                        double ry = gj * dy - pos.y;
                        double rz = gk * dz - pos.z;
                        double s = std::sqrt(rx * rx + ry * ry + rz * rz);

                        if (s < 1e-10 || s > rc_local) continue;

                        // Interpolate rVloc at s and s±δ for numerical derivative
                        double delta = std::min(1e-5, s * 0.001);
                        std::vector<double> spts = {s, s + delta, std::max(delta, s - delta)};
                        std::vector<double> vals;
                        Pseudopotential::spline_interp(r_grid, rVloc, rVloc_d, spts, vals);

                        double rVloc_s = vals[0];
                        double Vloc_s = rVloc_s / s;

                        // d(rVloc)/ds via central difference
                        double drVloc_ds = (vals[1] - vals[2]) / (spts[1] - spts[2]);

                        // dVloc/ds = [d(rVloc)/ds - Vloc] / s
                        double dVloc_ds = (drVloc_ds - Vloc_s) / s;

                        // ∇V = (dVloc/ds) * r_hat, where r_hat = (rx,ry,rz)/s
                        double inv_s = 1.0 / s;
                        double grad_Vx = dVloc_ds * rx * inv_s;
                        double grad_Vy = dVloc_ds * ry * inv_s;
                        double grad_Vz = dVloc_ds * rz * inv_s;

                        // F_J = -∫ ρ * ∇V dV  (∇ w.r.t. r, not R_J)
                        // But F = -dE/dR_J = +∫ ρ * ∇_r V dV  (∇_R V = -∇_r V)
                        // Actually: F_J = -∂E/∂R_J = -∫ ρ(r) · ∂V(r-R_J)/∂R_J dV
                        //                = +∫ ρ(r) · ∇_r V(r-R_J) dV
                        // Wait: ∂V(r-R)/∂R = -∇_r V, so F = +∫ ρ ∇_r V dV?
                        // No: E = ∫ ρ(r) V(r-R) dr, F = -dE/dR = -∫ ρ (∂V/∂R) = ∫ ρ ∇V
                        // Hmm, let me be more careful.
                        // V(r-R) depends on R through (r-R).
                        // ∂V(r-R)/∂R_x = -∂V(r-R)/∂(r-R)_x · 1 = -∂V/∂r_x
                        // So F_x = -∫ ρ · ∂V/∂R_x = +∫ ρ · ∂V/∂r_x
                        // Since ∂V/∂r_x = grad_Vx, we have F_x = ∫ ρ · grad_Vx

                        // WRONG: that gives repulsive force for Coulomb.
                        // For V = -Z/r, ∂V/∂r_x = Z·r_x/r³ (positive for positive r_x)
                        // So F_x = ∫ ρ · Z·r_x/r³ > 0 if electron is at positive x.
                        // But the physical force on nucleus at R from electron at r should
                        // be F = -Z·(R-r)/|R-r|³ · ρ = Z·(r-R)/|r-R|³ · ρ = Z·rx/s³ · ρ
                        // which is the same sign! So F_x = +∫ ρ · grad_Vx is correct.

                        int idx = li + lj * nx + lk * nx * ny;
                        fx += rho[idx] * grad_Vx * dV;
                        fy += rho[idx] * grad_Vy * dV;
                        fz += rho[idx] * grad_Vz * dV;
                    }
                }
            }

            f_local_[atom_count * 3 + 0] = fx;
            f_local_[atom_count * 3 + 1] = fy;
            f_local_[atom_count * 3 + 2] = fz;
            atom_count++;
        }
    }

    // Allreduce local forces across domain communicator
    if (!dmcomm.is_null() && dmcomm.size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, f_local_.data(), 3 * n_atom,
                      MPI_DOUBLE, MPI_SUM, dmcomm.comm());
    }
}

// ---------------------------------------------------------------------------
// Nonlocal force: KB projector contribution
//   F_J = -occfac · 2 · Σ_{n,k,s} w_k · g_n · Σ_{l,m,p} Γ · <χ|ψ> · <χ|∇ψ>
// ---------------------------------------------------------------------------
void Forces::compute_nonlocal(
    const Wavefunction& wfn,
    const Crystal& crystal,
    const std::vector<AtomNlocInfluence>& nloc_influence,
    const Gradient& gradient,
    const HaloExchange& halo,
    const Domain& domain,
    const FDGrid& grid,
    const std::vector<double>& kpt_weights,
    const MPIComm& dmcomm,
    const MPIComm& bandcomm,
    const MPIComm& kptcomm,
    const MPIComm& spincomm) {

    int n_atom = crystal.n_atom_total();
    int Nspin = wfn.Nspin();
    int Nkpts = wfn.Nkpts();
    int Nband = wfn.Nband();
    int Nd_d = domain.Nd_d();
    double dV = grid.dV();
    int ntypes = crystal.n_types();

    double occfac = (Nspin == 1) ? 2.0 : 1.0;

    f_nloc_.assign(3 * n_atom, 0.0);

    // Extended array dimensions for gradient
    int FDn = gradient.stencil().FDn();
    int nx = domain.Nx_d(), ny = domain.Ny_d(), nz = domain.Nz_d();
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nz_ex = nz + 2 * FDn;
    int Nd_ex = nx_ex * ny_ex * nz_ex;

    // Build per-type Gamma list and atom mapping
    struct TypeInfo {
        int nproj;
        std::vector<double> gamma;
    };
    std::vector<TypeInfo> type_info(ntypes);
    for (int it = 0; it < ntypes; ++it) {
        const auto& psd = crystal.types()[it].psd();
        type_info[it].nproj = psd.nproj_per_atom();
        for (int l = 0; l <= psd.lmax(); ++l) {
            if (l == psd.lloc()) continue;
            for (int p = 0; p < psd.ppl()[l]; ++p) {
                double g = psd.Gamma()[l][p];
                for (int m = -l; m <= l; ++m) {
                    type_info[it].gamma.push_back(g);
                }
            }
        }
    }

    for (int s = 0; s < Nspin; ++s) {
        for (int k = 0; k < Nkpts; ++k) {
            const NDArray<double>& psi_sk = wfn.psi(s, k);
            const NDArray<double>& occ_sk = wfn.occupations(s, k);
            double wk = kpt_weights[k];

            for (int n = 0; n < Nband; ++n) {
                double g_n = occ_sk(n);
                if (std::abs(g_n) < 1e-15) continue;

                const double* psi_n = psi_sk.data() + n * Nd_d;

                // Compute alpha = <χ|ψ> for all atoms
                std::vector<double> alpha;
                compute_chi_x(crystal, nloc_influence, domain, grid,
                              psi_n, dV, alpha, dmcomm);

                // For each direction, compute β = <χ|∇ψ> and accumulate force
                for (int dim = 0; dim < 3; ++dim) {
                    // Create extended psi and compute gradient
                    std::vector<double> psi_ex(Nd_ex, 0.0);
                    halo.execute(psi_n, psi_ex.data(), 1);

                    std::vector<double> Dpsi(Nd_d, 0.0);
                    gradient.apply(psi_ex.data(), Dpsi.data(), dim);

                    // Compute beta = <χ|∇ψ>
                    std::vector<double> beta;
                    compute_chi_x(crystal, nloc_influence, domain, grid,
                                  Dpsi.data(), dV, beta, dmcomm);

                    // Accumulate: F_J[dim] -= occfac * wk * 2 * g_n * Σ Γ * α * β
                    int proj_offset = 0;
                    for (int it = 0; it < ntypes; ++it) {
                        const auto& inf = nloc_influence[it];
                        int nproj = type_info[it].nproj;
                        if (nproj == 0) continue;
                        const auto& gamma = type_info[it].gamma;

                        for (int iat = 0; iat < inf.n_atom; ++iat) {
                            if (inf.ndc[iat] == 0) {
                                proj_offset += nproj;
                                continue;
                            }

                            int orig_atom = inf.atom_index[iat];
                            double fJ = 0.0;
                            for (int jp = 0; jp < nproj; ++jp) {
                                fJ += gamma[jp] * alpha[proj_offset + jp] * beta[proj_offset + jp];
                            }
                            f_nloc_[orig_atom * 3 + dim] -= occfac * wk * 2.0 * g_n * fJ;
                            proj_offset += nproj;
                        }
                    }
                }
            }
        }
    }

    // MPI reductions
    if (!bandcomm.is_null() && bandcomm.size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, f_nloc_.data(), 3 * n_atom,
                      MPI_DOUBLE, MPI_SUM, bandcomm.comm());
    }
    if (!kptcomm.is_null() && kptcomm.size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, f_nloc_.data(), 3 * n_atom,
                      MPI_DOUBLE, MPI_SUM, kptcomm.comm());
    }
    if (!spincomm.is_null() && spincomm.size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, f_nloc_.data(), 3 * n_atom,
                      MPI_DOUBLE, MPI_SUM, spincomm.comm());
    }
}

// ---------------------------------------------------------------------------
// Compute <χ_J|x> for all projectors of all atoms
// Result layout: [type0_atom0_proj0..nproj-1, type0_atom1_..., type1_...]
// ---------------------------------------------------------------------------
void Forces::compute_chi_x(
    const Crystal& crystal,
    const std::vector<AtomNlocInfluence>& nloc_influence,
    const Domain& domain,
    const FDGrid& grid,
    const double* x,
    double dV,
    std::vector<double>& result,
    const MPIComm& dmcomm) {

    int ntypes = crystal.n_types();
    int nx_d = domain.Nx_d(), ny_d = domain.Ny_d();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();

    // Count total projectors
    int total_nproj = 0;
    for (int it = 0; it < ntypes; ++it) {
        int nproj = crystal.types()[it].psd().nproj_per_atom();
        total_nproj += nproj * nloc_influence[it].n_atom;
    }
    result.assign(total_nproj, 0.0);

    int proj_offset = 0;
    for (int it = 0; it < ntypes; ++it) {
        const auto& psd = crystal.types()[it].psd();
        const auto& inf = nloc_influence[it];
        int nproj = psd.nproj_per_atom();
        if (nproj == 0) continue;

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            int ndc = inf.ndc[iat];
            if (ndc == 0) {
                proj_offset += nproj;
                continue;
            }

            const auto& gpos = inf.grid_pos[iat];
            Vec3 atom_pos = inf.coords[iat];

            // Compute distances and coordinates
            std::vector<double> rx(ndc), ry(ndc), rz(ndc), rr(ndc);
            for (int ig = 0; ig < ndc; ++ig) {
                int flat = gpos[ig];
                int li = flat % nx_d;
                int lj = (flat / nx_d) % ny_d;
                int lk = flat / (nx_d * ny_d);
                int gi = li + domain.vertices().xs;
                int gj = lj + domain.vertices().ys;
                int gk = lk + domain.vertices().zs;
                rx[ig] = gi * dx - atom_pos.x;
                ry[ig] = gj * dy - atom_pos.y;
                rz[ig] = gk * dz - atom_pos.z;
                rr[ig] = std::sqrt(rx[ig] * rx[ig] + ry[ig] * ry[ig] + rz[ig] * rz[ig]);
            }

            // Compute <χ|x> for each projector
            int col = 0;
            for (int l = 0; l <= psd.lmax(); ++l) {
                if (l == psd.lloc()) continue;
                for (int p = 0; p < psd.ppl()[l]; ++p) {
                    std::vector<double> udv_interp;
                    Pseudopotential::spline_interp(
                        psd.radial_grid(), psd.UdV()[l][p], psd.UdV_spline_d()[l][p],
                        rr, udv_interp);

                    for (int m = -l; m <= l; ++m) {
                        double dot = 0.0;
                        for (int ig = 0; ig < ndc; ++ig) {
                            double ylm = Ylm(l, m, rx[ig], ry[ig], rz[ig], rr[ig]);
                            double chi = ylm * udv_interp[ig];
                            dot += chi * x[gpos[ig]];
                        }
                        result[proj_offset + col] = dot * dV;
                        col++;
                    }
                }
            }
            proj_offset += nproj;
        }
    }

    // Allreduce across domain
    if (!dmcomm.is_null() && dmcomm.size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, result.data(), total_nproj,
                      MPI_DOUBLE, MPI_SUM, dmcomm.comm());
    }
}

// Real spherical harmonics (same as NonlocalProjector)
double Forces::Ylm(int l, int m, double x, double y, double z, double r) {
    if (r < 1e-14) return 0.0;
    double invr = 1.0 / r;
    double xn = x * invr, yn = y * invr, zn = z * invr;

    if (l == 0) return 1.0;
    if (l == 1) {
        if (m == -1) return yn;
        if (m ==  0) return zn;
        if (m ==  1) return xn;
    }
    if (l == 2) {
        double x2 = xn * xn, y2 = yn * yn, z2 = zn * zn;
        if (m == -2) return xn * yn * std::sqrt(3.0);
        if (m == -1) return yn * zn * std::sqrt(3.0);
        if (m ==  0) return 0.5 * (3.0 * z2 - 1.0);
        if (m ==  1) return xn * zn * std::sqrt(3.0);
        if (m ==  2) return 0.5 * std::sqrt(3.0) * (x2 - y2);
    }
    if (l == 3) {
        double x2 = xn * xn, y2 = yn * yn, z2 = zn * zn;
        if (m == -3) return 0.5 * std::sqrt(2.5) * yn * (3.0 * x2 - y2);
        if (m == -2) return xn * yn * zn * std::sqrt(15.0);
        if (m == -1) return 0.25 * std::sqrt(1.5) * yn * (5.0 * z2 - 1.0);
        if (m ==  0) return 0.5 * zn * (5.0 * z2 - 3.0);
        if (m ==  1) return 0.25 * std::sqrt(1.5) * xn * (5.0 * z2 - 1.0);
        if (m ==  2) return 0.5 * std::sqrt(15.0) * zn * (x2 - y2);
        if (m ==  3) return 0.5 * std::sqrt(2.5) * xn * (x2 - 3.0 * y2);
    }
    return 0.0;
}

void Forces::symmetrize(std::vector<double>& forces, int n_atom) {
    for (int d = 0; d < 3; ++d) {
        double avg = 0.0;
        for (int i = 0; i < n_atom; ++i) {
            avg += forces[i * 3 + d];
        }
        avg /= n_atom;
        for (int i = 0; i < n_atom; ++i) {
            forces[i * 3 + d] -= avg;
        }
    }
}

} // namespace sparc
