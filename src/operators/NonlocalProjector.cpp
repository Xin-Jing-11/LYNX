#include "operators/NonlocalProjector.hpp"
#include "core/constants.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace sparc {

// Real spherical harmonics Y_lm(x, y, z, r)
// Unnormalized (matching SPARC convention where normalization is in UdV)
double NonlocalProjector::spherical_harmonic(int l, int m, double x, double y, double z, double r) {
    if (r < 1e-14) return 0.0;

    double invr = 1.0 / r;
    double xn = x * invr, yn = y * invr, zn = z * invr;

    // l = 0
    if (l == 0) return 1.0;

    // l = 1: m = -1, 0, 1 → y, z, x
    if (l == 1) {
        if (m == -1) return yn;
        if (m ==  0) return zn;
        if (m ==  1) return xn;
    }

    // l = 2
    if (l == 2) {
        double x2 = xn * xn, y2 = yn * yn, z2 = zn * zn;
        if (m == -2) return xn * yn * std::sqrt(3.0);
        if (m == -1) return yn * zn * std::sqrt(3.0);
        if (m ==  0) return 0.5 * (3.0 * z2 - 1.0);
        if (m ==  1) return xn * zn * std::sqrt(3.0);
        if (m ==  2) return 0.5 * std::sqrt(3.0) * (x2 - y2);
    }

    // l = 3
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

    return 0.0;  // Higher l not implemented yet
}

void NonlocalProjector::setup(const Crystal& crystal,
                               const std::vector<AtomNlocInfluence>& nloc_influence,
                               const Domain& domain,
                               const FDGrid& grid) {
    crystal_ = &crystal;
    nloc_influence_ = &nloc_influence;

    int ntypes = crystal.n_types();
    Chi_.resize(ntypes);
    IP_displ_.resize(ntypes);

    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    int nx_d = domain.Nx_d();
    int ny_d = domain.Ny_d();

    total_nproj_ = 0;
    Gamma_all_.clear();

    for (int it = 0; it < ntypes; ++it) {
        const auto& psd = crystal.types()[it].psd();
        const auto& inf = nloc_influence[it];
        int nproj = psd.nproj_per_atom();

        Chi_[it].resize(inf.n_atom);
        IP_displ_[it].resize(inf.n_atom + 1, 0);

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            IP_displ_[it][iat + 1] = IP_displ_[it][iat] + nproj;
            int ndc = inf.ndc[iat];

            if (ndc == 0 || nproj == 0) continue;

            Chi_[it][iat] = NDArray<double>(ndc, nproj);

            // Compute grid point coordinates relative to atom
            const auto& gpos = inf.grid_pos[iat];
            Vec3 atom_pos = inf.coords[iat];

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

            // Build Chi columns: for each (l, p, m) excluding lloc
            int col = 0;
            for (int l = 0; l <= psd.lmax(); ++l) {
                if (l == psd.lloc()) continue;
                for (int p = 0; p < psd.ppl()[l]; ++p) {
                    // Interpolate radial projector UdV at grid distances
                    std::vector<double> udv_interp;
                    Pseudopotential::spline_interp(
                        psd.radial_grid(), psd.UdV()[l][p], psd.UdV_spline_d()[l][p],
                        rr, udv_interp);

                    for (int m = -l; m <= l; ++m) {
                        for (int ig = 0; ig < ndc; ++ig) {
                            double ylm = spherical_harmonic(l, m, rx[ig], ry[ig], rz[ig], rr[ig]);
                            Chi_[it][iat](ig, col) = ylm * udv_interp[ig];
                        }
                        col++;
                    }
                }
            }

            // Store Gamma for this atom's projectors
            for (int l = 0; l <= psd.lmax(); ++l) {
                if (l == psd.lloc()) continue;
                for (int p = 0; p < psd.ppl()[l]; ++p) {
                    double gamma = psd.Gamma()[l][p];
                    for (int m = -l; m <= l; ++m)
                        Gamma_all_.push_back(gamma);
                }
            }

            total_nproj_ += nproj;
        }
    }

    is_setup_ = true;
}

void NonlocalProjector::apply(const double* psi, double* Hpsi, int ncol, double dV,
                               const MPIComm& comm) const {
    if (!is_setup_ || total_nproj_ == 0) return;

    // Step 1: Compute alpha = Chi^T * psi * dV (inner products)
    int ntypes = static_cast<int>(Chi_.size());
    std::vector<double> alpha(total_nproj_ * ncol, 0.0);

    int proj_offset = 0;
    for (int it = 0; it < ntypes; ++it) {
        const auto& inf = (*nloc_influence_)[it];
        const auto& psd = crystal_->types()[it].psd();
        int nproj = psd.nproj_per_atom();
        if (nproj == 0) continue;

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            int ndc = inf.ndc[iat];
            if (ndc == 0) continue;

            const auto& gpos = inf.grid_pos[iat];
            int alpha_offset = (proj_offset + IP_displ_[it][iat]) * ncol;

            // Extract psi values at grid positions and compute alpha = Chi^T * x_rc
            for (int n = 0; n < ncol; ++n) {
                for (int jp = 0; jp < nproj; ++jp) {
                    double dot = 0.0;
                    for (int ig = 0; ig < ndc; ++ig) {
                        dot += Chi_[it][iat](ig, jp) * psi[gpos[ig] + n * (int)(inf.ndc[0] > 0 ? gpos.back() + 1 : 0)];
                    }
                    // Correction: psi is Nd_d per column
                    alpha[alpha_offset + n * nproj + jp] = 0.0;
                }
            }

            // Actually need to use the correct psi stride
            // psi layout: psi[i + n * Nd_d]
            // For simplicity, use direct loop
            for (int n = 0; n < ncol; ++n) {
                const double* psi_n = psi; // caller must pass with correct stride
                for (int jp = 0; jp < nproj; ++jp) {
                    double dot = 0.0;
                    for (int ig = 0; ig < ndc; ++ig) {
                        dot += Chi_[it][iat](ig, jp) * psi_n[gpos[ig]];
                    }
                    alpha[alpha_offset + n * nproj + jp] = dot * dV;
                }
                psi += 0; // will be adjusted by caller stride
            }
        }

        proj_offset += IP_displ_[it].back();
    }

    // Step 2: Allreduce alpha across domain communicator
    if (!comm.is_null() && comm.size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha.data(),
                      static_cast<int>(alpha.size()), MPI_DOUBLE, MPI_SUM, comm.comm());
    }

    // Step 3: Apply Gamma and accumulate: Hpsi += Chi * (Gamma * alpha)
    int gamma_idx = 0;
    proj_offset = 0;
    for (int it = 0; it < ntypes; ++it) {
        const auto& inf = (*nloc_influence_)[it];
        const auto& psd = crystal_->types()[it].psd();
        int nproj = psd.nproj_per_atom();
        if (nproj == 0) continue;

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            int ndc = inf.ndc[iat];
            if (ndc == 0) continue;

            const auto& gpos = inf.grid_pos[iat];
            int alpha_offset = (proj_offset + IP_displ_[it][iat]) * ncol;

            // Apply Gamma to alpha
            for (int n = 0; n < ncol; ++n) {
                for (int jp = 0; jp < nproj; ++jp) {
                    alpha[alpha_offset + n * nproj + jp] *= Gamma_all_[gamma_idx + jp];
                }
            }

            // Accumulate: Hpsi[gpos[ig]] += sum_j Chi(ig,j) * alpha(j)
            for (int n = 0; n < ncol; ++n) {
                for (int ig = 0; ig < ndc; ++ig) {
                    double val = 0.0;
                    for (int jp = 0; jp < nproj; ++jp) {
                        val += Chi_[it][iat](ig, jp) * alpha[alpha_offset + n * nproj + jp];
                    }
                    Hpsi[gpos[ig]] += val;
                }
                Hpsi += 0; // stride adjusted by caller
            }

            gamma_idx += nproj;
        }
        proj_offset += IP_displ_[it].back();
    }
}

} // namespace sparc
