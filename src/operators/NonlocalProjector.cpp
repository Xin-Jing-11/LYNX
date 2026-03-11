#include "operators/NonlocalProjector.hpp"
#include "core/constants.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace sparc {

// Real spherical harmonics Y_lm(x, y, z, r)
// Fully normalized: matching reference SPARC convention (tools.c RealSphericalHarmonic)
double NonlocalProjector::spherical_harmonic(int l, int m, double x, double y, double z, double r) {
    if (r < 1e-14) return 0.0;

    double invr = 1.0 / r;
    double xn = x * invr, yn = y * invr, zn = z * invr;

    // l = 0: Y00 = 0.5*sqrt(1/pi)
    if (l == 0) return 0.282094791773878;

    // l = 1: coefficients = sqrt(3/(4*pi))
    if (l == 1) {
        constexpr double C1 = 0.488602511902920;
        if (m == -1) return C1 * yn;
        if (m ==  0) return C1 * zn;
        if (m ==  1) return C1 * xn;
    }

    // l = 2
    if (l == 2) {
        double x2 = xn * xn, y2 = yn * yn, z2 = zn * zn;
        if (m == -2) return 1.092548430592079 * xn * yn;           // 0.5*sqrt(15/pi)
        if (m == -1) return 1.092548430592079 * yn * zn;           // 0.5*sqrt(15/pi)
        if (m ==  0) return 0.315391565252520 * (3.0 * z2 - 1.0); // 0.25*sqrt(5/pi)
        if (m ==  1) return 1.092548430592079 * xn * zn;           // 0.5*sqrt(15/pi)
        if (m ==  2) return 0.546274215296040 * (x2 - y2);         // 0.25*sqrt(15/pi)
    }

    // l = 3
    if (l == 3) {
        double x2 = xn * xn, y2 = yn * yn, z2 = zn * zn;
        if (m == -3) return 0.590043589926644 * yn * (3.0 * x2 - y2);  // 0.25*sqrt(35/(2*pi))
        if (m == -2) return 2.890611442640554 * xn * yn * zn;          // 0.5*sqrt(105/pi)
        if (m == -1) return 0.457045799464466 * yn * (5.0 * z2 - 1.0); // 0.25*sqrt(21/(2*pi))
        if (m ==  0) return 0.373176332590115 * zn * (5.0 * z2 - 3.0); // 0.25*sqrt(7/pi)
        if (m ==  1) return 0.457045799464466 * xn * (5.0 * z2 - 1.0); // 0.25*sqrt(21/(2*pi))
        if (m ==  2) return 1.445305721320277 * zn * (x2 - y2);        // 0.25*sqrt(105/pi)
        if (m ==  3) return 0.590043589926644 * xn * (x2 - 3.0 * y2);  // 0.25*sqrt(35/(2*pi))
    }

    return 0.0;  // Higher l not implemented yet
}

void NonlocalProjector::setup(const Crystal& crystal,
                               const std::vector<AtomNlocInfluence>& nloc_influence,
                               const Domain& domain,
                               const FDGrid& grid) {
    crystal_ = &crystal;
    nloc_influence_ = &nloc_influence;
    domain_ = &domain;

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

    int Nd_d = domain_->Nd_d();
    int ntypes = static_cast<int>(Chi_.size());
    int n_atom = crystal_->n_atom_total();

    // Global IP_displ: projector offset per physical atom
    // IP_displ_global[iat+1] = IP_displ_global[iat] + nproj_for_that_atom_type
    std::vector<int> IP_displ_global(n_atom + 1, 0);
    {
        int atom_idx = 0;
        for (int it = 0; it < crystal_->n_types(); ++it) {
            const auto& psd = crystal_->types()[it].psd();
            int nproj = psd.nproj_per_atom();
            int nat = crystal_->types()[it].n_atoms();
            for (int ia = 0; ia < nat; ++ia) {
                IP_displ_global[atom_idx + 1] = IP_displ_global[atom_idx] + nproj;
                atom_idx++;
            }
        }
    }
    int total_global_nproj = IP_displ_global[n_atom];

    // Step 1: Compute alpha = Chi^T * psi * dV (inner products)
    // Global alpha array: indexed by actual atom index (accumulates across periodic images)
    std::vector<double> alpha(total_global_nproj * ncol, 0.0);

    for (int it = 0; it < ntypes; ++it) {
        const auto& inf = (*nloc_influence_)[it];
        const auto& psd = crystal_->types()[it].psd();
        int nproj = psd.nproj_per_atom();
        if (nproj == 0) continue;

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            int ndc = inf.ndc[iat];
            if (ndc == 0) continue;

            int global_atom = inf.atom_index[iat];
            const auto& gpos = inf.grid_pos[iat];
            int a_off = IP_displ_global[global_atom] * ncol;

            for (int n = 0; n < ncol; ++n) {
                const double* psi_n = psi + n * Nd_d;
                for (int jp = 0; jp < nproj; ++jp) {
                    double dot = 0.0;
                    for (int ig = 0; ig < ndc; ++ig) {
                        dot += Chi_[it][iat](ig, jp) * psi_n[gpos[ig]];
                    }
                    alpha[a_off + n * nproj + jp] += dot * dV;  // += to accumulate across images
                }
            }
        }
    }

    // Step 2: Allreduce alpha across domain communicator
    if (!comm.is_null() && comm.size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha.data(),
                      static_cast<int>(alpha.size()), MPI_DOUBLE, MPI_SUM, comm.comm());
    }

    // Step 2b: Apply Gamma to alpha (matching reference: iterate over ALL atoms sequentially)
    {
        int count = 0;
        for (int it = 0; it < crystal_->n_types(); ++it) {
            const auto& psd = crystal_->types()[it].psd();
            int nat = crystal_->types()[it].n_atoms();
            for (int ia = 0; ia < nat; ++ia) {
                for (int n = 0; n < ncol; ++n) {
                    for (int l = 0; l <= psd.lmax(); ++l) {
                        if (l == psd.lloc()) continue;
                        for (int p = 0; p < psd.ppl()[l]; ++p) {
                            double gamma = psd.Gamma()[l][p];
                            for (int m = -l; m <= l; ++m) {
                                alpha[count++] *= gamma;
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 3: Accumulate: Hpsi += Chi * alpha (scatter back to all images)
    for (int it = 0; it < ntypes; ++it) {
        const auto& inf = (*nloc_influence_)[it];
        const auto& psd = crystal_->types()[it].psd();
        int nproj = psd.nproj_per_atom();
        if (nproj == 0) continue;

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            int ndc = inf.ndc[iat];
            if (ndc == 0) continue;

            int global_atom = inf.atom_index[iat];
            const auto& gpos = inf.grid_pos[iat];
            int a_off = IP_displ_global[global_atom] * ncol;

            for (int n = 0; n < ncol; ++n) {
                double* Hpsi_n = Hpsi + n * Nd_d;
                for (int ig = 0; ig < ndc; ++ig) {
                    double val = 0.0;
                    for (int jp = 0; jp < nproj; ++jp) {
                        val += Chi_[it][iat](ig, jp) * alpha[a_off + n * nproj + jp];
                    }
                    Hpsi_n[gpos[ig]] += val;
                }
            }
        }
    }
}

} // namespace sparc
