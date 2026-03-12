#include "operators/NonlocalProjector.hpp"
#include "core/constants.hpp"
#include "core/math_utils.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace sparc {

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

            bool is_orth = grid.lattice().is_orthogonal();
            const auto& lattice = grid.lattice();

            std::vector<double> rx(ndc), ry(ndc), rz(ndc), rr(ndc);
            for (int ig = 0; ig < ndc; ++ig) {
                int flat = gpos[ig];
                int li = flat % nx_d;
                int lj = (flat / nx_d) % ny_d;
                int lk = flat / (nx_d * ny_d);
                int gi = li + domain.vertices().xs;
                int gj = lj + domain.vertices().ys;
                int gk = lk + domain.vertices().zs;
                // Displacement in non-Cart coords
                double dx_nc = gi * dx - atom_pos.x;
                double dy_nc = gj * dy - atom_pos.y;
                double dz_nc = gk * dz - atom_pos.z;
                if (is_orth) {
                    // Non-Cart = Cartesian for orthogonal cells
                    rx[ig] = dx_nc;
                    ry[ig] = dy_nc;
                    rz[ig] = dz_nc;
                } else {
                    // Convert non-Cart displacement to Cartesian for Ylm
                    Vec3 cart = lattice.nonCart_to_cart({dx_nc, dy_nc, dz_nc});
                    rx[ig] = cart.x;
                    ry[ig] = cart.y;
                    rz[ig] = cart.z;
                }
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

void NonlocalProjector::apply(const double* psi, double* Hpsi, int ncol, double dV) const {
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

void NonlocalProjector::apply_kpt(const Complex* psi, Complex* Hpsi, int ncol, double dV) const {
    if (!is_setup_ || total_nproj_ == 0) return;

    int Nd_d = domain_->Nd_d();
    int ntypes = static_cast<int>(Chi_.size());
    int n_atom = crystal_->n_atom_total();

    // Global IP_displ: projector offset per physical atom
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

    // Step 1: Compute alpha = bloch_fac * dV * Chi^T * psi (complex inner products)
    std::vector<Complex> alpha(total_global_nproj * ncol, Complex(0.0, 0.0));

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

            // Bloch phase factor: e^{-i * k · R_image}
            const Vec3& shift = inf.image_shift[iat];
            double theta = -(kpt_cart_.x * shift.x + kpt_cart_.y * shift.y + kpt_cart_.z * shift.z);
            Complex bloch_fac(std::cos(theta), std::sin(theta));
            Complex alpha_scale = bloch_fac * dV;

            int a_off = IP_displ_global[global_atom] * ncol;

            for (int n = 0; n < ncol; ++n) {
                const Complex* psi_n = psi + n * Nd_d;
                for (int jp = 0; jp < nproj; ++jp) {
                    Complex dot(0.0, 0.0);
                    for (int ig = 0; ig < ndc; ++ig) {
                        // Chi is real, psi is complex
                        dot += Chi_[it][iat](ig, jp) * psi_n[gpos[ig]];
                    }
                    alpha[a_off + n * nproj + jp] += alpha_scale * dot;
                }
            }
        }
    }

    // Step 2: Apply Gamma to alpha
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

    // Step 3: Accumulate: Hpsi += conj(bloch_fac) * Chi * alpha
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

            // Conjugate Bloch phase: e^{+i * k · R_image}
            const Vec3& shift = inf.image_shift[iat];
            double theta = -(kpt_cart_.x * shift.x + kpt_cart_.y * shift.y + kpt_cart_.z * shift.z);
            Complex bloch_fac_conj(std::cos(theta), -std::sin(theta));

            int a_off = IP_displ_global[global_atom] * ncol;

            for (int n = 0; n < ncol; ++n) {
                Complex* Hpsi_n = Hpsi + n * Nd_d;
                for (int ig = 0; ig < ndc; ++ig) {
                    Complex val(0.0, 0.0);
                    for (int jp = 0; jp < nproj; ++jp) {
                        val += Chi_[it][iat](ig, jp) * alpha[a_off + n * nproj + jp];
                    }
                    Hpsi_n[gpos[ig]] += bloch_fac_conj * val;
                }
            }
        }
    }
}

} // namespace sparc
