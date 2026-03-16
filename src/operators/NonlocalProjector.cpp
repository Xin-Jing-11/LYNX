#include "operators/NonlocalProjector.hpp"
#include "core/constants.hpp"
#include "core/math_utils.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace lynx {

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

void NonlocalProjector::setup_soc(const Crystal& crystal,
                                    const std::vector<AtomNlocInfluence>& nloc_influence,
                                    const Domain& domain,
                                    const FDGrid& grid) {
    // Check if any atom type has SOC data
    bool any_soc = false;
    for (int it = 0; it < crystal.n_types(); ++it) {
        if (crystal.types()[it].psd().has_soc()) { any_soc = true; break; }
    }
    if (!any_soc) return;
    has_soc_ = true;

    int ntypes = crystal.n_types();
    Chi_soc_.resize(ntypes);
    soc_proj_info_.resize(ntypes);
    Gamma_soc_all_.clear();

    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    int nx_d = domain.Nx_d();
    int ny_d = domain.Ny_d();

    for (int it = 0; it < ntypes; ++it) {
        const auto& psd = crystal.types()[it].psd();
        if (!psd.has_soc()) {
            Chi_soc_[it].clear();
            soc_proj_info_[it].clear();
            continue;
        }

        const auto& inf = nloc_influence[it];

        // Compute nproj_soc per atom: sum_{l=1..lmax} ppl_soc[l] * (2*l+1)
        int nproj_soc = 0;
        for (int l = 1; l <= psd.lmax(); ++l) {
            nproj_soc += psd.ppl_soc()[l] * (2 * l + 1);
        }

        // Build per-type projector info
        soc_proj_info_[it].clear();
        for (int l = 1; l <= psd.lmax(); ++l) {
            for (int p = 0; p < psd.ppl_soc()[l]; ++p) {
                for (int m = -l; m <= l; ++m) {
                    soc_proj_info_[it].push_back({l, m, p});
                }
            }
        }

        Chi_soc_[it].resize(inf.n_atom);

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            int ndc = inf.ndc[iat];
            if (ndc == 0 || nproj_soc == 0) continue;

            Chi_soc_[it][iat] = NDArray<double>(ndc, nproj_soc);

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
                double dx_nc = gi * dx - atom_pos.x;
                double dy_nc = gj * dy - atom_pos.y;
                double dz_nc = gk * dz - atom_pos.z;
                if (is_orth) {
                    rx[ig] = dx_nc; ry[ig] = dy_nc; rz[ig] = dz_nc;
                } else {
                    Vec3 cart = lattice.nonCart_to_cart({dx_nc, dy_nc, dz_nc});
                    rx[ig] = cart.x; ry[ig] = cart.y; rz[ig] = cart.z;
                }
                rr[ig] = std::sqrt(rx[ig]*rx[ig] + ry[ig]*ry[ig] + rz[ig]*rz[ig]);
            }

            // Build Chi_soc columns
            int col = 0;
            for (int l = 1; l <= psd.lmax(); ++l) {
                for (int p = 0; p < psd.ppl_soc()[l]; ++p) {
                    std::vector<double> udv_interp;
                    Pseudopotential::spline_interp(
                        psd.radial_grid(), psd.UdV_soc()[l][p], psd.UdV_soc_spline_d()[l][p],
                        rr, udv_interp);

                    for (int m = -l; m <= l; ++m) {
                        for (int ig = 0; ig < ndc; ++ig) {
                            double ylm = spherical_harmonic(l, m, rx[ig], ry[ig], rz[ig], rr[ig]);
                            Chi_soc_[it][iat](ig, col) = ylm * udv_interp[ig];
                        }
                        col++;
                    }
                }
            }

            // Store SOC Gamma for this atom
            for (int l = 1; l <= psd.lmax(); ++l) {
                for (int p = 0; p < psd.ppl_soc()[l]; ++p) {
                    double gamma = psd.Gamma_soc()[l][p];
                    for (int m = -l; m <= l; ++m)
                        Gamma_soc_all_.push_back(gamma);
                }
            }
        }
    }
}

void NonlocalProjector::apply_soc_kpt(const Complex* psi, Complex* Hpsi, int ncol,
                                        int Nd_d, double dV) const {
    if (!has_soc_) return;

    int ntypes = static_cast<int>(Chi_soc_.size());
    int n_atom = crystal_->n_atom_total();
    int Nd_d_spinor = 2 * Nd_d;

    // Build global atom projector displacement for SOC
    std::vector<int> IP_displ_global(n_atom + 1, 0);
    {
        int atom_idx = 0;
        for (int it = 0; it < crystal_->n_types(); ++it) {
            const auto& psd = crystal_->types()[it].psd();
            int nproj_soc = 0;
            if (psd.has_soc()) {
                for (int l = 1; l <= psd.lmax(); ++l)
                    nproj_soc += psd.ppl_soc()[l] * (2 * l + 1);
            }
            int nat = crystal_->types()[it].n_atoms();
            for (int ia = 0; ia < nat; ++ia) {
                IP_displ_global[atom_idx + 1] = IP_displ_global[atom_idx] + nproj_soc;
                atom_idx++;
            }
        }
    }
    int total_soc_nproj = IP_displ_global[n_atom];
    if (total_soc_nproj == 0) return;

    // For each spinor component sigma (0=up, 1=down), compute alpha for SOC projectors
    // alpha_sigma[proj * ncol + n] = bloch_fac * dV * Chi_soc^T * psi_sigma
    std::vector<Complex> alpha_up(total_soc_nproj * ncol, Complex(0.0));
    std::vector<Complex> alpha_dn(total_soc_nproj * ncol, Complex(0.0));

    for (int it = 0; it < ntypes; ++it) {
        const auto& psd = crystal_->types()[it].psd();
        if (!psd.has_soc()) continue;
        const auto& inf = (*nloc_influence_)[it];
        int nproj_soc = 0;
        for (int l = 1; l <= psd.lmax(); ++l)
            nproj_soc += psd.ppl_soc()[l] * (2 * l + 1);
        if (nproj_soc == 0) continue;

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            int ndc = inf.ndc[iat];
            if (ndc == 0) continue;

            int global_atom = inf.atom_index[iat];
            const auto& gpos = inf.grid_pos[iat];

            const Vec3& shift = inf.image_shift[iat];
            double theta = -(kpt_cart_.x * shift.x + kpt_cart_.y * shift.y + kpt_cart_.z * shift.z);
            Complex bloch_fac(std::cos(theta), std::sin(theta));
            Complex alpha_scale = bloch_fac * dV;

            int a_off = IP_displ_global[global_atom] * ncol;

            for (int n = 0; n < ncol; ++n) {
                const Complex* psi_n = psi + n * Nd_d_spinor;
                const Complex* psi_up = psi_n;           // [0 .. Nd_d-1]
                const Complex* psi_dn = psi_n + Nd_d;    // [Nd_d .. 2*Nd_d-1]

                for (int jp = 0; jp < nproj_soc; ++jp) {
                    Complex dot_up(0.0), dot_dn(0.0);
                    for (int ig = 0; ig < ndc; ++ig) {
                        double chi_val = Chi_soc_[it][iat](ig, jp);
                        dot_up += chi_val * psi_up[gpos[ig]];
                        dot_dn += chi_val * psi_dn[gpos[ig]];
                    }
                    alpha_up[a_off + n * nproj_soc + jp] += alpha_scale * dot_up;
                    alpha_dn[a_off + n * nproj_soc + jp] += alpha_scale * dot_dn;
                }
            }
        }
    }

    // Apply SOC Term 1 (on-diagonal, Lz·Sz) and Term 2 (off-diagonal, ladder operators)
    // For each projector column with (l, m, p):
    //   Term 1: Hpsi_up += 0.5 * m * Gamma_soc * alpha_up * Chi_soc
    //           Hpsi_dn -= 0.5 * m * Gamma_soc * alpha_dn * Chi_soc
    //   Term 2: Hpsi_up += 0.5 * sqrt(l(l+1)-m(m+1)) * Gamma_soc_shifted * alpha_dn_shifted * Chi_soc
    //           Hpsi_dn += 0.5 * sqrt(l(l+1)-m(m-1)) * Gamma_soc_shifted * alpha_up_shifted * Chi_soc

    // First apply Gamma and m-factor to alpha arrays for Term 1
    // Then scatter back
    for (int it = 0; it < ntypes; ++it) {
        const auto& psd = crystal_->types()[it].psd();
        if (!psd.has_soc()) continue;
        const auto& inf = (*nloc_influence_)[it];
        const auto& proj_info = soc_proj_info_[it];
        int nproj_soc = static_cast<int>(proj_info.size());
        if (nproj_soc == 0) continue;

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            int ndc = inf.ndc[iat];
            if (ndc == 0) continue;

            int global_atom = inf.atom_index[iat];
            const auto& gpos = inf.grid_pos[iat];

            const Vec3& shift = inf.image_shift[iat];
            double theta = -(kpt_cart_.x * shift.x + kpt_cart_.y * shift.y + kpt_cart_.z * shift.z);
            Complex bloch_fac_conj(std::cos(theta), -std::sin(theta));

            int a_off = IP_displ_global[global_atom] * ncol;

            for (int n = 0; n < ncol; ++n) {
                Complex* Hpsi_n = Hpsi + n * Nd_d_spinor;
                Complex* Hpsi_up = Hpsi_n;
                Complex* Hpsi_dn = Hpsi_n + Nd_d;

                // Process each projector column
                for (int jp = 0; jp < nproj_soc; ++jp) {
                    int l = proj_info[jp].l;
                    int m = proj_info[jp].m;
                    int p = proj_info[jp].p;
                    double gamma_soc = psd.Gamma_soc()[l][p];

                    // Term 1: on-diagonal (Lz·Sz)
                    // Hpsi_up += 0.5 * m * gamma_soc * alpha_up * Chi_soc (conj bloch)
                    // Hpsi_dn -= 0.5 * m * gamma_soc * alpha_dn * Chi_soc (conj bloch)
                    if (m != 0) {
                        Complex coeff_up = 0.5 * static_cast<double>(m) * gamma_soc *
                                           alpha_up[a_off + n * nproj_soc + jp];
                        Complex coeff_dn = -0.5 * static_cast<double>(m) * gamma_soc *
                                           alpha_dn[a_off + n * nproj_soc + jp];
                        for (int ig = 0; ig < ndc; ++ig) {
                            double chi_val = Chi_soc_[it][iat](ig, jp);
                            Hpsi_up[gpos[ig]] += bloch_fac_conj * coeff_up * chi_val;
                            Hpsi_dn[gpos[ig]] += bloch_fac_conj * coeff_dn * chi_val;
                        }
                    }

                    // Term 2: L+ S- (raises m, acts on spin-down -> spin-up)
                    if (m + 1 <= l) {
                        double ladder = std::sqrt(static_cast<double>(l*(l+1) - m*(m+1)));
                        // Find the column index for (l, m+1, p)
                        int jp_shifted = jp + 1;  // m columns are sequential for same (l,p)
                        Complex coeff = 0.5 * ladder * gamma_soc *
                                        alpha_dn[a_off + n * nproj_soc + jp_shifted];
                        for (int ig = 0; ig < ndc; ++ig) {
                            Hpsi_up[gpos[ig]] += bloch_fac_conj * coeff * Chi_soc_[it][iat](ig, jp);
                        }
                    }

                    // Term 2: L- S+ (lowers m, acts on spin-up -> spin-down)
                    if (m - 1 >= -l) {
                        double ladder = std::sqrt(static_cast<double>(l*(l+1) - m*(m-1)));
                        int jp_shifted = jp - 1;  // column for (l, m-1, p)
                        Complex coeff = 0.5 * ladder * gamma_soc *
                                        alpha_up[a_off + n * nproj_soc + jp_shifted];
                        for (int ig = 0; ig < ndc; ++ig) {
                            Hpsi_dn[gpos[ig]] += bloch_fac_conj * coeff * Chi_soc_[it][iat](ig, jp);
                        }
                    }
                }
            }
        }
    }
}

} // namespace lynx
