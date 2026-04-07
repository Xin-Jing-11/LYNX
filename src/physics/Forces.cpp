#include "physics/Forces.hpp"
#include "physics/SCF.hpp"
#include "physics/Electrostatics.hpp"
#include "atoms/AtomSetup.hpp"
#include "io/InputParser.hpp"
#include "core/constants.hpp"
#include "core/Lattice.hpp"
#include "atoms/Pseudopotential.hpp"
#include <cmath>
#include <cstring>
#include <complex>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <cassert>

namespace lynx {

// ---------------------------------------------------------------------------
// High-level entry point: extract data from SCF/atoms and delegate.
// ---------------------------------------------------------------------------
void Forces::compute(
    const LynxContext& ctx,
    const SystemConfig& /*config*/,
    const Wavefunction& wfn,
    const SCF& scf,
    const AtomSetup& atoms,
    const NonlocalProjector& vnl) {

    // Set device for internal dispatch.
    // SOC forces stay on CPU: the GPU nonlocal force kernel doesn't handle
    // spinor layout, and SPARC also computes SOC forces on CPU.
    dev_ = Device::CPU;
#ifdef USE_CUDA
    if (scf.hamiltonian_ptr() && scf.hamiltonian_ptr()->gpu_state_ptr()
        && wfn.Nspinor() == 1) {
        dev_ = Device::GPU;
        hamiltonian_ = scf.hamiltonian_ptr();
        eigsolver_ = &scf.eigsolver();
    }
#endif

    compute_impl(ctx, wfn, atoms.crystal,
                 atoms.influence, atoms.nloc_influence, vnl,
                 scf.phi(), scf.density().rho_total().data(),
                 atoms.Vloc.data(),
                 atoms.elec.pseudocharge().data(),
                 atoms.elec.pseudocharge_ref().data(),
                 scf.Vxc(),
                 atoms.has_nlcc ? atoms.rho_core.data() : nullptr);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    print(rank, ctx.is_soc(), atoms.has_nlcc, atoms.Natom);
}

// ---------------------------------------------------------------------------
// Detailed implementation (private).
// ---------------------------------------------------------------------------
std::vector<double> Forces::compute_impl(
    const LynxContext& ctx,
    const Wavefunction& wfn,
    const Crystal& crystal,
    const std::vector<AtomInfluence>& influence,
    const std::vector<AtomNlocInfluence>& nloc_influence,
    const NonlocalProjector& vnl,
    const double* phi,
    const double* rho,
    const double* Vloc,
    const double* b,
    const double* b_ref,
    const double* Vxc,
    const double* rho_core) {
    ctx_ = &ctx;
    std::vector<double> kpt_weights = ctx.kpoints().normalized_weights();
    const auto& grid = ctx.grid();

    int n_atom = crystal.n_atom_total();
    f_local_.assign(3 * n_atom, 0.0);
    f_nloc_.assign(3 * n_atom, 0.0);
    f_soc_.assign(3 * n_atom, 0.0);
    f_xc_.assign(3 * n_atom, 0.0);
    f_total_.assign(3 * n_atom, 0.0);

    // Local force using pseudocharge-based formula (matches reference)
    compute_local(crystal, influence, phi, Vloc, b, b_ref);

    // Nonlocal force from KB projectors (dispatches to CPU or GPU)
    compute_nonlocal(wfn, crystal, nloc_influence, vnl, kpt_weights);

    // SOC nonlocal force from spin-orbit coupling projectors
    if (vnl.has_soc() && wfn.Nspinor() == 2) {
        compute_nonlocal_soc(ctx, wfn, crystal, nloc_influence, vnl, kpt_weights);
    }

    // Sum local + nonlocal + SOC
    for (int i = 0; i < 3 * n_atom; ++i) {
        f_total_[i] = f_local_[i] + f_nloc_[i] + f_soc_[i];
    }

    // NLCC XC force correction
    if (rho_core != nullptr) {
        compute_xc_nlcc(crystal, influence, Vxc);
        for (int i = 0; i < 3 * n_atom; ++i) {
            f_total_[i] += f_xc_[i];
        }
    }

    // For non-orthogonal cells: transform from non-Cart to Cartesian coordinates
    // F_cart = LatUVec^{-1} * F_nonCart
    if (!grid.lattice().is_orthogonal()) {
        const Mat3& uvec = grid.lattice().lat_uvec();
        Mat3 uvec_inv = uvec.inverse();
        for (int ia = 0; ia < n_atom; ++ia) {
            double fx = f_total_[ia*3+0];
            double fy = f_total_[ia*3+1];
            double fz = f_total_[ia*3+2];
            f_total_[ia*3+0] = uvec_inv(0,0)*fx + uvec_inv(0,1)*fy + uvec_inv(0,2)*fz;
            f_total_[ia*3+1] = uvec_inv(1,0)*fx + uvec_inv(1,1)*fy + uvec_inv(1,2)*fz;
            f_total_[ia*3+2] = uvec_inv(2,0)*fx + uvec_inv(2,1)*fy + uvec_inv(2,2)*fz;
        }
        // Also transform component forces for printing
        for (int ia = 0; ia < n_atom; ++ia) {
            double fx, fy, fz;
            fx = f_local_[ia*3+0]; fy = f_local_[ia*3+1]; fz = f_local_[ia*3+2];
            f_local_[ia*3+0] = uvec_inv(0,0)*fx + uvec_inv(0,1)*fy + uvec_inv(0,2)*fz;
            f_local_[ia*3+1] = uvec_inv(1,0)*fx + uvec_inv(1,1)*fy + uvec_inv(1,2)*fz;
            f_local_[ia*3+2] = uvec_inv(2,0)*fx + uvec_inv(2,1)*fy + uvec_inv(2,2)*fz;
            fx = f_nloc_[ia*3+0]; fy = f_nloc_[ia*3+1]; fz = f_nloc_[ia*3+2];
            f_nloc_[ia*3+0] = uvec_inv(0,0)*fx + uvec_inv(0,1)*fy + uvec_inv(0,2)*fz;
            f_nloc_[ia*3+1] = uvec_inv(1,0)*fx + uvec_inv(1,1)*fy + uvec_inv(1,2)*fz;
            f_nloc_[ia*3+2] = uvec_inv(2,0)*fx + uvec_inv(2,1)*fy + uvec_inv(2,2)*fz;
            fx = f_soc_[ia*3+0]; fy = f_soc_[ia*3+1]; fz = f_soc_[ia*3+2];
            f_soc_[ia*3+0] = uvec_inv(0,0)*fx + uvec_inv(0,1)*fy + uvec_inv(0,2)*fz;
            f_soc_[ia*3+1] = uvec_inv(1,0)*fx + uvec_inv(1,1)*fy + uvec_inv(1,2)*fz;
            f_soc_[ia*3+2] = uvec_inv(2,0)*fx + uvec_inv(2,1)*fy + uvec_inv(2,2)*fz;
            fx = f_xc_[ia*3+0]; fy = f_xc_[ia*3+1]; fz = f_xc_[ia*3+2];
            f_xc_[ia*3+0] = uvec_inv(0,0)*fx + uvec_inv(0,1)*fy + uvec_inv(0,2)*fz;
            f_xc_[ia*3+1] = uvec_inv(1,0)*fx + uvec_inv(1,1)*fy + uvec_inv(1,2)*fz;
            f_xc_[ia*3+2] = uvec_inv(2,0)*fx + uvec_inv(2,1)*fy + uvec_inv(2,2)*fz;
        }
    }

    // Symmetrize: ensure total force sums to zero
    symmetrize(f_total_, n_atom);

    return f_total_;
}

// ---------------------------------------------------------------------------
// Local force matching reference LYNX Calculate_local_forces_linear().
//
// Algorithm:
// 1. Compute ∇φ (gradient of electrostatic potential) on the DM domain
// 2. Compute ∇Vc (gradient of correction potential Vloc) on the DM domain
// 3. For each atom J:
//    a. Build VJ and VJ_ref on an FDn-extended grid around atom
//    b. Compute bJ = -1/(4π)·Lap(VJ) and bJ_ref = -1/(4π)·Lap(VJ_ref)
//    c. Compute VcJ = VJ_ref - VJ (correction potential for this atom)
//    d. Compute ∇VcJ using FD stencil
//    e. Accumulate:
//       F_J = -∫ bJ · ∇φ dV + 0.5 * ∫ [∇VcJ·(b+b_ref) - ∇Vc·(bJ+bJ_ref)] dV
// ---------------------------------------------------------------------------
void Forces::compute_local(
    const Crystal& crystal,
    const std::vector<AtomInfluence>& influence,
    const double* phi,
    const double* Vloc,
    const double* b_total,
    const double* b_ref_total) {

    const auto& stencil = ctx_->stencil();
    const auto& gradient = ctx_->gradient();
    const auto& halo = ctx_->halo();
    const auto& domain = ctx_->domain();
    const auto& grid = ctx_->grid();

    int n_atom = crystal.n_atom_total();
    f_local_.assign(3 * n_atom, 0.0);

    int FDn = stencil.FDn();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    double dV = grid.dV();
    int nx = domain.Nx_d(), ny = domain.Ny_d(), nz = domain.Nz_d();
    int DMnd = domain.Nd_d();
    int xs = domain.vertices().xs, ys = domain.vertices().ys, zs = domain.vertices().zs;
    bool is_orth = grid.lattice().is_orthogonal();
    const Lattice* lattice = &grid.lattice();

    double inv_4PI = 0.25 / constants::PI;
    double w2_diag = (stencil.D2_coeff_x()[0] + stencil.D2_coeff_y()[0] + stencil.D2_coeff_z()[0]) * (-inv_4PI);

    const double* D1_x = stencil.D1_coeff_x();
    const double* D1_y = stencil.D1_coeff_y();
    const double* D1_z = stencil.D1_coeff_z();
    const double* D2_x = stencil.D2_coeff_x();
    const double* D2_y = stencil.D2_coeff_y();
    const double* D2_z = stencil.D2_coeff_z();

    // Compute ∇φ on the DM domain
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nz_ex = nz + 2 * FDn;

    std::vector<double> phi_ex(nx_ex * ny_ex * nz_ex);
    halo.execute(phi, phi_ex.data(), 1);
    std::vector<double> Dphi_x(DMnd), Dphi_y(DMnd), Dphi_z(DMnd);
    gradient.apply(phi_ex.data(), Dphi_x.data(), 0);
    gradient.apply(phi_ex.data(), Dphi_y.data(), 1);
    gradient.apply(phi_ex.data(), Dphi_z.data(), 2);

    // Compute ∇Vc on the DM domain
    std::vector<double> Vc_ex(nx_ex * ny_ex * nz_ex);
    halo.execute(Vloc, Vc_ex.data(), 1);
    std::vector<double> DVc_x(DMnd), DVc_y(DMnd), DVc_z(DMnd);
    gradient.apply(Vc_ex.data(), DVc_x.data(), 0);
    gradient.apply(Vc_ex.data(), DVc_y.data(), 1);
    gradient.apply(Vc_ex.data(), DVc_z.data(), 2);

    // Loop over atom types and atoms
    int ntypes = crystal.n_types();
    for (int it = 0; it < ntypes; ++it) {
        const auto& psd = crystal.types()[it].psd();
        const auto& inf = influence[it];
        const auto& r_grid = psd.radial_grid();
        const auto& rVloc = psd.rVloc();
        const auto& rVloc_d = psd.rVloc_spline_d();
        double Znucl = psd.Zval();
        double rchrg = r_grid.back();
        double rc_ref = 0.5;  // REFERENCE_CUTOFF

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            Vec3 pos = inf.coords[iat];
            int atom_index = inf.atom_index[iat];

            // Overlap region
            int i_s = inf.xs[iat], i_e = inf.xe[iat];
            int j_s = inf.ys[iat], j_e = inf.ye[iat];
            int k_s = inf.zs[iat], k_e = inf.ze[iat];
            int lnx = i_e - i_s + 1;
            int lny = j_e - j_s + 1;
            int lnz = k_e - k_s + 1;

            // Extended grid for FD stencil
            int nxp = lnx + 2 * FDn;
            int nyp = lny + 2 * FDn;
            int nzp = lnz + 2 * FDn;
            int nd_ex = nxp * nyp * nzp;

            // Compute distances and interpolate VJ, VJ_ref on extended grid
            std::vector<double> VJ(nd_ex), VJ_ref_arr(nd_ex);

            int icor = i_s - FDn;
            int jcor = j_s - FDn;
            int kcor = k_s - FDn;
            double x0_shift = pos.x - dx * icor;
            double y0_shift = pos.y - dy * jcor;
            double z0_shift = pos.z - dz * kcor;

            // Collect radii and interpolation points
            std::vector<double> R(nd_ex);
            std::vector<int> ind_interp;
            std::vector<double> R_interp;

            int count = 0;
            for (int kk = 0; kk < nzp; ++kk) {
                double rz = kk * dz - z0_shift;
                for (int jj = 0; jj < nyp; ++jj) {
                    double ry = jj * dy - y0_shift;
                    for (int ii = 0; ii < nxp; ++ii) {
                        double rx = ii * dx - x0_shift;
                        double r = is_orth ? std::sqrt(rx*rx + ry*ry + rz*rz)
                                           : lattice->metric_distance(rx, ry, rz);
                        R[count] = r;
                        if (r <= rchrg) {
                            ind_interp.push_back(count);
                            R_interp.push_back(r);
                        } else {
                            VJ[count] = -Znucl / r;
                        }
                        count++;
                    }
                }
            }

            // Spline interpolation for points within rchrg
            if (!R_interp.empty()) {
                std::vector<double> VJ_interp;
                Pseudopotential::spline_interp(r_grid, rVloc, rVloc_d, R_interp, VJ_interp);
                for (size_t idx = 0; idx < ind_interp.size(); ++idx) {
                    int i_idx = ind_interp[idx];
                    if (R_interp[idx] < 1e-10) {
                        VJ[i_idx] = psd.Vloc_0();
                    } else {
                        VJ[i_idx] = VJ_interp[idx] / R_interp[idx];
                    }
                }
            }

            // Reference potential
            for (int i = 0; i < nd_ex; ++i) {
                VJ_ref_arr[i] = Electrostatics::V_ref(R[i], rc_ref, Znucl);
            }

            // Correction potential VcJ = VJ_ref - VJ
            std::vector<double> VcJ(nd_ex);
            for (int i = 0; i < nd_ex; ++i) {
                VcJ[i] = VJ_ref_arr[i] - VJ[i];
            }

            // Compute bJ and bJ_ref via Laplacian
            int nd_inner = lnx * lny * lnz;
            std::vector<double> bJ(nd_inner, 0.0);
            std::vector<double> bJ_ref(nd_inner, 0.0);

            if (is_orth) {
                Electrostatics::calc_lapV(VJ.data(), bJ.data(), lnx, lny, lnz,
                                          nxp, nyp, nzp, FDn, D2_x, D2_y, D2_z, -inv_4PI);
                Electrostatics::calc_lapV(VJ_ref_arr.data(), bJ_ref.data(), lnx, lny, lnz,
                                          nxp, nyp, nzp, FDn, D2_x, D2_y, D2_z, -inv_4PI);
            } else {
                Electrostatics::calc_lapV_nonorth(VJ.data(), bJ.data(), lnx, lny, lnz,
                                                   nxp, nyp, nzp, FDn, stencil, -inv_4PI);
                Electrostatics::calc_lapV_nonorth(VJ_ref_arr.data(), bJ_ref.data(), lnx, lny, lnz,
                                                   nxp, nyp, nzp, FDn, stencil, -inv_4PI);
            }

            // Shift vectors for stencil indexing
            // pshifty_ex[p] = p * nxp, pshiftz_ex[p] = p * nxp * nyp
            // We compute ∇VcJ using the FD stencil on the extended VcJ grid

            // DM domain offsets
            int dI = i_s - xs;
            int dJ = j_s - ys;
            int dK = k_s - zs;

            double force_x = 0.0, force_y = 0.0, force_z = 0.0;
            double corr_x = 0.0, corr_y = 0.0, corr_z = 0.0;

            for (int kk = 0; kk < lnz; ++kk) {
                int kp = kk + FDn;
                int k_DM = kk + dK;
                int kshift_DM = k_DM * nx * ny;
                int kshift_p = kp * nxp * nyp;
                int kshift = kk * lnx * lny;

                for (int jj = 0; jj < lny; ++jj) {
                    int jp = jj + FDn;
                    int j_DM = jj + dJ;
                    int jshift_DM = kshift_DM + j_DM * nx;
                    int jshift_p = kshift_p + jp * nxp;
                    int jshift = kshift + jj * lnx;

                    for (int ii = 0; ii < lnx; ++ii) {
                        int ip = ii + FDn;
                        int i_DM = ii + dI;
                        int ishift_DM = jshift_DM + i_DM;
                        int ishift_p = jshift_p + ip;
                        int ishift = jshift + ii;

                        // Gradient of VcJ at this point
                        double DVcJ_x = 0.0, DVcJ_y = 0.0, DVcJ_z = 0.0;
                        for (int p = 1; p <= FDn; ++p) {
                            DVcJ_x += (VcJ[ishift_p + p] - VcJ[ishift_p - p]) * D1_x[p];
                            DVcJ_y += (VcJ[ishift_p + p*nxp] - VcJ[ishift_p - p*nxp]) * D1_y[p];
                            DVcJ_z += (VcJ[ishift_p + p*nxp*nyp] - VcJ[ishift_p - p*nxp*nyp]) * D1_z[p];
                        }

                        double b_plus_bref = b_total[ishift_DM] + b_ref_total[ishift_DM];
                        double bJ_plus_bJref = bJ[ishift] + bJ_ref[ishift];

                        // Force = -bJ * ∇φ
                        force_x -= bJ[ishift] * Dphi_x[ishift_DM];
                        force_y -= bJ[ishift] * Dphi_y[ishift_DM];
                        force_z -= bJ[ishift] * Dphi_z[ishift_DM];

                        // Correction: ∇VcJ*(b+b_ref) - ∇Vc*(bJ+bJ_ref)
                        corr_x += DVcJ_x * b_plus_bref - DVc_x[ishift_DM] * bJ_plus_bJref;
                        corr_y += DVcJ_y * b_plus_bref - DVc_y[ishift_DM] * bJ_plus_bJref;
                        corr_z += DVcJ_z * b_plus_bref - DVc_z[ishift_DM] * bJ_plus_bJref;
                    }
                }
            }

            f_local_[atom_index * 3 + 0] += (force_x + 0.5 * corr_x) * dV;
            f_local_[atom_index * 3 + 1] += (force_y + 0.5 * corr_y) * dV;
            f_local_[atom_index * 3 + 2] += (force_z + 0.5 * corr_z) * dV;
        }
    }

}

// ---------------------------------------------------------------------------
// Nonlocal force matching reference LYNX Calculate_nonlocal_forces_linear().
//
// Algorithm:
// 1. For all bands at once: compute alpha = <χ|ψ>·dV
// Dispatcher: routes to CPU or GPU path based on dev_.
// ---------------------------------------------------------------------------
void Forces::compute_nonlocal(
    const Wavefunction& wfn,
    const Crystal& crystal,
    const std::vector<AtomNlocInfluence>& nloc_influence,
    const NonlocalProjector& vnl,
    const std::vector<double>& kpt_weights) {
#ifdef USE_CUDA
    if (dev_ == Device::GPU) {
        compute_nonlocal_gpu(wfn, crystal, nloc_influence, vnl, kpt_weights);
        return;
    }
#endif
    compute_nonlocal_cpu(wfn, crystal, nloc_influence, vnl, kpt_weights);
}

// ---------------------------------------------------------------------------
// CPU nonlocal force implementation.
// Algorithm:
// 1. For each (spin, kpt, band): alpha = dV · <χ|ψ>
// 2. For each direction dim: compute gradient ∇_dim ψ, then beta_dim = <χ|∇ψ>
// 3. Allreduce alpha and beta across domain
// 4. Assemble: F_J = -occfac·2·Σ_n g_n · Σ_lmp Γ · α·β
// 5. Allreduce across band, kpt, spin comms
// ---------------------------------------------------------------------------
void Forces::compute_nonlocal_cpu(
    const Wavefunction& wfn,
    const Crystal& crystal,
    const std::vector<AtomNlocInfluence>& nloc_influence,
    const NonlocalProjector& vnl,
    const std::vector<double>& kpt_weights) {

    const auto& gradient = ctx_->gradient();
    const auto& halo = ctx_->halo();
    const auto& domain = ctx_->domain();
    const auto& grid = ctx_->grid();
    const auto& bandcomm = ctx_->scf_bandcomm();
    const auto& kptcomm = ctx_->kpt_bridge();
    const auto& spincomm = ctx_->spin_bridge();
    const KPoints* kpoints = &ctx_->kpoints();
    int kpt_start = ctx_->kpt_start();
    int band_start = ctx_->band_start();

    int n_atom = crystal.n_atom_total();
    int Nspin_local = wfn.Nspin();
    int Nkpts = wfn.Nkpts();
    int Nband = wfn.Nband();  // local band count
    int Nd_d = domain.Nd_d();
    double dV = grid.dV();
    int ntypes = crystal.n_types();
    int FDn = gradient.stencil().FDn();
    bool is_kpt = wfn.is_complex();

    // Determine global Nspin from spincomm (spin_bridge)
    int Nspin = Nspin_local;
    if (!spincomm.is_null() && spincomm.size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &Nspin, 1, MPI_INT, MPI_SUM, spincomm.comm());
    }
    int Nspinor = wfn.Nspinor();
    // For SOC spinor: both spin components in one wfn, so occfac = 1.0
    double occfac = (Nspinor == 2) ? 1.0 : ((Nspin == 1) ? 2.0 : 1.0);
    double spn_fac = occfac * 2.0;
    f_nloc_.assign(3 * n_atom, 0.0);

    // Extended grid dimensions for gradient
    int nx = domain.Nx_d(), ny = domain.Ny_d(), nz = domain.Nz_d();
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nz_ex = nz + 2 * FDn;
    int Nd_ex = nx_ex * ny_ex * nz_ex;

    // Cell lengths for complex halo exchange
    Vec3 cell_lengths = grid.lattice().lengths();

    // Build per-unique-atom projector info and Gamma list
    std::vector<int> IP_displ(n_atom + 1, 0);
    for (int ia = 0; ia < n_atom; ++ia) {
        int it = crystal.type_indices()[ia];
        IP_displ[ia + 1] = IP_displ[ia] + crystal.types()[it].psd().nproj_per_atom();
    }
    int total_nproj = IP_displ[n_atom];

    // Build flat Gamma array indexed by unique atom projector offset
    std::vector<double> Gamma_flat(total_nproj, 0.0);
    for (int ia = 0; ia < n_atom; ++ia) {
        int it = crystal.type_indices()[ia];
        const auto& psd = crystal.types()[it].psd();
        int offset = IP_displ[ia];
        int col = 0;
        for (int l = 0; l <= psd.lmax(); ++l) {
            if (l == psd.lloc()) continue;
            for (int p = 0; p < psd.ppl()[l]; ++p) {
                double g = psd.Gamma()[l][p];
                for (int m = -l; m <= l; ++m) {
                    Gamma_flat[offset + col] = g;
                    col++;
                }
            }
        }
    }

    // Access precomputed Chi from NonlocalProjector
    const auto& Chi = vnl.Chi();

    for (int s = 0; s < Nspin_local; ++s) {
        for (int k = 0; k < Nkpts; ++k) {
            int k_glob = kpt_start + k;
            double wk = kpt_weights[k_glob];
            const DeviceArray<double>& occ_sk = wfn.occupations(s, k);

            if (is_kpt) {
                // ===== Complex k-point path =====
                const DeviceArray<Complex>& psi_sk = wfn.psi_kpt(s, k);
                Vec3 kpt_cart = kpoints->kpts_cart()[k_glob];

                for (int n = 0; n < Nband; ++n) {
                    double g_n = occ_sk(band_start + n);
                    if (std::abs(g_n) < 1e-15) continue;

                    const Complex* psi_n = psi_sk.col(n);

                    // For SOC spinor: psi_n = [psi_up(Nd_d) | psi_dn(Nd_d)].
                    // Scalar-relativistic KB projectors apply to each spinor
                    // component separately (same Chi, same Gamma).
                    int n_spinor_comp = (Nspinor == 2) ? 2 : 1;

                    // Compute complex alpha per spinor component
                    std::vector<std::vector<Complex>> alpha_sp(n_spinor_comp,
                        std::vector<Complex>(total_nproj, Complex(0.0, 0.0)));

                    for (int sp = 0; sp < n_spinor_comp; ++sp) {
                        const Complex* psi_sp = psi_n + sp * Nd_d;
                        for (int it = 0; it < ntypes; ++it) {
                            const auto& inf = nloc_influence[it];
                            int nproj = crystal.types()[it].psd().nproj_per_atom();
                            if (nproj == 0) continue;

                            for (int iat = 0; iat < inf.n_atom; ++iat) {
                                int ndc = inf.ndc[iat];
                                if (ndc == 0) continue;
                                int orig_atom = inf.atom_index[iat];
                                int offset = IP_displ[orig_atom];
                                const auto& gpos = inf.grid_pos[iat];
                                const auto& chi_iat = Chi[it][iat];

                                const Vec3& shift = inf.image_shift[iat];
                                double theta = -(kpt_cart.x * shift.x + kpt_cart.y * shift.y + kpt_cart.z * shift.z);
                                Complex bloch_fac(std::cos(theta), std::sin(theta));
                                Complex alpha_scale = bloch_fac * dV;

                                for (int jp = 0; jp < nproj; ++jp) {
                                    Complex dot(0.0, 0.0);
                                    for (int ig = 0; ig < ndc; ++ig)
                                        dot += chi_iat(ig, jp) * psi_sp[gpos[ig]];
                                    alpha_sp[sp][offset + jp] += alpha_scale * dot;
                                }
                            }
                        }
                    }

                    // For each direction, compute complex beta and accumulate force
                    for (int dim = 0; dim < 3; ++dim) {
                        std::vector<std::vector<Complex>> beta_sp(n_spinor_comp,
                            std::vector<Complex>(total_nproj, Complex(0.0, 0.0)));

                        for (int sp = 0; sp < n_spinor_comp; ++sp) {
                            const Complex* psi_sp = psi_n + sp * Nd_d;
                            std::vector<Complex> psi_ex(Nd_ex, Complex(0.0, 0.0));
                            halo.execute_kpt(psi_sp, psi_ex.data(), 1, kpt_cart, cell_lengths);
                            std::vector<Complex> Dpsi(Nd_d, Complex(0.0, 0.0));
                            gradient.apply(psi_ex.data(), Dpsi.data(), dim);

                            for (int it = 0; it < ntypes; ++it) {
                                const auto& inf = nloc_influence[it];
                                int nproj = crystal.types()[it].psd().nproj_per_atom();
                                if (nproj == 0) continue;

                                for (int iat = 0; iat < inf.n_atom; ++iat) {
                                    int ndc = inf.ndc[iat];
                                    if (ndc == 0) continue;
                                    int orig_atom = inf.atom_index[iat];
                                    int offset = IP_displ[orig_atom];
                                    const auto& gpos = inf.grid_pos[iat];
                                    const auto& chi_iat = Chi[it][iat];

                                    const Vec3& shift = inf.image_shift[iat];
                                    double theta = -(kpt_cart.x * shift.x + kpt_cart.y * shift.y + kpt_cart.z * shift.z);
                                    Complex bloch_fac(std::cos(theta), std::sin(theta));
                                    Complex beta_scale = bloch_fac * dV;

                                    for (int jp = 0; jp < nproj; ++jp) {
                                        Complex dot(0.0, 0.0);
                                        for (int ig = 0; ig < ndc; ++ig)
                                            dot += chi_iat(ig, jp) * Dpsi[gpos[ig]];
                                        beta_sp[sp][offset + jp] += beta_scale * dot;
                                    }
                                }
                            }
                        }

                        // Force: -spn_fac * wk * g_n * Re(Σ Gamma * conj(alpha) * beta)
                        // Sum over spinor components
                        for (int ia = 0; ia < n_atom; ++ia) {
                            int offset = IP_displ[ia];
                            int nproj = IP_displ[ia + 1] - offset;
                            double fJ = 0.0;
                            for (int sp = 0; sp < n_spinor_comp; ++sp) {
                                for (int jp = 0; jp < nproj; ++jp) {
                                    fJ += Gamma_flat[offset + jp] *
                                          std::real(std::conj(alpha_sp[sp][offset + jp]) * beta_sp[sp][offset + jp]);
                                }
                            }
                            f_nloc_[ia * 3 + dim] -= spn_fac * wk * g_n * fJ;
                        }
                    }
                }
            } else {
                // ===== Real gamma-point path =====
                const DeviceArray<double>& psi_sk = wfn.psi(s, k);

                for (int n = 0; n < Nband; ++n) {
                    double g_n = occ_sk(band_start + n);
                    if (std::abs(g_n) < 1e-15) continue;

                    const double* psi_n = psi_sk.col(n);

                    // Compute alpha = <χ|ψ>*dV, accumulated per unique atom
                    std::vector<double> alpha(total_nproj, 0.0);
                    for (int it = 0; it < ntypes; ++it) {
                        const auto& inf = nloc_influence[it];
                        int nproj = crystal.types()[it].psd().nproj_per_atom();
                        if (nproj == 0) continue;

                        for (int iat = 0; iat < inf.n_atom; ++iat) {
                            int ndc = inf.ndc[iat];
                            if (ndc == 0) continue;
                            int orig_atom = inf.atom_index[iat];
                            int offset = IP_displ[orig_atom];
                            const auto& gpos = inf.grid_pos[iat];
                            const auto& chi_iat = Chi[it][iat];

                            for (int jp = 0; jp < nproj; ++jp) {
                                double dot = 0.0;
                                for (int ig = 0; ig < ndc; ++ig) {
                                    dot += chi_iat(ig, jp) * psi_n[gpos[ig]];
                                }
                                alpha[offset + jp] += dot * dV;
                            }
                        }
                    }

                    // For each direction, compute beta = <χ|∇ψ> accumulated per unique atom
                    for (int dim = 0; dim < 3; ++dim) {
                        std::vector<double> psi_ex(Nd_ex, 0.0);
                        halo.execute(psi_n, psi_ex.data(), 1);
                        std::vector<double> Dpsi(Nd_d, 0.0);
                        gradient.apply(psi_ex.data(), Dpsi.data(), dim);

                        std::vector<double> beta(total_nproj, 0.0);
                        for (int it = 0; it < ntypes; ++it) {
                            const auto& inf = nloc_influence[it];
                            int nproj = crystal.types()[it].psd().nproj_per_atom();
                            if (nproj == 0) continue;

                            for (int iat = 0; iat < inf.n_atom; ++iat) {
                                int ndc = inf.ndc[iat];
                                if (ndc == 0) continue;
                                int orig_atom = inf.atom_index[iat];
                                int offset = IP_displ[orig_atom];
                                const auto& gpos = inf.grid_pos[iat];
                                const auto& chi_iat = Chi[it][iat];

                                for (int jp = 0; jp < nproj; ++jp) {
                                    double dot = 0.0;
                                    for (int ig = 0; ig < ndc; ++ig) {
                                        dot += chi_iat(ig, jp) * Dpsi[gpos[ig]];
                                    }
                                    beta[offset + jp] += dot * dV;
                                }
                            }
                        }

                        // Accumulate force per unique atom
                        for (int ia = 0; ia < n_atom; ++ia) {
                            int offset = IP_displ[ia];
                            int nproj = IP_displ[ia + 1] - offset;
                            double fJ = 0.0;
                            for (int jp = 0; jp < nproj; ++jp) {
                                fJ += Gamma_flat[offset + jp] * alpha[offset + jp] * beta[offset + jp];
                            }
                            f_nloc_[ia * 3 + dim] -= spn_fac * wk * g_n * fJ;
                        }
                    }
                }
            }
        }
    }

    // MPI reductions across band, kpt, spin
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
// SOC nonlocal force from spin-orbit coupling projectors.
//
// Mirrors compute_nonlocal's k-point path but uses Chi_soc, Gamma_soc,
// and the SOC coupling terms (Term 1: on-diagonal Lz·Sz, Term 2: ladder
// operators L+S-/L-S+) matching the apply_soc_kpt structure.
// ---------------------------------------------------------------------------
void Forces::compute_nonlocal_soc(
    const LynxContext& ctx,
    const Wavefunction& wfn,
    const Crystal& crystal,
    const std::vector<AtomNlocInfluence>& nloc_influence,
    const NonlocalProjector& vnl,
    const std::vector<double>& kpt_weights) {
    compute_nonlocal_soc(wfn, crystal, nloc_influence, vnl,
                         ctx.gradient(), ctx.halo(), ctx.domain(), ctx.grid(),
                         kpt_weights, ctx.scf_bandcomm(), ctx.kpt_bridge(),
                         &ctx.kpoints(), ctx.kpt_start(), ctx.band_start());
}

void Forces::compute_nonlocal_soc(
    const Wavefunction& wfn,
    const Crystal& crystal,
    const std::vector<AtomNlocInfluence>& nloc_influence,
    const NonlocalProjector& vnl,
    const Gradient& gradient,
    const HaloExchange& halo,
    const Domain& domain,
    const FDGrid& grid,
    const std::vector<double>& kpt_weights,
    const MPIComm& bandcomm,
    const MPIComm& kptcomm,
    const KPoints* kpoints,
    int kpt_start,
    int band_start) {

    int n_atom = crystal.n_atom_total();
    int Nkpts = wfn.Nkpts();
    int Nband = wfn.Nband();
    int Nd_d = domain.Nd_d();
    double dV = grid.dV();
    int ntypes = crystal.n_types();
    int FDn = gradient.stencil().FDn();

    // SOC spinor: both spin components present, so occfac = 1.0; spn_fac = occfac * 2.0 = 2.0
    double spn_fac = 2.0;

    f_soc_.assign(3 * n_atom, 0.0);

    // Extended grid dimensions for gradient
    int nx = domain.Nx_d(), ny = domain.Ny_d(), nz = domain.Nz_d();
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nz_ex = nz + 2 * FDn;
    int Nd_ex = nx_ex * ny_ex * nz_ex;

    Vec3 cell_lengths = grid.lattice().lengths();

    // Build per-unique-atom SOC projector displacement
    std::vector<int> IP_displ(n_atom + 1, 0);
    {
        int atom_idx = 0;
        for (int it = 0; it < ntypes; ++it) {
            const auto& psd = crystal.types()[it].psd();
            int nproj_soc = 0;
            if (psd.has_soc()) {
                for (int l = 1; l <= psd.lmax(); ++l)
                    nproj_soc += psd.ppl_soc()[l] * (2 * l + 1);
            }
            int nat = crystal.types()[it].n_atoms();
            for (int ia = 0; ia < nat; ++ia) {
                IP_displ[atom_idx + 1] = IP_displ[atom_idx] + nproj_soc;
                atom_idx++;
            }
        }
    }
    int total_soc_nproj = IP_displ[n_atom];
    if (total_soc_nproj == 0) return;

    const auto& Chi_soc = vnl.Chi_soc();
    const auto& soc_proj_info = vnl.soc_proj_info();

    // SOC is always k-point (spinor wavefunctions are complex)
    for (int k = 0; k < Nkpts; ++k) {
        int k_glob = kpt_start + k;
        double wk = kpt_weights[k_glob];
        // SOC: single spin channel (s=0), both components in spinor
        const DeviceArray<double>& occ_k = wfn.occupations(0, k);
        const DeviceArray<Complex>& psi_sk = wfn.psi_kpt(0, k);
        Vec3 kpt_cart = kpoints->kpts_cart()[k_glob];
        int Nd_d_spinor = 2 * Nd_d;

        for (int n = 0; n < Nband; ++n) {
            double g_n = occ_k(band_start + n);
            if (std::abs(g_n) < 1e-15) continue;

            const Complex* psi_n = psi_sk.col(n);
            const Complex* psi_up = psi_n;
            const Complex* psi_dn = psi_n + Nd_d;

            // Compute alpha_up/dn = bloch_fac * dV * Chi_soc^T * psi_up/dn
            std::vector<Complex> alpha_up(total_soc_nproj, Complex(0.0));
            std::vector<Complex> alpha_dn(total_soc_nproj, Complex(0.0));

            for (int it = 0; it < ntypes; ++it) {
                const auto& psd = crystal.types()[it].psd();
                if (!psd.has_soc()) continue;
                const auto& inf = nloc_influence[it];
                int nproj_soc = 0;
                for (int l = 1; l <= psd.lmax(); ++l)
                    nproj_soc += psd.ppl_soc()[l] * (2 * l + 1);
                if (nproj_soc == 0) continue;

                for (int iat = 0; iat < inf.n_atom; ++iat) {
                    int ndc = inf.ndc[iat];
                    if (ndc == 0) continue;
                    int orig_atom = inf.atom_index[iat];
                    int offset = IP_displ[orig_atom];
                    const auto& gpos = inf.grid_pos[iat];
                    const auto& chi_iat = Chi_soc[it][iat];

                    const Vec3& shift = inf.image_shift[iat];
                    double theta = -(kpt_cart.x * shift.x + kpt_cart.y * shift.y + kpt_cart.z * shift.z);
                    Complex bloch_fac(std::cos(theta), std::sin(theta));
                    Complex alpha_scale = bloch_fac * dV;

                    for (int jp = 0; jp < nproj_soc; ++jp) {
                        Complex dot_up(0.0), dot_dn(0.0);
                        for (int ig = 0; ig < ndc; ++ig) {
                            Complex chi_val = std::conj(chi_iat(ig, jp));
                            dot_up += chi_val * psi_up[gpos[ig]];
                            dot_dn += chi_val * psi_dn[gpos[ig]];
                        }
                        alpha_up[offset + jp] += alpha_scale * dot_up;
                        alpha_dn[offset + jp] += alpha_scale * dot_dn;
                    }
                }
            }

            // For each direction, compute beta_up/dn and accumulate SOC force
            for (int dim = 0; dim < 3; ++dim) {
                // Gradient of psi_up
                std::vector<Complex> psi_up_ex(Nd_ex, Complex(0.0));
                halo.execute_kpt(psi_up, psi_up_ex.data(), 1, kpt_cart, cell_lengths);
                std::vector<Complex> Dpsi_up(Nd_d, Complex(0.0));
                gradient.apply(psi_up_ex.data(), Dpsi_up.data(), dim);

                // Gradient of psi_dn
                std::vector<Complex> psi_dn_ex(Nd_ex, Complex(0.0));
                halo.execute_kpt(psi_dn, psi_dn_ex.data(), 1, kpt_cart, cell_lengths);
                std::vector<Complex> Dpsi_dn(Nd_d, Complex(0.0));
                gradient.apply(psi_dn_ex.data(), Dpsi_dn.data(), dim);

                // Compute beta_up/dn = bloch_fac * dV * Chi_soc^T * Dpsi_up/dn
                std::vector<Complex> beta_up(total_soc_nproj, Complex(0.0));
                std::vector<Complex> beta_dn(total_soc_nproj, Complex(0.0));

                for (int it = 0; it < ntypes; ++it) {
                    const auto& psd = crystal.types()[it].psd();
                    if (!psd.has_soc()) continue;
                    const auto& inf = nloc_influence[it];
                    int nproj_soc = 0;
                    for (int l = 1; l <= psd.lmax(); ++l)
                        nproj_soc += psd.ppl_soc()[l] * (2 * l + 1);
                    if (nproj_soc == 0) continue;

                    for (int iat = 0; iat < inf.n_atom; ++iat) {
                        int ndc = inf.ndc[iat];
                        if (ndc == 0) continue;
                        int orig_atom = inf.atom_index[iat];
                        int offset = IP_displ[orig_atom];
                        const auto& gpos = inf.grid_pos[iat];
                        const auto& chi_iat = Chi_soc[it][iat];

                        const Vec3& shift = inf.image_shift[iat];
                        double theta = -(kpt_cart.x * shift.x + kpt_cart.y * shift.y + kpt_cart.z * shift.z);
                        Complex bloch_fac(std::cos(theta), std::sin(theta));
                        Complex beta_scale = bloch_fac * dV;

                        for (int jp = 0; jp < nproj_soc; ++jp) {
                            Complex dot_up(0.0), dot_dn(0.0);
                            for (int ig = 0; ig < ndc; ++ig) {
                                Complex chi_val = std::conj(chi_iat(ig, jp));
                                dot_up += chi_val * Dpsi_up[gpos[ig]];
                                dot_dn += chi_val * Dpsi_dn[gpos[ig]];
                            }
                            beta_up[offset + jp] += beta_scale * dot_up;
                            beta_dn[offset + jp] += beta_scale * dot_dn;
                        }
                    }
                }

                // Accumulate SOC force per unique atom using Term 1 and Term 2
                for (int it = 0; it < ntypes; ++it) {
                    const auto& psd = crystal.types()[it].psd();
                    if (!psd.has_soc()) continue;
                    const auto& proj_info = soc_proj_info[it];
                    int nproj_soc = static_cast<int>(proj_info.size());
                    if (nproj_soc == 0) continue;

                    int nat = crystal.types()[it].n_atoms();
                    // Get the first global atom index for this type
                    int first_atom = 0;
                    for (int jt = 0; jt < it; ++jt)
                        first_atom += crystal.types()[jt].n_atoms();

                    for (int ia_local = 0; ia_local < nat; ++ia_local) {
                        int ia = first_atom + ia_local;
                        int offset = IP_displ[ia];
                        double fJ = 0.0;

                        for (int jp = 0; jp < nproj_soc; ++jp) {
                            int l = proj_info[jp].l;
                            int m = proj_info[jp].m;
                            int p = proj_info[jp].p;
                            double gamma_soc = psd.Gamma_soc()[l][p];

                            // Term 1: on-diagonal (Lz·Sz)
                            // F -= 0.5 * m * gamma_soc * [Re(conj(alpha_up)*beta_up) - Re(conj(alpha_dn)*beta_dn)]
                            if (m != 0) {
                                fJ += 0.5 * static_cast<double>(m) * gamma_soc *
                                    (std::real(std::conj(alpha_up[offset + jp]) * beta_up[offset + jp]) -
                                     std::real(std::conj(alpha_dn[offset + jp]) * beta_dn[offset + jp]));
                            }

                            // Term 2: L+S- (m -> m+1)
                            if (m + 1 <= l) {
                                double ladder = std::sqrt(static_cast<double>(l*(l+1) - m*(m+1)));
                                int jp_shifted = jp + 1;  // column for (l, m+1, p)
                                fJ += 0.5 * ladder * gamma_soc *
                                    std::real(std::conj(alpha_dn[offset + jp_shifted]) * beta_up[offset + jp] +
                                              std::conj(alpha_up[offset + jp]) * beta_dn[offset + jp_shifted]);
                            }

                            // Term 2: L-S+ (m -> m-1)
                            if (m - 1 >= -l) {
                                double ladder = std::sqrt(static_cast<double>(l*(l+1) - m*(m-1)));
                                int jp_shifted = jp - 1;  // column for (l, m-1, p)
                                fJ += 0.5 * ladder * gamma_soc *
                                    std::real(std::conj(alpha_up[offset + jp_shifted]) * beta_dn[offset + jp] +
                                              std::conj(alpha_dn[offset + jp]) * beta_up[offset + jp_shifted]);
                            }
                        }

                        f_soc_[ia * 3 + dim] -= spn_fac * wk * g_n * fJ;
                    }
                }
            }
        }
    }

    // MPI reductions across band and kpt comms
    if (!bandcomm.is_null() && bandcomm.size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, f_soc_.data(), 3 * n_atom,
                      MPI_DOUBLE, MPI_SUM, bandcomm.comm());
    }
    if (!kptcomm.is_null() && kptcomm.size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, f_soc_.data(), 3 * n_atom,
                      MPI_DOUBLE, MPI_SUM, kptcomm.comm());
    }
    // No spincomm reduction needed for SOC (single spin channel with spinor)
}

// ---------------------------------------------------------------------------
// NLCC XC force matching reference LYNX Calculate_forces_xc_linear().
//
// For each atom J with NLCC core charge:
//   1. Interpolate ρ_core_J on double-extended grid (rb + 2*FDn)
//   2. Compute ∇ρ_core_J on extended grid (rb + FDn) using FD stencil
//   3. Integrate: F_xc_J = ∫ Vxc(r) · ∇ρ_core_J(r) dV
// ---------------------------------------------------------------------------
void Forces::compute_xc_nlcc(
    const Crystal& crystal,
    const std::vector<AtomInfluence>& influence,
    const double* Vxc) {

    const auto& stencil = ctx_->stencil();
    const auto& domain = ctx_->domain();
    const auto& grid = ctx_->grid();

    int n_atom = crystal.n_atom_total();
    f_xc_.assign(3 * n_atom, 0.0);

    int FDn = stencil.FDn();
    double dx = grid.dx(), dy = grid.dy(), dz = grid.dz();
    double dV = grid.dV();
    int DMnx = domain.Nx_d(), DMny = domain.Ny_d();
    int xs = domain.vertices().xs, ys = domain.vertices().ys, zs = domain.vertices().zs;
    bool is_orth_xc = grid.lattice().is_orthogonal();
    const Lattice* lattice_xc = &grid.lattice();

    const double* D1_x = stencil.D1_coeff_x();
    const double* D1_y = stencil.D1_coeff_y();
    const double* D1_z = stencil.D1_coeff_z();

    int ntypes = crystal.n_types();
    for (int it = 0; it < ntypes; ++it) {
        const auto& psd = crystal.types()[it].psd();
        if (!psd.has_nlcc()) continue;

        const auto& inf = influence[it];
        const auto& r_grid = psd.radial_grid();
        double rchrg = r_grid.back();

        for (int iat = 0; iat < inf.n_atom; ++iat) {
            Vec3 pos = inf.coords[iat];
            int atom_index = inf.atom_index[iat];

            // Overlap region
            int i_s = inf.xs[iat], i_e = inf.xe[iat];
            int j_s = inf.ys[iat], j_e = inf.ye[iat];
            int k_s = inf.zs[iat], k_e = inf.ze[iat];
            int nx = i_e - i_s + 1;
            int ny = j_e - j_s + 1;
            int nz = k_e - k_s + 1;

            // Extended grid (rb + FDn) for gradient output
            int nxp = nx + 2 * FDn;
            int nyp = ny + 2 * FDn;
            int nzp = nz + 2 * FDn;
            int nd_ex = nxp * nyp * nzp;

            // Double-extended grid (rb + 2*FDn) for interpolation
            int nx2p = nxp + 2 * FDn;
            int ny2p = nyp + 2 * FDn;
            int nz2p = nzp + 2 * FDn;
            int nd_2ex = nx2p * ny2p * nz2p;

            // Corner of the double-extended region
            int icor = i_s - 2 * FDn;
            int jcor = j_s - 2 * FDn;
            int kcor = k_s - 2 * FDn;
            double x0_shift = pos.x - dx * icor;
            double y0_shift = pos.y - dy * jcor;
            double z0_shift = pos.z - dz * kcor;

            // Compute distances on double-extended grid
            std::vector<double> R(nd_2ex);
            std::vector<int> ind_interp;
            std::vector<double> R_interp;

            int count = 0;
            for (int kk = 0; kk < nz2p; ++kk) {
                double rz = kk * dz - z0_shift;
                for (int jj = 0; jj < ny2p; ++jj) {
                    double ry = jj * dy - y0_shift;
                    for (int ii = 0; ii < nx2p; ++ii) {
                        double rx = ii * dx - x0_shift;
                        double r = is_orth_xc ? std::sqrt(rx*rx + ry*ry + rz*rz)
                                              : lattice_xc->metric_distance(rx, ry, rz);
                        R[count] = r;
                        if (r <= rchrg) {
                            ind_interp.push_back(count);
                            R_interp.push_back(r);
                        }
                        count++;
                    }
                }
            }

            // Interpolate rho_core_J on double-extended grid
            std::vector<double> rhocJ(nd_2ex, 0.0);
            if (!R_interp.empty()) {
                std::vector<double> rhoc_interp;
                Pseudopotential::spline_interp(r_grid, psd.rho_c_table(),
                                               psd.rho_c_spline_d(), R_interp, rhoc_interp);
                for (size_t idx = 0; idx < ind_interp.size(); ++idx) {
                    rhocJ[ind_interp[idx]] = rhoc_interp[idx];
                }
            }

            // Compute gradient of rhocJ on extended grid (rb + FDn)
            std::vector<double> drhocJ_x(nd_ex, 0.0);
            std::vector<double> drhocJ_y(nd_ex, 0.0);
            std::vector<double> drhocJ_z(nd_ex, 0.0);

            for (int kk = 0; kk < nzp; ++kk) {
                int k2p = kk + FDn;
                int kshift_2p = k2p * nx2p * ny2p;
                int kshift_p = kk * nxp * nyp;
                for (int jj = 0; jj < nyp; ++jj) {
                    int j2p = jj + FDn;
                    int jshift_2p = kshift_2p + j2p * nx2p;
                    int jshift_p = kshift_p + jj * nxp;
                    for (int ii = 0; ii < nxp; ++ii) {
                        int i2p = ii + FDn;
                        int ishift_2p = jshift_2p + i2p;
                        int ishift_p = jshift_p + ii;
                        double dx_val = 0.0, dy_val = 0.0, dz_val = 0.0;
                        for (int p = 1; p <= FDn; ++p) {
                            dx_val += (rhocJ[ishift_2p + p] - rhocJ[ishift_2p - p]) * D1_x[p];
                            dy_val += (rhocJ[ishift_2p + p*nx2p] - rhocJ[ishift_2p - p*nx2p]) * D1_y[p];
                            dz_val += (rhocJ[ishift_2p + p*nx2p*ny2p] - rhocJ[ishift_2p - p*nx2p*ny2p]) * D1_z[p];
                        }
                        drhocJ_x[ishift_p] = dx_val;
                        drhocJ_y[ishift_p] = dy_val;
                        drhocJ_z[ishift_p] = dz_val;
                    }
                }
            }

            // DM domain offsets
            int dI = i_s - xs;
            int dJ = j_s - ys;
            int dK = k_s - zs;

            // Integrate Vxc * drhocJ over overlap region (rb)
            double force_x = 0.0, force_y = 0.0, force_z = 0.0;
            for (int kk = 0; kk < nz; ++kk) {
                int kp = kk + FDn;
                int k_DM = kk + dK;
                int kshift_DM = k_DM * DMnx * DMny;
                int kshift_p = kp * nxp * nyp;
                for (int jj = 0; jj < ny; ++jj) {
                    int jp = jj + FDn;
                    int j_DM = jj + dJ;
                    int jshift_DM = kshift_DM + j_DM * DMnx;
                    int jshift_p = kshift_p + jp * nxp;
                    for (int ii = 0; ii < nx; ++ii) {
                        int ip = ii + FDn;
                        int i_DM = ii + dI;
                        int ishift_DM = jshift_DM + i_DM;
                        int ishift_p = jshift_p + ip;

                        double Vxc_val = Vxc[ishift_DM];
                        force_x += Vxc_val * drhocJ_x[ishift_p];
                        force_y += Vxc_val * drhocJ_y[ishift_p];
                        force_z += Vxc_val * drhocJ_z[ishift_p];
                    }
                }
            }

            f_xc_[atom_index * 3 + 0] += force_x * dV;
            f_xc_[atom_index * 3 + 1] += force_y * dV;
            f_xc_[atom_index * 3 + 2] += force_z * dV;
        }
    }

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

void Forces::print(int rank, bool is_soc, bool has_nlcc, int Natom) const {
    if (rank != 0) return;

    std::printf("\nLocal forces (Ha/Bohr):\n");
    const auto& fl = f_local_;
    for (int i = 0; i < Natom; ++i)
        std::printf("  Atom %3d: %14.10f %14.10f %14.10f\n",
                    i + 1, fl[3*i], fl[3*i+1], fl[3*i+2]);

    std::printf("\nNonlocal forces (Ha/Bohr):\n");
    const auto& fn = f_nloc_;
    for (int i = 0; i < Natom; ++i)
        std::printf("  Atom %3d: %14.10f %14.10f %14.10f\n",
                    i + 1, fn[3*i], fn[3*i+1], fn[3*i+2]);

    if (is_soc) {
        std::printf("\nSOC forces (Ha/Bohr):\n");
        const auto& fs = f_soc_;
        for (int i = 0; i < Natom; ++i)
            std::printf("  Atom %3d: %14.10f %14.10f %14.10f\n",
                        i + 1, fs[3*i], fs[3*i+1], fs[3*i+2]);
    }

    if (has_nlcc) {
        std::printf("\nNLCC XC forces (Ha/Bohr):\n");
        const auto& fxc = f_xc_;
        for (int i = 0; i < Natom; ++i)
            std::printf("  Atom %3d: %14.10f %14.10f %14.10f\n",
                        i + 1, fxc[3*i], fxc[3*i+1], fxc[3*i+2]);
    }

    std::printf("\nTotal forces (Ha/Bohr):\n");
    const auto& ft = f_total_;
    for (int i = 0; i < Natom; ++i)
        std::printf("  Atom %3d: %14.10f %14.10f %14.10f\n",
                    i + 1, ft[3*i], ft[3*i+1], ft[3*i+2]);
}

} // namespace lynx
