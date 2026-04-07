#include "electronic/KineticEnergyDensity.hpp"
#include <cstring>
#include <cmath>
#include <complex>
#include <mpi.h>

namespace lynx {

#ifndef USE_CUDA
KineticEnergyDensity::~KineticEnergyDensity() = default;
#endif

void KineticEnergyDensity::allocate(int Nd_d, int Nspin) {
    int tau_size = (Nspin == 2) ? 3 * Nd_d : Nd_d;
    tau_ = DeviceArray<double>(tau_size);
    valid_ = false;
}

void KineticEnergyDensity::compute(const Wavefunction& wfn,
                                    const std::vector<double>& kpt_weights,
                                    const FDGrid& grid,
                                    const Domain& domain,
                                    const HaloExchange& halo,
                                    const Gradient& gradient,
                                    const KPoints* kpoints,
                                    const MPIComm& bandcomm,
                                    const MPIComm& kptcomm,
                                    const MPIComm* spincomm,
                                    int spin_start,
                                    int kpt_start,
                                    int band_start,
                                    int Nspin_global) {
    int Nd_d = domain.Nd_d();
    int Nband_loc = wfn.Nband();
    int Nspin_local = wfn.Nspin();
    int Nkpts = wfn.Nkpts();

    // Zero tau
    std::memset(tau_.data(), 0, tau_.size() * sizeof(double));

    // NOTE: Unlike density which uses spin_fac (2 for non-spin, 1 for spin),
    // tau uses g_nk = occ[n] (no spin_fac) in the accumulation loop.

    // Gradient operator and halo exchange setup
    int nd_ex = halo.nx_ex() * halo.ny_ex() * halo.nz_ex();
    bool is_orth = grid.lattice().is_orthogonal();
    const Mat3& lapcT = grid.lattice().lapc_T();
    Vec3 cell_lengths = grid.lattice().lengths();

    for (int s = 0; s < Nspin_local; ++s) {
        int s_glob = spin_start + s;
        double* tau_s = tau_.data() + s_glob * Nd_d;

        for (int k = 0; k < Nkpts; ++k) {
            const auto& occ = wfn.occupations(s, k);
            double wk = kpt_weights[kpt_start + k];

            if (wfn.is_complex()) {
                // k-point (complex) path
                const auto& psi_c = wfn.psi_kpt(s, k);
                std::vector<Complex> psi_ex(nd_ex);
                std::vector<Complex> dpsi_x(Nd_d), dpsi_y(Nd_d), dpsi_z(Nd_d);
                Vec3 kpt = kpoints->kpts_cart()[kpt_start + k];

                for (int n = 0; n < Nband_loc; ++n) {
                    double fn = occ(band_start + n);
                    if (fn < 1e-16) continue;
                    double g_nk = wk * fn;

                    const Complex* col = psi_c.col(n);
                    halo.execute_kpt(col, psi_ex.data(), 1, kpt, cell_lengths);

                    gradient.apply(psi_ex.data(), dpsi_x.data(), 0, 1);
                    gradient.apply(psi_ex.data(), dpsi_y.data(), 1, 1);
                    gradient.apply(psi_ex.data(), dpsi_z.data(), 2, 1);

                    if (is_orth) {
                        for (int i = 0; i < Nd_d; ++i) {
                            tau_s[i] += g_nk * (std::norm(dpsi_x[i]) + std::norm(dpsi_y[i]) + std::norm(dpsi_z[i]));
                        }
                    } else {
                        for (int i = 0; i < Nd_d; ++i) {
                            Complex dx = dpsi_x[i], dy = dpsi_y[i], dz = dpsi_z[i];
                            double val = lapcT(0,0) * std::norm(dx) + lapcT(1,1) * std::norm(dy) + lapcT(2,2) * std::norm(dz)
                                       + 2.0 * lapcT(0,1) * (std::conj(dx) * dy).real()
                                       + 2.0 * lapcT(0,2) * (std::conj(dx) * dz).real()
                                       + 2.0 * lapcT(1,2) * (std::conj(dy) * dz).real();
                            tau_s[i] += g_nk * val;
                        }
                    }
                }
            } else {
                // Gamma-point (real) path
                const auto& psi = wfn.psi(s, k);
                std::vector<double> psi_ex(nd_ex);
                std::vector<double> dpsi_x(Nd_d), dpsi_y(Nd_d), dpsi_z(Nd_d);

                for (int n = 0; n < Nband_loc; ++n) {
                    double fn = occ(band_start + n);
                    if (fn < 1e-16) continue;
                    double g_nk = wk * fn;

                    const double* col = psi.col(n);
                    halo.execute(col, psi_ex.data(), 1);

                    gradient.apply(psi_ex.data(), dpsi_x.data(), 0, 1);
                    gradient.apply(psi_ex.data(), dpsi_y.data(), 1, 1);
                    gradient.apply(psi_ex.data(), dpsi_z.data(), 2, 1);

                    if (is_orth) {
                        for (int i = 0; i < Nd_d; ++i) {
                            tau_s[i] += g_nk * (dpsi_x[i]*dpsi_x[i] + dpsi_y[i]*dpsi_y[i] + dpsi_z[i]*dpsi_z[i]);
                        }
                    } else {
                        for (int i = 0; i < Nd_d; ++i) {
                            double dx = dpsi_x[i], dy = dpsi_y[i], dz = dpsi_z[i];
                            double val = lapcT(0,0)*dx*dx + lapcT(1,1)*dy*dy + lapcT(2,2)*dz*dz
                                       + 2.0*lapcT(0,1)*dx*dy + 2.0*lapcT(0,2)*dx*dz + 2.0*lapcT(1,2)*dy*dz;
                            tau_s[i] += g_nk * val;
                        }
                    }
                }
            }
        }
    }

    // Allreduce over band communicator
    if (!bandcomm.is_null() && bandcomm.size() > 1) {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start + s;
            bandcomm.allreduce_sum(tau_.data() + s_glob * Nd_d, Nd_d);
        }
    }

    // Allreduce over kpt communicator
    if (!kptcomm.is_null() && kptcomm.size() > 1) {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start + s;
            kptcomm.allreduce_sum(tau_.data() + s_glob * Nd_d, Nd_d);
        }
    }

    // Exchange spin channels across spin communicator
    if (spincomm && !spincomm->is_null() && spincomm->size() > 1 && Nspin_global == 2) {
        int my_spin = spin_start;
        int other_spin = 1 - my_spin;
        int partner = (spincomm->rank() == 0) ? 1 : 0;
        MPI_Sendrecv(tau_.data() + my_spin * Nd_d, Nd_d, MPI_DOUBLE, partner, 0,
                     tau_.data() + other_spin * Nd_d, Nd_d, MPI_DOUBLE, partner, 0,
                     spincomm->comm(), MPI_STATUS_IGNORE);
    }

    // For spin-polarized: apply 0.5 factor then compute total = up + dn
    if (Nspin_global == 2) {
        double* tau_up = tau_.data();
        double* tau_dn = tau_.data() + Nd_d;
        double* tau_tot = tau_.data() + 2 * Nd_d;
        for (int i = 0; i < Nd_d; ++i) {
            tau_up[i] *= 0.5;
            tau_dn[i] *= 0.5;
            tau_tot[i] = tau_up[i] + tau_dn[i];
        }
    }

    valid_ = true;
}

// CPU compute helper
static void compute_cpu_impl(KineticEnergyDensity& self, const LynxContext& ctx,
                              const Wavefunction& wfn,
                              const std::vector<double>& kpt_weights) {
    self.compute(wfn, kpt_weights,
                 ctx.grid(), ctx.domain(), ctx.halo(), ctx.gradient(),
                 &ctx.kpoints(), ctx.scf_bandcomm(), ctx.kpt_bridge(),
                 &ctx.spin_bridge(), ctx.spin_start(), ctx.kpt_start(),
                 ctx.band_start(), ctx.Nspin());
}

// ============================================================
// Dispatching compute — checks dev_ member
// ============================================================

void KineticEnergyDensity::compute(const LynxContext& ctx,
                                    const Wavefunction& wfn,
                                    const std::vector<double>& kpt_weights)
{
    // GPU tau computation requires device-resident psi pointers.
    // Use compute_gpu_from_device() for GPU builds (called directly by SCF).
    // This dispatcher only handles the CPU path.
    compute_cpu_impl(*this, ctx, wfn, kpt_weights);
}

} // namespace lynx
