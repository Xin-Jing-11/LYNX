#include "electronic/ElectronDensity.hpp"
#include <cstring>
#include <cmath>

namespace lynx {

#ifndef USE_CUDA
ElectronDensity::~ElectronDensity() = default;
#endif

void ElectronDensity::allocate(int Nd_d, int Nspin) {
    Nd_d_ = Nd_d;
    Nspin_ = Nspin;

    rho_.clear();
    for (int s = 0; s < Nspin; ++s) {
        rho_.emplace_back(Nd_d);
    }
    rho_total_ = DeviceArray<double>(Nd_d);
    if (Nspin > 1) {
        mag_ = DeviceArray<double>(Nd_d);
    }
}

void ElectronDensity::initialize_uniform(int Nd_d, int Nspin, int Nelectron, double cell_volume) {
    allocate(Nd_d, Nspin);
    double rho0 = Nelectron / cell_volume;

    if (Nspin == 1) {
        double* rho = rho_[0].data();
        for (int i = 0; i < Nd_d; ++i) rho[i] = rho0;
    } else {
        double* rho_up = rho_[0].data();
        double* rho_dn = rho_[1].data();
        for (int i = 0; i < Nd_d; ++i) {
            rho_up[i] = rho0 * 0.5;
            rho_dn[i] = rho0 * 0.5;
        }
    }
    double* rho_t = rho_total_.data();
    for (int i = 0; i < Nd_d; ++i) rho_t[i] = rho0;
}

void ElectronDensity::initialize_uniform_noncollinear(int Nd_d, int Nelectron, double cell_volume) {
    allocate_noncollinear(Nd_d);
    double rho0 = Nelectron / cell_volume;
    double* rho = rho_total_.data();
    for (int i = 0; i < Nd_d; ++i) rho[i] = rho0;
    std::memcpy(rho_[0].data(), rho, Nd_d * sizeof(double));
    mag_x_.zero();
    mag_y_.zero();
    mag_z_.zero();
}

void ElectronDensity::compute(const Wavefunction& wfn,
                               const std::vector<double>& kpt_weights,
                               double dV,
                               const MPIComm& bandcomm,
                               const MPIComm& kptcomm,
                               int Nspin_global,
                               int spin_start,
                               const MPIComm* spincomm,
                               int kpt_start,
                               int band_start) {
    int Nband_loc = wfn.Nband();           // local bands (psi columns)
    int Nband_glob = wfn.Nband_global();   // global bands (occupation array size)
    int Nspin_local = wfn.Nspin();
    int Nkpts = wfn.Nkpts();

    // If Nspin_global not specified, infer from wfn (backward compat)
    if (Nspin_global <= 0) Nspin_global = Nspin_local;

    // Zero out all spin densities (Nspin_ = Nspin_global from allocate)
    rho_total_.zero();
    for (int s = 0; s < Nspin_; ++s) {
        rho_[s].zero();
    }

    // Spin multiplier: 2 for non-spin-polarized, 1 for collinear
    double spin_fac = (Nspin_global == 1) ? 2.0 : 1.0;

    // Compute density for LOCAL spin channels only.
    // Iterate over Nband_loc (local psi columns). Occupation is indexed by
    // global band index: occ(band_start + n) for local band n.
    for (int s = 0; s < Nspin_local; ++s) {
        int s_glob = spin_start + s;  // global spin index
        for (int k = 0; k < Nkpts; ++k) {
            const auto& occ = wfn.occupations(s, k);
            double wk = kpt_weights[kpt_start + k];

            if (wfn.is_complex()) {
                const auto& psi_c = wfn.psi_kpt(s, k);
                for (int n = 0; n < Nband_loc; ++n) {
                    double fn = occ(band_start + n);  // global band index
                    if (fn < 1e-16) continue;

                    const Complex* col = psi_c.col(n);
                    double* rho_s = rho_[s_glob].data();
                    double w = spin_fac * wk * fn;

                    for (int i = 0; i < Nd_d_; ++i) {
                        rho_s[i] += w * std::norm(col[i]);
                    }
                }
            } else {
                const auto& psi = wfn.psi(s, k);
                for (int n = 0; n < Nband_loc; ++n) {
                    double fn = occ(band_start + n);  // global band index
                    if (fn < 1e-16) continue;

                    const double* col = psi.col(n);
                    double* rho_s = rho_[s_glob].data();
                    double w = spin_fac * wk * fn;

                    for (int i = 0; i < Nd_d_; ++i) {
                        rho_s[i] += w * col[i] * col[i];
                    }
                }
            }
        }
    }

    // Allreduce over band communicator (partial band sums — only local spin channels)
    if (!bandcomm.is_null() && bandcomm.size() > 1) {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start + s;
            bandcomm.allreduce_sum(rho_[s_glob].data(), Nd_d_);
        }
    }

    // Allreduce over kpt communicator (kpt_bridge)
    if (!kptcomm.is_null() && kptcomm.size() > 1) {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start + s;
            kptcomm.allreduce_sum(rho_[s_glob].data(), Nd_d_);
        }
    }

    // Exchange spin densities across spin communicator (spin_bridge)
    // Each process computed rho_[spin_start], need to get the other spin channel
    if (spincomm && !spincomm->is_null() && spincomm->size() > 1 && Nspin_global == 2) {
        // Process with spin_start=0 has rho_[0], needs rho_[1] from partner
        // Process with spin_start=1 has rho_[1], needs rho_[0] from partner
        int my_spin = spin_start;
        int other_spin = 1 - my_spin;
        int partner = (spincomm->rank() == 0) ? 1 : 0;

        MPI_Sendrecv(rho_[my_spin].data(), Nd_d_, MPI_DOUBLE, partner, 0,
                     rho_[other_spin].data(), Nd_d_, MPI_DOUBLE, partner, 0,
                     spincomm->comm(), MPI_STATUS_IGNORE);
    }

    // Compute total density (sum over ALL spins — now available on all processes)
    rho_total_.zero();
    for (int s = 0; s < Nspin_; ++s) {
        double* rt = rho_total_.data();
        const double* rs = rho_[s].data();
        for (int i = 0; i < Nd_d_; ++i) {
            rt[i] += rs[i];
        }
    }

    // Magnetization for collinear spin
    if (Nspin_ == 2) {
        double* m = mag_.data();
        const double* ru = rho_[0].data();
        const double* rd = rho_[1].data();
        for (int i = 0; i < Nd_d_; ++i) {
            m[i] = ru[i] - rd[i];
        }
    }
}

void ElectronDensity::allocate_noncollinear(int Nd_d) {
    Nd_d_ = Nd_d;
    Nspin_ = 1;  // single density channel for noncollinear

    rho_.clear();
    rho_.emplace_back(Nd_d);
    rho_total_ = DeviceArray<double>(Nd_d);
    mag_x_ = DeviceArray<double>(Nd_d);
    mag_y_ = DeviceArray<double>(Nd_d);
    mag_z_ = DeviceArray<double>(Nd_d);
}

void ElectronDensity::compute_spinor(const Wavefunction& wfn,
                                      const std::vector<double>& kpt_weights,
                                      double dV,
                                      const MPIComm& bandcomm,
                                      const MPIComm& kptcomm,
                                      int kpt_start,
                                      int band_start) {
    int Nband_loc = wfn.Nband();
    int Nd_d = wfn.Nd_d();
    int Nkpts = wfn.Nkpts();

    // Zero arrays
    rho_total_.zero();
    rho_[0].zero();
    mag_x_.zero();
    mag_y_.zero();
    mag_z_.zero();

    // spin_fac = 1.0 for spinor (both components present)
    double spin_fac = 1.0;

    for (int k = 0; k < Nkpts; ++k) {
        const auto& occ = wfn.occupations(0, k);
        double wk = kpt_weights[kpt_start + k];

        const auto& psi_c = wfn.psi_kpt(0, k);

        for (int n = 0; n < Nband_loc; ++n) {
            double fn = occ(band_start + n);
            if (fn < 1e-16) continue;

            const Complex* col = psi_c.col(n);
            const Complex* psi_up = col;
            const Complex* psi_dn = col + Nd_d;
            double w = spin_fac * wk * fn;

            double* rho = rho_total_.data();
            double* mx = mag_x_.data();
            double* my = mag_y_.data();
            double* mz = mag_z_.data();

            for (int i = 0; i < Nd_d; ++i) {
                double up2 = std::norm(psi_up[i]);
                double dn2 = std::norm(psi_dn[i]);
                Complex cross = std::conj(psi_up[i]) * psi_dn[i];

                rho[i] += w * (up2 + dn2);
                mx[i]  += w * 2.0 * cross.real();
                my[i]  -= w * 2.0 * cross.imag();
                mz[i]  += w * (up2 - dn2);
            }
        }
    }

    // Allreduce over band communicator
    if (!bandcomm.is_null() && bandcomm.size() > 1) {
        bandcomm.allreduce_sum(rho_total_.data(), Nd_d);
        bandcomm.allreduce_sum(mag_x_.data(), Nd_d);
        bandcomm.allreduce_sum(mag_y_.data(), Nd_d);
        bandcomm.allreduce_sum(mag_z_.data(), Nd_d);
    }

    // Allreduce over kpt communicator
    if (!kptcomm.is_null() && kptcomm.size() > 1) {
        kptcomm.allreduce_sum(rho_total_.data(), Nd_d);
        kptcomm.allreduce_sum(mag_x_.data(), Nd_d);
        kptcomm.allreduce_sum(mag_y_.data(), Nd_d);
        kptcomm.allreduce_sum(mag_z_.data(), Nd_d);
    }

    // Keep rho_[0] in sync with rho_total_
    std::memcpy(rho_[0].data(), rho_total_.data(), Nd_d * sizeof(double));
}

// CPU compute implementation (private helper)
static void compute_cpu_impl(ElectronDensity& self, const LynxContext& ctx,
                              const Wavefunction& wfn,
                              const std::vector<double>& kpt_weights) {
    self.compute(wfn, kpt_weights, ctx.dV(),
                 ctx.scf_bandcomm(), ctx.kpt_bridge(),
                 ctx.Nspin(), ctx.spin_start(),
                 &ctx.spin_bridge(), ctx.kpt_start(), ctx.band_start());
}

// CPU spinor compute implementation (private helper)
static void compute_spinor_cpu_impl(ElectronDensity& self, const LynxContext& ctx,
                                     const Wavefunction& wfn,
                                     const std::vector<double>& kpt_weights) {
    self.compute_spinor(wfn, kpt_weights, ctx.dV(),
                        ctx.scf_bandcomm(), ctx.kpt_bridge(),
                        ctx.kpt_start(), ctx.band_start());
}

double ElectronDensity::integrate(double dV) const {
    double sum = 0.0;
    const double* rt = rho_total_.data();
    for (int i = 0; i < Nd_d_; ++i) {
        sum += rt[i];
    }
    return sum * dV;
}

// ============================================================
// Dispatching compute — checks dev_ member
// ============================================================

void ElectronDensity::compute(const LynxContext& ctx,
                               const Wavefunction& wfn,
                               const std::vector<double>& kpt_weights)
{
    // GPU density computation requires device-resident psi pointers.
    // Use compute_from_device_ptrs() for GPU builds (called directly by SCF).
    // This dispatcher only handles the CPU path.
    compute_cpu_impl(*this, ctx, wfn, kpt_weights);
}

void ElectronDensity::compute_spinor(const LynxContext& ctx,
                                      const Wavefunction& wfn,
                                      const std::vector<double>& kpt_weights)
{
#ifdef USE_CUDA
    if (dev_ == Device::GPU) {
        compute_spinor_gpu(ctx, wfn, kpt_weights);
        return;
    }
#endif
    compute_spinor_cpu_impl(*this, ctx, wfn, kpt_weights);
}

} // namespace lynx
