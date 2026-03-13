#include "electronic/ElectronDensity.hpp"
#include <cstring>
#include <cmath>

namespace sparc {

void ElectronDensity::allocate(int Nd_d, int Nspin) {
    Nd_d_ = Nd_d;
    Nspin_ = Nspin;

    rho_.clear();
    for (int s = 0; s < Nspin; ++s) {
        rho_.emplace_back(Nd_d);
    }
    rho_total_ = NDArray<double>(Nd_d);
    if (Nspin > 1) {
        mag_ = NDArray<double>(Nd_d);
    }
}

void ElectronDensity::compute(const Wavefunction& wfn,
                               const std::vector<double>& kpt_weights,
                               double dV,
                               const MPIComm& bandcomm,
                               const MPIComm& kptcomm,
                               int Nspin_global,
                               int spin_start,
                               const MPIComm* spincomm,
                               int kpt_start) {
    int Nband = wfn.Nband();
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

    // Compute density for LOCAL spin channels only
    for (int s = 0; s < Nspin_local; ++s) {
        int s_glob = spin_start + s;  // global spin index
        for (int k = 0; k < Nkpts; ++k) {
            const auto& occ = wfn.occupations(s, k);
            double wk = kpt_weights[kpt_start + k];

            if (wfn.is_complex()) {
                const auto& psi_c = wfn.psi_kpt(s, k);
                for (int n = 0; n < Nband; ++n) {
                    double fn = occ(n);
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
                for (int n = 0; n < Nband; ++n) {
                    double fn = occ(n);
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

double ElectronDensity::integrate(double dV) const {
    double sum = 0.0;
    const double* rt = rho_total_.data();
    for (int i = 0; i < Nd_d_; ++i) {
        sum += rt[i];
    }
    return sum * dV;
}

} // namespace sparc
