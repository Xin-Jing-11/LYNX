#include "electronic/ElectronDensity.hpp"
#include <cstring>

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
                               const MPIComm& kptcomm) {
    int Nband = wfn.Nband();
    int Nspin = wfn.Nspin();
    int Nkpts = wfn.Nkpts();

    // Zero out densities
    rho_total_.zero();
    for (int s = 0; s < Nspin_; ++s) {
        rho_[s].zero();
    }

    // Spin multiplier: 2 for non-spin-polarized, 1 for collinear
    double spin_fac = (Nspin == 1) ? 2.0 : 1.0;

    for (int s = 0; s < Nspin; ++s) {
        for (int k = 0; k < Nkpts; ++k) {
            const auto& psi = wfn.psi(s, k);
            const auto& occ = wfn.occupations(s, k);
            double wk = kpt_weights[k];

            // rho_s(r) += sum_n spin_fac * w_k * f_n * |psi_n(r)|^2
            for (int n = 0; n < Nband; ++n) {
                double fn = occ(n);
                if (fn < 1e-16) continue;

                const double* col = psi.col(n);
                double* rho_s = rho_[s].data();
                double w = spin_fac * wk * fn;

                for (int i = 0; i < Nd_d_; ++i) {
                    rho_s[i] += w * col[i] * col[i];
                }
            }
        }
    }

    // Allreduce over band communicator (partial band sums)
    if (!bandcomm.is_null() && bandcomm.size() > 1) {
        for (int s = 0; s < Nspin_; ++s) {
            bandcomm.allreduce_sum(rho_[s].data(), Nd_d_);
        }
    }

    // Allreduce over kpt communicator
    if (!kptcomm.is_null() && kptcomm.size() > 1) {
        for (int s = 0; s < Nspin_; ++s) {
            kptcomm.allreduce_sum(rho_[s].data(), Nd_d_);
        }
    }

    // Compute total density
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
