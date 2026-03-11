#include "physics/Energy.hpp"
#include "electronic/Occupation.hpp"
#include <cmath>

namespace sparc {

double Energy::band_energy(const Wavefunction& wfn,
                             const std::vector<double>& kpt_weights,
                             int Nspin) {
    double Eband = 0.0;
    double spin_fac = (Nspin == 1) ? 2.0 : 1.0;

    for (int s = 0; s < wfn.Nspin(); ++s) {
        for (int k = 0; k < wfn.Nkpts(); ++k) {
            const auto& eig = wfn.eigenvalues(s, k);
            const auto& occ = wfn.occupations(s, k);
            double wk = kpt_weights[k] * spin_fac;
            for (int n = 0; n < wfn.Nband(); ++n) {
                Eband += wk * occ(n) * eig(n);
            }
        }
    }
    return Eband;
}

double Energy::xc_energy(const double* rho, const double* exc,
                           int Nd_d, double dV, const MPIComm& dmcomm) {
    double local_sum = 0.0;
    for (int i = 0; i < Nd_d; ++i) {
        local_sum += rho[i] * exc[i];
    }
    local_sum *= dV;

    if (!dmcomm.is_null() && dmcomm.size() > 1) {
        return dmcomm.allreduce_sum(local_sum);
    }
    return local_sum;
}

double Energy::hartree_energy(const double* rho, const double* phi,
                                int Nd_d, double dV, const MPIComm& dmcomm) {
    double local_sum = 0.0;
    for (int i = 0; i < Nd_d; ++i) {
        local_sum += rho[i] * phi[i];
    }
    local_sum *= 0.5 * dV;

    if (!dmcomm.is_null() && dmcomm.size() > 1) {
        return dmcomm.allreduce_sum(local_sum);
    }
    return local_sum;
}

double Energy::total_energy(const EnergyComponents& E) {
    // Etotal = Eband + Exc - E2 + Ehart + Eself + Ec + Entropy
    // where E2 = integral rho * Vxc dV (double counting correction)
    // and E3 = integral rho * phi dV (already in Eband via Veff)
    // Simplified: Etotal = Eband - E2 + Ehart + Exc + Eself + Ec + Entropy
    return E.Etotal;
}

EnergyComponents Energy::compute_all(
    const Wavefunction& wfn,
    const ElectronDensity& density,
    const double* Veff,
    const double* phi,
    const double* exc,
    const double* Vxc,
    const double* rho_b,
    double Eself,
    double Ec,
    double beta,
    SmearingType smearing,
    const std::vector<double>& kpt_weights,
    int Nd_d, double dV,
    const MPIComm& dmcomm,
    const double* rho_core,
    double Ef) {

    EnergyComponents E;
    E.Eself = Eself;
    E.Ec = Ec;

    int Nspin = wfn.Nspin();

    // Band energy
    E.Eband = band_energy(wfn, kpt_weights, Nspin);

    // XC energy: Exc = ∫ (rho + rho_core) * exc dV (matching reference SPARC)
    const double* rho = density.rho_total().data();
    if (rho_core) {
        // With NLCC, integrate (rho_valence + rho_core) * exc
        double local_sum = 0.0;
        for (int i = 0; i < Nd_d; ++i) {
            local_sum += (rho[i] + rho_core[i]) * exc[i];
        }
        local_sum *= dV;
        if (!dmcomm.is_null() && dmcomm.size() > 1) {
            E.Exc = dmcomm.allreduce_sum(local_sum);
        } else {
            E.Exc = local_sum;
        }
    } else {
        E.Exc = xc_energy(rho, exc, Nd_d, dV, dmcomm);
    }

    // Double counting: E2 = integral rho * Vxc dV
    double E2 = 0.0;
    for (int i = 0; i < Nd_d; ++i) {
        E2 += rho[i] * Vxc[i];
    }
    E2 *= dV;
    if (!dmcomm.is_null() && dmcomm.size() > 1) {
        E2 = dmcomm.allreduce_sum(E2);
    }

    // Electrostatic energy: E3 = integral rho * phi dV
    double E3 = 0.0;
    for (int i = 0; i < Nd_d; ++i) {
        E3 += rho[i] * phi[i];
    }
    E3 *= dV;
    if (!dmcomm.is_null() && dmcomm.size() > 1) {
        E3 = dmcomm.allreduce_sum(E3);
    }

    // Hartree energy = 0.5 * integral (rho + b) * phi dV
    E.Ehart = 0.0;
    if (rho_b) {
        for (int i = 0; i < Nd_d; ++i) {
            E.Ehart += (rho[i] + rho_b[i]) * phi[i];
        }
        E.Ehart *= 0.5 * dV;
        if (!dmcomm.is_null() && dmcomm.size() > 1) {
            E.Ehart = dmcomm.allreduce_sum(E.Ehart);
        }
    }

    // Entropy
    E.Entropy = Occupation::entropy(wfn, beta, smearing, kpt_weights, Ef);

    // Total free energy:
    // Etot = Eband + E1 - E2 - E3 + Exc + Eself + Ec + Entropy
    // where E1 = Ehart = 0.5 * int (rho+b)*phi dV
    // Eband already contains kinetic + Veff contribution
    // Veff = Vxc + phi + ... so we need to subtract double counting
    E.Etotal = E.Eband - E2 + E.Exc - E3 + E.Ehart + E.Eself + E.Ec + E.Entropy;

    // Debug: print energy components
    if (dmcomm.rank() == 0) {
        std::printf("  Energy: Eband=%.6f Exc=%.6f Ehart=%.6f Eself=%.6f Ec=%.6f\n",
                    E.Eband, E.Exc, E.Ehart, E.Eself, E.Ec);
        std::printf("  Energy: E2(rho*Vxc)=%.6f E3(rho*phi)=%.6f Entropy=%.6f\n",
                    E2, E3, E.Entropy);
    }

    return E;
}

} // namespace sparc
