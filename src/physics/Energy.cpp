#include "physics/Energy.hpp"
#include "electronic/Occupation.hpp"
#include <cmath>
#include <cstdio>
#include <mpi.h>

namespace lynx {

double Energy::band_energy(const Wavefunction& wfn,
                             const std::vector<double>& kpt_weights,
                             int Nspin,
                             int kpt_start) {
    double Eband = 0.0;
    double spin_fac = (Nspin == 1 && wfn.Nspinor() == 1) ? 2.0 : 1.0;
    int Nband = wfn.Nband_global();  // use global band count (eigenvalue/occupation size)

    for (int s = 0; s < wfn.Nspin(); ++s) {
        for (int k = 0; k < wfn.Nkpts(); ++k) {
            const auto& eig = wfn.eigenvalues(s, k);
            const auto& occ = wfn.occupations(s, k);
            double wk = kpt_weights[kpt_start + k] * spin_fac;
            for (int n = 0; n < Nband; ++n) {
                Eband += wk * occ(n) * eig(n);
            }
        }
    }
    return Eband;
}

double Energy::xc_energy(const double* rho, const double* exc,
                           int Nd_d, double dV) {
    double local_sum = 0.0;
    for (int i = 0; i < Nd_d; ++i) {
        local_sum += rho[i] * exc[i];
    }
    local_sum *= dV;
    return local_sum;
}

double Energy::hartree_energy(const double* rho, const double* phi,
                                int Nd_d, double dV) {
    double local_sum = 0.0;
    for (int i = 0; i < Nd_d; ++i) {
        local_sum += rho[i] * phi[i];
    }
    local_sum *= 0.5 * dV;
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
    const double* rho_core,
    double Ef,
    int kpt_start,
    const MPIComm* kptcomm,
    const MPIComm* spincomm,
    int Nspin_global,
    const MPIComm* bandcomm,
    const double* tau,
    const double* vtau) {

    EnergyComponents E;
    E.Eself = Eself;
    E.Ec = Ec;

    // Use Nspin_global for spin_fac; fall back to wfn.Nspin() if not specified
    int Nspin = (Nspin_global > 0) ? Nspin_global : wfn.Nspin();

    // Band energy (local contribution — will be reduced below)
    E.Eband = band_energy(wfn, kpt_weights, Nspin, kpt_start);

    // XC energy: Exc = ∫ (rho + rho_core) * exc dV (matching reference LYNX)
    const double* rho = density.rho_total().data();
    if (rho_core) {
        // With NLCC, integrate (rho_valence + rho_core) * exc
        double local_sum = 0.0;
        for (int i = 0; i < Nd_d; ++i) {
            local_sum += (rho[i] + rho_core[i]) * exc[i];
        }
        local_sum *= dV;
        E.Exc = local_sum;
    } else {
        E.Exc = xc_energy(rho, exc, Nd_d, dV);
    }

    // Double counting: E2 = integral sum_s rho_s * Vxc_s dV
    // For non-spin: E2 = integral rho * Vxc dV
    // For spin: E2 = integral (rho_up * Vxc_up + rho_dn * Vxc_dn) dV
    double E2 = 0.0;
    if (Nspin == 2) {
        const double* rho_up = density.rho(0).data();
        const double* rho_dn = density.rho(1).data();
        for (int i = 0; i < Nd_d; ++i) {
            E2 += rho_up[i] * Vxc[i] + rho_dn[i] * Vxc[Nd_d + i];
        }
    } else {
        for (int i = 0; i < Nd_d; ++i) {
            E2 += rho[i] * Vxc[i];
        }
    }
    E2 *= dV;

    // Electrostatic energy: E3 = integral rho * phi dV
    double E3 = 0.0;
    for (int i = 0; i < Nd_d; ++i) {
        E3 += rho[i] * phi[i];
    }
    E3 *= dV;

    // mGGA double-counting correction: E3_mgga = integral tau * vtau dV
    // (Eband already includes the vtau contribution through the Hamiltonian)
    if (tau && vtau) {
        double E3_mgga = 0.0;
        if (Nspin == 2) {
            // tau layout: [up|dn|total], vtau layout: [up|dn]
            // E3_mgga = integral (tau_up * vtau_up + tau_dn * vtau_dn) dV
            for (int i = 0; i < 2 * Nd_d; ++i) {
                E3_mgga += tau[i] * vtau[i];
            }
        } else {
            for (int i = 0; i < Nd_d; ++i) {
                E3_mgga += tau[i] * vtau[i];
            }
        }
        E3_mgga *= dV;
        E3 += E3_mgga;
    }

    // Hartree energy = 0.5 * integral (rho + b) * phi dV
    E.Ehart = 0.0;
    if (rho_b) {
        for (int i = 0; i < Nd_d; ++i) {
            E.Ehart += (rho[i] + rho_b[i]) * phi[i];
        }
        E.Ehart *= 0.5 * dV;
    }

    // Entropy (local contribution)
    E.Entropy = Occupation::entropy(wfn, beta, smearing, kpt_weights, Ef, kpt_start, Nspin);

    // Allreduce Eband and Entropy across bandcomm, kptcomm, and spincomm
    // Note: Eband and Entropy are computed from eigenvalues/occupations which are global
    // (all procs have all eigenvalues), but band_energy iterates over them redundantly
    // on each band proc. So with band parallelism, do NOT allreduce over bandcomm
    // since each proc already computed the full Eband from global eigenvalues.
    // The bandcomm allreduce is only needed for density, forces, stress.
    if (kptcomm && !kptcomm->is_null() && kptcomm->size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &E.Eband, 1, MPI_DOUBLE, MPI_SUM, kptcomm->comm());
        MPI_Allreduce(MPI_IN_PLACE, &E.Entropy, 1, MPI_DOUBLE, MPI_SUM, kptcomm->comm());
    }
    if (spincomm && !spincomm->is_null() && spincomm->size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &E.Eband, 1, MPI_DOUBLE, MPI_SUM, spincomm->comm());
        MPI_Allreduce(MPI_IN_PLACE, &E.Entropy, 1, MPI_DOUBLE, MPI_SUM, spincomm->comm());
    }

    // Total free energy:
    // Etot = Eband + E1 - E2 - E3 + Exc + Eself + Ec + Entropy
    // where E1 = Ehart = 0.5 * int (rho+b)*phi dV
    // Eband already contains kinetic + Veff contribution
    // Veff = Vxc + phi + ... so we need to subtract double counting
    E.Etotal = E.Eband - E2 + E.Exc - E3 + E.Ehart + E.Eself + E.Ec + E.Entropy;

    {
        int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            std::printf("LYNX_CMP Etot=%.15e Eband=%.15e E1=%.15e E2=%.15e E3=%.15e Exc=%.15e Eself=%.15e Ec=%.15e Entropy=%.15e\n",
                E.Etotal, E.Eband, E.Ehart, E2, E3, E.Exc, E.Eself, E.Ec, E.Entropy);
    }

    return E;
}

} // namespace lynx
