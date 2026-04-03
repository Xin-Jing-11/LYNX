#include "physics/Energy.hpp"
#include "core/LynxContext.hpp"
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

double Energy::xc_energy_with_nlcc(const double* rho, const double* rho_core,
                                     const double* exc, int Nd_d, double dV) {
    double local_sum = 0.0;
    if (rho_core) {
        for (int i = 0; i < Nd_d; ++i)
            local_sum += (rho[i] + rho_core[i]) * exc[i];
    } else {
        for (int i = 0; i < Nd_d; ++i)
            local_sum += rho[i] * exc[i];
    }
    return local_sum * dV;
}

double Energy::double_counting_correction(const ElectronDensity& density,
                                            const double* Vxc, int Nd_d, double dV, int Nspin) {
    double E2 = 0.0;
    if (Nspin == 2) {
        const double* rho_up = density.rho(0).data();
        const double* rho_dn = density.rho(1).data();
        for (int i = 0; i < Nd_d; ++i)
            E2 += rho_up[i] * Vxc[i] + rho_dn[i] * Vxc[Nd_d + i];
    } else {
        const double* rho = density.rho_total().data();
        for (int i = 0; i < Nd_d; ++i)
            E2 += rho[i] * Vxc[i];
    }
    return E2 * dV;
}

double Energy::electrostatic_energy(const double* rho, const double* phi, int Nd_d, double dV) {
    double E3 = 0.0;
    for (int i = 0; i < Nd_d; ++i)
        E3 += rho[i] * phi[i];
    return E3 * dV;
}

double Energy::mgga_correction(const double* tau, const double* vtau,
                                 int Nd_d, double dV, int Nspin) {
    double E3_mgga = 0.0;
    if (Nspin == 2) {
        // tau layout: [up|dn|total], vtau layout: [up|dn]
        for (int i = 0; i < 2 * Nd_d; ++i)
            E3_mgga += tau[i] * vtau[i];
    } else {
        for (int i = 0; i < Nd_d; ++i)
            E3_mgga += tau[i] * vtau[i];
    }
    return E3_mgga * dV;
}

double Energy::hartree_energy_with_pseudocharge(const double* rho, const double* rho_b,
                                                  const double* phi, int Nd_d, double dV) {
    if (!rho_b) return 0.0;
    double local_sum = 0.0;
    for (int i = 0; i < Nd_d; ++i)
        local_sum += (rho[i] + rho_b[i]) * phi[i];
    return 0.5 * local_sum * dV;
}

double Energy::self_consistency_correction(
    const ElectronDensity& density,
    const double* Veff_out, const double* Veff_in,
    int Nd_d, double dV, int Nspin) {
    double Escc = 0.0;
    if (Nspin == 2) {
        for (int s = 0; s < Nspin; ++s) {
            const double* rho_s = density.rho(s).data();
            for (int i = 0; i < Nd_d; ++i)
                Escc += rho_s[i] * (Veff_out[s * Nd_d + i] - Veff_in[s * Nd_d + i]);
        }
    } else {
        const double* rho_tot = density.rho_total().data();
        for (int i = 0; i < Nd_d; ++i)
            Escc += rho_tot[i] * (Veff_out[i] - Veff_in[i]);
    }
    return Escc * dV;
}

double Energy::total_energy(const EnergyComponents& E) {
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

    const double* rho = density.rho_total().data();

    // Band energy (local contribution — will be reduced below)
    E.Eband = band_energy(wfn, kpt_weights, Nspin, kpt_start);

    // XC energy: Exc = integral (rho + rho_core) * exc dV
    E.Exc = xc_energy_with_nlcc(rho, rho_core, exc, Nd_d, dV);

    // Double-counting correction: E2 = integral sum_s rho_s * Vxc_s dV
    double E2 = double_counting_correction(density, Vxc, Nd_d, dV, Nspin);

    // Electrostatic energy: E3 = integral rho * phi dV
    double E3 = electrostatic_energy(rho, phi, Nd_d, dV);

    // mGGA double-counting correction (Eband already includes vtau via Hamiltonian)
    if (tau && vtau)
        E3 += mgga_correction(tau, vtau, Nd_d, dV, Nspin);

    // Hartree energy = 0.5 * integral (rho + b) * phi dV
    E.Ehart = hartree_energy_with_pseudocharge(rho, rho_b, phi, Nd_d, dV);

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

    return E;
}

EnergyComponents Energy::compute_all(
    const LynxContext& ctx,
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
    const double* rho_core,
    double Ef,
    const double* tau,
    const double* vtau) {

    return compute_all(wfn, density, Veff, phi, exc, Vxc, rho_b,
                       Eself, Ec, beta, smearing, kpt_weights,
                       ctx.domain().Nd_d(), ctx.dV(),
                       rho_core, Ef, ctx.kpt_start(),
                       &ctx.kpt_bridge(), &ctx.spin_bridge(),
                       ctx.Nspin(), nullptr, tau, vtau);
}

} // namespace lynx
