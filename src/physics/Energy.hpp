#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "core/LynxContext.hpp"
#include "electronic/Wavefunction.hpp"
#include "electronic/ElectronDensity.hpp"
#include "xc/XCFunctional.hpp"

#include <vector>

namespace lynx {

// Energy components of the Kohn-Sham DFT calculation.
struct EnergyComponents {
    double Eband = 0.0;      // Band structure energy: sum f_n * epsilon_n
    double Exc = 0.0;        // Exchange-correlation energy
    double Ehart = 0.0;      // Hartree (electrostatic) energy
    double Eself = 0.0;      // Self energy of pseudocharges
    double Ec = 0.0;         // Correction energy (overlap of b)
    double Entropy = 0.0;    // Electronic entropy contribution (-T*S)
    double Eexx = 0.0;       // Exact exchange energy (hybrid functionals)
    double Etotal = 0.0;     // Total free energy
    double Eatom = 0.0;      // Energy per atom
};

class Energy {
public:
    Energy() = default;

    // Compute band energy: Eband = sum_{n,k,s} w_k * f_nks * epsilon_nks
    // kpt_start: global index offset for kpt_weights lookup
    static double band_energy(const Wavefunction& wfn,
                               const std::vector<double>& kpt_weights,
                               int Nspin,
                               int kpt_start = 0);

    // Compute XC energy: Exc = integral rho * exc dV
    static double xc_energy(const double* rho, const double* exc,
                             int Nd_d, double dV);

    // Compute Hartree energy: Ehart = 0.5 * integral rho * phi dV
    static double hartree_energy(const double* rho, const double* phi,
                                  int Nd_d, double dV);

    // XC energy with NLCC: Exc = integral (rho + rho_core) * exc dV
    static double xc_energy_with_nlcc(const double* rho, const double* rho_core,
                                       const double* exc, int Nd_d, double dV);

    // Double-counting correction: E2 = integral sum_s rho_s * Vxc_s dV
    static double double_counting_correction(const ElectronDensity& density,
                                              const double* Vxc, int Nd_d, double dV, int Nspin);

    // Electrostatic energy: E3 = integral rho * phi dV
    static double electrostatic_energy(const double* rho, const double* phi, int Nd_d, double dV);

    // mGGA double-counting correction: E3_mgga = integral tau * vtau dV
    static double mgga_correction(const double* tau, const double* vtau,
                                   int Nd_d, double dV, int Nspin);

    // Hartree energy with pseudocharge: 0.5 * integral (rho + rho_b) * phi dV
    static double hartree_energy_with_pseudocharge(const double* rho, const double* rho_b,
                                                    const double* phi, int Nd_d, double dV);

    // Self-consistency correction: Escc = integral sum_s rho_s * (Veff_out_s - Veff_in_s) dV
    // Used in potential mixing to correct the total energy for the Veff mismatch.
    static double self_consistency_correction(
        const ElectronDensity& density,
        const double* Veff_out, const double* Veff_in,
        int Nd_d, double dV, int Nspin);

    // Compute total energy from components
    static double total_energy(const EnergyComponents& E);

    // Compute all energy components (with LynxContext — preferred for CPU callers).
    // Extracts Nd_d, dV, kpt_start, kptcomm, spincomm, Nspin_global from ctx.
    static EnergyComponents compute_all(
        const LynxContext& ctx,
        const Wavefunction& wfn,
        const ElectronDensity& density,
        const double* Veff,
        const double* phi,          // electrostatic potential
        const double* exc,          // XC energy density
        const double* Vxc,          // XC potential
        const double* rho_b,        // pseudocharge density
        double Eself,
        double Ec,
        double beta,
        SmearingType smearing,
        const std::vector<double>& kpt_weights,
        const double* rho_core = nullptr,
        double Ef = 0.0,
        const double* tau = nullptr,      // mGGA kinetic energy density
        const double* vtau = nullptr);    // mGGA d(nε)/dτ potential

    // Compute all energy components (explicit params — used by GPU code paths).
    static EnergyComponents compute_all(
        const Wavefunction& wfn,
        const ElectronDensity& density,
        const double* Veff,
        const double* phi,          // electrostatic potential
        const double* exc,          // XC energy density
        const double* Vxc,          // XC potential
        const double* rho_b,        // pseudocharge density
        double Eself,
        double Ec,
        double beta,
        SmearingType smearing,
        const std::vector<double>& kpt_weights,
        int Nd_d, double dV,
        const double* rho_core = nullptr,
        double Ef = 0.0,
        int kpt_start = 0,
        const MPIComm* kptcomm = nullptr,
        const MPIComm* spincomm = nullptr,
        int Nspin_global = 0,
        const MPIComm* bandcomm = nullptr,
        const double* tau = nullptr,      // mGGA kinetic energy density
        const double* vtau = nullptr);    // mGGA d(nε)/dτ potential
};

} // namespace lynx
