#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "electronic/Wavefunction.hpp"
#include "electronic/ElectronDensity.hpp"
#include "xc/XCFunctional.hpp"

#include <vector>

namespace sparc {

// Energy components of the Kohn-Sham DFT calculation.
struct EnergyComponents {
    double Eband = 0.0;      // Band structure energy: sum f_n * epsilon_n
    double Exc = 0.0;        // Exchange-correlation energy
    double Ehart = 0.0;      // Hartree (electrostatic) energy
    double Eself = 0.0;      // Self energy of pseudocharges
    double Ec = 0.0;         // Correction energy (overlap of b)
    double Entropy = 0.0;    // Electronic entropy contribution (-T*S)
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

    // Compute total energy from components
    static double total_energy(const EnergyComponents& E);

    // Compute all energy components
    // Nspin_global: global spin count (for correct spin_fac and E2)
    // bandcomm: for band-parallel allreduce of Eband (optional, nullptr for serial)
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
        const MPIComm* bandcomm = nullptr);
};

} // namespace sparc
