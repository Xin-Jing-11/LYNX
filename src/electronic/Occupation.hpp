#pragma once

#include "core/types.hpp"
#include "electronic/Wavefunction.hpp"
#include "parallel/MPIComm.hpp"

#include <vector>

namespace lynx {

// Fermi level determination and occupation calculation.
// Supports Gaussian and Fermi-Dirac smearing.
class Occupation {
public:
    Occupation() = default;

    // Compute Fermi level and occupations from eigenvalues.
    // Nelectron: total number of electrons
    // beta: 1/(kB*T) in atomic units
    // smearing: type of smearing function
    // After this call, wfn.occupations() are updated.
    // Returns the Fermi energy.
    static double compute(Wavefunction& wfn,
                          double Nelectron,
                          double beta,
                          SmearingType smearing,
                          const std::vector<double>& kpt_weights,
                          const MPIComm& kptcomm,
                          const MPIComm& spincomm,
                          int kpt_start = 0);

    // Smearing functions
    static double fermi_dirac(double x, double beta);
    static double gaussian_smearing(double x, double beta);

    // Occupation function dispatch
    static double smearing_function(double x, double beta, SmearingType type);

    // Entropy contribution: -kB*T * sum_n [f_n * ln(f_n) + (1-f_n)*ln(1-f_n)]
    // for Fermi-Dirac, or equivalent for Gaussian
    // Nspin_global: global spin count for correct spin_fac (0 = use wfn.Nspin())
    static double entropy(const Wavefunction& wfn,
                          double beta,
                          SmearingType smearing,
                          const std::vector<double>& kpt_weights,
                          double Ef = 0.0,
                          int kpt_start = 0,
                          int Nspin_global = 0);

private:
    // Find Fermi energy via Brent's method
    static double find_fermi_level(const std::vector<double>& all_eigs,
                                   const std::vector<double>& all_weights,
                                   double Nelectron,
                                   double beta,
                                   SmearingType smearing);

    // Evaluate total occupation for a given Fermi level
    static double total_occupation(const std::vector<double>& eigs,
                                   const std::vector<double>& weights,
                                   double Ef,
                                   double beta,
                                   SmearingType smearing);
};

} // namespace lynx
