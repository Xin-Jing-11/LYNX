#pragma once

#include "core/types.hpp"
#include "core/KPoints.hpp"
#include "operators/Hamiltonian.hpp"
#include "operators/NonlocalProjector.hpp"
#include "electronic/Wavefunction.hpp"
#include "electronic/ElectronDensity.hpp"
#include "electronic/Occupation.hpp"
#include "solvers/EigenSolver.hpp"
#include "solvers/Mixer.hpp"
#include "physics/Energy.hpp"
#include "physics/EffectivePotential.hpp"
#include "physics/SCFInitializer.hpp"  // for SCFState
#include "physics/SCF.hpp"             // for SCFParams
#include "parallel/MPIComm.hpp"

#include <vector>

namespace lynx {

class ExactExchange;

// Runs the outer Fock iteration loop for hybrid DFT (PBE0, HSE06).
// Wraps inner SCF cycles with ACE operator building and EXX energy convergence.
class HybridSCF {
public:
    HybridSCF() = default;

    // Run the Fock outer loop.
    // Modifies: wfn, density, arrays, energy
    // Requires: inner SCF has already converged (standard DFT), exx is set up.
    // Returns: total energy including exact exchange correction.
    void run(Wavefunction& wfn,
             ElectronDensity& density,
             VeffArrays& arrays,
             EffectivePotential& veff_builder,
             EnergyComponents& energy,
             double& Ef,
             bool& converged,
             ExactExchange* exx,
             const Hamiltonian* hamiltonian,
             const NonlocalProjector* vnl,
             EigenSolver& eigsolver,
             Mixer& mixer,
             const SCFParams& params,
             const FDGrid& grid,
             const Domain& domain,
             const MPIComm& bandcomm,
             const MPIComm& kptcomm,
             const MPIComm& spincomm,
             const KPoints* kpoints,
             int Nelectron,
             int Natom,
             int Nspin_global,
             int Nspin_local,
             int spin_start,
             int kpt_start,
             int band_start,
             const double* rho_b,
             const double* rho_core,
             XCType xc_type,
             bool is_kpt,
             double Eself,
             double Ec,
             const std::vector<double>& kpt_weights,
             SCFState& state);

private:
    // Run one inner SCF cycle with EXX active
    void run_inner_scf(Wavefunction& wfn,
                       ElectronDensity& density,
                       VeffArrays& arrays,
                       EffectivePotential& veff_builder,
                       EnergyComponents& energy,
                       double& Ef,
                       bool& converged,
                       double Eexx_est,
                       EigenSolver& eigsolver,
                       Mixer& mixer,
                       const SCFParams& params,
                       const FDGrid& grid,
                       const Domain& domain,
                       const Hamiltonian* hamiltonian,
                       const NonlocalProjector* vnl,
                       const MPIComm& bandcomm,
                       const MPIComm& kptcomm,
                       const MPIComm& spincomm,
                       const KPoints* kpoints,
                       int Nelectron,
                       int Nspin_global,
                       int Nspin_local,
                       int spin_start,
                       int kpt_start,
                       int band_start,
                       const double* rho_b,
                       const double* rho_core,
                       XCType xc_type,
                       bool is_kpt,
                       double Eself,
                       double Ec,
                       const std::vector<double>& kpt_weights,
                       SCFState& state);
};

} // namespace lynx
