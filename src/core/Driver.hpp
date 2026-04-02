#pragma once

#include "core/LynxContext.hpp"
#include "io/InputParser.hpp"
#include "atoms/Crystal.hpp"
#include "atoms/AtomType.hpp"
#include "operators/Hamiltonian.hpp"
#include "operators/NonlocalProjector.hpp"
#include "electronic/Wavefunction.hpp"
#include "physics/SCF.hpp"
#include "physics/Electrostatics.hpp"

#include <vector>

namespace lynx {

/// Result of atom/electrostatics setup.
struct AtomSetup {
    Crystal crystal;
    Electrostatics elec;
    std::vector<double> Vloc;
    std::vector<double> rho_core;
    std::vector<AtomInfluence> influence;
    std::vector<AtomNlocInfluence> nloc_influence;
    bool has_nlcc = false;
    int Nelectron = 0;
    int Natom = 0;
};

/// Result of operator setup.
struct OperatorSetup {
    Hamiltonian hamiltonian;
    NonlocalProjector vnl;
};

/// Result of an SCF calculation.
struct SCFResult {
    Wavefunction wfn;
    SCF scf;
};

/// High-level driver functions that orchestrate LYNX calculations.
/// These were extracted from main.cpp to keep it concise.
class Driver {
public:
    /// Load atoms, create Crystal, compute electrostatics (pseudocharge, Vloc, NLCC).
    static AtomSetup setup_atoms(SystemConfig& config, const LynxContext& ctx);

    /// Setup Hamiltonian and NonlocalProjector.
    static OperatorSetup setup_operators(const SystemConfig& config,
                                         const LynxContext& ctx,
                                         const Crystal& crystal,
                                         const std::vector<AtomNlocInfluence>& nloc_influence);

    /// Initialize density and run SCF to convergence.
    static SCFResult run_scf(const SystemConfig& config,
                              const LynxContext& ctx,
                              const Crystal& crystal,
                              AtomSetup& atoms,
                              const Hamiltonian& hamiltonian,
                              const NonlocalProjector& vnl);
};

} // namespace lynx
