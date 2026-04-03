#pragma once

#include "atoms/Crystal.hpp"
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

} // namespace lynx
