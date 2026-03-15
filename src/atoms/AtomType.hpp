#pragma once

#include <string>
#include <memory>
#include "atoms/Pseudopotential.hpp"

namespace lynx {

// Represents one element type in the simulation
class AtomType {
public:
    AtomType() = default;
    AtomType(const std::string& element, double mass, double Zval, int n_atoms);

    const std::string& element() const { return element_; }
    double mass() const { return mass_; }
    double Zval() const { return Zval_; }
    int n_atoms() const { return n_atoms_; }

    Pseudopotential& psd() { return psd_; }
    const Pseudopotential& psd() const { return psd_; }

private:
    std::string element_;
    double mass_ = 0.0;          // atomic mass in amu
    double Zval_ = 0.0;          // valence charge
    int n_atoms_ = 0;            // number of atoms of this type
    Pseudopotential psd_;
};

} // namespace lynx
