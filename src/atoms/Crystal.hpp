#pragma once

#include <vector>
#include "core/types.hpp"
#include "core/Lattice.hpp"
#include "core/Domain.hpp"
#include "atoms/AtomType.hpp"

namespace sparc {

// Influence of an atom type on a local domain — atoms whose rc-sphere
// overlaps with this process's domain (includes periodic images)
struct AtomInfluence {
    int n_atom = 0;                  // number of influencing atoms
    std::vector<Vec3> coords;        // Cartesian coordinates
    std::vector<int> atom_index;     // original atom index (before imaging)
    // Overlap region corners (in local domain indices)
    std::vector<int> xs, xe, ys, ye, zs, ze;
};

// Nonlocal influence: similar but with rc-based spherical region
struct AtomNlocInfluence {
    int n_atom = 0;
    std::vector<Vec3> coords;
    std::vector<int> atom_index;
    std::vector<int> xs, xe, ys, ye, zs, ze;
    std::vector<int> ndc;                    // grid points in rc-domain per atom
    std::vector<std::vector<int>> grid_pos;  // local grid positions in rc-sphere
};

// Stores all atoms in the simulation: positions, types, and influence info
class Crystal {
public:
    Crystal() = default;
    Crystal(std::vector<AtomType> types, std::vector<Vec3> positions,
            std::vector<int> type_indices, const Lattice& lattice);

    int n_atom_total() const { return static_cast<int>(positions_.size()); }
    int n_types() const { return static_cast<int>(types_.size()); }

    const std::vector<AtomType>& types() const { return types_; }
    std::vector<AtomType>& types() { return types_; }
    const std::vector<Vec3>& positions() const { return positions_; }
    const std::vector<int>& type_indices() const { return type_indices_; }

    double total_Zval() const;

    // Move atoms to [0, L) for periodic directions
    void wrap_positions();

    // Find all atom images within rc of the domain (for local/nonlocal influence)
    void compute_atom_influence(const Domain& domain, double rc_max,
                                std::vector<AtomInfluence>& influence) const;

    // Find all atom images within nonlocal rc of the domain
    // Also compute spherical grid_pos for each influencing atom
    void compute_nloc_influence(const Domain& domain,
                                std::vector<AtomNlocInfluence>& nloc_influence) const;

private:
    std::vector<AtomType> types_;
    std::vector<Vec3> positions_;         // Cartesian positions
    std::vector<int> type_indices_;       // type_indices_[iatom] = which AtomType
    const Lattice* lattice_ = nullptr;
};

} // namespace sparc
