#pragma once

#include <vector>
#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "atoms/Crystal.hpp"

namespace sparc {

// Stores precomputed nonlocal projectors Chi for all influencing atoms.
// Chi[ityp][iat] has shape (ndc, nproj) in column-major layout.
//
// The nonlocal operation:
//   alpha = Chi^T * psi * dV       (inner products)
//   MPI_Allreduce(alpha)           (sum over domain)
//   alpha *= Gamma                 (apply KB energy)
//   Hpsi += Chi * alpha            (accumulate)
class NonlocalProjector {
public:
    NonlocalProjector() = default;

    // Setup: compute projector values on the grid for all influencing atoms
    void setup(const Crystal& crystal,
               const std::vector<AtomNlocInfluence>& nloc_influence,
               const Domain& domain,
               const FDGrid& grid);

    // Apply Vnl to psi: Hpsi += Vnl * psi
    // psi: local domain array, shape = (Nd_d, ncol) column-major
    // Hpsi: output, same shape (accumulated into)
    void apply(const double* psi, double* Hpsi, int ncol, double dV) const;

    bool is_setup() const { return is_setup_; }

    // Total number of projectors across all atoms
    int total_nproj() const { return total_nproj_; }

    // Access Chi for diagnostics
    const std::vector<std::vector<NDArray<double>>>& Chi() const { return Chi_; }

private:
    bool is_setup_ = false;
    int total_nproj_ = 0;

    // Chi_[ityp][iat] = NDArray<double> of shape (ndc, nproj)
    std::vector<std::vector<NDArray<double>>> Chi_;

    // IP_displ_[ityp][iat+1] = cumulative projector displacement
    std::vector<std::vector<int>> IP_displ_;

    // Gamma coefficients per projector (flattened across all atoms)
    std::vector<double> Gamma_all_;

    // Back-references
    const Crystal* crystal_ = nullptr;
    const std::vector<AtomNlocInfluence>* nloc_influence_ = nullptr;
    const Domain* domain_ = nullptr;

public:
    // Spherical harmonics (public for use by Forces/Stress)
    static double spherical_harmonic(int l, int m, double x, double y, double z, double r);
};

} // namespace sparc
