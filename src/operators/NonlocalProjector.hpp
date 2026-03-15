#pragma once

#include <vector>
#include <complex>
#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "atoms/Crystal.hpp"

namespace lynx {

using Complex = std::complex<double>;

// Stores precomputed nonlocal projectors Chi for all influencing atoms.
// Chi[ityp][iat] has shape (ndc, nproj) in column-major layout.
//
// The nonlocal operation:
//   alpha = Chi^T * psi * dV       (inner products)
//   MPI_Allreduce(alpha)           (sum over domain)
//   alpha *= Gamma                 (apply KB energy)
//   Hpsi += Chi * alpha            (accumulate)
//
// For k-points: Chi is REAL, Bloch phase factors are applied as scalars
// during matrix operations (matching reference LYNX).
class NonlocalProjector {
public:
    NonlocalProjector() = default;
    NonlocalProjector(const NonlocalProjector&) = delete;
    NonlocalProjector& operator=(const NonlocalProjector&) = delete;
    NonlocalProjector(NonlocalProjector&&) = default;
    NonlocalProjector& operator=(NonlocalProjector&&) = default;

    // Setup: compute projector values on the grid for all influencing atoms
    void setup(const Crystal& crystal,
               const std::vector<AtomNlocInfluence>& nloc_influence,
               const Domain& domain,
               const FDGrid& grid);

    // Apply Vnl to psi: Hpsi += Vnl * psi (real, Gamma-point)
    void apply(const double* psi, double* Hpsi, int ncol, double dV) const;

    // Apply Vnl to complex psi with Bloch phase factors (k-point)
    // kpt_cart: k-point in Cartesian reciprocal coords
    void apply_kpt(const Complex* psi, Complex* Hpsi, int ncol, double dV) const;

    // Set k-point for Bloch phase computation
    void set_kpoint(const Vec3& kpt_cart) { kpt_cart_ = kpt_cart; }

    bool is_setup() const { return is_setup_; }

    // Total number of projectors across all atoms
    int total_nproj() const { return total_nproj_; }

    // Access Chi for diagnostics
    const std::vector<std::vector<NDArray<double>>>& Chi() const { return Chi_; }

private:
    bool is_setup_ = false;
    int total_nproj_ = 0;

    // Chi_[ityp][iat] = NDArray<double> of shape (ndc, nproj)
    // Chi is REAL for both Gamma and k-point calculations
    std::vector<std::vector<NDArray<double>>> Chi_;

    // IP_displ_[ityp][iat+1] = cumulative projector displacement
    std::vector<std::vector<int>> IP_displ_;

    // Gamma coefficients per projector (flattened across all atoms)
    std::vector<double> Gamma_all_;

    // K-point for Bloch phase
    Vec3 kpt_cart_ = {0.0, 0.0, 0.0};

    // Back-references
    const Crystal* crystal_ = nullptr;
    const std::vector<AtomNlocInfluence>* nloc_influence_ = nullptr;
    const Domain* domain_ = nullptr;

};

} // namespace lynx
