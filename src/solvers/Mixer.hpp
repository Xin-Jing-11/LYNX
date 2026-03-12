#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "operators/Laplacian.hpp"
#include "parallel/HaloExchange.hpp"
#include "solvers/LinearSolver.hpp"

#include <vector>

namespace sparc {

// Anderson/Pulay mixing with optional Kerker preconditioner.
// Matches reference SPARC Mixing_periodic_pulay exactly.
class Mixer {
public:
    Mixer() = default;

    void setup(int Nd_d,
               MixingVariable var,
               MixingPrecond precond_type,
               int history_depth,
               double mixing_param,
               const Laplacian* laplacian = nullptr,
               const HaloExchange* halo = nullptr,
               const FDGrid* grid = nullptr);  // no domain comm needed

    // Mix density: x_k is the current input, g_k is the output from SCF.
    // After mixing, x_k is updated in-place with the new mixed density.
    // ncol: number of density columns (1 for non-spin, 3 for collinear spin [total,up,down])
    // Reference: Mixing_periodic_pulay
    void mix(double* x_k_inout, const double* g_k, int Nd_d, int ncol = 1);

    // Reset history (e.g., at start of new SCF)
    void reset();

private:
    int Nd_d_ = 0;
    int Nd_ = 0;           // global grid size (for renormalization)
    MixingVariable var_ = MixingVariable::Density;
    MixingPrecond precond_type_ = MixingPrecond::None;
    int m_ = 7;            // history depth
    double beta_ = 0.3;    // mixing parameter (Pulay)
    int iter_ = 0;

    // State from previous iteration
    std::vector<double> x_km1_;    // x_{k-1}
    std::vector<double> f_k_;      // current residual f_k = g_k - x_k
    std::vector<double> f_km1_;    // previous residual f_{k-1}

    // History matrices (column-major, Nd_d x m)
    std::vector<double> R_;    // [x_k - x_{k-1}] differences
    std::vector<double> F_;    // [f_k - f_{k-1}] differences

    // Kerker preconditioner components
    const Laplacian* laplacian_ = nullptr;
    const HaloExchange* halo_ = nullptr;
    const FDGrid* grid_ = nullptr;
    double precond_tol_ = 1e-4;  // TOL_PRECOND: default h_eff^2 * 1e-3

    // Apply Kerker preconditioner: solve -(Lap - kTF²)*Pf = (Lap - idiemac*kTF²)*f
    // Then Pf *= -amix
    void apply_kerker(const double* f, double amix, double* Pf) const;
};

} // namespace sparc
