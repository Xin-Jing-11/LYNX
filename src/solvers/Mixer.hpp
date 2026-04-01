#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "solvers/Preconditioner.hpp"

#include <vector>
#include <functional>

namespace lynx {

// Anderson/Pulay mixing with pluggable preconditioner.
// Matches reference LYNX Mixing_periodic_pulay exactly.
class Mixer {
public:
    Mixer() = default;

    void setup(int Nd_d,
               MixingVariable var,
               MixingPrecond precond_type,
               int history_depth,
               double mixing_param,
               Preconditioner* preconditioner = nullptr);

    // Set density constraint: after mixing, clamp negative densities and
    // renormalize to Nelectron.  Only used when mixing_var == Density.
    // Nd_global: total grid points (for MPI-global renormalization)
    // dV: volume element per grid point
    void set_density_constraint(int Nelectron, int Nd_global, double dV);

    // Set potential mean-shift mode: remove per-spin mean before mixing,
    // restore after.  Only used when mixing_var == Potential.
    // Nd_global: total grid points (for mean computation)
    void set_potential_mean_shift(int Nd_global);

    // Mix density: x_k is the current input, g_k is the output from SCF.
    // After mixing, x_k is updated in-place with the new mixed density.
    // ncol: number of density columns (1 for non-spin, 3 for collinear spin [total,up,down])
    // Reference: Mixing_periodic_pulay
    void mix(double* x_k_inout, const double* g_k, int Nd_d, int ncol = 1);

    // Reset history (e.g., at start of new SCF)
    void reset();

    // Access saved Veff mean (per spin) for potential mixing.
    // Only valid after mix() when potential mean-shift is enabled.
    const std::vector<double>& veff_mean() const { return veff_mean_; }

private:
    int Nd_d_ = 0;
    int Nd_ = 0;           // global grid size (for renormalization / mean)
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

    // Pluggable preconditioner (externally owned, may be null)
    Preconditioner* preconditioner_ = nullptr;

    // Density post-processing
    bool density_constraint_ = false;
    int Nelectron_ = 0;
    double dV_ = 0.0;

    // Potential mean-shift
    bool potential_mean_shift_ = false;
    std::vector<double> veff_mean_;  // per-spin mean values

    // Apply density clamping + renormalization in-place.
    // x points to ncol columns of Nd_d_ each.
    // For ncol==1: clamp total density, renormalize.
    // For ncol==2: unpack [total,mag] -> [up,dn], clamp, repack, renormalize.
    // For ncol==4: clamp col 0, renormalize col 0 (SOC magnetization untouched).
    void apply_density_constraint(double* x, int ncol);

    // Remove per-column mean from x (ncol columns), save means in veff_mean_.
    void remove_mean(double* x, int ncol);

    // Restore per-column mean to x (ncol columns) from veff_mean_.
    void restore_mean(double* x, int ncol);
};

} // namespace lynx
