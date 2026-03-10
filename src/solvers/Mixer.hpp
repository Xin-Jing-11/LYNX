#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "operators/Laplacian.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"
#include "solvers/LinearSolver.hpp"

#include <vector>

namespace sparc {

// Anderson/Pulay mixing with optional Kerker preconditioner.
// For SCF convergence: mixes density or potential between iterations.
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
               const FDGrid* grid = nullptr,
               const MPIComm* dmcomm = nullptr);

    // Mix: given input x_in (new) and x_out (old output / current guess),
    // produce the next input.
    // f = x_out - x_in is the residual
    // x_in is updated in-place with the mixed result.
    void mix(double* x_in, const double* x_out, int Nd_d);

    // Reset history (e.g., at start of new SCF)
    void reset();

private:
    int Nd_d_ = 0;
    MixingVariable var_ = MixingVariable::Density;
    MixingPrecond precond_type_ = MixingPrecond::None;
    int m_ = 7;            // history depth
    double beta_ = 0.3;    // mixing parameter
    int iter_ = 0;

    // History: ring buffer of residuals and input differences
    std::vector<NDArray<double>> dF_;   // residual differences: f_k - f_{k-1}
    std::vector<NDArray<double>> dX_;   // input differences: x_k - x_{k-1}

    NDArray<double> f_prev_;      // previous residual
    NDArray<double> x_prev_;      // previous input

    // Kerker preconditioner components
    const Laplacian* laplacian_ = nullptr;
    const HaloExchange* halo_ = nullptr;
    const FDGrid* grid_ = nullptr;
    const MPIComm* dmcomm_ = nullptr;

    // Apply Kerker preconditioner: P*r where P ≈ |k|^2 / (|k|^2 + k_TF^2)
    // In real space: solve (-Lap + k_TF^2)*z = -Lap*r
    void apply_kerker(const double* r, double* Pr) const;
};

} // namespace sparc
