#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "operators/Laplacian.hpp"
#include "operators/FDStencil.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"
#include "solvers/LinearSolver.hpp"

namespace sparc {

// Solves the Poisson equation: -Lap(phi) = 4*pi*(rho + b)
// where b is the pseudocharge density.
// Uses AAR iterative solver with Jacobi preconditioner.
class PoissonSolver {
public:
    PoissonSolver() = default;

    void setup(const Laplacian& laplacian,
               const FDStencil& stencil,
               const Domain& domain,
               const FDGrid& grid,
               const HaloExchange& halo,
               const MPIComm& dmcomm);

    // Solve: -Lap(phi) = rhs
    // rhs: (Nd_d,) right-hand side = 4*pi*(rho + b)
    // phi: (Nd_d,) output electrostatic potential
    // Returns number of iterations
    int solve(const double* rhs, double* phi, double tol = 1e-8) const;

    // Set AAR parameters
    void set_aar_params(const AARParams& params) { aar_params_ = params; }

private:
    const Laplacian* laplacian_ = nullptr;
    const FDStencil* stencil_ = nullptr;
    const Domain* domain_ = nullptr;
    const FDGrid* grid_ = nullptr;
    const HaloExchange* halo_ = nullptr;
    const MPIComm* dmcomm_ = nullptr;

    AARParams aar_params_;
    double jacobi_weight_ = 0.0;  // Jacobi preconditioner weight
};

} // namespace sparc
