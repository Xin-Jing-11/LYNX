#pragma once

#include "core/types.hpp"
#include "core/DeviceArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "core/DeviceTag.hpp"
#include "operators/Laplacian.hpp"
#include "operators/FDStencil.hpp"
#include "parallel/HaloExchange.hpp"
#include "solvers/LinearSolver.hpp"

namespace lynx {

// Solves the Poisson equation: -Lap(phi) = 4*pi*(rho + b)
// where b is the pseudocharge density.
// Uses AAR iterative solver with Jacobi preconditioner.
class PoissonSolver {
public:
    PoissonSolver() = default;
    ~PoissonSolver();
    PoissonSolver(PoissonSolver&&) noexcept = default;
    PoissonSolver& operator=(PoissonSolver&&) noexcept = default;
    PoissonSolver(const PoissonSolver&) = delete;
    PoissonSolver& operator=(const PoissonSolver&) = delete;

    void setup(const Laplacian& laplacian,
               const FDStencil& stencil,
               const Domain& domain,
               const FDGrid& grid,
               const HaloExchange& halo);  // no domain comm needed

    // Solve: -Lap(phi) = rhs
    // rhs: (Nd_d,) right-hand side = 4*pi*(rho + b)
    // phi: (Nd_d,) output electrostatic potential
    // Returns number of iterations
    int solve(const double* rhs, double* phi, double tol = 1e-8) const;

    // ── Device-dispatching overload ─────────────────────────────
    int solve(const double* rhs, double* phi, double tol, Device dev) const;

#ifdef USE_CUDA
    void* gpu_state_raw_ = nullptr;  // Opaque pointer to GPUPoissonState (defined in .cu)
public:
    void setup_gpu(const class LynxContext& ctx);
    void cleanup_gpu();
#endif

    // Set AAR parameters
    void set_aar_params(const AARParams& params) { aar_params_ = params; }

private:
    const Laplacian* laplacian_ = nullptr;
    const FDStencil* stencil_ = nullptr;
    const Domain* domain_ = nullptr;
    const FDGrid* grid_ = nullptr;
    const HaloExchange* halo_ = nullptr;

    AARParams aar_params_;
    double jacobi_weight_ = 0.0;  // Jacobi preconditioner weight
};

} // namespace lynx
