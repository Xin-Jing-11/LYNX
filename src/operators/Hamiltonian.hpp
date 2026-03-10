#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "operators/FDStencil.hpp"
#include "operators/NonlocalProjector.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"

namespace sparc {

// Hamiltonian operator: H*psi = -0.5*Lap*psi + Veff*psi + Vnl*psi
//
// Takes raw pointers for hot-path compatibility (CUDA-ready).
// All data must be on the local domain.
class Hamiltonian {
public:
    Hamiltonian() = default;

    // Setup with all required components
    void setup(const FDStencil& stencil,
               const Domain& domain,
               const FDGrid& grid,
               const HaloExchange& halo,
               const NonlocalProjector* vnl,  // may be null
               const MPIComm& dmcomm);

    // Apply H*psi = y
    // psi: (Nd_d, ncol) column-major
    // Veff: (Nd_d,) effective potential on local domain
    // y: (Nd_d, ncol) output
    void apply(const double* psi, const double* Veff, double* y,
               int ncol, double c = 0.0) const;

    // Apply only kinetic + Veff (no nonlocal) — for Poisson solver etc.
    void apply_local(const double* psi, const double* Veff, double* y,
                     int ncol, double c = 0.0) const;

    const FDStencil& stencil() const { return *stencil_; }
    const Domain& domain() const { return *domain_; }

private:
    const FDStencil* stencil_ = nullptr;
    const Domain* domain_ = nullptr;
    const FDGrid* grid_ = nullptr;
    const HaloExchange* halo_ = nullptr;
    const NonlocalProjector* vnl_ = nullptr;
    const MPIComm* dmcomm_ = nullptr;

    // Apply -0.5*Lap*psi + Veff*psi + c*psi (orthogonal cell)
    void lap_plus_diag_orth(const double* x_ex, const double* Veff,
                            double* y, int ncol, double c) const;
};

} // namespace sparc
