#pragma once

#include "FDStencil.hpp"
#include "core/Domain.hpp"

namespace sparc {

// Applies (a*Laplacian + b*diag(V) + c*I) to a vector on a local domain.
// x_ex must be an extended array with ghost nodes already filled.
// Supports both orthogonal and non-orthogonal cells.
class Laplacian {
public:
    Laplacian() = default;
    Laplacian(const FDStencil& stencil, const Domain& domain);

    // Apply (a*Lap + c*I) * x_ex = y
    // x_ex: extended array with ghosts, size (nx+2*FDn)*(ny+2*FDn)*(nz+2*FDn) per col
    // y: local array, size nx*ny*nz per col
    void apply(const double* x_ex, double* y, double a, double c, int ncol = 1) const;

    // Apply (a*Lap + b*diag(V) + c*I) * x_ex = y
    void apply_with_diag(const double* x_ex, const double* V, double* y,
                         double a, double b, double c, int ncol = 1) const;

    const FDStencil& stencil() const { return *stencil_; }
    const Domain& domain() const { return *domain_; }

private:
    const FDStencil* stencil_ = nullptr;
    const Domain* domain_ = nullptr;

    void apply_orth(const double* x_ex, const double* V, double* y,
                    double a, double b, double c, int ncol) const;

    void apply_nonorth(const double* x_ex, const double* V, double* y,
                       double a, double b, double c, int ncol) const;
};

} // namespace sparc
