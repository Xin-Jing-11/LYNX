#pragma once

#include "FDStencil.hpp"
#include "core/Domain.hpp"

namespace sparc {

// Computes gradient (first derivative) in a specified direction
class Gradient {
public:
    Gradient() = default;
    Gradient(const FDStencil& stencil, const Domain& domain);

    // direction: 0=x, 1=y, 2=z
    void apply(const double* x, double* y, int direction, int ncol = 1) const;

    const FDStencil& stencil() const { return *stencil_; }
    const Domain& domain() const { return *domain_; }

private:
    const FDStencil* stencil_ = nullptr;
    const Domain* domain_ = nullptr;
};

} // namespace sparc
