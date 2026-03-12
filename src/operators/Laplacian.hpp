#pragma once

#include "FDStencil.hpp"
#include "core/Domain.hpp"
#include <complex>

namespace sparc {

using Complex = std::complex<double>;

// Applies (a*Laplacian + b*diag(V) + c*I) to a vector on a local domain.
// x_ex must be an extended array with ghost nodes already filled.
// Supports both orthogonal and non-orthogonal cells.
// Overloads for both real (Gamma-point) and complex (k-point) data.
class Laplacian {
public:
    Laplacian() = default;
    Laplacian(const FDStencil& stencil, const Domain& domain);

    // Real versions (Gamma-point)
    void apply(const double* x_ex, double* y, double a, double c, int ncol = 1) const;
    void apply_with_diag(const double* x_ex, const double* V, double* y,
                         double a, double b, double c, int ncol = 1) const;

    // Complex versions (k-point) — same stencil, complex data
    void apply(const Complex* x_ex, Complex* y, double a, double c, int ncol = 1) const;
    void apply_with_diag(const Complex* x_ex, const double* V, Complex* y,
                         double a, double b, double c, int ncol = 1) const;

    const FDStencil& stencil() const { return *stencil_; }
    const Domain& domain() const { return *domain_; }

private:
    const FDStencil* stencil_ = nullptr;
    const Domain* domain_ = nullptr;

    // Templated internals to avoid duplicating loop logic
    template<typename T>
    void apply_orth_impl(const T* x_ex, const double* V, T* y,
                         double a, double b, double c, int ncol) const;

    template<typename T>
    void apply_nonorth_impl(const T* x_ex, const double* V, T* y,
                            double a, double b, double c, int ncol) const;

    // Keep old non-template declarations for backward compat (delegate to template)
    void apply_orth(const double* x_ex, const double* V, double* y,
                    double a, double b, double c, int ncol) const;
    void apply_nonorth(const double* x_ex, const double* V, double* y,
                       double a, double b, double c, int ncol) const;
};

} // namespace sparc
