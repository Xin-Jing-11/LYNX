#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "operators/FDStencil.hpp"
#include "operators/NonlocalProjector.hpp"
#include "parallel/HaloExchange.hpp"
#include <complex>

namespace sparc {

using Complex = std::complex<double>;

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
               const NonlocalProjector* vnl);  // may be null (Gamma-point)

    // --- Real (Gamma-point) interface ---

    void apply(const double* psi, const double* Veff, double* y,
               int ncol, double c = 0.0) const;

    void apply_local(const double* psi, const double* Veff, double* y,
                     int ncol, double c = 0.0) const;

    // --- Complex (k-point) interface ---

    // Set the nonlocal projector for k-point (complex Chi with Bloch phases)
    void set_vnl_kpt(const NonlocalProjector* vnl_kpt) { vnl_kpt_ = vnl_kpt; }

    // Apply H*psi for a specific k-point
    // kpt_cart: k-point in Cartesian reciprocal coords
    // cell_lengths: (Lx, Ly, Lz)
    void apply_kpt(const Complex* psi, const double* Veff, Complex* y,
                   int ncol, const Vec3& kpt_cart, const Vec3& cell_lengths,
                   double c = 0.0) const;

    void apply_local_kpt(const Complex* psi, const double* Veff, Complex* y,
                         int ncol, const Vec3& kpt_cart, const Vec3& cell_lengths,
                         double c = 0.0) const;

    const FDStencil& stencil() const { return *stencil_; }
    const Domain& domain() const { return *domain_; }

private:
    const FDStencil* stencil_ = nullptr;
    const Domain* domain_ = nullptr;
    const FDGrid* grid_ = nullptr;
    const HaloExchange* halo_ = nullptr;
    const NonlocalProjector* vnl_ = nullptr;      // Gamma-point nonlocal
    const NonlocalProjector* vnl_kpt_ = nullptr;   // k-point nonlocal (complex Chi)

    // Templated stencil application (shared between real and complex)
    template<typename T>
    void lap_plus_diag_orth_impl(const T* x_ex, const double* Veff,
                                  T* y, int ncol, double c) const;
    template<typename T>
    void lap_plus_diag_nonorth_impl(const T* x_ex, const double* Veff,
                                     T* y, int ncol, double c) const;

    // Legacy real wrappers
    void lap_plus_diag_orth(const double* x_ex, const double* Veff,
                            double* y, int ncol, double c) const;
    void lap_plus_diag_nonorth(const double* x_ex, const double* Veff,
                               double* y, int ncol, double c) const;
};

} // namespace sparc
