#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "operators/Laplacian.hpp"
#include "operators/Gradient.hpp"
#include "operators/FDStencil.hpp"
#include "operators/Hamiltonian.hpp"
#include "parallel/HaloExchange.hpp"
#include "electronic/ElectronDensity.hpp"
#include "xc/XCFunctional.hpp"
#include "solvers/PoissonSolver.hpp"

namespace lynx {

class ExactExchange;

// Work arrays produced by EffectivePotential computation.
// These are inputs/outputs that persist across SCF iterations.
struct VeffArrays {
    NDArray<double> Veff;       // effective potential (Nd_d * Nspin)
    NDArray<double> Vxc;        // XC potential (Nd_d * Nspin)
    NDArray<double> exc;        // XC energy density (Nd_d)
    NDArray<double> phi;        // electrostatic potential (Nd_d)
    NDArray<double> Dxcdgrho;   // GGA: dExc/d(|grad rho|^2) (Nd_d * dxc_ncol)
    NDArray<double> vtau;       // d(n*exc)/d(tau) (mGGA) (Nd_d or 2*Nd_d for spin)
    NDArray<double> Veff_spinor; // spinor Veff [V_uu|V_dd|Re(V_ud)|Im(V_ud)] (4*Nd_d)

    // Allocate arrays for given system parameters
    void allocate(int Nd_d, int Nspin, XCType xc_type, bool is_soc);
};

// Builds the effective potential from electron density.
// Handles XC evaluation, Poisson solve, NLCC, mGGA fallback, and spinor variants.
class EffectivePotential {
public:
    EffectivePotential() = default;

    void setup(const Domain& domain,
               const FDGrid& grid,
               const FDStencil& stencil,
               const Laplacian& laplacian,
               const Gradient& gradient,
               const Hamiltonian& hamiltonian,
               const HaloExchange& halo,
               int Nspin_global);

    // Compute Veff for standard (scalar) case.
    // density: electron density (with spin-resolved components)
    // rho_b: pseudocharge density (may be null)
    // rho_core: NLCC core density (may be null)
    // xc_type: XC functional type
    // exx_frac_scale: if > 0, scale exchange by (1 - exx_frac_scale)
    // arrays: work arrays (modified in place)
    void compute(const ElectronDensity& density,
                 const double* rho_b,
                 const double* rho_core,
                 XCType xc_type,
                 double exx_frac_scale,
                 double poisson_tol,
                 VeffArrays& arrays,
                 const double* tau = nullptr,
                 bool tau_valid = false);

    // Compute spinor Veff from noncollinear density (rho, mx, my, mz).
    void compute_spinor(const ElectronDensity& density,
                        const double* rho_b,
                        const double* rho_core,
                        XCType xc_type,
                        double poisson_tol,
                        VeffArrays& arrays);

private:
    const Domain* domain_ = nullptr;
    const FDGrid* grid_ = nullptr;
    const FDStencil* stencil_ = nullptr;
    const Laplacian* laplacian_ = nullptr;
    const Gradient* gradient_ = nullptr;
    const Hamiltonian* hamiltonian_ = nullptr;
    const HaloExchange* halo_ = nullptr;
    int Nspin_global_ = 1;

    // Solve Poisson equation and shift phi to zero mean
    void solve_poisson(const double* rho, const double* rho_b,
                       int Nd_d, double poisson_tol, double* phi);
};

} // namespace lynx
