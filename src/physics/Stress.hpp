#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "atoms/Crystal.hpp"
#include "operators/Gradient.hpp"
#include "operators/Laplacian.hpp"
#include "operators/NonlocalProjector.hpp"
#include "electronic/Wavefunction.hpp"
#include "xc/XCFunctional.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"

#include <vector>
#include <array>

namespace sparc {

// Stress tensor calculation for single-point DFT.
// Components: kinetic, XC, electrostatic, nonlocal.
// Stored as 6-component Voigt notation:
//   [σ_xx, σ_xy, σ_xz, σ_yy, σ_yz, σ_zz]
class Stress {
public:
    Stress() = default;

    // Compute full stress tensor
    // Returns 6-component stress in Ha/Bohr³
    std::array<double, 6> compute(
        const Wavefunction& wfn,
        const Crystal& crystal,
        const std::vector<AtomNlocInfluence>& nloc_influence,
        const NonlocalProjector& vnl,
        const Gradient& gradient,
        const HaloExchange& halo,
        const Domain& domain,
        const FDGrid& grid,
        const double* phi,
        const double* rho,
        const double* rho_b,
        const double* exc,
        const double* Vxc,
        double Exc,
        XCType xc_type,
        const std::vector<double>& kpt_weights,
        const MPIComm& dmcomm,
        const MPIComm& bandcomm,
        const MPIComm& kptcomm,
        const MPIComm& spincomm);

    // Compute pressure: P = -(1/3) * Tr(σ)
    double pressure() const;

    // Access components
    const std::array<double, 6>& kinetic_stress() const { return stress_k_; }
    const std::array<double, 6>& xc_stress() const { return stress_xc_; }
    const std::array<double, 6>& electrostatic_stress() const { return stress_el_; }
    const std::array<double, 6>& nonlocal_stress() const { return stress_nl_; }
    const std::array<double, 6>& total_stress() const { return stress_total_; }

private:
    std::array<double, 6> stress_k_ = {};
    std::array<double, 6> stress_xc_ = {};
    std::array<double, 6> stress_el_ = {};
    std::array<double, 6> stress_nl_ = {};
    std::array<double, 6> stress_total_ = {};
    double cell_measure_ = 0.0;

    // Kinetic stress: -Σ g_n <∂ψ/∂x_α | ∂ψ/∂x_β>
    void compute_kinetic(
        const Wavefunction& wfn,
        const Gradient& gradient,
        const HaloExchange& halo,
        const Domain& domain,
        const FDGrid& grid,
        const std::vector<double>& kpt_weights,
        const MPIComm& dmcomm,
        const MPIComm& bandcomm,
        const MPIComm& kptcomm,
        const MPIComm& spincomm);

    // XC stress: (Exc)*δ_αβ - GGA gradient correction
    void compute_xc(
        const double* rho,
        const double* exc,
        const double* Vxc,
        double Exc,
        XCType xc_type,
        const Gradient& gradient,
        const HaloExchange& halo,
        const Domain& domain,
        const FDGrid& grid,
        const MPIComm& dmcomm);

    // Electrostatic stress: (1/4π)|∇φ|² terms + correction
    void compute_electrostatic(
        const double* phi,
        const double* rho,
        const double* rho_b,
        const Gradient& gradient,
        const HaloExchange& halo,
        const Domain& domain,
        const FDGrid& grid,
        const MPIComm& dmcomm);

    // Nonlocal stress: Γ * <χ|ψ> * <χ|(x-R_J)·∂ψ/∂x>
    void compute_nonlocal(
        const Wavefunction& wfn,
        const Crystal& crystal,
        const std::vector<AtomNlocInfluence>& nloc_influence,
        const NonlocalProjector& vnl,
        const Gradient& gradient,
        const HaloExchange& halo,
        const Domain& domain,
        const FDGrid& grid,
        const std::vector<double>& kpt_weights,
        const MPIComm& dmcomm,
        const MPIComm& bandcomm,
        const MPIComm& kptcomm,
        const MPIComm& spincomm);

    // Helper: compute <χ|x> for all projectors
    static void compute_chi_x_local(
        const Crystal& crystal,
        const std::vector<AtomNlocInfluence>& nloc_influence,
        const Domain& domain,
        const FDGrid& grid,
        const double* x,
        double dV,
        std::vector<double>& result,
        const MPIComm& dmcomm);

    // Real spherical harmonics
    static double Ylm_stress(int l, int m, double x, double y, double z, double r);
};

} // namespace sparc
