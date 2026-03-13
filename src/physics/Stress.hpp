#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "core/KPoints.hpp"
#include "atoms/Crystal.hpp"
#include "operators/FDStencil.hpp"
#include "operators/Gradient.hpp"
#include "operators/NonlocalProjector.hpp"
#include "electronic/Wavefunction.hpp"
#include "xc/XCFunctional.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"

#include <vector>
#include <array>

namespace sparc {

// Stress tensor calculation matching reference SPARC.
// Components: kinetic, XC, electrostatic (local), nonlocal.
// Stored as 6-component Voigt notation:
//   [σ_xx, σ_xy, σ_xz, σ_yy, σ_yz, σ_zz] (indices 0-5)
//
// Total: stress = stress_k + stress_xc + stress_nl + stress_el
// All components normalized by cell_measure (volume for 3D periodic).
class Stress {
public:
    Stress() = default;

    // Compute full stress tensor. Returns 6-component Voigt stress in Ha/Bohr³.
    std::array<double, 6> compute(
        const Wavefunction& wfn,
        const Crystal& crystal,
        const std::vector<AtomInfluence>& influence,
        const std::vector<AtomNlocInfluence>& nloc_influence,
        const NonlocalProjector& vnl,
        const FDStencil& stencil,
        const Gradient& gradient,
        const HaloExchange& halo,
        const Domain& domain,
        const FDGrid& grid,
        const double* phi,           // electrostatic potential
        const double* rho,           // total electron density
        const double* rho_up,        // spin-up density (nullptr for non-spin)
        const double* rho_dn,        // spin-down density (nullptr for non-spin)
        const double* Vloc,          // correction potential Vc
        const double* b,             // pseudocharge density
        const double* b_ref,         // reference pseudocharge density
        const double* exc,           // XC energy density
        const double* Vxc,           // XC potential (Nspin*Nd_d for spin)
        const double* Dxcdgrho,      // GGA: v2x+v2c (1*Nd_d non-spin, 3*Nd_d spin)
        double Exc,                  // total XC energy
        double Esc,                  // self + correction energy (Eself + Ec)
        XCType xc_type,
        int Nspin,                   // 1 or 2
        const double* rho_core,      // NLCC core density (nullptr if no NLCC)
        const std::vector<double>& kpt_weights,
        const MPIComm& bandcomm,
        const MPIComm& kptcomm,
        const MPIComm& spincomm,
        const KPoints* kpoints = nullptr,
        int kpt_start = 0,
        int band_start = 0);

    double pressure() const;

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

    // XC stress: diagonal = (Exc - Exc_corr), GGA correction = -∫ Dxcdgrho * ∂ρ/∂x_α * ∂ρ/∂x_β dV
    void compute_xc_stress(
        const double* rho,
        const double* rho_up,
        const double* rho_dn,
        const double* exc,
        const double* Vxc,
        const double* Dxcdgrho,
        double Exc,
        XCType xc_type,
        int Nspin,
        const double* rho_core,
        const Gradient& gradient,
        const HaloExchange& halo,
        const Domain& domain,
        const FDGrid& grid);

    // Electrostatic stress: uses ∇φ, ∇bJ, ∇VJ, etc. (matching reference)
    void compute_electrostatic(
        const Crystal& crystal,
        const std::vector<AtomInfluence>& influence,
        const FDStencil& stencil,
        const Gradient& gradient,
        const HaloExchange& halo,
        const Domain& domain,
        const FDGrid& grid,
        const double* phi,
        const double* rho,
        const double* Vloc,
        const double* b,
        const double* b_ref,
        double Esc);

    // NLCC XC stress correction: ∫ ∇(ρ_core_J) · (x-R_J) · Vxc dV
    void compute_xc_nlcc_stress(
        const Crystal& crystal,
        const std::vector<AtomInfluence>& influence,
        const FDStencil& stencil,
        const Domain& domain,
        const FDGrid& grid,
        const double* Vxc,
        int Nspin);

    // Nonlocal+kinetic stress combined (matching reference)
    void compute_nonlocal_kinetic(
        const Wavefunction& wfn,
        const Crystal& crystal,
        const std::vector<AtomNlocInfluence>& nloc_influence,
        const NonlocalProjector& vnl,
        const Gradient& gradient,
        const HaloExchange& halo,
        const Domain& domain,
        const FDGrid& grid,
        const std::vector<double>& kpt_weights,
        const MPIComm& bandcomm,
        const MPIComm& kptcomm,
        const MPIComm& spincomm,
        const KPoints* kpoints = nullptr,
        int kpt_start = 0,
        int band_start = 0);
};

} // namespace sparc
