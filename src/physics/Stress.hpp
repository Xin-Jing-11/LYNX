#pragma once

#include "core/types.hpp"
#include "core/DeviceArray.hpp"
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
#include "core/LynxContext.hpp"

#include <vector>
#include <array>

namespace lynx {

struct AtomSetup;      // forward declaration (defined in AtomSetup.hpp)
struct SystemConfig;   // forward declaration (defined in InputParser.hpp)
class SCF;             // forward declaration

// Stress tensor calculation matching reference LYNX.
// Components: kinetic, XC, electrostatic (local), nonlocal.
// Stored as 6-component Voigt notation:
//   [σ_xx, σ_xy, σ_xz, σ_yy, σ_yz, σ_zz] (indices 0-5)
//
// Total: stress = stress_k + stress_xc + stress_nl + stress_el
// All components normalized by cell_measure (volume for 3D periodic).
class Stress {
public:
    Stress() = default;

    /// High-level entry point: compute stress (including GPU mGGA and EXX) and print results.
    void compute(const LynxContext& ctx,
                 const SystemConfig& config,
                 const Wavefunction& wfn,
                 SCF& scf,
                 const AtomSetup& atoms,
                 const NonlocalProjector& vnl);

    double pressure() const;

    const std::array<double, 6>& kinetic_stress() const { return stress_k_; }
    const std::array<double, 6>& xc_stress() const { return stress_xc_; }
    const std::array<double, 6>& electrostatic_stress() const { return stress_el_; }
    const std::array<double, 6>& nonlocal_stress() const { return stress_nl_; }
    const std::array<double, 6>& total_stress() const { return stress_total_; }
    const std::array<double, 6>& soc_stress() const { return stress_soc_; }
    double soc_energy() const { return energy_soc_; }
    void set_cell_measure(double cm) { cell_measure_ = cm; }

    // Nonlocal+kinetic stress combined (with LynxContext — preferred).
    void compute_nonlocal_kinetic(
        const LynxContext& ctx,
        const Wavefunction& wfn,
        const Crystal& crystal,
        const std::vector<AtomNlocInfluence>& nloc_influence,
        const NonlocalProjector& vnl,
        const std::vector<double>& kpt_weights);

    // Nonlocal+kinetic stress combined (explicit params — used by GPUSCF).
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

private:
    const LynxContext* ctx_ = nullptr;

    std::array<double, 6> stress_k_ = {};
    std::array<double, 6> stress_xc_ = {};
    std::array<double, 6> stress_el_ = {};
    std::array<double, 6> stress_nl_ = {};
    std::array<double, 6> stress_soc_ = {};
    std::array<double, 6> stress_total_ = {};
    double cell_measure_ = 0.0;
    double energy_soc_ = 0.0;

    /// Detailed compute (implementation -- called by the high-level overload).
    std::array<double, 6> compute_impl(
        const LynxContext& ctx,
        const Wavefunction& wfn,
        const Crystal& crystal,
        const std::vector<AtomInfluence>& influence,
        const std::vector<AtomNlocInfluence>& nloc_influence,
        const NonlocalProjector& vnl,
        const double* phi,
        const double* rho,
        const double* rho_up,
        const double* rho_dn,
        const double* Vloc,
        const double* b,
        const double* b_ref,
        const double* exc,
        const double* Vxc,
        const double* Dxcdgrho,
        double Exc,
        double Esc,
        XCType xc_type,
        int Nspin,
        const double* rho_core,
        const double* vtau = nullptr,
        const double* tau = nullptr,
        const double* gpu_mgga_psi_stress = nullptr,
        const double* gpu_tau_vtau_dot = nullptr);

    /// Print computed stress tensor to stdout (called at end of compute).
    void print(int rank) const;

    void add_to_total(const std::array<double, 6>& extra) {
        for (int i = 0; i < 6; ++i) stress_total_[i] += extra[i];
    }

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
        const double* vtau = nullptr,
        const double* tau = nullptr,
        const double* gpu_tau_vtau_dot = nullptr);

    // Electrostatic stress: uses ∇φ, ∇bJ, ∇VJ, etc. (matching reference)
    void compute_electrostatic(
        const Crystal& crystal,
        const std::vector<AtomInfluence>& influence,
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
        const double* Vxc,
        int Nspin);
};

} // namespace lynx
