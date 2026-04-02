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
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"
#include "core/LynxContext.hpp"

#include <vector>

namespace lynx {

// Force calculation matching reference LYNX.
// Three contributions:
//   F_local  = -∫ b_J · ∇φ dV + 0.5 * correction  (electrostatic + pseudocharge)
//   F_nloc   = -occfac·2·Σ g_n Γ <χ|ψ><χ|∇ψ>      (nonlocal KB projectors)
//   F_xc     = ∫ Vxc · ∇ρ_core_J dV                (NLCC correction)
class Forces {
public:
    Forces() = default;

    /// Simplified compute using LynxContext for infrastructure.
    std::vector<double> compute(
        const LynxContext& ctx,
        const Wavefunction& wfn,
        const Crystal& crystal,
        const std::vector<AtomInfluence>& influence,
        const std::vector<AtomNlocInfluence>& nloc_influence,
        const NonlocalProjector& vnl,
        const double* phi,
        const double* rho,
        const double* Vloc,
        const double* b,
        const double* b_ref,
        const double* Vxc,
        const double* rho_core);

    /// Legacy compute with explicit infrastructure parameters.
    std::vector<double> compute(
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
        const double* rho,           // electron density
        const double* Vloc,          // correction potential Vc = V_ref - V_J
        const double* b,             // pseudocharge density
        const double* b_ref,         // reference pseudocharge density
        const double* Vxc,           // XC potential (for NLCC force)
        const double* rho_core,      // NLCC core density (nullptr if no NLCC)
        const std::vector<double>& kpt_weights,
        const MPIComm& bandcomm,
        const MPIComm& kptcomm,
        const MPIComm& spincomm,
        const KPoints* kpoints = nullptr,
        int kpt_start = 0,
        int band_start = 0);

    const std::vector<double>& local_forces() const { return f_local_; }
    const std::vector<double>& nonlocal_forces() const { return f_nloc_; }
    const std::vector<double>& soc_forces() const { return f_soc_; }
    const std::vector<double>& xc_forces() const { return f_xc_; }
    const std::vector<double>& total_forces() const { return f_total_; }

private:
    std::vector<double> f_local_;
    std::vector<double> f_nloc_;
    std::vector<double> f_soc_;
    std::vector<double> f_xc_;
    std::vector<double> f_total_;

    // Local force: F = -∫ bJ · ∇φ dV + 0.5 * ∫ [∇VcJ·(b+b_ref) - ∇Vc·(bJ+bJ_ref)] dV
    void compute_local(
        const Crystal& crystal,
        const std::vector<AtomInfluence>& influence,
        const FDStencil& stencil,
        const Gradient& gradient,
        const HaloExchange& halo,
        const Domain& domain,
        const FDGrid& grid,
        const double* phi,
        const double* Vloc,
        const double* b,
        const double* b_ref);

    // Nonlocal force: F = -occfac·2·Σ g_n Γ <χ|ψ><χ|∇ψ>
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
        const MPIComm& bandcomm,
        const MPIComm& kptcomm,
        const MPIComm& spincomm,
        const KPoints* kpoints = nullptr,
        int kpt_start = 0,
        int band_start = 0);

public:
    // SOC nonlocal force from spin-orbit coupling projectors
    void compute_nonlocal_soc(
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
        const KPoints* kpoints = nullptr,
        int kpt_start = 0,
        int band_start = 0);

    // NLCC XC force: F = ∫ Vxc · ∇ρ_core_J dV
    void compute_xc_nlcc(
        const Crystal& crystal,
        const std::vector<AtomInfluence>& influence,
        const FDStencil& stencil,
        const Domain& domain,
        const FDGrid& grid,
        const double* Vxc);

    // Symmetrize forces: subtract mean to ensure sum = 0
    static void symmetrize(std::vector<double>& forces, int n_atom);
};

} // namespace lynx
