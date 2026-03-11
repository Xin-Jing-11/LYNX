#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "atoms/Crystal.hpp"
#include "operators/Gradient.hpp"
#include "operators/NonlocalProjector.hpp"
#include "electronic/Wavefunction.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"

#include <vector>

namespace sparc {

// Force calculation for Kohn-Sham DFT.
// Two main contributions:
//   F_local  = -∫ ρ(r) ∇V_J^loc(r-R_J) dV  (Hellmann-Feynman local)
//   F_nloc   = -2·occfac·Σ_n f_n Γ <χ|ψ><χ|∇ψ>  (nonlocal KB projectors)
class Forces {
public:
    Forces() = default;

    // Compute all forces. Returns total forces [n_atom * 3].
    std::vector<double> compute(
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
        const std::vector<double>& kpt_weights,
        const MPIComm& dmcomm,
        const MPIComm& bandcomm,
        const MPIComm& kptcomm,
        const MPIComm& spincomm);

    const std::vector<double>& local_forces() const { return f_local_; }
    const std::vector<double>& nonlocal_forces() const { return f_nloc_; }
    const std::vector<double>& total_forces() const { return f_total_; }

private:
    std::vector<double> f_local_;
    std::vector<double> f_nloc_;
    std::vector<double> f_total_;

    // Local force via Hellmann-Feynman: F_J = -∫ ρ(r) ∇V_J^loc dV
    void compute_local(
        const Crystal& crystal,
        const Domain& domain,
        const FDGrid& grid,
        const double* rho,
        const MPIComm& dmcomm);

    // Nonlocal force: F_J = -2·occfac·Σ_n f_n Γ <χ|ψ><χ|∇ψ>
    void compute_nonlocal(
        const Wavefunction& wfn,
        const Crystal& crystal,
        const std::vector<AtomNlocInfluence>& nloc_influence,
        const Gradient& gradient,
        const HaloExchange& halo,
        const Domain& domain,
        const FDGrid& grid,
        const std::vector<double>& kpt_weights,
        const MPIComm& dmcomm,
        const MPIComm& bandcomm,
        const MPIComm& kptcomm,
        const MPIComm& spincomm);

    // Compute <χ_J|x> for all projectors of all atoms
    static void compute_chi_x(
        const Crystal& crystal,
        const std::vector<AtomNlocInfluence>& nloc_influence,
        const Domain& domain,
        const FDGrid& grid,
        const double* x,
        double dV,
        std::vector<double>& result,
        const MPIComm& dmcomm);

    // Real spherical harmonics (same convention as NonlocalProjector)
    static double Ylm(int l, int m, double x, double y, double z, double r);

    // Symmetrize forces: subtract mean to ensure sum = 0
    static void symmetrize(std::vector<double>& forces, int n_atom);
};

} // namespace sparc
