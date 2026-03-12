#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "atoms/Crystal.hpp"
#include "operators/FDStencil.hpp"

#include <vector>

namespace sparc {

// Electrostatics: pseudocharge density, self energy, correction energy,
// local potential on the grid.
class Electrostatics {
public:
    Electrostatics() = default;

    // Compute pseudocharge density b(r) = Σ_J b_J(r) on the local domain.
    // Also computes Eself and Ec.
    void compute_pseudocharge(
        const Crystal& crystal,
        const std::vector<AtomInfluence>& influence,
        const Domain& domain,
        const FDGrid& grid,
        const FDStencil& stencil);

    // Access results
    const NDArray<double>& pseudocharge() const { return b_; }
    double Eself_Ec() const { return Eself_Ec_; }
    double Eself() const { return Eself_; }
    double Ec() const { return Ec_; }
    double int_b() const { return int_b_; }

    // Compute local potential V_loc(r) = Σ_J (V_J - V_ref)(r) on the domain
    // This is the correction potential Vc in the reference code.
    void compute_Vloc(
        const Crystal& crystal,
        const std::vector<AtomInfluence>& influence,
        const Domain& domain,
        const FDGrid& grid,
        double* Vloc);

    // Compute correction energy Ec = 0.5 * ∫(b + b_ref) * Vc * dV
    // Must be called after compute_pseudocharge and compute_Vloc.
    void compute_Ec(const double* Vloc, int Nd_d, double dV);

    // Compute initial atomic density (superposition of atomic densities)
    void compute_atomic_density(
        const Crystal& crystal,
        const std::vector<AtomInfluence>& influence,
        const Domain& domain,
        const FDGrid& grid,
        double* rho_at,
        int Nelectron);

    const NDArray<double>& pseudocharge_ref() const { return b_ref_; }

    // Compute NLCC core charge density on the grid
    // rho_core: output array of size Nd_d, stores interpolated core charge
    // Returns true if any atom type has NLCC (fchrg > 0)
    bool compute_core_density(
        const Crystal& crystal,
        const std::vector<AtomInfluence>& influence,
        const Domain& domain,
        const FDGrid& grid,
        double* rho_core);

    // Reference potential: smooth version of -Z/r (no singularity)
    static double V_ref(double r, double rc, double Znucl);

    // Compute Laplacian of a 3D function on a small grid using FD stencil
    static void calc_lapV(
        const double* V,
        double* lapV,
        int nx, int ny, int nz,
        int nxp, int nyp, int nzp,
        int FDn,
        const double* D2_x, const double* D2_y, const double* D2_z,
        double coef);

    // Non-orthogonal version with mixed derivatives
    static void calc_lapV_nonorth(
        const double* V, double* lapV,
        int nx, int ny, int nz,
        int nxp, int nyp, int nzp,
        int FDn,
        const FDStencil& stencil,
        double coef);

private:
    NDArray<double> b_;      // total pseudocharge density
    NDArray<double> b_ref_;  // reference pseudocharge density (from -Z/r)
    double Eself_Ec_ = 0.0;  // combined self + correction energy
    double Eself_ = 0.0;
    double Ec_ = 0.0;
    double int_b_ = 0.0;     // integral of b (should equal -total_Z)
};

} // namespace sparc
