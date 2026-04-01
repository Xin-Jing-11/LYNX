#pragma once

#include "core/types.hpp"

namespace lynx {

struct SystemConfig;  // forward declaration (io/InputParser.hpp)
class FDGrid;

namespace ParameterDefaults {

/// Effective mesh spacing from grid spacings dx, dy, dz.
/// If all three are equal, returns dx; otherwise harmonic mean.
double compute_h_eff(double dx, double dy, double dz);

/// Chebyshev polynomial degree from effective mesh spacing.
/// Uses polynomial fit: p3=-700/3, p2=1240/3, p1=-773/3, p0=1078/15.
/// For h_eff > 0.7, returns 14.
int compute_cheb_degree(double h_eff);

/// Electronic temperature (Kelvin) from smearing type.
/// Gaussian -> 0.2 eV, Fermi-Dirac -> 0.1 eV, converted to Kelvin.
double compute_elec_temp(SmearingType smearing);

/// Poisson solver tolerance from SCF tolerance: tol * 0.01.
double compute_poisson_tol(double scf_tol);

/// Kerker preconditioner tolerance: h_eff^2 * 1e-3.
double compute_precond_tol(double h_eff);

/// Number of Kohn-Sham states from electron count and spin/SOC flags.
int compute_nstates(int Nelectron, bool is_spin, bool is_soc);

/// Resolve ALL auto-default parameters in SystemConfig using the grid.
/// Must be called ONCE from main.cpp after parsing and grid creation.
/// After this call, all parameters (elec_temp, cheb_degree, poisson_tol,
/// precond_tol, Nstates) are guaranteed valid — no sentinel values remain.
void update_default(SystemConfig& config, const FDGrid& grid,
                 int Nelectron, bool is_spin, bool is_soc);

}  // namespace ParameterDefaults
}  // namespace lynx
