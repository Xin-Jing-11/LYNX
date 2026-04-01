#include "core/ParameterDefaults.hpp"
#include "core/constants.hpp"
#include "core/FDGrid.hpp"
#include "io/InputParser.hpp"
#include <cmath>
#include <cstdio>

namespace lynx::ParameterDefaults {

double compute_h_eff(double dx, double dy, double dz) {
    if (std::abs(dx - dy) < 1e-12 && std::abs(dy - dz) < 1e-12) {
        return dx;
    }
    double dx2i = 1.0 / (dx * dx);
    double dy2i = 1.0 / (dy * dy);
    double dz2i = 1.0 / (dz * dz);
    return std::sqrt(3.0 / (dx2i + dy2i + dz2i));
}

int compute_cheb_degree(double h_eff) {
    double p3 = -700.0 / 3.0;
    double p2 =  1240.0 / 3.0;
    double p1 = -773.0 / 3.0;
    double p0 =  1078.0 / 15.0;

    double npl;
    if (h_eff > 0.7) {
        npl = 14.0;
    } else {
        npl = ((p3 * h_eff + p2) * h_eff + p1) * h_eff + p0;
    }
    return static_cast<int>(std::round(npl));
}

double compute_elec_temp(SmearingType smearing) {
    double smearing_eV = (smearing == SmearingType::GaussianSmearing) ? 0.2 : 0.1;
    double beta_au = constants::EH / smearing_eV;
    return 1.0 / (constants::KB * beta_au);
}

double compute_poisson_tol(double scf_tol) {
    return scf_tol * 0.01;
}

double compute_precond_tol(double h_eff) {
    return h_eff * h_eff * 1e-3;
}

int compute_nstates(int Nelectron, bool is_spin, bool is_soc) {
    if (is_soc) {
        // SOC: spinor holds both spin components, no /2
        return Nelectron + 20;
    } else {
        return Nelectron / 2 + 10;
    }
}

void update_default(SystemConfig& config, const FDGrid& grid,
                 int Nelectron, bool is_spin, bool is_soc) {
    double h_eff = compute_h_eff(grid.dx(), grid.dy(), grid.dz());

    // Electronic temperature
    if (config.elec_temp < 0.0)
        config.elec_temp = compute_elec_temp(config.smearing);

    // Chebyshev polynomial degree
    if (config.cheb_degree < 0)
        config.cheb_degree = compute_cheb_degree(h_eff);

    // Poisson solver tolerance
    if (config.poisson_tol < 0.0)
        config.poisson_tol = compute_poisson_tol(config.scf_tol);

    // Kerker preconditioner tolerance
    if (config.precond_tol < 0.0)
        config.precond_tol = compute_precond_tol(h_eff);

    // Number of states
    if (config.Nstates <= 0)
        config.Nstates = compute_nstates(Nelectron, is_spin, is_soc);
}

}  // namespace lynx::ParameterDefaults
