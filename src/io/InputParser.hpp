#pragma once

#include <string>
#include <vector>
#include <array>
#include "core/types.hpp"
#include "parallel/Parallelization.hpp"

namespace lynx {

struct SystemConfig {
    // Lattice
    Mat3 latvec;
    CellType cell_type = CellType::Orthogonal;

    // Grid
    int Nx = 0, Ny = 0, Nz = 0;
    double mesh_spacing = 0.0;
    int fd_order = 12;
    BCType bcx = BCType::Periodic, bcy = BCType::Periodic, bcz = BCType::Periodic;

    // Atoms
    struct AtomTypeInput {
        std::string element;
        std::string pseudo_file;
        std::vector<Vec3> coords;
        bool fractional = true;
        std::vector<double> spin;
        std::vector<std::array<int, 3>> relax_constraint;
    };
    std::vector<AtomTypeInput> atom_types;

    // Electronic
    int Nstates = 0;
    SpinType spin_type = SpinType::None;
    double elec_temp = -1.0;         // -1 = auto (resolved by ParameterDefaults::resolve_all)
    SmearingType smearing = SmearingType::GaussianSmearing;
    XCType xc = XCType::GGA_PBE;
    EXXParams exx_params;    // exact exchange params (hybrid functionals)
    int Nelectron = 0;

    // K-points
    int Kx = 1, Ky = 1, Kz = 1;
    Vec3 kpt_shift = {0, 0, 0};

    // SCF
    int max_scf_iter = 100;
    int min_scf_iter = 2;
    double scf_tol = 1e-6;
    MixingVariable mixing_var = MixingVariable::Density;
    MixingPrecond mixing_precond = MixingPrecond::Kerker;
    int mixing_history = 7;
    double mixing_param = 0.3;
    int cheb_degree = -1;            // -1 = auto (resolved by ParameterDefaults::resolve_all)
    int rho_trigger = 4;
    double poisson_tol = -1.0;       // -1 = auto (resolved by ParameterDefaults::resolve_all)
    double precond_tol = -1.0;       // -1 = auto (resolved by ParameterDefaults::resolve_all)

    // Parallelization
    ParallelParams parallel;

    // Output
    bool print_forces = false;
    bool print_atoms = true;
    bool print_eigen = false;
    bool calc_stress = false;
    bool calc_pressure = false;

    // Density I/O (checkpoint/restart)
    std::string density_restart_file;   // read initial density from this file
    std::string density_output_file;    // write converged density to this file

    // MD / Relax
    bool md_flag = false;
    bool relax_flag = false;
    std::string md_method = "NVT_NH";
    std::string relax_method = "LBFGS";
};

class InputParser {
public:
    static SystemConfig parse(const std::string& json_file);
    static void validate(const SystemConfig& config);

    // Auto-resolve missing pseudo_file paths from PseudoDojo submodules.
    // Uses XC type and spin type to select SR or FR pseudopotentials.
    // psps_root: path to the psps/ directory (default: "psps" relative to executable).
    static void resolve_pseudopotentials(SystemConfig& config,
                                          const std::string& psps_root = "psps");
};

} // namespace lynx
