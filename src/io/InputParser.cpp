#include "io/InputParser.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <map>

using json = nlohmann::json;

namespace lynx {

namespace {

CellType parse_cell_type(const std::string& s) {
    if (s == "orthogonal") return CellType::Orthogonal;
    if (s == "nonorthogonal") return CellType::NonOrthogonal;
    if (s == "helical") return CellType::Helical;
    throw std::runtime_error("Unknown cell_type: " + s);
}

BCType parse_bc(const std::string& s) {
    if (s == "periodic" || s == "P") return BCType::Periodic;
    if (s == "dirichlet" || s == "D") return BCType::Dirichlet;
    throw std::runtime_error("Unknown boundary condition: " + s);
}

SpinType parse_spin(const std::string& s) {
    if (s == "none") return SpinType::None;
    if (s == "collinear") return SpinType::Collinear;
    if (s == "noncollinear") return SpinType::NonCollinear;
    throw std::runtime_error("Unknown spin type: " + s);
}

SmearingType parse_smearing(const std::string& s) {
    if (s == "gaussian") return SmearingType::GaussianSmearing;
    if (s == "fermi-dirac" || s == "fd") return SmearingType::FermiDirac;
    throw std::runtime_error("Unknown smearing type: " + s);
}

XCType parse_xc(const std::string& s) {
    if (s == "LDA_PZ") return XCType::LDA_PZ;
    if (s == "LDA_PW") return XCType::LDA_PW;
    if (s == "GGA_PBE") return XCType::GGA_PBE;
    if (s == "GGA_PBEsol") return XCType::GGA_PBEsol;
    if (s == "GGA_RPBE") return XCType::GGA_RPBE;
    if (s == "SCAN") return XCType::MGGA_SCAN;
    if (s == "RSCAN") return XCType::MGGA_RSCAN;
    if (s == "R2SCAN") return XCType::MGGA_R2SCAN;
    if (s == "PBE0" || s == "HYB_PBE0") return XCType::HYB_PBE0;
    if (s == "HSE" || s == "HSE06" || s == "HYB_HSE") return XCType::HYB_HSE;
    throw std::runtime_error("Unknown XC functional: " + s);
}

MixingVariable parse_mixing_var(const std::string& s) {
    if (s == "density") return MixingVariable::Density;
    if (s == "potential") return MixingVariable::Potential;
    throw std::runtime_error("Unknown mixing variable: " + s);
}

MixingPrecond parse_mixing_precond(const std::string& s) {
    if (s == "none") return MixingPrecond::None;
    if (s == "kerker") return MixingPrecond::Kerker;
    throw std::runtime_error("Unknown mixing preconditioner: " + s);
}

} // anonymous namespace

SystemConfig InputParser::parse(const std::string& json_file) {
    std::ifstream ifs(json_file);
    if (!ifs.is_open())
        throw std::runtime_error("Cannot open input file: " + json_file);

    json j;
    ifs >> j;

    SystemConfig config;

    // Lattice
    if (j.contains("lattice")) {
        auto& lat = j["lattice"];
        if (lat.contains("vectors")) {
            auto& vecs = lat["vectors"];
            for (int i = 0; i < 3; ++i)
                for (int k = 0; k < 3; ++k)
                    config.latvec(i, k) = vecs[i][k].get<double>();
        }
        if (lat.contains("cell_type"))
            config.cell_type = parse_cell_type(lat["cell_type"].get<std::string>());
    }

    // Grid
    if (j.contains("grid")) {
        auto& grid = j["grid"];
        if (grid.contains("Nx")) config.Nx = grid["Nx"].get<int>();
        if (grid.contains("Ny")) config.Ny = grid["Ny"].get<int>();
        if (grid.contains("Nz")) config.Nz = grid["Nz"].get<int>();
        if (grid.contains("mesh_spacing")) config.mesh_spacing = grid["mesh_spacing"].get<double>();
        if (grid.contains("fd_order")) config.fd_order = grid["fd_order"].get<int>();
        if (grid.contains("boundary_conditions")) {
            auto& bc = grid["boundary_conditions"];
            config.bcx = parse_bc(bc[0].get<std::string>());
            config.bcy = parse_bc(bc[1].get<std::string>());
            config.bcz = parse_bc(bc[2].get<std::string>());
        }
    }

    // Atoms
    if (j.contains("atoms")) {
        for (auto& atom : j["atoms"]) {
            SystemConfig::AtomTypeInput at;
            at.element = atom["element"].get<std::string>();
            if (atom.contains("pseudo_file"))
                at.pseudo_file = atom["pseudo_file"].get<std::string>();
            if (atom.contains("fractional"))
                at.fractional = atom["fractional"].get<bool>();

            for (auto& coord : atom["coordinates"]) {
                at.coords.push_back({coord[0].get<double>(),
                                     coord[1].get<double>(),
                                     coord[2].get<double>()});
            }

            if (atom.contains("spin")) {
                for (auto& s : atom["spin"])
                    at.spin.push_back(s.get<double>());
            }

            config.atom_types.push_back(std::move(at));
        }
    }

    // Electronic
    if (j.contains("electronic")) {
        auto& elec = j["electronic"];
        if (elec.contains("xc"))
            config.xc = parse_xc(elec["xc"].get<std::string>());
        if (elec.contains("spin"))
            config.spin_type = parse_spin(elec["spin"].get<std::string>());
        if (elec.contains("temperature"))
            config.elec_temp = elec["temperature"].get<double>();
        if (elec.contains("smearing"))
            config.smearing = parse_smearing(elec["smearing"].get<std::string>());
        if (elec.contains("Nstates"))
            config.Nstates = elec["Nstates"].get<int>();
        if (elec.contains("Nelectron"))
            config.Nelectron = elec["Nelectron"].get<int>();

        // EXX parameters for hybrid functionals
        if (elec.contains("exx_frac"))
            config.exx_params.exx_frac = elec["exx_frac"].get<double>();
        if (elec.contains("exx_range_fock"))
            config.exx_params.hyb_range_fock = elec["exx_range_fock"].get<double>();
        if (elec.contains("exx_div_flag"))
            config.exx_params.exx_div_flag = elec["exx_div_flag"].get<int>();
        if (elec.contains("maxit_fock"))
            config.exx_params.maxit_fock = elec["maxit_fock"].get<int>();
        if (elec.contains("tol_fock"))
            config.exx_params.tol_fock = elec["tol_fock"].get<double>();
    }

    // Set default EXX parameters based on XC type
    if (config.xc == XCType::HYB_PBE0) {
        if (config.exx_params.hyb_range_fock < 0)
            config.exx_params.hyb_range_fock = 0.0;  // unscreened
    } else if (config.xc == XCType::HYB_HSE) {
        if (config.exx_params.hyb_range_fock < 0)
            config.exx_params.hyb_range_fock = 0.11;  // HSE06 screening
    }

    // K-points
    if (j.contains("kpoints")) {
        auto& kp = j["kpoints"];
        if (kp.contains("grid")) {
            config.Kx = kp["grid"][0].get<int>();
            config.Ky = kp["grid"][1].get<int>();
            config.Kz = kp["grid"][2].get<int>();
        }
        if (kp.contains("shift")) {
            config.kpt_shift = {kp["shift"][0].get<double>(),
                                kp["shift"][1].get<double>(),
                                kp["shift"][2].get<double>()};
        }
    }

    // SCF
    if (j.contains("scf")) {
        auto& scf = j["scf"];
        if (scf.contains("max_iter")) config.max_scf_iter = scf["max_iter"].get<int>();
        if (scf.contains("min_iter")) config.min_scf_iter = scf["min_iter"].get<int>();
        if (scf.contains("tolerance")) config.scf_tol = scf["tolerance"].get<double>();
        if (scf.contains("mixing"))
            config.mixing_var = parse_mixing_var(scf["mixing"].get<std::string>());
        if (scf.contains("preconditioner"))
            config.mixing_precond = parse_mixing_precond(scf["preconditioner"].get<std::string>());
        if (scf.contains("mixing_history"))
            config.mixing_history = scf["mixing_history"].get<int>();
        if (scf.contains("mixing_parameter"))
            config.mixing_param = scf["mixing_parameter"].get<double>();
        if (scf.contains("cheb_degree"))
            config.cheb_degree = scf["cheb_degree"].get<int>();
        if (scf.contains("rho_trigger"))
            config.rho_trigger = scf["rho_trigger"].get<int>();
    }

    // Parallelization
    if (j.contains("parallel")) {
        auto& par = j["parallel"];
        if (par.contains("npspin")) config.parallel.npspin = par["npspin"].get<int>();
        if (par.contains("npkpt")) config.parallel.npkpt = par["npkpt"].get<int>();
        if (par.contains("npband")) config.parallel.npband = par["npband"].get<int>();
        if (par.contains("num_threads")) config.parallel.num_threads = par["num_threads"].get<int>();
        // npNdx/y/z removed (no domain decomposition)
    }

    // Output
    if (j.contains("output")) {
        auto& out = j["output"];
        if (out.contains("print_forces")) config.print_forces = out["print_forces"].get<bool>();
        if (out.contains("print_atoms")) config.print_atoms = out["print_atoms"].get<bool>();
        if (out.contains("print_eigen")) config.print_eigen = out["print_eigen"].get<bool>();
        if (out.contains("calc_stress")) config.calc_stress = out["calc_stress"].get<bool>();
        if (out.contains("calc_pressure")) config.calc_pressure = out["calc_pressure"].get<bool>();
        if (out.contains("density_restart_file"))
            config.density_restart_file = out["density_restart_file"].get<std::string>();
        if (out.contains("density_output_file"))
            config.density_output_file = out["density_output_file"].get<std::string>();
    }

    // MD / Relax
    if (j.contains("md")) {
        config.md_flag = true;
        if (j["md"].contains("method"))
            config.md_method = j["md"]["method"].get<std::string>();
    }
    if (j.contains("relax")) {
        config.relax_flag = true;
        if (j["relax"].contains("method"))
            config.relax_method = j["relax"]["method"].get<std::string>();
    }

    return config;
}

// ---------------------------------------------------------------------------
// Auto-resolve pseudopotential paths from PseudoDojo submodules.
//
// Directory structure:
//   psps/ONCVPSP-PBE-PDv0.4/standard.txt       (SR, GGA_PBE)
//   psps/ONCVPSP-LDA-PDv0.4/standard.txt       (SR, LDA)
//   psps/ONCVPSP-PBE-FR-PDv0.4/standard.txt    (FR, GGA_PBE, SOC)
//   psps/ONCVPSP-LDA-FR-PDv0.4/standard.txt    (FR, LDA, SOC)
//
// standard.txt contains one line per element: "Element/Element[-suffix][_r].psp8"
// ---------------------------------------------------------------------------
void InputParser::resolve_pseudopotentials(SystemConfig& config,
                                            const std::string& psps_root) {
    // Check if any atom type needs auto-resolution
    bool need_resolve = false;
    for (const auto& at : config.atom_types) {
        if (at.pseudo_file.empty()) { need_resolve = true; break; }
    }
    if (!need_resolve) return;

    // Determine which PSP table to use based on XC and spin type
    bool is_soc = (config.spin_type == SpinType::NonCollinear);
    bool is_lda = (config.xc == XCType::LDA_PZ || config.xc == XCType::LDA_PW);

    std::string table_dir;
    if (is_lda) {
        table_dir = psps_root + (is_soc ? "/ONCVPSP-LDA-FR-PDv0.4" : "/ONCVPSP-LDA-PDv0.4");
    } else {
        // GGA (PBE, PBEsol, RPBE) — use PBE table
        table_dir = psps_root + (is_soc ? "/ONCVPSP-PBE-FR-PDv0.4" : "/ONCVPSP-PBE-PDv0.4");
    }

    // Read standard.txt mapping: each line is "Element/filename.psp8"
    std::string standard_file = table_dir + "/standard.txt";
    std::ifstream ifs(standard_file);
    if (!ifs.is_open()) {
        // Try with ../ prefix (running from build/ directory)
        standard_file = "../" + table_dir + "/standard.txt";
        table_dir = "../" + table_dir;
        ifs.open(standard_file);
    }
    if (!ifs.is_open()) {
        throw std::runtime_error(
            "Cannot find pseudopotential table: " + standard_file +
            "\nRun 'git submodule update --init' to fetch PseudoDojo pseudopotentials.");
    }

    // Parse standard.txt into element -> relative_path map
    std::map<std::string, std::string> elem_to_psp;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        // Line format: "Si/Si_r.psp8" or "Au/Au-sp.psp8"
        // Extract element name from directory part
        auto slash = line.find('/');
        if (slash != std::string::npos) {
            std::string elem = line.substr(0, slash);
            elem_to_psp[elem] = table_dir + "/" + line;
        }
    }

    // Resolve each atom type that has no pseudo_file
    for (auto& at : config.atom_types) {
        if (!at.pseudo_file.empty()) continue;

        auto it = elem_to_psp.find(at.element);
        if (it != elem_to_psp.end()) {
            at.pseudo_file = it->second;
        } else {
            throw std::runtime_error(
                "No pseudopotential found for element '" + at.element +
                "' in " + standard_file +
                (is_soc ? " (FR/SOC table)" : " (SR table)"));
        }
    }
}

void InputParser::validate(const SystemConfig& config) {
    // Lattice vectors must have non-zero volume
    if (std::abs(config.latvec.determinant()) < 1e-14)
        throw std::runtime_error("Lattice vectors have zero volume");

    // Grid must be specified
    if (config.Nx <= 0 || config.Ny <= 0 || config.Nz <= 0) {
        if (config.mesh_spacing <= 0)
            throw std::runtime_error("Grid dimensions or mesh_spacing must be specified");
    }

    // Must have at least one atom type
    if (config.atom_types.empty())
        throw std::runtime_error("No atom types specified");

    // Each atom type must have at least one coordinate
    for (const auto& at : config.atom_types) {
        if (at.coords.empty())
            throw std::runtime_error("Atom type " + at.element + " has no coordinates");
    }

    // FD order must be even and positive
    if (config.fd_order <= 0 || config.fd_order % 2 != 0)
        throw std::runtime_error("FD order must be a positive even integer");

    // Nstates
    if (config.Nstates <= 0)
        throw std::runtime_error("Nstates must be positive");
}

} // namespace lynx
