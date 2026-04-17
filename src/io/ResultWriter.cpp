#include "io/ResultWriter.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace lynx {

namespace {
std::string iso8601_utc_now() {
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &time);
#else
    gmtime_r(&time, &tm_utc);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}
} // namespace

void ResultWriter::write(const std::string& path) const {
    nlohmann::json j;
    j["lynx_version"]  = lynx_version_;
    j["test_name"]     = test_name_;
    j["input_file"]    = input_file_;
    j["timestamp_utc"] = iso8601_utc_now();
    j["device"]        = device_;
    j["mpi_ranks"]     = mpi_ranks_;

    // --- system ---
    {
        nlohmann::json s;
        s["n_atoms"]     = system_.n_atoms;
        s["n_electrons"] = system_.n_electrons;
        s["atom_types"]  = nlohmann::json::array();
        for (const auto& t : system_.atom_types) {
            s["atom_types"].push_back({
                {"element",     t.element},
                {"Zval",        t.Zval},
                {"mass_amu",    t.mass_amu},
                {"n_atoms",     t.n_atoms},
                {"pseudo_file", t.pseudo_file},
            });
        }
        s["atoms_cartesian_bohr"] = system_.atoms_cartesian_bohr;
        s["atoms_fractional"]     = system_.atoms_fractional;
        j["system"] = std::move(s);
    }

    // --- cell ---
    j["cell"] = {
        {"lattice_vectors_bohr", cell_.lattice_vectors_bohr},
        {"volume_bohr3",         cell_.volume_bohr3},
        {"orthogonal",           cell_.orthogonal},
    };

    // --- grid ---
    j["grid"] = {
        {"Nx", grid_.Nx}, {"Ny", grid_.Ny}, {"Nz", grid_.Nz},
        {"dx_bohr", grid_.dx_bohr},
        {"dy_bohr", grid_.dy_bohr},
        {"dz_bohr", grid_.dz_bohr},
        {"fd_order", grid_.fd_order},
        {"boundary_conditions", grid_.boundary_conditions},
    };

    // --- electronic ---
    j["electronic"] = {
        {"xc",                       electronic_.xc},
        {"spin",                     electronic_.spin},
        {"soc",                      electronic_.soc},
        {"smearing",                 electronic_.smearing},
        {"electronic_temperature_K", electronic_.electronic_temperature_K},
        {"kBT_Ha",                   electronic_.kBT_Ha},
        {"n_states",                 electronic_.n_states},
        {"cheb_degree",              electronic_.cheb_degree},
    };

    // --- kpoints ---
    {
        nlohmann::json kp;
        kp["grid"]          = kpoints_.grid;
        kp["shift"]         = kpoints_.shift;
        kp["n_irreducible"] = static_cast<int>(kpoints_.list.size());
        kp["list"]          = nlohmann::json::array();
        for (const auto& k : kpoints_.list) {
            kp["list"].push_back({
                {"index",   k.index},
                {"reduced", k.reduced},
                {"weight",  k.weight},
            });
        }
        j["kpoints"] = std::move(kp);
    }

    // --- scf ---
    {
        nlohmann::json s;
        s["max_iter"]         = scf_.max_iter;
        s["tolerance"]        = scf_.tolerance;
        s["mixing_variable"]  = scf_.mixing_variable;
        s["preconditioner"]   = scf_.preconditioner;
        s["mixing_history"]   = scf_.mixing_history;
        s["mixing_parameter"] = scf_.mixing_parameter;
        s["converged"]        = scf_.converged;
        s["n_iterations"]     = scf_.n_iterations;
        s["history"]          = nlohmann::json::array();
        for (const auto& r : scf_.history) {
            s["history"].push_back({
                {"iter",                     r.iter},
                {"free_energy_per_atom_Ha",  r.free_energy_per_atom_Ha},
                {"scf_error",                r.scf_error},
                {"time_s",                   r.time_s},
            });
        }
        j["scf"] = std::move(s);
    }

    // --- energies_Ha ---
    j["energies_Ha"] = {
        {"total_free_energy",     energies_.total_free_energy},
        {"free_energy_per_atom",  energies_.free_energy_per_atom},
        {"band_structure",        energies_.band_structure},
        {"exchange_correlation",  energies_.exchange_correlation},
        {"hartree",               energies_.hartree},
        {"self_energy",           energies_.self_energy},
        {"correction_energy",     energies_.correction_energy},
        {"self_plus_correction",  energies_.self_plus_correction},
        {"entropy_term",          energies_.entropy_term},
        {"fermi_level",           energies_.fermi_level},
    };

    // --- forces_Ha_per_bohr ---
    j["forces_Ha_per_bohr"] = {
        {"max",      forces_.max_Ha_per_bohr},
        {"rms",      forces_.rms_Ha_per_bohr},
        {"per_atom", forces_.per_atom},
    };

    // --- stress ---
    j["stress"] = {
        {"computed",             stress_.computed},
        {"tensor_Ha_per_bohr3",  stress_.tensor_Ha_per_bohr3},
        {"tensor_GPa",           stress_.tensor_GPa},
        {"voigt_Ha_per_bohr3",   stress_.voigt_Ha_per_bohr3},
        {"voigt_GPa",            stress_.voigt_GPa},
        {"pressure_GPa",         stress_.pressure_GPa},
        {"max_abs_stress_GPa",   stress_.max_abs_stress_GPa},
    };

    // --- eigenvalues ---
    {
        nlohmann::json ev;
        ev["units"]      = "Ha";
        ev["n_states"]   = eigenvalues_.n_states;
        ev["per_kpoint"] = nlohmann::json::array();
        for (const auto& e : eigenvalues_.per_kpoint) {
            nlohmann::json entry;
            entry["kpoint_index"] = e.kpoint_index;
            entry["reduced"]      = e.reduced;
            entry["weight"]       = e.weight;
            entry["spin_up"] = {
                {"eigenvalues", e.spin_up.eigenvalues},
                {"occupations", e.spin_up.occupations},
            };
            if (!e.spin_down.empty()) {
                entry["spin_down"] = {
                    {"eigenvalues", e.spin_down.eigenvalues},
                    {"occupations", e.spin_down.occupations},
                };
            }
            ev["per_kpoint"].push_back(std::move(entry));
        }
        j["eigenvalues"] = std::move(ev);
    }

    // --- timing_s ---
    j["timing_s"] = {
        {"scf_total",      timing_.scf_total_s},
        {"force",          timing_.force_s},
        {"stress",         timing_.stress_s},
        {"walltime_total", timing_.walltime_total_s},
    };

    std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("ResultWriter: cannot open " + path);
    out << j.dump(2);
}

// Stub until Task 5 adds the real implementation.
void ResultWriter::populate_eigenvalues_from(const LynxContext& /*ctx*/,
                                             const Wavefunction& /*wfn*/) {
    // Implementation added in Task 5.
}

} // namespace lynx
