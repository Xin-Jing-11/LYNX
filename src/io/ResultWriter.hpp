#pragma once

#include <array>
#include <string>
#include <vector>

namespace lynx {

class LynxContext;
class Wavefunction;

class ResultWriter {
public:
    struct AtomTypeInfo {
        std::string element;
        double      Zval = 0.0;
        double      mass_amu = 0.0;
        int         n_atoms = 0;
        std::string pseudo_file;
    };

    struct SystemInfo {
        int n_atoms = 0;
        int n_electrons = 0;
        std::vector<AtomTypeInfo> atom_types;
        std::vector<std::array<double, 3>> atoms_cartesian_bohr;
        std::vector<std::array<double, 3>> atoms_fractional;
    };

    struct CellInfo {
        std::array<std::array<double, 3>, 3> lattice_vectors_bohr{};
        double volume_bohr3 = 0.0;
        bool   orthogonal = true;
    };

    struct GridInfo {
        int Nx = 0, Ny = 0, Nz = 0;
        double dx_bohr = 0.0, dy_bohr = 0.0, dz_bohr = 0.0;
        int fd_order = 0;
        std::array<std::string, 3> boundary_conditions{{"P", "P", "P"}};
    };

    struct ElectronicInfo {
        std::string xc;
        std::string spin = "none";
        bool        soc = false;
        std::string smearing;
        double      electronic_temperature_K = 0.0;
        double      kBT_Ha = 0.0;
        int         n_states = 0;
        int         cheb_degree = 0;
    };

    struct KpointInfo {
        int                    index = 0;
        std::array<double, 3>  reduced{};
        double                 weight = 0.0;
    };

    struct KpointsInfo {
        std::array<int, 3>       grid{{1, 1, 1}};
        std::array<double, 3>    shift{{0.0, 0.0, 0.0}};
        std::vector<KpointInfo>  list;
    };

    struct ScfIterRecord {
        int    iter = 0;
        double free_energy_per_atom_Ha = 0.0;
        double scf_error = 0.0;
        double time_s = 0.0;
    };

    struct ScfInfo {
        int         max_iter = 0;
        double      tolerance = 0.0;
        std::string mixing_variable;
        std::string preconditioner;
        int         mixing_history = 0;
        double      mixing_parameter = 0.0;
        bool        converged = false;
        int         n_iterations = 0;
        std::vector<ScfIterRecord> history;
    };

    struct EnergiesInfo {
        double total_free_energy = 0.0;
        double free_energy_per_atom = 0.0;
        double band_structure = 0.0;
        double exchange_correlation = 0.0;
        double hartree = 0.0;
        double self_energy = 0.0;
        double correction_energy = 0.0;
        double self_plus_correction = 0.0;
        double entropy_term = 0.0;
        double fermi_level = 0.0;
    };

    struct ForcesInfo {
        double max_Ha_per_bohr = 0.0;
        double rms_Ha_per_bohr = 0.0;
        std::vector<std::array<double, 3>> per_atom;
    };

    struct StressInfo {
        bool computed = false;
        std::array<std::array<double, 3>, 3> tensor_Ha_per_bohr3{};
        std::array<std::array<double, 3>, 3> tensor_GPa{};
        std::array<double, 6> voigt_Ha_per_bohr3{};
        std::array<double, 6> voigt_GPa{};
        double pressure_GPa = 0.0;
        double max_abs_stress_GPa = 0.0;
    };

    struct KpointEigenBlock {
        std::vector<double> eigenvalues;
        std::vector<double> occupations;
        bool empty() const { return eigenvalues.empty() && occupations.empty(); }
    };

    struct KpointEigenEntry {
        int                   kpoint_index = 0;
        std::array<double, 3> reduced{};
        double                weight = 0.0;
        KpointEigenBlock      spin_up;
        KpointEigenBlock      spin_down;  // empty when spin=none
    };

    struct EigenvaluesInfo {
        int n_states = 0;
        std::vector<KpointEigenEntry> per_kpoint;
    };

    struct TimingInfo {
        double scf_total_s = 0.0;
        double force_s = 0.0;
        double stress_s = 0.0;
        double walltime_total_s = 0.0;
    };

    ResultWriter() = default;

    void set_test_name(const std::string& name)    { test_name_ = name; }
    void set_input_file(const std::string& path)   { input_file_ = path; }
    void set_device(const std::string& dev)        { device_ = dev; }
    void set_mpi_ranks(int n)                      { mpi_ranks_ = n; }
    void set_lynx_version(const std::string& v)    { lynx_version_ = v; }

    void set_system(const SystemInfo& v)           { system_ = v; }
    void set_cell(const CellInfo& v)               { cell_ = v; }
    void set_grid(const GridInfo& v)               { grid_ = v; }
    void set_electronic(const ElectronicInfo& v)   { electronic_ = v; }
    void set_kpoints(const KpointsInfo& v)         { kpoints_ = v; }
    void set_scf(const ScfInfo& v)                 { scf_ = v; }
    void set_energies(const EnergiesInfo& v)       { energies_ = v; }
    void set_forces(const ForcesInfo& v)           { forces_ = v; }
    void set_stress(const StressInfo& v)           { stress_ = v; }
    void set_eigenvalues(const EigenvaluesInfo& v) { eigenvalues_ = v; }
    void set_timing(const TimingInfo& v)           { timing_ = v; }

    /// Populate the eigenvalues section by pulling (eigvals, occs) out of `wfn`
    /// and gathering across spin/kpt MPI communicators exposed by `ctx`.
    /// Must be called by every rank in MPI_COMM_WORLD; only rank 0 retains data.
    void populate_eigenvalues_from(const LynxContext& ctx, const Wavefunction& wfn);

    /// Serialise the accumulated state to `path` as JSON.
    /// Creates parent directories if missing.
    void write(const std::string& path) const;

private:
    std::string test_name_;
    std::string input_file_;
    std::string device_ = "CPU";
    int         mpi_ranks_ = 1;
    std::string lynx_version_;

    SystemInfo      system_;
    CellInfo        cell_;
    GridInfo        grid_;
    ElectronicInfo  electronic_;
    KpointsInfo     kpoints_;
    ScfInfo         scf_;
    EnergiesInfo    energies_;
    ForcesInfo      forces_;
    StressInfo      stress_;
    EigenvaluesInfo eigenvalues_;
    TimingInfo      timing_;
};

} // namespace lynx
