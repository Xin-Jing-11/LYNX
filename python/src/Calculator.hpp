#pragma once

#include <string>
#include <vector>
#include <memory>
#include <array>

#include "io/InputParser.hpp"
#include "atoms/AtomSetup.hpp"
#include "operators/Hamiltonian.hpp"
#include "operators/NonlocalProjector.hpp"
#include "electronic/Wavefunction.hpp"
#include "physics/SCF.hpp"
#include "physics/Forces.hpp"
#include "physics/Stress.hpp"
#include "core/LynxContext.hpp"

namespace pylynx {

/// High-level Calculator that owns all LYNX objects and replicates main.cpp workflow.
/// Delegates grid/parallel infrastructure to LynxContext singleton.
class Calculator {
public:
    Calculator() = default;

    /// Load configuration from JSON file
    void load_config(const std::string& json_file);

    /// Set configuration directly (no JSON file needed)
    void set_config(const lynx::SystemConfig& config);

    /// Set up all internal objects (lattice, grid, operators, etc.)
    /// Uses MPI_COMM_SELF for single-process mode.
    void setup(MPI_Comm comm = MPI_COMM_SELF);

    /// Run the full SCF calculation
    double run();

    /// Compute atomic forces (after SCF)
    std::vector<double> compute_forces();

    /// Compute stress tensor (after SCF)
    std::array<double, 6> compute_stress();

    /// Compute hydrostatic pressure (after stress)
    double compute_pressure();

    // --- Accessors ---
    bool is_setup() const { return setup_done_; }
    bool is_converged() const { return scf_converged_; }
    const lynx::SystemConfig& config() const { return config_; }

    // Grid/spatial -- delegate to LynxContext
    const lynx::Lattice& lattice() const;
    const lynx::FDGrid& grid() const;
    const lynx::FDStencil& stencil() const;
    const lynx::Domain& domain() const;
    const lynx::KPoints& kpoints() const;
    const lynx::HaloExchange& halo() const;
    const lynx::Laplacian& laplacian() const;
    const lynx::Gradient& gradient() const;

    // Physics objects -- owned by Calculator
    const lynx::Hamiltonian& hamiltonian() const { return hamiltonian_; }
    const lynx::NonlocalProjector& nonlocal_projector() const { return vnl_; }
    const lynx::Crystal& crystal() const { return atoms_.crystal; }
    const lynx::Electrostatics& electrostatics() const { return atoms_.elec; }
    const lynx::SCF& scf() const { return scf_; }
    lynx::Wavefunction& wavefunction() { return wfn_; }
    const lynx::Wavefunction& wavefunction() const { return wfn_; }
    const lynx::EnergyComponents& energy() const { return scf_.energy(); }
    double fermi_energy() const { return scf_.fermi_energy(); }

    int Nd_d() const;
    int Nelectron() const;
    int Natom() const;
    int Nspin() const;
    bool use_gpu() const { return use_gpu_; }
    void set_use_gpu(bool v) { use_gpu_ = v; }
    int n_iterations() const { return scf_.n_iterations(); }

    static bool cuda_available() {
#ifdef USE_CUDA
        return true;
#else
        return false;
#endif
    }

    const double* Vloc_data() const { return atoms_.Vloc.data(); }
    int Vloc_size() const { return static_cast<int>(atoms_.Vloc.size()); }
    const double* rho_core_data() const { return atoms_.has_nlcc ? atoms_.rho_core.data() : nullptr; }
    const double* atomic_density_data() const { return rho_atomic_.data(); }
    int atomic_density_size() const { return static_cast<int>(rho_atomic_.size()); }

    /// Set initial electron density for SCF (e.g., from a previous converged calculation).
    /// rho_init: total density array of size Nd_d.
    /// mag_init: magnetization density (Nd_d) for spin-polarized, or nullptr.
    void set_initial_density(const double* rho_init, int Nd_d,
                             const double* mag_init = nullptr) {
        scf_.set_initial_density(rho_init, Nd_d, mag_init);
    }

private:
    lynx::SystemConfig config_;
    lynx::AtomSetup atoms_;
    lynx::Hamiltonian hamiltonian_;
    lynx::NonlocalProjector vnl_;
    lynx::SCF scf_;
    lynx::Wavefunction wfn_;

    std::vector<double> rho_atomic_;

    bool setup_done_ = false;
    bool scf_converged_ = false;
    bool use_gpu_ = false;
};

} // namespace pylynx
