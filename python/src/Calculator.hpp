#pragma once

#include <string>
#include <vector>
#include <memory>

#include "io/InputParser.hpp"
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include "core/Domain.hpp"
#include "core/KPoints.hpp"
#include "operators/FDStencil.hpp"
#include "operators/Laplacian.hpp"
#include "operators/Gradient.hpp"
#include "operators/Hamiltonian.hpp"
#include "operators/NonlocalProjector.hpp"
#include "atoms/Crystal.hpp"
#include "atoms/AtomType.hpp"
#include "electronic/Wavefunction.hpp"
#include "electronic/ElectronDensity.hpp"
#include "physics/SCF.hpp"
#include "physics/Energy.hpp"
#include "physics/Forces.hpp"
#include "physics/Stress.hpp"
#include "physics/Electrostatics.hpp"
#include "parallel/Parallelization.hpp"
#include "parallel/HaloExchange.hpp"
#include "parallel/MPIComm.hpp"
#include "solvers/PoissonSolver.hpp"
#include "solvers/Mixer.hpp"
#include "solvers/EigenSolver.hpp"
#include "xc/XCFunctional.hpp"

namespace pylynx {

/// High-level Calculator that owns all LYNX objects and replicates main.cpp workflow
class Calculator {
public:
    Calculator() = default;

    /// Load configuration from JSON file
    void load_config(const std::string& json_file);

    /// Set up all internal objects (lattice, grid, operators, etc.)
    /// Uses MPI_COMM_SELF for single-process mode.
    void setup(MPI_Comm comm = MPI_COMM_SELF);

    /// Run the full SCF calculation
    double run();

    /// Compute atomic forces (after SCF)
    std::vector<double> compute_forces();

    /// Compute stress tensor (after SCF)
    std::array<double, 6> compute_stress();

    // --- Accessors for mid-level usage ---
    bool is_setup() const { return setup_done_; }
    bool is_converged() const { return scf_converged_; }

    const lynx::SystemConfig& config() const { return config_; }
    const lynx::Lattice& lattice() const { return lattice_; }
    const lynx::FDGrid& grid() const { return grid_; }
    const lynx::FDStencil& stencil() const { return stencil_; }
    const lynx::Domain& domain() const;
    const lynx::KPoints& kpoints() const { return kpoints_; }
    const lynx::HaloExchange& halo() const { return halo_; }
    const lynx::Laplacian& laplacian() const { return laplacian_; }
    const lynx::Gradient& gradient() const { return gradient_; }
    const lynx::Hamiltonian& hamiltonian() const { return hamiltonian_; }
    const lynx::NonlocalProjector& nonlocal_projector() const { return vnl_; }
    const lynx::Crystal& crystal() const { return crystal_; }
    const lynx::Electrostatics& electrostatics() const { return elec_; }
    const lynx::SCF& scf() const { return scf_; }
    lynx::Wavefunction& wavefunction() { return wfn_; }
    const lynx::Wavefunction& wavefunction() const { return wfn_; }

    const lynx::EnergyComponents& energy() const { return scf_.energy(); }
    double fermi_energy() const { return scf_.fermi_energy(); }

    // Internal state
    int Nd_d() const;
    int Nelectron() const { return Nelectron_; }
    int Natom() const { return Natom_; }
    int Nspin() const { return Nspin_; }

    const double* Vloc_data() const { return Vloc_.data(); }
    const double* rho_core_data() const { return has_nlcc_ ? rho_core_.data() : nullptr; }

private:
    lynx::SystemConfig config_;
    lynx::Lattice lattice_;
    lynx::FDGrid grid_;
    lynx::FDStencil stencil_;
    lynx::KPoints kpoints_;

    std::unique_ptr<lynx::Parallelization> parallel_;
    lynx::HaloExchange halo_;
    lynx::Laplacian laplacian_;
    lynx::Gradient gradient_;
    lynx::Hamiltonian hamiltonian_;
    lynx::NonlocalProjector vnl_;

    lynx::Crystal crystal_;
    lynx::Electrostatics elec_;
    lynx::SCF scf_;
    lynx::Wavefunction wfn_;

    std::vector<lynx::AtomInfluence> influence_;
    std::vector<lynx::AtomNlocInfluence> nloc_influence_;
    std::vector<double> Vloc_;
    std::vector<double> rho_core_;

    int Nelectron_ = 0;
    int Natom_ = 0;
    int Nspin_ = 1;
    int Nstates_ = 0;
    bool has_nlcc_ = false;
    bool setup_done_ = false;
    bool scf_converged_ = false;
};

} // namespace pylynx
