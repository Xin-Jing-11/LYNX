#pragma once

#include "core/types.hpp"
#include "core/DeviceArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "core/KPoints.hpp"
#include "operators/Hamiltonian.hpp"
#include "operators/FDStencil.hpp"
#include "operators/Laplacian.hpp"
#include "operators/Gradient.hpp"
#include "operators/NonlocalProjector.hpp"
#include "electronic/Wavefunction.hpp"
#include "electronic/ElectronDensity.hpp"
#include "electronic/Occupation.hpp"
#include "electronic/KineticEnergyDensity.hpp"
#include "xc/XCFunctional.hpp"
#include "solvers/EigenSolver.hpp"
#include "solvers/PoissonSolver.hpp"
#include "solvers/Mixer.hpp"
#include "physics/Energy.hpp"
#include "physics/EffectivePotential.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"
#include "io/InputParser.hpp"
#include "core/LynxContext.hpp"
#include "core/DeviceTag.hpp"

#include <vector>
#include <functional>
#include <memory>

#ifdef USE_CUDA
#include "physics/Electrostatics.hpp"
#endif

namespace lynx {

class ExactExchange;  // forward declaration
struct AtomSetup;     // forward declaration (defined in AtomSetup.hpp)
struct SCFResult;     // forward declaration (defined below SCF class)

// Bundles the initialized state needed to start the SCF loop.
struct SCFState {
    // Spectral bounds per local spin channel
    std::vector<double> eigval_min;
    std::vector<double> eigval_max;
    double lambda_cutoff = 0.0;

    // K-point weights (global)
    std::vector<double> kpt_weights;

    // Derived parameters
    double kBT = 0.0;
    double beta = 0.0;  // 1/(kBT)
    int Nband = 0;       // global band count
    int Nband_loc = 0;   // local bands on this process
    int Nspin_local = 0; // spins on this process
    int Nkpts = 0;       // local k-points
    int Nelectron = 0;   // total electron count (for occupations)

    // Potential mixing state
    bool use_potential_mixing = false;
    DeviceArray<double> Veff_mixed;   // zero-mean Veff for mixer (persistent)
    std::vector<double> Veff_mean; // per-spin Veff mean
    DeviceArray<double> Veff_out;     // Veff from rho_out (potential mixing), shared between energy & convergence
};

/// All fields must be fully resolved before passing to SCF.
/// Call ParameterDefaults::update_default() in main.cpp after parsing to fill auto-defaults.
struct SCFParams {
    int max_iter = 100;
    int min_iter = 2;
    double tol = 1e-6;           // SCF convergence tolerance (||rho_out-rho_in||/||rho_out||)
    MixingVariable mixing_var = MixingVariable::Density;
    MixingPrecond mixing_precond = MixingPrecond::Kerker;
    int mixing_history = 7;
    double mixing_param = 0.3;
    SmearingType smearing = SmearingType::GaussianSmearing;
    double elec_temp = 0.0;     // Electronic temperature in Kelvin (must be set before use)
    int cheb_degree = 0;        // Chebyshev polynomial degree (must be set before use)
    int rho_trigger = 4;         // CheFSI passes before first density (from random guess)
    int nchefsi = 1;             // CheFSI passes per subsequent SCF iteration
    double poisson_tol = 0.0;    // Poisson solver tolerance (must be set before use)
    double precond_tol = 0.0;    // Kerker preconditioner tolerance (must be set before use)
    bool print_eigen = false;
    EXXParams exx_params;     // exact exchange parameters (hybrid functionals)

    /// Build SCFParams from a fully-resolved SystemConfig.
    static SCFParams from_config(const SystemConfig& config);
};

class SCF {
public:
    SCF() = default;
    SCF(const SCF&) = delete;
    SCF& operator=(const SCF&) = delete;
    SCF(SCF&&) = default;
    SCF& operator=(SCF&&) = default;

    /// High-level entry point: initialize density and run SCF to convergence.
    static SCFResult run_calculation(const SystemConfig& config,
                                     const LynxContext& ctx,
                                     const Crystal& crystal,
                                     AtomSetup& atoms,
                                     Hamiltonian& hamiltonian,
                                     const NonlocalProjector& vnl);

    /// Setup using LynxContext for all infrastructure.
    void setup(const LynxContext& ctx,
               Hamiltonian& hamiltonian,
               const NonlocalProjector* vnl,
               const SCFParams& params);

    // Run self-consistent field loop
    // Returns total energy
    double run(Wavefunction& wfn,
               int Nelectron,
               int Natom,
               const double* rho_b,       // pseudocharge density (may be null)
               const double* Vloc,         // local pseudopotential correction (may be null)
               double Eself,              // self energy
               double Ec,                 // correction energy
               XCType xc_type = XCType::GGA_PBE,
               const double* rho_core = nullptr);  // NLCC core density (may be null)

    // Access results
    const EnergyComponents& energy() const { return energy_; }
    const ElectronDensity& density() const { return density_; }
    double fermi_energy() const { return Ef_; }
    bool converged() const { return converged_; }

    // Access internal potentials (needed for Forces/Stress post-SCF)
    const double* phi() const { return arrays_.phi.data(); }
    const double* Vxc() const { return arrays_.Vxc.data(); }
    const double* exc() const { return arrays_.exc.data(); }
    const double* Veff() const { return arrays_.Veff.data(); }
    const double* Dxcdgrho() const { return arrays_.Dxcdgrho.data(); }
    const double* vtau() const { return arrays_.vtau.data(); }
    const double* tau() const { return tau_.data(); }

#ifdef USE_CUDA
    // GPU mGGA stress: computed post-SCF using Hamiltonian's GPU state
    // (psi stays on device; see Stress.cpp for usage)
    const Hamiltonian* hamiltonian_ptr() const { return hamiltonian_; }

    // Post-SCF device psi access (for forces/stress on GPU).
    // Returns device psi pointer for a specific (spin, kpt).
    const double* device_psi_real(int spin, int kpt) const { return eigsolver_.device_psi_real(spin, kpt); }
    const void* device_psi_z(int spin, int kpt) const { return eigsolver_.device_psi_z(spin, kpt); }

    // Clean up GPU state (called at end of run(), after GPU force+stress)
    void cleanup_gpu();

    // ── GPU force+stress (psi stays on device) ──────────────────
    struct GPUForceStressResult {
        std::vector<double> f_nloc;          // [3 * n_atom]
        std::array<double, 6> stress_k = {};
        std::array<double, 6> stress_nl = {};
        double energy_nl = 0.0;
        bool computed = false;
    };

    // Compute nonlocal forces + kinetic/nonlocal stress on GPU.
    // Loops over all (spin, kpt), accumulates results on host.
    // Requires that device psi buffers are still live (call BEFORE cleanup_gpu).
    void compute_gpu_force_stress(const Wavefunction& wfn);

    const GPUForceStressResult& gpu_force_stress() const { return gpu_fs_; }
    bool has_gpu_force_stress() const { return gpu_fs_.computed; }
#endif

private:
    // Context reference (non-owning, for sub-component setup)
    const LynxContext* ctx_ = nullptr;

    // Operator references (non-owning)
    const FDGrid* grid_ = nullptr;
    const Domain* domain_ = nullptr;
    const FDStencil* stencil_ = nullptr;
    const Laplacian* laplacian_ = nullptr;
    const Gradient* gradient_ = nullptr;
    Hamiltonian* hamiltonian_ = nullptr;
    const HaloExchange* halo_ = nullptr;
    const NonlocalProjector* vnl_ = nullptr;
    const MPIComm* bandcomm_ = nullptr;
    const MPIComm* kptcomm_ = nullptr;
    const MPIComm* spincomm_ = nullptr;

    // Device tag: CPU or GPU (unified dispatch — operators use this to select code path)
    Device dev_ = Device::CPU;

    // Parameters and results
    SCFParams params_;
    EnergyComponents energy_;
    ElectronDensity density_;
    double Ef_ = 0.0;
    bool converged_ = false;

    // Work arrays (owned by VeffArrays)
    VeffArrays arrays_;

    // XC and external potential state
    XCType xc_type_ = XCType::GGA_PBE;
    const double* rho_core_ = nullptr;

    // Parallel decomposition
    int Nspin_global_ = 1;
    int Nspin_local_ = 1;
    int spin_start_ = 0;
    const KPoints* kpoints_ = nullptr;
    bool is_kpt_ = false;
    int kpt_start_ = 0;
    int Nband_global_ = 0;
    int band_start_ = 0;

    // SOC / noncollinear
    bool is_soc_ = false;

    // Cached SCF dimensions (for post-SCF device psi access)
    int Nband_loc_ = 0;
    int Nkpts_ = 0;

    // Exact exchange (hybrid functionals)
    ExactExchange* exx_ = nullptr;

    // Components
    EffectivePotential veff_builder_;

    // SCF loop sub-steps (extracted from run())
    void solve_eigenproblem(Wavefunction& wfn, SCFState& state, int scf_iter);
    void compute_new_density(const Wavefunction& wfn, const SCFState& state,
                             ElectronDensity& rho_new);
    void compute_scf_energy(const Wavefunction& wfn, const ElectronDensity& rho_new,
                            const double* rho_b, double Eself, double Ec, SCFState& state);
    bool check_convergence(const Wavefunction& wfn, const ElectronDensity& rho_new,
                           const SCFState& state, int scf_iter);
    void mix_and_update(const ElectronDensity& rho_new, Mixer& mixer,
                        const double* rho_b, const double* rho_core, int Nelectron, SCFState& state);

    // Kinetic energy density (mGGA)
    KineticEnergyDensity tau_;

    // EigenSolver (member so device psi persists for post-SCF forces/stress)
    EigenSolver eigsolver_;

    // SCF initialization helpers (merged from SCFInitializer)
    static SCFState initialize_scf(
        const LynxContext& ctx,
        Wavefunction& wfn, ElectronDensity& density,
        VeffArrays& arrays, EffectivePotential& veff_builder,
        SCFParams& params, Hamiltonian& hamiltonian,
        const NonlocalProjector* vnl, EigenSolver& eigsolver, Mixer& mixer,
        int Nelectron, XCType xc_type,
        const double* rho_b, const double* rho_core);

    static void init_density(ElectronDensity& density, int Nd_d, int Nelectron,
                             int Nspin_global, const FDGrid& grid, bool is_soc);
    static void randomize_wavefunctions(Wavefunction& wfn, int Nspin_local, int spin_start,
                                         const MPIComm& spincomm, const MPIComm& bandcomm, bool is_kpt);
    static void estimate_spectral_bounds(
        SCFState& state, EigenSolver& eigsolver,
        const double* Veff, const double* Veff_spinor,
        int Nd_d, int Nspin_local, int spin_start,
        bool is_kpt, bool is_soc, const KPoints* kpoints, int kpt_start,
        const Vec3& cell_lengths, Hamiltonian& hamiltonian,
        const NonlocalProjector* vnl, int rank_world);

#ifdef USE_CUDA
    // GPU force+stress result (accumulated across spins/kpts)
    GPUForceStressResult gpu_fs_;

    // GPU-specific data (set via set_gpu_data before run)
    const Crystal* crystal_ = nullptr;
    const std::vector<AtomNlocInfluence>* nloc_influence_ = nullptr;
    const std::vector<AtomInfluence>* influence_ = nullptr;
    const Electrostatics* elec_ = nullptr;
    bool gpu_enabled_ = false;

public:
    // Call before run() to enable GPU path
    void set_gpu_data(const Crystal& crystal,
                      const std::vector<AtomNlocInfluence>& nloc_influence,
                      const std::vector<AtomInfluence>& influence,
                      const Electrostatics& elec) {
        crystal_ = &crystal;
        nloc_influence_ = &nloc_influence;
        influence_ = &influence;
        elec_ = &elec;
        gpu_enabled_ = true;
    }
private:
#endif

public:
    // Set initial density from external source (e.g., atomic superposition)
    // For spin: rho_init is total density (Nd_d), mag_init is magnetization (Nd_d, may be null)
    void set_initial_density(const double* rho_init, int Nd_d,
                             const double* mag_init = nullptr);

    // Set exact exchange operator for hybrid functionals
    void set_exx(ExactExchange* exx) { exx_ = exx; }
    const ExactExchange& exx() const { return *exx_; }
    ExactExchange& exx() { return *exx_; }
};

/// Result of an SCF calculation.
struct SCFResult {
    Wavefunction wfn;
    SCF scf;
};

} // namespace lynx
