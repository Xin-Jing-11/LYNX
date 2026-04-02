#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
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
#include "xc/XCFunctional.hpp"
#include "solvers/EigenSolver.hpp"
#include "solvers/PoissonSolver.hpp"
#include "solvers/Mixer.hpp"
#include "physics/Energy.hpp"
#include "physics/EffectivePotential.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"
#include "io/InputParser.hpp"

#include <vector>
#include <functional>
#include <memory>

#ifdef USE_CUDA
#include "physics/GPUSCF.cuh"
#include "physics/Electrostatics.hpp"
#endif

namespace lynx {

class ExactExchange;  // forward declaration
struct SCFState;      // forward declaration (defined in SCFInitializer.hpp)

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
};

class SCF {
public:
    SCF() = default;
    SCF(const SCF&) = delete;
    SCF& operator=(const SCF&) = delete;
    SCF(SCF&&) = default;
    SCF& operator=(SCF&&) = default;

    void setup(const FDGrid& grid,
               const Domain& domain,
               const FDStencil& stencil,
               const Laplacian& laplacian,
               const Gradient& gradient,
               const Hamiltonian& hamiltonian,
               const HaloExchange& halo,
               const NonlocalProjector* vnl,
               const MPIComm& bandcomm,
               const MPIComm& kptcomm,
               const MPIComm& spincomm,
               const SCFParams& params,
               int Nspin_global = 1,
               int Nspin_local = 1,
               int spin_start = 0,
               const KPoints* kpoints = nullptr,
               int kpt_start = 0,
               int Nband_global = 0,
               int band_start = 0);

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
    const double* tau() const { return arrays_.tau.data(); }

#ifdef USE_CUDA
    GPUSCFRunner* gpu_runner() { return gpu_runner_.get(); }
#endif

private:
    // Operator references (non-owning)
    const FDGrid* grid_ = nullptr;
    const Domain* domain_ = nullptr;
    const FDStencil* stencil_ = nullptr;
    const Laplacian* laplacian_ = nullptr;
    const Gradient* gradient_ = nullptr;
    const Hamiltonian* hamiltonian_ = nullptr;
    const HaloExchange* halo_ = nullptr;
    const NonlocalProjector* vnl_ = nullptr;
    const MPIComm* bandcomm_ = nullptr;
    const MPIComm* kptcomm_ = nullptr;
    const MPIComm* spincomm_ = nullptr;

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

    // Exact exchange (hybrid functionals)
    ExactExchange* exx_ = nullptr;

    // Components
    EffectivePotential veff_builder_;

    // SCF loop sub-steps (extracted from run())
    void solve_eigenproblem(Wavefunction& wfn, EigenSolver& eigsolver, SCFState& state, int scf_iter);
    void compute_new_density(const Wavefunction& wfn, const SCFState& state, ElectronDensity& rho_new);
    void compute_scf_energy(const Wavefunction& wfn, const ElectronDensity& rho_new,
                            const double* rho_b, double Eself, double Ec, SCFState& state);
    bool check_convergence(const Wavefunction& wfn, const ElectronDensity& rho_new,
                           const SCFState& state, int scf_iter);
    void mix_and_update(const ElectronDensity& rho_new, Mixer& mixer,
                        const double* rho_b, const double* rho_core, int Nelectron, SCFState& state);

    // Compute kinetic energy density tau from wavefunctions
    void compute_tau(const Wavefunction& wfn,
                     const std::vector<double>& kpt_weights,
                     int kpt_start, int band_start);

#ifdef USE_CUDA
    std::unique_ptr<GPUSCFRunner> gpu_runner_;

    // GPU-specific data (set via set_gpu_data before run)
    const Crystal* crystal_ = nullptr;
    const std::vector<AtomNlocInfluence>* nloc_influence_ = nullptr;
    const std::vector<AtomInfluence>* influence_ = nullptr;
    const Electrostatics* elec_ = nullptr;
    bool gpu_enabled_ = false;

    double run_gpu(Wavefunction& wfn, int Nelectron, int Natom,
                   const double* rho_b, double Eself, double Ec,
                   XCType xc_type, const double* rho_core);
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

} // namespace lynx
