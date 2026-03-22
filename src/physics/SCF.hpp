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

struct SCFParams {
    int max_iter = 100;
    int min_iter = 2;
    double tol = 1e-6;           // SCF convergence tolerance (||rho_out-rho_in||/||rho_out||)
    MixingVariable mixing_var = MixingVariable::Density;
    MixingPrecond mixing_precond = MixingPrecond::Kerker;
    int mixing_history = 7;
    double mixing_param = 0.3;
    SmearingType smearing = SmearingType::GaussianSmearing;
    double elec_temp = -1.0;    // K (auto: 0.2eV for Gaussian, 0.1eV for FD)
    int cheb_degree = -1;       // auto: Mesh2ChebDegree(h_eff)
    int rho_trigger = 4;         // CheFSI passes before first density (from random guess)
    int nchefsi = 1;             // CheFSI passes per subsequent SCF iteration
    double poisson_tol = -1.0;   // Default: computed as tol * 0.01
    bool print_eigen = false;
    EXXParams exx_params;        // Exact exchange parameters (for hybrid functionals)
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
               Hamiltonian& hamiltonian,
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
    const double* phi() const { return phi_.data(); }
    const double* Vxc() const { return Vxc_.data(); }
    const double* exc() const { return exc_.data(); }
    const double* Veff() const { return Veff_.data(); }
    const double* Dxcdgrho() const { return Dxcdgrho_.data(); }

private:
    const FDGrid* grid_ = nullptr;
    const Domain* domain_ = nullptr;
    const FDStencil* stencil_ = nullptr;
    const Laplacian* laplacian_ = nullptr;
    const Gradient* gradient_ = nullptr;
    Hamiltonian* hamiltonian_ = nullptr;  // mutable for set_exx_context
    const HaloExchange* halo_ = nullptr;
    const NonlocalProjector* vnl_ = nullptr;
    const MPIComm* bandcomm_ = nullptr;
    const MPIComm* kptcomm_ = nullptr;
    const MPIComm* spincomm_ = nullptr;

    SCFParams params_;
    EnergyComponents energy_;
    ElectronDensity density_;
    double Ef_ = 0.0;
    bool converged_ = false;

    // Work arrays
    NDArray<double> Veff_;      // effective potential
    NDArray<double> Vxc_;       // XC potential
    NDArray<double> exc_;       // XC energy density
    NDArray<double> phi_;       // electrostatic potential
    NDArray<double> Dxcdgrho_;  // GGA: dExc/d(|∇ρ|²) = v2x + v2c (stored like reference)

    XCType xc_type_ = XCType::GGA_PBE;
    const double* Vloc_ = nullptr;
    const double* rho_core_ = nullptr;  // NLCC core density (non-owning)

    // Exact exchange support
    class ExactExchange* exx_ = nullptr;  // non-owning pointer, set by main.cpp
    bool in_fock_loop_ = false;           // true during outer Fock iterations

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

    int Nspin_global_ = 1;  // global number of spin channels (1 or 2)
    int Nspin_local_ = 1;   // spin channels on this process (1 or Nspin_global)
    int spin_start_ = 0;    // global spin index of first local spin channel
    const KPoints* kpoints_ = nullptr;  // k-point info (null = gamma-only)
    bool is_kpt_ = false;               // true if using k-points (complex wavefunctions)
    int kpt_start_ = 0;                 // global index of first local k-point
    int Nband_global_ = 0;  // total bands across all band-parallel procs (0 = use Nband)
    int band_start_ = 0;    // global index of first local band

    // SOC / noncollinear
    bool is_soc_ = false;
    NDArray<double> Veff_spinor_;  // [V_uu | V_dd | Re(V_ud) | Im(V_ud)], 4*Nd_d

    // Compute effective potential: Veff = Vxc + phi + Vloc
    // For spin-polarized: Veff has Nspin columns, Vxc has Nspin columns
    void compute_Veff(const double* rho, const double* rho_b, const double* Vloc);

    // Compute spinor Veff from noncollinear density (rho, mx, my, mz)
    void compute_Veff_spinor(const ElectronDensity& density,
                              const double* rho_b, const double* Vloc);

    // Initialize density from superposition of atomic densities (simplified)
    void init_density(int Nd_d, int Nelectron);

public:
    // Set initial density from external source (e.g., atomic superposition)
    // For spin: rho_init is total density (Nd_d), mag_init is magnetization (Nd_d, may be null)
    void set_initial_density(const double* rho_init, int Nd_d,
                             const double* mag_init = nullptr);

    // Set exact exchange operator for hybrid functionals
    void set_exx(ExactExchange* exx) { exx_ = exx; }
};

} // namespace lynx
