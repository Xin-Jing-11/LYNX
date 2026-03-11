#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
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

namespace sparc {

struct SCFParams {
    int max_iter = 100;
    int min_iter = 3;
    double tol = 1e-6;           // SCF energy tolerance (Ha/atom)
    MixingVariable mixing_var = MixingVariable::Density;
    MixingPrecond mixing_precond = MixingPrecond::Kerker;
    int mixing_history = 7;
    double mixing_param = 0.3;
    SmearingType smearing = SmearingType::GaussianSmearing;
    double elec_temp = 300.0;    // K
    int cheb_degree = 20;
    int rho_trigger = 4;         // CheFSI passes before first density (from random guess)
    int nchefsi = 1;             // CheFSI passes per subsequent SCF iteration
    double poisson_tol = 1e-8;
    bool print_eigen = false;
};

class SCF {
public:
    SCF() = default;

    void setup(const FDGrid& grid,
               const Domain& domain,
               const FDStencil& stencil,
               const Laplacian& laplacian,
               const Gradient& gradient,
               const Hamiltonian& hamiltonian,
               const HaloExchange& halo,
               const NonlocalProjector* vnl,
               const MPIComm& dmcomm,
               const MPIComm& bandcomm,
               const MPIComm& kptcomm,
               const MPIComm& spincomm,
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
    const double* phi() const { return phi_.data(); }
    const double* Vxc() const { return Vxc_.data(); }
    const double* exc() const { return exc_.data(); }
    const double* Veff() const { return Veff_.data(); }

private:
    const FDGrid* grid_ = nullptr;
    const Domain* domain_ = nullptr;
    const FDStencil* stencil_ = nullptr;
    const Laplacian* laplacian_ = nullptr;
    const Gradient* gradient_ = nullptr;
    const Hamiltonian* hamiltonian_ = nullptr;
    const HaloExchange* halo_ = nullptr;
    const NonlocalProjector* vnl_ = nullptr;
    const MPIComm* dmcomm_ = nullptr;
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

    XCType xc_type_ = XCType::GGA_PBE;
    const double* Vloc_ = nullptr;
    const double* rho_core_ = nullptr;  // NLCC core density (non-owning)

    // Compute effective potential: Veff = Vxc + phi + Vloc
    void compute_Veff(const double* rho, const double* rho_b, const double* Vloc);

    // Initialize density from superposition of atomic densities (simplified)
    void init_density(int Nd_d, int Nelectron);

public:
    // Set initial density from external source (e.g., atomic superposition)
    void set_initial_density(const double* rho_init, int Nd_d);
};

} // namespace sparc
