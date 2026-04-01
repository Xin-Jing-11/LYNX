#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "core/KPoints.hpp"
#include "core/constants.hpp"
#include "operators/Hamiltonian.hpp"
#include "operators/NonlocalProjector.hpp"
#include "electronic/Wavefunction.hpp"
#include "electronic/ElectronDensity.hpp"
#include "solvers/EigenSolver.hpp"
#include "solvers/Mixer.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"
#include "physics/EffectivePotential.hpp"
#include "physics/SCF.hpp"

#include <vector>

namespace lynx {

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

    // Potential mixing state
    bool use_potential_mixing = false;
    NDArray<double> Veff_mixed;   // zero-mean Veff for mixer (persistent)
    std::vector<double> Veff_mean; // per-spin Veff mean
};

// Handles all SCF initialization: parameter auto-computation, array allocation,
// density initialization, wavefunction randomization, Lanczos spectral bounds.
class SCFInitializer {
public:
    SCFInitializer() = default;

    // Initialize SCF state: allocate arrays, set initial density, randomize wavefunctions,
    // compute initial Veff, estimate spectral bounds via Lanczos.
    // Modifies: wfn (randomized), density (initialized), arrays (allocated + initial Veff)
    // Modifies params in place (auto-computes poisson_tol, elec_temp, cheb_degree).
    static SCFState initialize(
        Wavefunction& wfn,
        ElectronDensity& density,
        VeffArrays& arrays,
        EffectivePotential& veff_builder,
        SCFParams& params,
        const FDGrid& grid,
        const Domain& domain,
        const Hamiltonian& hamiltonian,
        const HaloExchange& halo,
        const NonlocalProjector* vnl,
        const MPIComm& bandcomm,
        const MPIComm& kptcomm,
        const MPIComm& spincomm,
        EigenSolver& eigsolver,
        Mixer& mixer,
        int Nelectron,
        int Nspin_global,
        int Nspin_local,
        int spin_start,
        const KPoints* kpoints,
        int kpt_start,
        int band_start,
        XCType xc_type,
        const double* rho_b,
        const double* rho_core,
        bool is_kpt,
        bool is_soc);

private:
    // Auto-compute parameters that have sentinel values
    static void auto_compute_params(SCFParams& params, const FDGrid& grid, int rank_world);

    // Initialize density if not already set
    static void init_density(ElectronDensity& density, int Nd_d, int Nelectron,
                             int Nspin_global, const FDGrid& grid, bool is_soc);

    // Randomize wavefunctions with appropriate seeds
    static void randomize_wavefunctions(Wavefunction& wfn, int Nspin_local, int spin_start,
                                         const MPIComm& spincomm, const MPIComm& bandcomm,
                                         bool is_kpt);

    // Estimate spectral bounds via Lanczos
    static void estimate_spectral_bounds(
        SCFState& state,
        EigenSolver& eigsolver,
        const double* Veff,
        const double* Veff_spinor,
        int Nd_d,
        int Nspin_local, int spin_start,
        bool is_kpt, bool is_soc,
        const KPoints* kpoints, int kpt_start,
        const Vec3& cell_lengths,
        const Hamiltonian& hamiltonian,
        const NonlocalProjector* vnl,
        int rank_world);
};

} // namespace lynx
