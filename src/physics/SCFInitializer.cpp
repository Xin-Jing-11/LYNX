#include "physics/SCFInitializer.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mpi.h>

namespace lynx {

static bool is_mgga_type(XCType t) {
    return t == XCType::MGGA_SCAN || t == XCType::MGGA_RSCAN || t == XCType::MGGA_R2SCAN;
}

void SCFInitializer::auto_compute_params(SCFParams& params, const FDGrid& grid, int rank_world) {
    // All parameters should be pre-resolved by ParameterDefaults::update_default() in main.cpp.
    // This function is now a no-op; kept for backward compatibility with test code
    // that may not call ParameterDefaults.
    (void)grid;
    (void)rank_world;
}

void SCFInitializer::init_density(ElectronDensity& density, int Nd_d, int Nelectron,
                                   int Nspin_global, const FDGrid& grid, bool is_soc) {
    if (is_soc) {
        if (density.Nd_d() == 0) {
            density.allocate_noncollinear(Nd_d);
            double volume = grid.Nd() * grid.dV();
            double rho0 = Nelectron / volume;
            double* rho = density.rho_total().data();
            for (int i = 0; i < Nd_d; ++i) rho[i] = rho0;
            std::memcpy(density.rho(0).data(), rho, Nd_d * sizeof(double));
            density.mag_x().zero();
            density.mag_y().zero();
            density.mag_z().zero();
        } else if (density.mag_x().size() == 0) {
            NDArray<double> rho_save = density.rho_total().clone();
            density.allocate_noncollinear(Nd_d);
            std::memcpy(density.rho_total().data(), rho_save.data(), Nd_d * sizeof(double));
            std::memcpy(density.rho(0).data(), rho_save.data(), Nd_d * sizeof(double));
            density.mag_x().zero();
            density.mag_y().zero();
            density.mag_z().zero();
        }
    } else {
        if (density.Nd_d() == 0) {
            density.allocate(Nd_d, Nspin_global);
            double volume = grid.Nd() * grid.dV();
            double rho0 = Nelectron / volume;

            if (Nspin_global == 1) {
                double* rho = density.rho(0).data();
                for (int i = 0; i < Nd_d; ++i) rho[i] = rho0;
            } else {
                double* rho_up = density.rho(0).data();
                double* rho_dn = density.rho(1).data();
                for (int i = 0; i < Nd_d; ++i) {
                    rho_up[i] = rho0 * 0.5;
                    rho_dn[i] = rho0 * 0.5;
                }
            }
            double* rho_t = density.rho_total().data();
            for (int i = 0; i < Nd_d; ++i) rho_t[i] = rho0;
        }
    }
}

void SCFInitializer::randomize_wavefunctions(Wavefunction& wfn, int Nspin_local, int spin_start,
                                               const MPIComm& spincomm, const MPIComm& bandcomm,
                                               bool is_kpt) {
    int spincomm_rank = spincomm.is_null() ? 0 : spincomm.rank();
    int bandcomm_rank = bandcomm.is_null() ? 0 : bandcomm.rank();
    int Nkpts = wfn.Nkpts();

    for (int s = 0; s < Nspin_local; ++s) {
        int s_glob = spin_start + s;
        unsigned rand_seed = spincomm_rank * 100 + bandcomm_rank * 10 + s_glob * 1000 + 1;
        for (int k = 0; k < Nkpts; ++k) {
            if (is_kpt) {
                wfn.randomize_kpt(s, k, rand_seed);
            } else {
                wfn.randomize(s, k, rand_seed);
            }
        }
    }
}

void SCFInitializer::estimate_spectral_bounds(
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
    int rank_world) {

    state.eigval_min.resize(Nspin_local, 0.0);
    state.eigval_max.resize(Nspin_local, 0.0);

    if (is_soc) {
        Vec3 kpt0 = kpoints->kpts_cart()[kpt_start];
        // Set k-point on nonlocal projector BEFORE Lanczos (required for SOC)
        if (vnl && vnl->is_setup()) {
            const_cast<NonlocalProjector*>(vnl)->set_kpoint(kpt0);
            const_cast<Hamiltonian&>(hamiltonian).set_vnl_kpt(vnl);
        }
        double eigmin_spinor, eigmax_spinor;
        eigsolver.lanczos_bounds_spinor_kpt(Veff_spinor, Nd_d,
                                             kpt0, cell_lengths,
                                             eigmin_spinor, eigmax_spinor);
        double eigmin_scalar, eigmax_scalar;
        eigsolver.lanczos_bounds_kpt(Veff, Nd_d, kpt0, cell_lengths,
                                      eigmin_scalar, eigmax_scalar);
        state.eigval_min[0] = eigmin_spinor;
        state.eigval_max[0] = eigmax_scalar;

        // For SOC: cutoff uses scalar midpoint
        double eigmin_s2, eigmax_s2;
        eigsolver.lanczos_bounds_kpt(Veff, Nd_d, kpt0, cell_lengths,
                                      eigmin_s2, eigmax_s2);
        state.lambda_cutoff = 0.5 * (eigmin_s2 + state.eigval_max[0]);
    } else {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start + s;
            if (is_kpt) {
                Vec3 kpt0 = kpoints->kpts_cart()[kpt_start];
                eigsolver.lanczos_bounds_kpt(Veff + s_glob * Nd_d, Nd_d,
                                              kpt0, cell_lengths,
                                              state.eigval_min[s], state.eigval_max[s]);
            } else {
                eigsolver.lanczos_bounds(Veff + s_glob * Nd_d, Nd_d,
                                          state.eigval_min[s], state.eigval_max[s]);
            }
        }
        state.lambda_cutoff = 0.5 * (state.eigval_min[0] + state.eigval_max[0]);
    }

    if (rank_world == 0) {
        for (int s = 0; s < Nspin_local; ++s)
            std::printf("Lanczos bounds (spin %d): eigmin=%.6e, eigmax=%.6e\n",
                        spin_start + s, state.eigval_min[s], state.eigval_max[s]);
    }
}

SCFState SCFInitializer::initialize(
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
    bool is_soc) {

    int rank_world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
    int Nd_d = domain.Nd_d();
    int Nspin = Nspin_global;

    // 1. Auto-compute parameters
    auto_compute_params(params, grid, rank_world);

    // 2. Populate state scalars
    SCFState state;
    state.Nband = wfn.Nband_global();
    state.Nband_loc = wfn.Nband();
    state.Nspin_local = wfn.Nspin();
    state.Nkpts = wfn.Nkpts();
    state.kBT = params.elec_temp * constants::KB;
    state.beta = 1.0 / state.kBT;

    // K-point weights
    int Nkpts_global = kpoints ? kpoints->Nkpts() : state.Nkpts;
    if (kpoints) {
        state.kpt_weights = kpoints->normalized_weights();
    } else {
        state.kpt_weights.assign(Nkpts_global, 1.0 / Nkpts_global);
    }

    // 3. Allocate work arrays
    arrays.allocate(Nd_d, Nspin, xc_type, is_soc);

    // 4. Initialize density
    init_density(density, Nd_d, Nelectron, Nspin_global, grid, is_soc);

    // 5. Setup potential mixing state
    state.use_potential_mixing = (params.mixing_var == MixingVariable::Potential) && !is_soc;
    if (state.use_potential_mixing) {
        state.Veff_mean.resize(Nspin, 0.0);
    }

    // 6. Compute initial Veff from initial density
    if (is_soc) {
        veff_builder.compute_spinor(density, rho_b, rho_core, xc_type, params.poisson_tol, arrays);
    } else {
        veff_builder.compute(density, rho_b, rho_core, xc_type, 0.0, params.poisson_tol, arrays);
    }

    // 7. For potential mixing: initialize zero-mean copy
    if (state.use_potential_mixing) {
        state.Veff_mixed = NDArray<double>(Nd_d * Nspin);
        std::memcpy(state.Veff_mixed.data(), arrays.Veff.data(), Nd_d * Nspin * sizeof(double));
        for (int s = 0; s < Nspin; ++s) {
            double mean = 0;
            for (int i = 0; i < Nd_d; ++i) mean += state.Veff_mixed.data()[s*Nd_d + i];
            mean /= grid.Nd();
            state.Veff_mean[s] = mean;
            for (int i = 0; i < Nd_d; ++i) state.Veff_mixed.data()[s*Nd_d + i] -= mean;
        }
    }

    // 8. Estimate spectral bounds via Lanczos
    Vec3 cell_lengths = grid.lattice().lengths();
    estimate_spectral_bounds(
        state, eigsolver,
        arrays.Veff.data(),
        is_soc ? arrays.Veff_spinor.data() : nullptr,
        Nd_d, Nspin_local, spin_start,
        is_kpt, is_soc, kpoints, kpt_start,
        cell_lengths, hamiltonian, vnl, rank_world);

    if (rank_world == 0 && params.cheb_degree > 0)
        std::printf("Chebyshev degree: %d\n", params.cheb_degree);

    // 9. Randomize wavefunctions
    randomize_wavefunctions(wfn, Nspin_local, spin_start, spincomm, bandcomm, is_kpt);

    return state;
}

} // namespace lynx
