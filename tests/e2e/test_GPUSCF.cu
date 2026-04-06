// GPU SCF End-to-End Test
// Uses the real LYNX SCF infrastructure with GPU dispatch (same pattern as
// test_EndToEnd.cpp) to run BaTiO3 and verify convergence + energy.

#ifdef USE_CUDA

#include <gtest/gtest.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>

#include "io/InputParser.hpp"
#include "core/LynxContext.hpp"
#include "core/ParameterDefaults.hpp"
#include "operators/Hamiltonian.hpp"
#include "operators/NonlocalProjector.hpp"
#include "atoms/Crystal.hpp"
#include "atoms/AtomType.hpp"
#include "electronic/Wavefunction.hpp"
#include "electronic/ElectronDensity.hpp"
#include "electronic/Occupation.hpp"
#include "xc/XCFunctional.hpp"
#include "physics/SCF.hpp"
#include "physics/Energy.hpp"
#include "physics/Electrostatics.hpp"
#include "atoms/AtomSetup.hpp"

using namespace lynx;

// Check that a GPU is available
static bool gpu_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

// Check that a file exists
static bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

TEST(GPUSCF, BaTiO3_SCF) {
    if (!gpu_available()) {
        GTEST_SKIP() << "No GPU available, skipping GPU SCF test";
    }

    const std::string json_file = "examples/BaTiO3.json";
    if (!file_exists(json_file)) {
        GTEST_SKIP() << "BaTiO3.json not found, skipping";
    }

    // Parse config and check PSP files exist
    auto config = InputParser::parse(json_file);
    InputParser::resolve_pseudopotentials(config);
    for (const auto& at : config.atom_types) {
        if (!file_exists(at.pseudo_file)) {
            GTEST_SKIP() << "PSP file not found: " << at.pseudo_file;
        }
    }
    InputParser::validate(config);

    // Initialize LynxContext
    auto& ctx = LynxContext::instance();
    ctx.reset();
    ctx.initialize(config, MPI_COMM_WORLD);

    const auto& lattice = ctx.lattice();
    const auto& grid = ctx.grid();
    const auto& domain = ctx.domain();
    const auto& stencil = ctx.stencil();
    int Nspin = ctx.Nspin();
    bool is_soc = ctx.is_soc();

    // Load pseudopotentials and build Crystal
    std::vector<AtomType> atom_types;
    std::vector<Vec3> all_positions;
    std::vector<int> type_indices;
    int total_Nelectron = 0;

    for (size_t it = 0; it < config.atom_types.size(); ++it) {
        const auto& at_in = config.atom_types[it];
        int n_atoms = static_cast<int>(at_in.coords.size());

        Pseudopotential psd_tmp;
        psd_tmp.load_psp8(at_in.pseudo_file);
        double Zval = psd_tmp.Zval();

        AtomType atype(at_in.element, 1.0, Zval, n_atoms);
        atype.psd().load_psp8(at_in.pseudo_file);

        for (int ia = 0; ia < n_atoms; ++ia) {
            Vec3 pos = at_in.coords[ia];
            if (at_in.fractional) {
                if (lattice.is_orthogonal()) {
                    pos = lattice.frac_to_cart(pos);
                } else {
                    Vec3 L = lattice.lengths();
                    pos = {pos.x * L.x, pos.y * L.y, pos.z * L.z};
                }
            } else if (!lattice.is_orthogonal()) {
                pos = lattice.cart_to_nonCart(pos);
            }
            all_positions.push_back(pos);
            type_indices.push_back(static_cast<int>(it));
        }

        total_Nelectron += static_cast<int>(Zval) * n_atoms;
        atom_types.push_back(std::move(atype));
    }

    int Nelectron = (config.Nelectron > 0) ? config.Nelectron : total_Nelectron;
    int Natom = static_cast<int>(all_positions.size());

    ParameterDefaults::update_default(config, grid, Nelectron, (Nspin == 2), is_soc);

    Crystal crystal(std::move(atom_types), all_positions, type_indices, lattice);

    // Atom influence
    double rc_max = 0.0;
    for (int it = 0; it < crystal.n_types(); ++it) {
        const auto& psd = crystal.types()[it].psd();
        for (auto rc : psd.rc()) rc_max = std::max(rc_max, rc);
        if (!psd.radial_grid().empty())
            rc_max = std::max(rc_max, psd.radial_grid().back());
    }
    rc_max += 8.0 * std::max({grid.dx(), grid.dy(), grid.dz()});

    std::vector<AtomInfluence> influence;
    crystal.compute_atom_influence(domain, rc_max, influence);

    std::vector<AtomNlocInfluence> nloc_influence;
    crystal.compute_nloc_influence(domain, nloc_influence);

    // Electrostatics
    Electrostatics elec;
    elec.compute_pseudocharge(crystal, influence, domain, grid, stencil);

    int Nd_d = domain.Nd_d();
    std::vector<double> Vloc(Nd_d, 0.0);
    elec.compute_Vloc(crystal, influence, domain, grid, Vloc.data());
    elec.compute_Ec(Vloc.data(), Nd_d, grid.dV());

    // Operators
    NonlocalProjector vnl;
    vnl.setup(crystal, nloc_influence, domain, grid);
    if (is_soc) vnl.setup_soc(crystal, nloc_influence, domain, grid);

    Hamiltonian hamiltonian;
    hamiltonian.setup(stencil, domain, grid, ctx.halo(), &vnl);

    // SCF setup with GPU dispatch
    auto scf_params = SCFParams::from_config(config);

    SCF scf;
    scf.setup(ctx, hamiltonian, &vnl, scf_params);
    scf.set_gpu_data(crystal, nloc_influence, influence, elec);

    Wavefunction wfn;
    wfn.allocate(Nd_d, ctx.Nband_local(), ctx.Nstates(),
                 ctx.Nspin_local(), ctx.Nkpts_local(),
                 ctx.is_kpt(), ctx.Nspinor());

    // Initialize density from atomic superposition
    {
        std::vector<double> rho_at(Nd_d, 0.0);
        elec.compute_atomic_density(crystal, influence, domain, grid,
                                    rho_at.data(), Nelectron);
        scf.set_initial_density(rho_at.data(), Nd_d, nullptr);
    }

    // Compute NLCC core density
    std::vector<double> rho_core(Nd_d, 0.0);
    bool has_nlcc = elec.compute_core_density(crystal, influence, domain, grid,
                                               rho_core.data());

    // Run SCF
    scf.run(wfn, Nelectron, Natom,
            elec.pseudocharge().data(), Vloc.data(),
            elec.Eself(), elec.Ec(), config.xc,
            has_nlcc ? rho_core.data() : nullptr);

    // Verify results
    double Etotal = scf.energy().Etotal;
    bool converged = scf.converged();

    std::printf("\n=== GPU SCF BaTiO3 Results ===\n");
    std::printf("  Converged: %s\n", converged ? "YES" : "NO");
    std::printf("  Etotal = %.10f Ha\n", Etotal);

    EXPECT_TRUE(converged) << "GPU SCF did not converge";
    // Reference ~ -136.92 Ha; coarse 15x15x15 grid so use 1.5 Ha tolerance
    EXPECT_NEAR(Etotal, -136.92, 1.5)
        << "GPU SCF energy too far from reference";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    lynx::LynxContext::instance().reset();
    MPI_Finalize();
    return result;
}

#else  // !USE_CUDA

#include <cstdio>
int main() {
    std::printf("GPU SCF test skipped: not compiled with USE_CUDA\n");
    return 0;
}

#endif  // USE_CUDA
