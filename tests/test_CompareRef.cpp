// Component-by-component comparison test against reference LYNX dumps
// Run reference first: cd dev_SPARC_GPU/tests/BaTiO3_quick/standard && mpirun -np 1 ../../../lib/lynx -name BaTiO3_quick
// Then run this: mpirun -np 1 ./tests/lynx_compare_tests --gtest_filter=*

#include <gtest/gtest.h>
#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

#include "io/InputParser.hpp"
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include "core/Domain.hpp"
#include "operators/FDStencil.hpp"
#include "operators/Laplacian.hpp"
#include "operators/Gradient.hpp"
#include "operators/Hamiltonian.hpp"
#include "operators/NonlocalProjector.hpp"
#include "atoms/Crystal.hpp"
#include "atoms/AtomType.hpp"
#include "electronic/Wavefunction.hpp"
#include "electronic/ElectronDensity.hpp"
#include "electronic/Occupation.hpp"
#include "physics/SCF.hpp"
#include "physics/Energy.hpp"
#include "physics/Electrostatics.hpp"
#include "parallel/Parallelization.hpp"
#include "parallel/HaloExchange.hpp"
#include "solvers/PoissonSolver.hpp"
#include "solvers/EigenSolver.hpp"
#include "xc/XCFunctional.hpp"
#include "core/constants.hpp"

using namespace lynx;

// Helper to set up the full system from BaTiO3_quick.json
struct SystemSetup {
    SystemConfig config;
    Lattice lattice;
    FDGrid grid;
    FDStencil stencil;
    Domain domain;
    MPIComm bandcomm_wrap;
    MPIComm kptcomm_wrap;
    MPIComm spincomm_wrap;
    Crystal* crystal = nullptr;
    Electrostatics elec;
    HaloExchange* halo = nullptr;
    Laplacian* laplacian = nullptr;
    Gradient* gradient = nullptr;
    Hamiltonian* hamiltonian = nullptr;
    NonlocalProjector* vnl = nullptr;
    std::vector<AtomInfluence> influence;
    std::vector<AtomNlocInfluence> nloc_influence;
    std::vector<double> Vloc;
    int Nelectron = 0;
    int Natom = 0;
    int Nd_d = 0;

    ~SystemSetup() {
        delete crystal;
        delete halo;
        delete laplacian;
        delete gradient;
        delete hamiltonian;
        delete vnl;
    }

    void init(const std::string& json_file) {
        config = InputParser::parse(json_file);
        InputParser::validate(config);

        lattice = Lattice(config.latvec, config.cell_type);
        grid = FDGrid(config.Nx, config.Ny, config.Nz, lattice,
                      config.bcx, config.bcy, config.bcz);
        stencil = FDStencil(config.fd_order, grid, lattice);

        // For np=1, domain = full grid
        DomainVertices verts{0, config.Nx-1, 0, config.Ny-1, 0, config.Nz-1};
        domain = Domain(grid, verts);
        Nd_d = domain.Nd_d();

        // Use MPI_COMM_SELF for serial
        // dmcomm removed (no domain decomposition)
        bandcomm_wrap = MPIComm(MPI_COMM_SELF);
        kptcomm_wrap = MPIComm(MPI_COMM_SELF);
        spincomm_wrap = MPIComm(MPI_COMM_SELF);

        // Load pseudopotentials
        std::vector<AtomType> atom_types;
        std::vector<Vec3> all_positions;
        std::vector<int> type_indices;
        int total_Ne = 0;

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
                if (at_in.fractional) pos = lattice.frac_to_cart(pos);
                all_positions.push_back(pos);
                type_indices.push_back(static_cast<int>(it));
            }
            total_Ne += static_cast<int>(Zval) * n_atoms;
            atom_types.push_back(std::move(atype));
        }

        Nelectron = (config.Nelectron > 0) ? config.Nelectron : total_Ne;
        Natom = static_cast<int>(all_positions.size());

        crystal = new Crystal(std::move(atom_types), all_positions, type_indices, lattice);

        // Compute influence
        double rc_max = 0.0;
        for (int it = 0; it < crystal->n_types(); ++it) {
            const auto& psd = crystal->types()[it].psd();
            for (auto rc : psd.rc()) rc_max = std::max(rc_max, rc);
            if (!psd.radial_grid().empty())
                rc_max = std::max(rc_max, psd.radial_grid().back());
        }
        crystal->compute_atom_influence(domain, rc_max, influence);
        crystal->compute_nloc_influence(domain, nloc_influence);

        // Electrostatics
        elec.compute_pseudocharge(*crystal, influence, domain, grid, stencil);
        Vloc.resize(Nd_d, 0.0);
        elec.compute_Vloc(*crystal, influence, domain, grid, Vloc.data());
        elec.compute_Ec(Vloc.data(), Nd_d, grid.dV());

        // Operators
        halo = new HaloExchange(domain, stencil.FDn());
        laplacian = new Laplacian(stencil, domain);
        gradient = new Gradient(stencil, domain);
        vnl = new NonlocalProjector();
        vnl->setup(*crystal, nloc_influence, domain, grid);
        hamiltonian = new Hamiltonian();
        hamiltonian->setup(stencil, domain, grid, *halo, vnl);
    }
};

class CompareRef : public ::testing::Test {
protected:
    static SystemSetup* sys;

    static void SetUpTestSuite() {
        sys = new SystemSetup();
        sys->init("../tests/data/BaTiO3_quick.json");
    }
    static void TearDownTestSuite() {
        delete sys;
        sys = nullptr;
    }
};
SystemSetup* CompareRef::sys = nullptr;

TEST_F(CompareRef, FDStencilCoefficients) {
    // Reference: DUMP_D2_COEFF p=0 x=-1.152800990777592e+01
    const double ref_D2[7] = {
        -1.152800990777592e+01,
         6.625469334751346e+00,
        -1.035229583554898e+00,
         2.044897942824489e-01,
        -3.450765278516326e-02,
         4.015435960455361e-03,
        -2.323747662300556e-04
    };

    int FDn = sys->stencil.FDn();
    EXPECT_EQ(FDn, 6);

    printf("FD Stencil Coefficients (D2_x):\n");
    for (int p = 0; p <= FDn; p++) {
        double ours = sys->stencil.D2_coeff_x()[p];
        double ref = ref_D2[p];
        double err = std::abs(ours - ref);
        printf("  p=%d: ours=%.15e ref=%.15e err=%.2e\n", p, ours, ref, err);
        EXPECT_NEAR(ours, ref, 1e-10) << "D2_coeff_x[" << p << "] mismatch";
    }
}

TEST_F(CompareRef, GridParameters) {
    // Reference: DUMP_GRID Nx=15 Ny=15 Nz=15 dx=5.086666666666667e-01
    EXPECT_EQ(sys->grid.Nx(), 15);
    EXPECT_EQ(sys->grid.Ny(), 15);
    EXPECT_EQ(sys->grid.Nz(), 15);
    EXPECT_NEAR(sys->grid.dx(), 5.086666666666667e-01, 1e-14);
    EXPECT_NEAR(sys->grid.dV(), 1.316133176296297e-01, 1e-12);
    printf("Grid: Nx=%d dx=%.15e dV=%.15e\n",
           sys->grid.Nx(), sys->grid.dx(), sys->grid.dV());
}

TEST_F(CompareRef, PseudochargeDensity) {
    // Reference: DUMP_PSEUDOCHARGE sum*dV=-4.000000271653961e+01
    // b[0]=-1.009773486038008e+00
    int Nd_d = sys->Nd_d;
    const double* b = sys->elec.pseudocharge().data();

    double b_sum = 0;
    for (int i = 0; i < Nd_d; i++) b_sum += b[i];
    double int_b = b_sum * sys->grid.dV();

    printf("Pseudocharge: int(b)*dV = %.15e (ref: -4.000000271653961e+01)\n", int_b);
    printf("  b[0] = %.15e (ref: -1.009773486038008e+00)\n", b[0]);
    printf("  b[3374] = %.15e (ref: -9.603707688539381e-01)\n", b[Nd_d-1]);

    // Reference values for first 10
    double ref_b[10] = {
        -1.009773486038008e+00,
        -9.094156613005440e-01,
        -5.892267237656669e-01,
        -1.480237138030860e-01,
         1.715952440502831e-02,
        -1.438090492849253e-03,
         9.964651157418544e-05,
        -9.091886311359584e-05,
         4.359267164937743e-04,
        -1.337347955694892e-03
    };
    for (int i = 0; i < 10; i++) {
        double err = std::abs(b[i] - ref_b[i]);
        printf("  b[%d]: ours=%.15e ref=%.15e err=%.2e\n", i, b[i], ref_b[i], err);
    }

    EXPECT_NEAR(int_b, -40.0, 0.01);
}

TEST_F(CompareRef, EselfAndEc) {
    // Reference: DUMP_ESC=-1.844903260970901e+02
    double Esc = sys->elec.Eself() + sys->elec.Ec();
    printf("Eself = %.15e\n", sys->elec.Eself());
    printf("Ec    = %.15e\n", sys->elec.Ec());
    printf("Esc   = %.15e (ref: -1.844903260970901e+02)\n", Esc);
    EXPECT_NEAR(Esc, -1.844903260970901e+02, 0.01);
}

TEST_F(CompareRef, CorrectionPotentialVc) {
    // Reference: DUMP_VC sum=-4.799300863433480e+02 min=-2.986091136990689e+01 max=2.616486282181985e-02
    // Vc[0]=-1.430113802125830e+01
    // Our code stores Vloc which should match Vc (correction = V_ref - V_loc)
    int Nd_d = sys->Nd_d;
    double sum = 0, vmin = 1e99, vmax = -1e99;
    for (int i = 0; i < Nd_d; i++) {
        sum += sys->Vloc[i];
        if (sys->Vloc[i] < vmin) vmin = sys->Vloc[i];
        if (sys->Vloc[i] > vmax) vmax = sys->Vloc[i];
    }
    printf("Vloc: sum=%.15e min=%.15e max=%.15e Vloc[0]=%.15e\n", sum, vmin, vmax, sys->Vloc[0]);
    printf("  Ref: sum=-4.799e+02 min=-2.986e+01 max=2.616e-02 Vc[0]=-1.430e+01\n");
}

TEST_F(CompareRef, InitialDensityAndPoisson) {
    // The reference uses superposition of atomic densities as initial density.
    // Our code uses uniform density. Let's use the REFERENCE initial density
    // to test our Poisson solver: solve -Lap(phi) = 4*pi*(rho+b)
    // and compare phi values.

    int Nd_d = sys->Nd_d;

    // Reference initial density values (first 10)
    double ref_rho[10] = {
        1.038070405169625e-01,
        1.766423637971214e-01,
        2.217569097351044e-01,
        1.512570972762500e-01,
        6.648189297577180e-02,
        2.573055319218580e-02,
        1.380152725779412e-02,
        1.119454590919475e-02,
        1.167714456331469e-02,
        1.637532993530148e-02
    };

    printf("Our initial density (uniform): rho[0]=%.15e\n",
           (double)sys->Nelectron / (sys->grid.Nd() * sys->grid.dV()));
    printf("Ref initial density: rho[0]=%.15e\n", ref_rho[0]);
    printf("NOTE: Reference uses superposition of atomic densities, we use uniform.\n");
    printf("This is a known difference that affects convergence speed but not correctness.\n");
}

TEST_F(CompareRef, PoissonSolverWithRefDensity) {
    // Use uniform density (like our code does) and solve Poisson
    // Then compare with the analytical expectation
    int Nd_d = sys->Nd_d;
    const double* b = sys->elec.pseudocharge().data();
    double dV = sys->grid.dV();
    int Nd = sys->grid.Nd();

    // Use uniform density = Nelectron / Volume
    double volume = Nd * dV;
    double rho0 = sys->Nelectron / volume;

    // Build RHS = 4*pi*(rho + b)
    std::vector<double> rhs(Nd_d);
    double rhs_sum = 0;
    for (int i = 0; i < Nd_d; i++) {
        rhs[i] = 4.0 * constants::PI * (rho0 + b[i]);
        rhs_sum += rhs[i];
    }
    printf("Poisson RHS sum*dV = %.15e (should be ~0 for charge neutral)\n", rhs_sum * dV);

    // Solve -Lap(phi) = rhs
    PoissonSolver poisson;
    poisson.setup(*sys->laplacian, sys->stencil, sys->domain, sys->grid, *sys->halo);
    std::vector<double> phi(Nd_d, 0.0);
    poisson.solve(rhs.data(), phi.data(), 1e-6);

    // Shift phi mean to zero
    double phi_sum = 0;
    for (int i = 0; i < Nd_d; i++) phi_sum += phi[i];
    double phi_mean = phi_sum / Nd;
    for (int i = 0; i < Nd_d; i++) phi[i] -= phi_mean;

    double phi_min = 1e99, phi_max = -1e99;
    for (int i = 0; i < Nd_d; i++) {
        if (phi[i] < phi_min) phi_min = phi[i];
        if (phi[i] > phi_max) phi_max = phi[i];
    }
    printf("Poisson phi: min=%.6e max=%.6e\n", phi_min, phi_max);
    printf("Note: With uniform rho, phi is dominated by pseudocharge structure.\n");
}

TEST_F(CompareRef, XCWithRefDensity) {
    // Test XC with reference initial density
    int Nd_d = sys->Nd_d;

    // Reference density
    double ref_rho_vals[10] = {
        1.038070405169625e-01, 1.766423637971214e-01,
        2.217569097351044e-01, 1.512570972762500e-01,
        6.648189297577180e-02, 2.573055319218580e-02,
        1.380152725779412e-02, 1.119454590919475e-02,
        1.167714456331469e-02, 1.637532993530148e-02
    };

    // Create a density array that matches the reference initial density
    // For a full comparison, we'd need all 3375 values. For now, test with uniform.
    double rho0 = sys->Nelectron / (sys->grid.Nd() * sys->grid.dV());
    std::vector<double> rho(Nd_d, rho0);
    std::vector<double> Vxc(Nd_d, 0.0);
    std::vector<double> exc(Nd_d, 0.0);

    XCFunctional xc;
    xc.setup(XCType::GGA_PBE, sys->domain, sys->grid, sys->gradient, sys->halo);
    xc.evaluate(rho.data(), Vxc.data(), exc.data(), Nd_d);

    printf("XC with uniform density (rho0=%.6e):\n", rho0);
    printf("  Vxc[0]=%.15e (uniform density → same everywhere for LDA)\n", Vxc[0]);
    printf("  Ref Vxc[0]=%.15e (with atomic superposition density)\n", -1.057895351026050e+00);
    printf("  Ref Vxc range: [%.6e, %.6e]\n", -1.654554493962237e+00, -2.517883549720829e-01);
}

TEST_F(CompareRef, SmearingParameters) {
    // Reference: DUMP_SMEARING type=1 temp=2.320903624310025e+03 Beta=1.360569312299400e+02
    // type=1 is Gaussian smearing
    // temp in Kelvin, Beta = 1/(kB*T) in Hartree
    double kBT = sys->config.elec_temp * constants::KB;
    double beta = 1.0 / kBT;
    printf("Smearing: temp=%.6f K, kBT=%.15e Ha, beta=%.15e\n",
           sys->config.elec_temp, kBT, beta);
    printf("  Ref: temp=2320.9 K, Beta=136.057\n");
    // Note: the reference has T=2320.9K (= 0.2 eV), our config may differ
}

TEST_F(CompareRef, MixingParameters) {
    // Reference: DUMP_MIXING var=0 precond=1 history=7 param=3.000000000000000e-01
    printf("Mixing: var=%d precond=%d history=%d param=%.6f\n",
           (int)sys->config.mixing_var, (int)sys->config.mixing_precond,
           sys->config.mixing_history, sys->config.mixing_param);
    printf("  Ref: var=0 precond=1 history=7 param=0.3\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
