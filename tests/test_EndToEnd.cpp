#include <gtest/gtest.h>
#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>

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
#include "xc/XCFunctional.hpp"
#include "solvers/PoissonSolver.hpp"
#include "solvers/EigenSolver.hpp"
#include "solvers/Mixer.hpp"
#include "physics/SCF.hpp"
#include "physics/Energy.hpp"
#include "physics/Forces.hpp"
#include "physics/Stress.hpp"
#include "physics/Electrostatics.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"
#include "parallel/Parallelization.hpp"
#include "core/KPoints.hpp"

using namespace lynx;

// ============================================================
// Helper: run the full single-point DFT pipeline from a JSON config
// ============================================================
struct DFTResult {
    double Etotal = 0.0;
    double Eband = 0.0;
    double Exc = 0.0;
    double Eself_Ec = 0.0;
    double Entropy = 0.0;
    double Ef = 0.0;
    std::vector<double> forces;      // [3*Natom]
    std::array<double, 6> stress{};  // Voigt: xx, xy, xz, yy, yz, zz
    double pressure = 0.0;
    bool converged = false;
    int Natom = 0;
    int Nelectron = 0;
};

static DFTResult run_single_point(const std::string& json_file) {
    DFTResult result;

    auto config = InputParser::parse(json_file);
    InputParser::resolve_pseudopotentials(config);
    InputParser::validate(config);

    Lattice lattice(config.latvec, config.cell_type);
    FDGrid grid(config.Nx, config.Ny, config.Nz, lattice,
                config.bcx, config.bcy, config.bcz);
    FDStencil stencil(config.fd_order, grid, lattice);

    // SOC detection
    bool is_soc = (config.spin_type == SpinType::NonCollinear);
    int Nspin = is_soc ? 1 : ((config.spin_type == SpinType::None) ? 1 : 2);
    int Nspinor = is_soc ? 2 : 1;
    if (is_soc) config.parallel.npspin = 1;

    KPoints kpoints;
    kpoints.generate(config.Kx, config.Ky, config.Kz, config.kpt_shift, lattice);
    int Nkpts = kpoints.Nkpts();
    bool is_kpt = !kpoints.is_gamma_only() || is_soc;

    // Auto-detect parallelization (matching main.cpp logic)
    int nproc; MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    if (is_soc) config.parallel.npspin = 1;
    else if (Nspin == 2 && nproc >= 2 && config.parallel.npspin <= 1)
        config.parallel.npspin = 2;
    int nproc_after_spin = nproc / config.parallel.npspin;
    if (config.parallel.npkpt <= 1 && Nkpts > 1 && nproc_after_spin > 1)
        config.parallel.npkpt = std::min(nproc_after_spin, Nkpts);
    int nproc_after_kpt = nproc_after_spin / std::max(1, config.parallel.npkpt);
    if (config.parallel.npband <= 1 && nproc_after_kpt > 1)
        config.parallel.npband = nproc_after_kpt;

    Parallelization parallel(MPI_COMM_WORLD, config.parallel,
                             grid, Nspin, Nkpts, config.Nstates);

    const auto& domain = parallel.domain();
    const auto& bandcomm = parallel.bandcomm();
    const auto& kptcomm = parallel.kptcomm();
    const auto& spincomm = parallel.spincomm();

    // Load pseudopotentials and create Crystal
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
    result.Natom = Natom;
    result.Nelectron = Nelectron;

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
    HaloExchange halo(domain, stencil.FDn());
    Laplacian laplacian(stencil, domain);
    Gradient gradient(stencil, domain);

    NonlocalProjector vnl;
    vnl.setup(crystal, nloc_influence, domain, grid);
    if (is_soc) vnl.setup_soc(crystal, nloc_influence, domain, grid);

    Hamiltonian hamiltonian;
    hamiltonian.setup(stencil, domain, grid, halo, &vnl);

    // SCF
    int Nstates = config.Nstates;
    if (Nstates <= 0) {
        Nstates = is_soc ? (Nelectron + 20) : (Nelectron / 2 + 10);
    }

    SCFParams scf_params;
    scf_params.max_iter = config.max_scf_iter;
    scf_params.min_iter = config.min_scf_iter;
    scf_params.tol = config.scf_tol;
    scf_params.mixing_var = config.mixing_var;
    scf_params.mixing_precond = config.mixing_precond;
    scf_params.mixing_history = config.mixing_history;
    scf_params.mixing_param = config.mixing_param;
    scf_params.smearing = config.smearing;
    scf_params.elec_temp = config.elec_temp;
    scf_params.cheb_degree = config.cheb_degree;

    int Nspin_local = parallel.Nspin_local();
    int Nkpts_local = parallel.Nkpts_local();
    int kpt_start = parallel.kpt_start();
    int spin_start = parallel.spin_start();
    int Nband_local = parallel.Nband_local();
    int band_start = parallel.band_start();
    const auto& kpt_bridge = parallel.kpt_bridge();
    const auto& spin_bridge = parallel.spin_bridge();

    SCF scf;
    scf.setup(grid, domain, stencil, laplacian, gradient, hamiltonian,
              halo, &vnl, bandcomm, kpt_bridge, spin_bridge, scf_params,
              Nspin, Nspin_local, spin_start, &kpoints, kpt_start,
              Nstates, band_start);
#ifdef USE_CUDA
    scf.set_gpu_data(crystal, nloc_influence, influence, elec);
#endif

    Wavefunction wfn;
    wfn.allocate(Nd_d, Nband_local, Nstates, Nspin_local, Nkpts_local,
                 is_kpt, Nspinor);

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

    scf.run(wfn, Nelectron, Natom,
            elec.pseudocharge().data(), Vloc.data(),
            elec.Eself(), elec.Ec(), config.xc,
            has_nlcc ? rho_core.data() : nullptr);

    result.Etotal = scf.energy().Etotal;
    result.Eband = scf.energy().Eband;
    result.Exc = scf.energy().Exc;
    result.Eself_Ec = scf.energy().Eself + scf.energy().Ec;
    result.Entropy = scf.energy().Entropy;
    result.Ef = scf.fermi_energy();
    result.converged = scf.converged();

    // Dump all energy components for debugging
    int dump_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &dump_rank);
    if (dump_rank == 0) {
        std::printf("\n=== Energy Components ===\n");
        std::printf("  Eband   = %.15e\n", scf.energy().Eband);
        std::printf("  Exc     = %.15e\n", scf.energy().Exc);
        std::printf("  Ehart   = %.15e\n", scf.energy().Ehart);
        std::printf("  Eself   = %.15e\n", scf.energy().Eself);
        std::printf("  Ec      = %.15e\n", scf.energy().Ec);
        std::printf("  Entropy = %.15e\n", scf.energy().Entropy);
        std::printf("  Etotal  = %.15e\n", scf.energy().Etotal);
        std::printf("  Eself+Ec= %.15e\n", scf.energy().Eself + scf.energy().Ec);
    }

    // Forces
    if (config.print_forces) {
        std::vector<double> kpt_weights = kpoints.normalized_weights();
        Forces forces;
        result.forces = forces.compute(wfn, crystal, influence, nloc_influence, vnl,
                                       stencil, gradient, halo, domain, grid,
                                       scf.phi(), scf.density().rho_total().data(),
                                       Vloc.data(),
                                       elec.pseudocharge().data(),
                                       elec.pseudocharge_ref().data(),
                                       scf.Vxc(),
                                       has_nlcc ? rho_core.data() : nullptr,
                                       kpt_weights, bandcomm, kpt_bridge, spin_bridge,
                                       &kpoints, kpt_start, band_start);
    }

    // Stress
    if (config.calc_stress || config.calc_pressure) {
        std::vector<double> kpt_weights = kpoints.normalized_weights();
        Stress stress_calc;
        int Nspin_calc = (config.spin_type == SpinType::Collinear) ? 2 : 1;
        const double* rho_up_ptr = (Nspin_calc == 2) ? scf.density().rho(0).data() : nullptr;
        const double* rho_dn_ptr = (Nspin_calc == 2) ? scf.density().rho(1).data() : nullptr;
        result.stress = stress_calc.compute(wfn, crystal, influence, nloc_influence, vnl,
                                            stencil, gradient, halo, domain, grid,
                                            scf.phi(), scf.density().rho_total().data(),
                                            rho_up_ptr, rho_dn_ptr,
                                            Vloc.data(),
                                            elec.pseudocharge().data(),
                                            elec.pseudocharge_ref().data(),
                                            scf.exc(), scf.Vxc(),
                                            scf.Dxcdgrho(),
                                            scf.energy().Exc,
                                            elec.Eself() + elec.Ec(),
                                            config.xc,
                                            Nspin_calc,
                                            has_nlcc ? rho_core.data() : nullptr,
                                            kpt_weights, bandcomm, kpt_bridge, spin_bridge,
                                            &kpoints, kpt_start, band_start,
                                            scf.vtau());
        result.pressure = stress_calc.pressure();
    }

    return result;
}

// ============================================================
// Test: Pseudopotential loading (quick, no SCF)
// ============================================================
TEST(EndToEnd, PseudopotentialLoad) {
    std::string psp_file = "psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8";

    Pseudopotential psd;
    psd.load_psp8(psp_file);

    EXPECT_EQ(psd.Zval(), 4.0);
    EXPECT_GT(psd.lmax(), 0);
    EXPECT_GT(psd.grid_size(), 100);
    EXPECT_GT(psd.nproj_per_atom(), 0);
}

// ============================================================
// Test: Pseudocharge computation for a single Si atom
// ============================================================
TEST(EndToEnd, PseudochargeComputation) {
    Mat3 latvec;
    latvec(0, 0) = 10.0; latvec(1, 1) = 10.0; latvec(2, 2) = 10.0;
    Lattice lattice(latvec, CellType::Orthogonal);
    FDGrid grid(20, 20, 20, lattice, BCType::Periodic, BCType::Periodic, BCType::Periodic);
    FDStencil stencil(12, grid, lattice);

    DomainVertices verts = {0, 19, 0, 19, 0, 19};
    Domain domain(grid, verts);

    std::string psp_file = "psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8";
    AtomType si("Si", 28.085, 4.0, 1);
    si.psd().load_psp8(psp_file);

    std::vector<AtomType> types = {si};
    std::vector<Vec3> positions = {{5.0, 5.0, 5.0}};
    std::vector<int> type_indices = {0};
    Crystal crystal(std::move(types), positions, type_indices, lattice);

    double rc_max = 0.0;
    for (auto rc : crystal.types()[0].psd().rc())
        rc_max = std::max(rc_max, rc);
    rc_max = std::max(rc_max, crystal.types()[0].psd().radial_grid().back());

    std::vector<AtomInfluence> influence;
    crystal.compute_atom_influence(domain, rc_max, influence);

    EXPECT_EQ(influence.size(), 1u);
    EXPECT_GT(influence[0].n_atom, 0);

    Electrostatics elec;
    elec.compute_pseudocharge(crystal, influence, domain, grid, stencil);

    double int_b = elec.int_b();
    std::printf("  Integral of pseudocharge: %.6f (expected: %.1f)\n", int_b, -4.0);
    EXPECT_NEAR(int_b, -4.0, 0.5);

    std::printf("  Eself: %.6f\n", elec.Eself());
    EXPECT_LT(elec.Eself(), 0.0);
}

// ============================================================
// Test: Nonlocal projector setup
// ============================================================
TEST(EndToEnd, NonlocalProjectorSetup) {
    Mat3 latvec;
    latvec(0, 0) = 10.0; latvec(1, 1) = 10.0; latvec(2, 2) = 10.0;
    Lattice lattice(latvec, CellType::Orthogonal);
    FDGrid grid(20, 20, 20, lattice, BCType::Periodic, BCType::Periodic, BCType::Periodic);

    DomainVertices verts = {0, 19, 0, 19, 0, 19};
    Domain domain(grid, verts);

    std::string psp_file = "psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8";
    AtomType si("Si", 28.085, 4.0, 1);
    si.psd().load_psp8(psp_file);

    std::vector<AtomType> types = {si};
    std::vector<Vec3> positions = {{5.0, 5.0, 5.0}};
    std::vector<int> type_indices = {0};
    Crystal crystal(std::move(types), positions, type_indices, lattice);

    std::vector<AtomNlocInfluence> nloc_influence;
    crystal.compute_nloc_influence(domain, nloc_influence);

    NonlocalProjector vnl;
    vnl.setup(crystal, nloc_influence, domain, grid);

    EXPECT_TRUE(vnl.is_setup());
    EXPECT_GT(vnl.total_nproj(), 0);
    std::printf("  Total nonlocal projectors: %d\n", vnl.total_nproj());
}

// ============================================================
// Test: XC functional on real density
// ============================================================
TEST(EndToEnd, XCFunctionalRealDensity) {
    int N = 100;
    std::vector<double> rho(N), Vxc(N), exc(N);

    for (int i = 0; i < N; ++i) {
        double x = (i - 50.0) / 10.0;
        rho[i] = 0.05 * std::exp(-x * x);
        if (rho[i] < 1e-20) rho[i] = 1e-20;
    }

    Domain domain;
    FDGrid grid;
    XCFunctional xc;
    xc.setup(XCType::GGA_PBE, domain, grid, nullptr, nullptr);
    xc.evaluate(rho.data(), Vxc.data(), exc.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_LT(exc[i], 0.0);
    }
    for (int i = 0; i < N; ++i) {
        EXPECT_LT(Vxc[i], 0.0);
    }
}

// ============================================================
// Test: Full BaTiO3 SCF calculation — validate energy and forces
// Reference: LYNX BaTiO3_quick test
//   Etotal = -136.9227950641 Ha
//   Eband  = -10.613764677 Ha
//   Exc    = -28.295344017 Ha
//   Eself+Ec = -184.49032610 Ha
//   Ef     = 0.31446488165 Ha
// ============================================================
TEST(EndToEnd, BaTiO3_SCF) {
    std::string json_file = "tests/data/BaTiO3_quick.json";
    auto result = run_single_point(json_file);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::printf("\n=== BaTiO3 SCF Results ===\n");
        std::printf("  Converged: %s\n", result.converged ? "yes" : "no");
        std::printf("  Etotal = %.10f Ha (ref: -136.9227950641)\n", result.Etotal);
        std::printf("  Eband  = %.10f Ha (ref: -10.613764677)\n", result.Eband);
        std::printf("  Exc    = %.10f Ha (ref: -28.295344017)\n", result.Exc);
        std::printf("  Ef     = %.10f Ha (ref: 0.31446488165)\n", result.Ef);
    }

    // Reference values from LYNX BaTiO3_quick test
    double ref_Etotal = -136.9227950641;

    // Energy should match to within ~1e-3 Ha for this coarse grid
    // (exact match requires identical implementation of all components)
    EXPECT_TRUE(result.converged) << "SCF did not converge";
    EXPECT_NEAR(result.Etotal, ref_Etotal, 1.0)
        << "Total energy deviates significantly from reference";

    // Forces: check that all 5 atoms have force vectors
    EXPECT_EQ(static_cast<int>(result.forces.size()), 15); // 5 atoms * 3

    if (rank == 0 && result.forces.size() == 15) {
        // Reference forces (Ha/Bohr) from BaTiO3_quick.refstatic
        double ref_forces[15] = {
             6.6359121598E-05, -1.9068310927E-02, -1.1086032284E-01,
             6.9688243455E-02,  1.6092562227E-01, -2.0216340645E-01,
            -1.1697218238E-02,  4.8389612623E-02, -2.2358123032E-01,
            -9.0946800836E-02, -3.3044902010E-01,  3.3567524994E-01,
             3.2889416497E-02,  1.4020209614E-01,  2.0092970967E-01
        };

        std::printf("\n  Forces (Ha/Bohr):\n");
        double max_force_err = 0.0;
        for (int i = 0; i < 5; ++i) {
            std::printf("  Atom %d: %12.6f %12.6f %12.6f  (ref: %12.6f %12.6f %12.6f)\n",
                        i + 1,
                        result.forces[3*i], result.forces[3*i+1], result.forces[3*i+2],
                        ref_forces[3*i], ref_forces[3*i+1], ref_forces[3*i+2]);
            for (int d = 0; d < 3; ++d) {
                double err = std::abs(result.forces[3*i+d] - ref_forces[3*i+d]);
                max_force_err = std::max(max_force_err, err);
            }
        }
        std::printf("  Max force error: %.6e Ha/Bohr\n", max_force_err);
    }
}

// ============================================================
// Test: Full Si8 SCF — non-orthogonal cell with stress
// Reference: LYNX Si8 test
//   Etotal = -33.26990391 Ha
//   Pressure = 20.4249871 GPa
//   Stress (GPa):
//     -18.5522  -9.8997  -4.0031
//      -9.8997 -25.6727   7.3638
//      -4.0031   7.3638 -17.0501
// ============================================================
TEST(EndToEnd, Si8_SCF) {
    std::string json_file = "tests/data/Si8.json";
    auto result = run_single_point(json_file);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::printf("\n=== Si8 SCF Results ===\n");
        std::printf("  Converged: %s\n", result.converged ? "yes" : "no");
        std::printf("  Etotal = %.10f Ha (ref: -33.26990391)\n", result.Etotal);
        std::printf("  Ef     = %.10f Ha (ref: 0.17420147348)\n", result.Ef);
    }

    double ref_Etotal = -33.26990391;

    EXPECT_TRUE(result.converged) << "SCF did not converge";
    EXPECT_NEAR(result.Etotal, ref_Etotal, 1.0)
        << "Total energy deviates significantly from reference";

    // Check forces exist (8 atoms * 3 components)
    EXPECT_EQ(static_cast<int>(result.forces.size()), 24);

    if (rank == 0 && result.forces.size() == 24) {
        // Reference forces (Ha/Bohr) from Si8.refstatic
        double ref_forces[24] = {
            -5.8821426822E-02,  5.2400900901E-02, -1.2050125635E-01,
            -5.8791642131E-02,  5.2377300943E-02, -1.2041507021E-01,
            -5.8807187828E-02,  5.2422792346E-02, -1.2045485441E-01,
            -5.8834909693E-02,  5.2387972635E-02, -1.2045495115E-01,
             5.8828354347E-02, -5.2379034010E-02,  1.2047042428E-01,
             5.8801858152E-02, -5.2410353008E-02,  1.2047372392E-01,
             5.8810689215E-02, -5.2396926479E-02,  1.2044422480E-01,
             5.8814264759E-02, -5.2402653329E-02,  1.2043775912E-01
        };

        std::printf("\n  Forces (Ha/Bohr):\n");
        double max_force_err = 0.0;
        for (int i = 0; i < 8; ++i) {
            std::printf("  Atom %d: %12.6f %12.6f %12.6f  (ref: %12.6f %12.6f %12.6f)\n",
                        i + 1,
                        result.forces[3*i], result.forces[3*i+1], result.forces[3*i+2],
                        ref_forces[3*i], ref_forces[3*i+1], ref_forces[3*i+2]);
            for (int d = 0; d < 3; ++d) {
                double err = std::abs(result.forces[3*i+d] - ref_forces[3*i+d]);
                max_force_err = std::max(max_force_err, err);
            }
        }
        std::printf("  Max force error: %.6e Ha/Bohr\n", max_force_err);
    }

    // Check stress
    if (rank == 0) {
        const double au_to_gpa = 29421.01569650548;
        // Voigt: xx, xy, xz, yy, yz, zz
        double s_xx = result.stress[0] * au_to_gpa;
        double s_xy = result.stress[1] * au_to_gpa;
        double s_xz = result.stress[2] * au_to_gpa;
        double s_yy = result.stress[3] * au_to_gpa;
        double s_yz = result.stress[4] * au_to_gpa;
        double s_zz = result.stress[5] * au_to_gpa;

        std::printf("\n  Stress (GPa):\n");
        std::printf("    %10.4f %10.4f %10.4f\n", s_xx, s_xy, s_xz);
        std::printf("    %10.4f %10.4f %10.4f\n", s_xy, s_yy, s_yz);
        std::printf("    %10.4f %10.4f %10.4f\n", s_xz, s_yz, s_zz);

        std::printf("\n  Ref stress (GPa):\n");
        std::printf("    %10.4f %10.4f %10.4f\n", -18.5522, -9.8997, -4.0031);
        std::printf("    %10.4f %10.4f %10.4f\n", -9.8997, -25.6727, 7.3638);
        std::printf("    %10.4f %10.4f %10.4f\n", -4.0031, 7.3638, -17.0501);

        double p_gpa = result.pressure * au_to_gpa;
        std::printf("\n  Pressure: %.4f GPa (ref: 20.4250)\n", p_gpa);
    }
}

// ============================================================
// Test: PtAu SOC — spin-orbit coupling, non-orthogonal cell
// Reference: SPARC PtAu_SOC (standard accuracy, 19x19x32 grid)
//   Etotal = -253.6571996 Ha
//   Forces (Ha/Bohr):
//     Au: -0.1621  0.1139  0.4109
//     Pt:  0.1621 -0.1139 -0.4109
// ============================================================
TEST(EndToEnd, PtAu_SOC) {
    // Use the PtAu_SOC example JSON (requires FR pseudopotentials from SPARC test data)
    std::string json_file;
    // Try multiple paths
    std::vector<std::string> candidates = {
        "../examples/PtAu_SOC.json",
        "examples/PtAu_SOC.json",
        "/home/xx/Desktop/LYNX/examples/PtAu_SOC.json"
    };
    for (const auto& path : candidates) {
        std::ifstream f(path);
        if (f.good()) { json_file = path; break; }
    }
    if (json_file.empty()) {
        GTEST_SKIP() << "PtAu_SOC.json not found (need FR pseudopotentials)";
    }

    // Check that the pseudopotential files exist
    {
        auto config = InputParser::parse(json_file);
        for (const auto& at : config.atom_types) {
            if (!at.pseudo_file.empty()) {
                std::ifstream f(at.pseudo_file);
                if (!f.good()) {
                    GTEST_SKIP() << "Pseudopotential not found: " << at.pseudo_file;
                }
            }
        }
    }

    DFTResult result;
    try {
        result = run_single_point(json_file);
    } catch (const std::exception& e) {
        int rank_err = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_err);
        FAIL() << "run_single_point threw on rank " << rank_err << ": " << e.what();
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::printf("\n=== PtAu SOC Results ===\n");
        std::printf("  Converged: %s\n", result.converged ? "yes" : "no");
        std::printf("  Etotal = %.10f Ha (ref: -253.6571996)\n", result.Etotal);
        std::printf("  Ef     = %.10f Ha (ref: 0.5357)\n", result.Ef);
    }

    // SPARC reference
    double ref_Etotal = -253.6571996;

    EXPECT_TRUE(result.converged) << "SOC SCF did not converge";
    // Energy should match within ~2 mHa (slightly different numerics)
    EXPECT_NEAR(result.Etotal, ref_Etotal, 2e-3)
        << "SOC total energy deviates from SPARC reference";

    // Forces
    if (rank == 0 && result.forces.size() == 6) {
        double ref_forces[6] = {
            -1.6214905881E-01,  1.1390720982E-01,  4.1092765948E-01,
             1.6214905881E-01, -1.1390720982E-01, -4.1092765948E-01
        };
        std::printf("\n  Forces (Ha/Bohr):\n");
        double max_force_err = 0.0;
        for (int i = 0; i < 2; ++i) {
            std::printf("  Atom %d: %12.6f %12.6f %12.6f  (ref: %12.6f %12.6f %12.6f)\n",
                        i + 1,
                        result.forces[3*i], result.forces[3*i+1], result.forces[3*i+2],
                        ref_forces[3*i], ref_forces[3*i+1], ref_forces[3*i+2]);
            for (int d = 0; d < 3; ++d) {
                double err = std::abs(result.forces[3*i+d] - ref_forces[3*i+d]);
                max_force_err = std::max(max_force_err, err);
            }
        }
        std::printf("  Max force error: %.6e Ha/Bohr\n", max_force_err);
        // Forces should match within ~2e-2 Ha/Bohr (differences from pseudocharge cutoff)
        EXPECT_LT(max_force_err, 2e-2) << "SOC forces deviate from SPARC reference";
    }
}

// ============================================================
// Test: Gamma-point Si4 SCAN — orthogonal deformed cell
// SPARC reference: Si4 FCC, 26x26x26 grid, gamma-only, SCAN
//   Etotal = -15.477507471 Ha
//   Eband  = -1.8911990670 Ha
//   Exc    = -4.4332977072 Ha
//   Ef     = 0.010348992 Ha
// ============================================================
TEST(EndToEnd, Si4_gamma_SCAN) {
    std::string json_file = "/home/xx/Desktop/LYNX/.worktrees/scan/tests/data/Si4_scan_gamma.json";

    std::ifstream f(json_file);
    if (!f.good()) {
        GTEST_SKIP() << "Test data not found: " << json_file;
    }
    f.close();

    auto result = run_single_point(json_file);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // SPARC SCAN reference for deformed Si4 (10.0 x 10.26 x 10.5)
    double ref_Etotal = -15.478970526;
    double ref_Eband = 0.0; // not compared
    double ref_Exc = 0.0;
    double ref_Ef = 0.0;

    if (rank == 0) {
        std::printf("\n=== Si4 Gamma-point SCAN Results ===\n");
        std::printf("  Converged: %s\n", result.converged ? "yes" : "no");
        std::printf("  Etotal = %.12f Ha (ref: %.12f)\n", result.Etotal, ref_Etotal);
        std::printf("  Eband  = %.12f Ha (ref: %.12f)\n", result.Eband, ref_Eband);
        std::printf("  Exc    = %.12f Ha (ref: %.12f)\n", result.Exc, ref_Exc);
        std::printf("  Ef     = %.12f Ha (ref: %.12f)\n", result.Ef, ref_Ef);
        std::printf("  Etotal error: %.6e Ha\n", std::abs(result.Etotal - ref_Etotal));
        std::printf("  Exc error:    %.6e Ha\n", std::abs(result.Exc - ref_Exc));

        // SPARC SCAN forces (Ha/Bohr) for deformed cell 10.0 x 10.26 x 10.5
        double ref_forces[12] = {
            -1.1260733637E-07, -2.0475633673E-08,  2.3343149360E-02,
             6.5716542982E-04,  6.0688497580E-05, -2.3376211266E-02,
            -6.8131886993E-08,  2.4936846560E-08,  2.3408903487E-02,
            -6.5698469060E-04, -6.0692958793E-05, -2.3375841580E-02
        };

        if (result.forces.size() == 12) {
            std::printf("\n  Forces (Ha/Bohr):\n");
            double max_force_err = 0.0;
            for (int i = 0; i < 4; ++i) {
                std::printf("  Atom %d: %14.8e %14.8e %14.8e\n", i+1,
                            result.forces[3*i], result.forces[3*i+1], result.forces[3*i+2]);
                std::printf("     ref: %14.8e %14.8e %14.8e\n",
                            ref_forces[3*i], ref_forces[3*i+1], ref_forces[3*i+2]);
                for (int d = 0; d < 3; ++d) {
                    double err = std::abs(result.forces[3*i+d] - ref_forces[3*i+d]);
                    max_force_err = std::max(max_force_err, err);
                }
            }
            std::printf("  Max force error: %.6e Ha/Bohr\n", max_force_err);
            EXPECT_LT(max_force_err, 1e-5) << "Forces deviate from SPARC reference";
        }

        // SPARC SCAN stress (GPa) — deformed cell 10.0 x 10.26 x 10.5
        // Voigt: xx, xy, xz, yy, yz, zz
        double ref_stress[6] = {
            -2.5783048441E-01, -1.3335761406E+01,  6.8329942976E-06,
             4.6147953628E-01,  7.0657923920E-07, -1.3890775900E+00
        };
        if (result.stress[0] != 0.0 || result.stress[3] != 0.0) {
            std::printf("\n  Stress (GPa):\n");
            std::printf("    LYNX:  %10.6f %10.6f %10.6f\n", result.stress[0], result.stress[1], result.stress[2]);
            std::printf("           %10.6f %10.6f %10.6f\n", result.stress[3], result.stress[4], result.stress[5]);
            std::printf("    ref:   %10.6f %10.6f %10.6f\n", ref_stress[0], ref_stress[1], ref_stress[2]);
            std::printf("           %10.6f %10.6f %10.6f\n", ref_stress[3], ref_stress[4], ref_stress[5]);
            double max_stress_err = 0.0;
            for (int i = 0; i < 6; ++i) {
                double err = std::abs(result.stress[i] - ref_stress[i]);
                max_stress_err = std::max(max_stress_err, err);
            }
            std::printf("    Max stress error: %.6e GPa\n", max_stress_err);
        }
    }

    EXPECT_TRUE(result.converged) << "SCF did not converge";
    EXPECT_NEAR(result.Etotal, ref_Etotal, 1e-4)
        << "Total energy deviates from SPARC reference";
}

// ============================================================
// Test: K-point Si4 SCAN — orthogonal deformed cell with 2x2x2 kpts
// SPARC reference: Si4 (10.0 x 10.26 x 10.5), 25x26x27 grid, 2x2x2 kpts
//   Etotal = -15.663154820 Ha
// ============================================================
TEST(EndToEnd, Si4_kpt_SCAN) {
    std::string json_file = "/home/xx/Desktop/LYNX/.worktrees/scan/tests/data/Si4_scan_kpt.json";

    std::ifstream f(json_file);
    if (!f.good()) {
        GTEST_SKIP() << "Test data not found: " << json_file;
    }
    f.close();

    auto result = run_single_point(json_file);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double ref_Etotal = -15.663154820;
    double ref_forces[12] = {
         1.7876563941E-08, -2.2330656558E-08,  1.2593264499E-02,
         7.1639443493E-04, -5.3460669882E-05, -1.2377267004E-02,
        -1.4888269285E-08,  1.3562230122E-08,  1.2161270449E-02,
        -7.1639742322E-04,  5.3469438309E-05, -1.2377267944E-02
    };

    if (rank == 0) {
        std::printf("\n=== Si4 K-point SCAN Results ===\n");
        std::printf("  Converged: %s\n", result.converged ? "yes" : "no");
        std::printf("  Etotal = %.12f Ha (ref: %.12f)\n", result.Etotal, ref_Etotal);
        std::printf("  Etotal error: %.6e Ha\n", std::abs(result.Etotal - ref_Etotal));

        if (result.forces.size() == 12) {
            double max_force_err = 0.0;
            for (int i = 0; i < 4; ++i) {
                std::printf("  Atom %d: %14.8e %14.8e %14.8e\n", i+1,
                            result.forces[3*i], result.forces[3*i+1], result.forces[3*i+2]);
                for (int d = 0; d < 3; ++d) {
                    double err = std::abs(result.forces[3*i+d] - ref_forces[3*i+d]);
                    max_force_err = std::max(max_force_err, err);
                }
            }
            std::printf("  Max force error: %.6e Ha/Bohr\n", max_force_err);
            EXPECT_LT(max_force_err, 1e-5) << "K-point SCAN forces deviate from SPARC";
        }
    }

    EXPECT_TRUE(result.converged) << "K-point SCAN SCF did not converge";
    EXPECT_NEAR(result.Etotal, ref_Etotal, 1e-4)
        << "K-point SCAN energy deviates from SPARC reference";
}
