#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>

#include "io/ResultWriter.hpp"

using namespace lynx;
using nlohmann::json;

TEST(ResultWriter, EmitsMinimalValidJson) {
    ResultWriter rw;
    rw.set_test_name("Unit.Minimal");
    rw.set_input_file("dummy.json");
    rw.set_device("CPU");
    rw.set_mpi_ranks(1);
    rw.set_lynx_version("test");

    const std::string path = "/tmp/lynx_rw_test_minimal.json";
    std::remove(path.c_str());
    rw.write(path);

    std::ifstream f(path);
    ASSERT_TRUE(f.good()) << "File not created: " << path;
    json j;
    f >> j;
    EXPECT_EQ(j.at("test_name").get<std::string>(), "Unit.Minimal");
    EXPECT_EQ(j.at("input_file").get<std::string>(), "dummy.json");
    EXPECT_EQ(j.at("device").get<std::string>(), "CPU");
    EXPECT_EQ(j.at("mpi_ranks").get<int>(), 1);
    EXPECT_EQ(j.at("lynx_version").get<std::string>(), "test");
    EXPECT_TRUE(j.contains("timestamp_utc"));
}

TEST(ResultWriter, EmitsAllSections) {
    ResultWriter rw;
    rw.set_test_name("Unit.AllSections");
    rw.set_input_file("in.json");
    rw.set_lynx_version("test");

    ResultWriter::SystemInfo sys;
    sys.n_atoms = 2;
    sys.n_electrons = 8;
    sys.atoms_cartesian_bohr = {{0, 0, 0}, {1, 1, 1}};
    sys.atoms_fractional     = {{0, 0, 0}, {0.5, 0.5, 0.5}};
    ResultWriter::AtomTypeInfo at;
    at.element = "Si";
    at.Zval = 4.0;
    at.mass_amu = 28.0855;
    at.n_atoms = 2;
    at.pseudo_file = "/path/Si.psp8";
    sys.atom_types.push_back(at);
    rw.set_system(sys);

    ResultWriter::CellInfo cell;
    cell.lattice_vectors_bohr = {{ {{2,0,0}}, {{0,2,0}}, {{0,0,2}} }};
    cell.volume_bohr3 = 8.0;
    cell.orthogonal = true;
    rw.set_cell(cell);

    ResultWriter::GridInfo grid;
    grid.Nx = 10; grid.Ny = 10; grid.Nz = 10;
    grid.dx_bohr = 0.2; grid.dy_bohr = 0.2; grid.dz_bohr = 0.2;
    grid.fd_order = 12;
    grid.boundary_conditions = {{"P","P","P"}};
    rw.set_grid(grid);

    ResultWriter::ElectronicInfo elec;
    elec.xc = "PBE"; elec.spin = "none"; elec.soc = false;
    elec.smearing = "gaussian";
    elec.electronic_temperature_K = 315.775;
    elec.kBT_Ha = 1e-3;
    elec.n_states = 6;
    elec.cheb_degree = 20;
    rw.set_electronic(elec);

    ResultWriter::KpointsInfo kp;
    kp.grid = {{1, 1, 1}};
    kp.shift = {{0, 0, 0}};
    ResultWriter::KpointInfo ki;
    ki.index = 1;
    ki.reduced = {{0, 0, 0}};
    ki.weight = 1.0;
    kp.list.push_back(ki);
    rw.set_kpoints(kp);

    ResultWriter::ScfInfo scf;
    scf.max_iter = 100; scf.tolerance = 1e-6;
    scf.mixing_variable = "density"; scf.preconditioner = "kerker";
    scf.mixing_history = 7; scf.mixing_parameter = 0.3;
    scf.converged = true; scf.n_iterations = 3;
    scf.history = {
        {1, -3.87, 1.8e-1, 0.5},
        {2, -3.87, 2.3e-2, 0.4},
        {3, -3.87, 5.1e-7, 0.4},
    };
    rw.set_scf(scf);

    ResultWriter::EnergiesInfo en;
    en.total_free_energy    = -15.48;
    en.free_energy_per_atom =  -7.74;
    en.band_structure       =  -1.89;
    en.exchange_correlation =  -4.43;
    en.hartree              =   0.0;
    en.self_energy          = -50.0;
    en.correction_energy    =  -7.6;
    en.self_plus_correction = -57.6;
    en.entropy_term         = -0.017;
    en.fermi_level          =  0.008;
    rw.set_energies(en);

    ResultWriter::ForcesInfo fr;
    fr.max_Ha_per_bohr = 0.023;
    fr.rms_Ha_per_bohr = 0.020;
    fr.per_atom = {{0.01, 0.02, 0.03}, {-0.01, -0.02, -0.03}};
    rw.set_forces(fr);

    ResultWriter::StressInfo st;
    st.computed = true;
    st.voigt_Ha_per_bohr3 = {{1, 2, 3, 4, 5, 6}};
    st.voigt_GPa          = {{1, 2, 3, 4, 5, 6}};
    st.tensor_Ha_per_bohr3 = {{ {{1,2,3}}, {{2,4,5}}, {{3,5,6}} }};
    st.tensor_GPa          = {{ {{1,2,3}}, {{2,4,5}}, {{3,5,6}} }};
    st.pressure_GPa = 1.0;
    st.max_abs_stress_GPa = 6.0;
    rw.set_stress(st);

    ResultWriter::EigenvaluesInfo ev;
    ev.n_states = 2;
    ResultWriter::KpointEigenEntry e;
    e.kpoint_index = 1;
    e.reduced = {{0, 0, 0}};
    e.weight = 1.0;
    e.spin_up.eigenvalues = {-0.5, -0.4};
    e.spin_up.occupations = {2.0, 2.0};
    // spin_down intentionally empty (spin=none)
    ev.per_kpoint.push_back(e);
    rw.set_eigenvalues(ev);

    ResultWriter::TimingInfo tm;
    tm.scf_total_s = 5.0; tm.force_s = 0.1; tm.stress_s = 0.3; tm.walltime_total_s = 6.0;
    rw.set_timing(tm);

    const std::string path = "/tmp/lynx_rw_test_all.json";
    std::remove(path.c_str());
    rw.write(path);

    std::ifstream f(path);
    ASSERT_TRUE(f.good());
    json j;
    f >> j;

    EXPECT_EQ(j["system"]["n_atoms"].get<int>(), 2);
    EXPECT_EQ(j["cell"]["volume_bohr3"].get<double>(), 8.0);
    EXPECT_EQ(j["grid"]["fd_order"].get<int>(), 12);
    EXPECT_EQ(j["electronic"]["xc"].get<std::string>(), "PBE");
    EXPECT_EQ(j["kpoints"]["list"].size(), 1u);
    EXPECT_EQ(j["scf"]["history"].size(), 3u);
    EXPECT_DOUBLE_EQ(j["energies_Ha"]["total_free_energy"].get<double>(), -15.48);
    EXPECT_EQ(j["forces_Ha_per_bohr"]["per_atom"].size(), 2u);
    EXPECT_EQ(j["stress"]["computed"].get<bool>(), true);
    EXPECT_EQ(j["stress"]["voigt_GPa"].size(), 6u);
    EXPECT_EQ(j["eigenvalues"]["per_kpoint"][0]["spin_up"]["eigenvalues"].size(), 2u);
    EXPECT_FALSE(j["eigenvalues"]["per_kpoint"][0].contains("spin_down"))
        << "spin_down must be omitted when empty";
    EXPECT_DOUBLE_EQ(j["timing_s"]["walltime_total"].get<double>(), 6.0);
}
