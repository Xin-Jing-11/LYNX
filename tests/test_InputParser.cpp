#include <gtest/gtest.h>
#include "io/InputParser.hpp"
#include <fstream>
#include <cstdio>

using namespace lynx;

namespace {

// Helper: write a temp JSON file and return path
std::string write_temp_json(const std::string& content) {
    std::string path = "/tmp/lynx_test_input.json";
    std::ofstream ofs(path);
    ofs << content;
    ofs.close();
    return path;
}

const char* BATIO3_JSON = R"({
  "lattice": {
    "vectors": [[7.63, 0, 0], [0, 7.63, 0], [0, 0, 7.63]],
    "cell_type": "orthogonal"
  },
  "grid": {
    "Nx": 30, "Ny": 30, "Nz": 30,
    "fd_order": 12,
    "boundary_conditions": ["periodic", "periodic", "periodic"]
  },
  "atoms": [
    {
      "element": "Ba",
      "pseudo_file": "Ba.psp8",
      "fractional": true,
      "coordinates": [[0.0, 0.0, 0.0]]
    },
    {
      "element": "Ti",
      "pseudo_file": "Ti.psp8",
      "fractional": true,
      "coordinates": [[0.5, 0.5, 0.5]]
    },
    {
      "element": "O",
      "pseudo_file": "O.psp8",
      "fractional": true,
      "coordinates": [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
    }
  ],
  "electronic": {
    "xc": "GGA_PBE",
    "spin": "none",
    "temperature": 300,
    "smearing": "gaussian",
    "Nstates": 30
  },
  "kpoints": {
    "grid": [4, 4, 4],
    "shift": [0.5, 0.5, 0.5]
  },
  "scf": {
    "max_iter": 100,
    "tolerance": 1e-6,
    "mixing": "density",
    "preconditioner": "kerker",
    "mixing_history": 7,
    "mixing_parameter": 0.3
  },
  "output": {
    "print_forces": true,
    "print_atoms": true
  }
})";

} // anonymous namespace

TEST(InputParser, ParseBaTiO3) {
    std::string path = write_temp_json(BATIO3_JSON);
    auto config = InputParser::parse(path);

    // Lattice
    EXPECT_DOUBLE_EQ(config.latvec(0, 0), 7.63);
    EXPECT_DOUBLE_EQ(config.latvec(1, 1), 7.63);
    EXPECT_DOUBLE_EQ(config.latvec(2, 2), 7.63);
    EXPECT_EQ(config.cell_type, CellType::Orthogonal);

    // Grid
    EXPECT_EQ(config.Nx, 30);
    EXPECT_EQ(config.Ny, 30);
    EXPECT_EQ(config.Nz, 30);
    EXPECT_EQ(config.fd_order, 12);
    EXPECT_EQ(config.bcx, BCType::Periodic);

    // Atoms
    EXPECT_EQ(config.atom_types.size(), 3u);
    EXPECT_EQ(config.atom_types[0].element, "Ba");
    EXPECT_EQ(config.atom_types[0].coords.size(), 1u);
    EXPECT_EQ(config.atom_types[2].element, "O");
    EXPECT_EQ(config.atom_types[2].coords.size(), 3u);

    // Electronic
    EXPECT_EQ(config.xc, XCType::GGA_PBE);
    EXPECT_EQ(config.spin_type, SpinType::None);
    EXPECT_EQ(config.Nstates, 30);
    EXPECT_DOUBLE_EQ(config.elec_temp, 300.0);

    // K-points
    EXPECT_EQ(config.Kx, 4);
    EXPECT_EQ(config.Ky, 4);
    EXPECT_EQ(config.Kz, 4);

    // SCF
    EXPECT_EQ(config.max_scf_iter, 100);
    EXPECT_DOUBLE_EQ(config.scf_tol, 1e-6);
    EXPECT_EQ(config.mixing_var, MixingVariable::Density);
    EXPECT_EQ(config.mixing_precond, MixingPrecond::Kerker);

    // Output
    EXPECT_TRUE(config.print_forces);
    EXPECT_TRUE(config.print_atoms);
}

TEST(InputParser, Validate) {
    std::string path = write_temp_json(BATIO3_JSON);
    auto config = InputParser::parse(path);
    EXPECT_NO_THROW(InputParser::validate(config));
}

TEST(InputParser, ValidateMissingNstates) {
    const char* json = R"({
      "lattice": {"vectors": [[10,0,0],[0,10,0],[0,0,10]]},
      "grid": {"Nx": 10, "Ny": 10, "Nz": 10},
      "atoms": [{"element": "H", "coordinates": [[0,0,0]]}],
      "electronic": {"Nstates": 0}
    })";
    std::string path = write_temp_json(json);
    auto config = InputParser::parse(path);
    EXPECT_THROW(InputParser::validate(config), std::runtime_error);
}

TEST(InputParser, ValidateMissingAtoms) {
    const char* json = R"({
      "lattice": {"vectors": [[10,0,0],[0,10,0],[0,0,10]]},
      "grid": {"Nx": 10, "Ny": 10, "Nz": 10},
      "electronic": {"Nstates": 10}
    })";
    std::string path = write_temp_json(json);
    auto config = InputParser::parse(path);
    EXPECT_THROW(InputParser::validate(config), std::runtime_error);
}

TEST(InputParser, FileNotFound) {
    EXPECT_THROW(InputParser::parse("/nonexistent/file.json"), std::runtime_error);
}

TEST(InputParser, DirichletBC) {
    const char* json = R"({
      "lattice": {"vectors": [[10,0,0],[0,10,0],[0,0,10]]},
      "grid": {"Nx": 20, "Ny": 20, "Nz": 20,
               "boundary_conditions": ["dirichlet", "dirichlet", "dirichlet"]},
      "atoms": [{"element": "H", "coordinates": [[0.5, 0.5, 0.5]]}],
      "electronic": {"Nstates": 5}
    })";
    std::string path = write_temp_json(json);
    auto config = InputParser::parse(path);
    EXPECT_EQ(config.bcx, BCType::Dirichlet);
    EXPECT_EQ(config.bcy, BCType::Dirichlet);
    EXPECT_EQ(config.bcz, BCType::Dirichlet);
}

TEST(InputParser, SpinCollinear) {
    const char* json = R"({
      "lattice": {"vectors": [[10,0,0],[0,10,0],[0,0,10]]},
      "grid": {"Nx": 20, "Ny": 20, "Nz": 20},
      "atoms": [{"element": "Fe", "coordinates": [[0,0,0]]}],
      "electronic": {"Nstates": 10, "spin": "collinear"}
    })";
    std::string path = write_temp_json(json);
    auto config = InputParser::parse(path);
    EXPECT_EQ(config.spin_type, SpinType::Collinear);
}
