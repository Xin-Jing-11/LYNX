#include <gtest/gtest.h>
#include "io/DensityIO.hpp"
#include "electronic/ElectronDensity.hpp"
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include <cmath>
#include <cstdio>
#include <string>

using namespace lynx;

namespace {

// Create a simple test lattice and grid
Lattice make_lattice() {
    Mat3 lv{};
    lv(0,0) = 10.0; lv(1,1) = 10.0; lv(2,2) = 10.0;
    return Lattice(lv, CellType::Orthogonal);
}

} // namespace

TEST(DensityIO, WriteReadRoundtrip_Nspin1) {
    auto lattice = make_lattice();
    FDGrid grid(8, 8, 8, lattice, BCType::Periodic, BCType::Periodic, BCType::Periodic);
    int Nd = grid.Nx() * grid.Ny() * grid.Nz();

    // Create density with known pattern
    ElectronDensity rho_out;
    rho_out.allocate(Nd, 1);
    for (int i = 0; i < Nd; ++i) {
        rho_out.rho(0).data()[i] = 0.01 + 0.001 * std::sin(i * 0.1);
        rho_out.rho_total().data()[i] = rho_out.rho(0).data()[i];
    }

    std::string filename = "/tmp/test_density_nspin1.lynxrho";
    ASSERT_TRUE(DensityIO::write(filename, rho_out, grid, lattice));

    // Read back
    ElectronDensity rho_in;
    rho_in.allocate(Nd, 1);
    ASSERT_TRUE(DensityIO::read(filename, rho_in, grid, lattice));

    // Compare
    for (int i = 0; i < Nd; ++i) {
        EXPECT_DOUBLE_EQ(rho_out.rho(0).data()[i], rho_in.rho(0).data()[i]);
        EXPECT_DOUBLE_EQ(rho_out.rho_total().data()[i], rho_in.rho_total().data()[i]);
    }

    std::remove(filename.c_str());
}

TEST(DensityIO, WriteReadRoundtrip_Nspin2) {
    auto lattice = make_lattice();
    FDGrid grid(6, 6, 6, lattice, BCType::Periodic, BCType::Periodic, BCType::Periodic);
    int Nd = grid.Nx() * grid.Ny() * grid.Nz();

    ElectronDensity rho_out;
    rho_out.allocate(Nd, 2);
    for (int i = 0; i < Nd; ++i) {
        double up = 0.006 + 0.001 * std::sin(i * 0.2);
        double dn = 0.004 + 0.001 * std::cos(i * 0.3);
        rho_out.rho(0).data()[i] = up;
        rho_out.rho(1).data()[i] = dn;
        rho_out.rho_total().data()[i] = up + dn;
        rho_out.mag().data()[i] = up - dn;
    }

    std::string filename = "/tmp/test_density_nspin2.lynxrho";
    ASSERT_TRUE(DensityIO::write(filename, rho_out, grid, lattice));

    ElectronDensity rho_in;
    rho_in.allocate(Nd, 2);
    ASSERT_TRUE(DensityIO::read(filename, rho_in, grid, lattice));

    for (int i = 0; i < Nd; ++i) {
        EXPECT_DOUBLE_EQ(rho_out.rho(0).data()[i], rho_in.rho(0).data()[i]);
        EXPECT_DOUBLE_EQ(rho_out.rho(1).data()[i], rho_in.rho(1).data()[i]);
        EXPECT_DOUBLE_EQ(rho_out.rho_total().data()[i], rho_in.rho_total().data()[i]);
        EXPECT_DOUBLE_EQ(rho_out.mag().data()[i], rho_in.mag().data()[i]);
    }

    std::remove(filename.c_str());
}

TEST(DensityIO, ReadNonexistentFile) {
    auto lattice = make_lattice();
    FDGrid grid(4, 4, 4, lattice, BCType::Periodic, BCType::Periodic, BCType::Periodic);

    ElectronDensity rho;
    rho.allocate(64, 1);
    EXPECT_FALSE(DensityIO::read("/tmp/nonexistent_density.lynxrho", rho, grid, lattice));
}

TEST(DensityIO, ReadGridMismatch) {
    auto lattice = make_lattice();
    FDGrid grid8(8, 8, 8, lattice, BCType::Periodic, BCType::Periodic, BCType::Periodic);
    FDGrid grid6(6, 6, 6, lattice, BCType::Periodic, BCType::Periodic, BCType::Periodic);

    // Write with 8x8x8
    ElectronDensity rho_out;
    rho_out.allocate(512, 1);
    for (int i = 0; i < 512; ++i) rho_out.rho(0).data()[i] = 0.01;
    std::memcpy(rho_out.rho_total().data(), rho_out.rho(0).data(), 512 * sizeof(double));

    std::string filename = "/tmp/test_density_mismatch.lynxrho";
    ASSERT_TRUE(DensityIO::write(filename, rho_out, grid8, lattice));

    // Try to read with 6x6x6 — should fail
    ElectronDensity rho_in;
    rho_in.allocate(216, 1);
    EXPECT_FALSE(DensityIO::read(filename, rho_in, grid6, lattice));

    std::remove(filename.c_str());
}

TEST(DensityIO, ReadSpinMismatch) {
    auto lattice = make_lattice();
    FDGrid grid(4, 4, 4, lattice, BCType::Periodic, BCType::Periodic, BCType::Periodic);
    int Nd = 64;

    // Write with Nspin=1
    ElectronDensity rho_out;
    rho_out.allocate(Nd, 1);
    for (int i = 0; i < Nd; ++i) rho_out.rho(0).data()[i] = 0.01;
    std::memcpy(rho_out.rho_total().data(), rho_out.rho(0).data(), Nd * sizeof(double));

    std::string filename = "/tmp/test_density_spinmismatch.lynxrho";
    ASSERT_TRUE(DensityIO::write(filename, rho_out, grid, lattice));

    // Try to read with Nspin=2 — should fail
    ElectronDensity rho_in;
    rho_in.allocate(Nd, 2);
    EXPECT_FALSE(DensityIO::read(filename, rho_in, grid, lattice));

    std::remove(filename.c_str());
}

TEST(DensityIO, HeaderSize) {
    EXPECT_EQ(sizeof(DensityIO::FileHeader), 128u);
}
