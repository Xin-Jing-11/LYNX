#include <gtest/gtest.h>
#include "parallel/HaloExchange.hpp"
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include "core/Domain.hpp"
#include <vector>
#include <complex>
#include <cmath>

using namespace lynx;

TEST(HaloExchange, PeriodicSingleProcess) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);
    FDGrid grid(10, 10, 10, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);
    DomainVertices v{0, 9, 0, 9, 0, 9};
    Domain domain(grid, v);

    int FDn = 3;
    HaloExchange halo(domain, FDn);

    EXPECT_EQ(halo.nx_ex(), 16);
    EXPECT_EQ(halo.ny_ex(), 16);
    EXPECT_EQ(halo.nz_ex(), 16);

    int nd = 10 * 10 * 10;
    std::vector<double> x(nd);
    for (int i = 0; i < nd; ++i) x[i] = i + 1.0;

    int nd_ex = halo.nd_ex();
    std::vector<double> x_ex(nd_ex, -1.0);
    halo.execute(x.data(), x_ex.data(), 1);

    // Check interior is preserved
    int nx = 10, ny = 10;
    int nx_ex = 16, ny_ex = 16;
    for (int k = 0; k < 10; ++k)
        for (int j = 0; j < 10; ++j)
            for (int i = 0; i < 10; ++i) {
                int loc = i + j * nx + k * nx * ny;
                int ext = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nx_ex * ny_ex;
                EXPECT_DOUBLE_EQ(x_ex[ext], x[loc])
                    << "Interior mismatch at (" << i << "," << j << "," << k << ")";
            }

    // Check periodic ghost in x direction:
    // x_ex[FDn-1, FDn, FDn] should equal x[nx-1, 0, 0]
    {
        int ext = (FDn - 1) + FDn * nx_ex + FDn * nx_ex * ny_ex;
        int loc = (nx - 1) + 0 * nx + 0 * nx * ny;
        EXPECT_DOUBLE_EQ(x_ex[ext], x[loc]) << "Periodic ghost x-low failed";
    }

    // x_ex[FDn+nx, FDn, FDn] should equal x[0, 0, 0]
    {
        int ext = (FDn + nx) + FDn * nx_ex + FDn * nx_ex * ny_ex;
        int loc = 0 + 0 * nx + 0 * nx * ny;
        EXPECT_DOUBLE_EQ(x_ex[ext], x[loc]) << "Periodic ghost x-high failed";
    }
}

TEST(HaloExchange, ExtendedDimensions) {
    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);
    FDGrid grid(20, 30, 40, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);
    DomainVertices v{0, 19, 0, 29, 0, 39};
    Domain domain(grid, v);

    HaloExchange halo(domain, 6);
    EXPECT_EQ(halo.nx_ex(), 32);
    EXPECT_EQ(halo.ny_ex(), 42);
    EXPECT_EQ(halo.nz_ex(), 52);
}

// Test complex halo exchange with Bloch phase factors
TEST(HaloExchange, ComplexBlochPhase) {
    using Complex = std::complex<double>;
    constexpr double PI = 3.14159265358979323846;

    Mat3 lv;
    double L = 10.0;
    lv(0, 0) = L; lv(1, 1) = L; lv(2, 2) = L;
    Lattice lat(lv, CellType::Orthogonal);
    int N = 10;
    FDGrid grid(N, N, N, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);
    DomainVertices v{0, N-1, 0, N-1, 0, N-1};
    Domain domain(grid, v);

    int FDn = 3;
    HaloExchange halo(domain, FDn);

    int nd = N * N * N;
    int nd_ex = halo.nd_ex();
    int nx_ex = halo.nx_ex(), ny_ex = halo.ny_ex();

    // Fill with a known complex function
    std::vector<Complex> x(nd);
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                int idx = i + j * N + k * N * N;
                x[idx] = Complex(i + 1.0, j + k * 0.1);
            }

    // K-point at (0.25 * 2π/L, 0, 0) — only x-phase is non-trivial
    Vec3 kpt = {0.25 * 2.0 * PI / L, 0.0, 0.0};
    Vec3 cell_lengths = {L, L, L};

    std::vector<Complex> x_ex(nd_ex);
    halo.execute_kpt(x.data(), x_ex.data(), 1, kpt, cell_lengths);

    // Phase factors
    double theta_x = kpt.x * L;  // = 0.25 * 2π = π/2
    Complex phase_l_x(std::cos(theta_x), -std::sin(theta_x));  // e^{-iπ/2} = -i
    Complex phase_r_x(std::cos(theta_x),  std::sin(theta_x));  // e^{+iπ/2} = +i

    // Verify phase_l_x = -i, phase_r_x = +i
    EXPECT_NEAR(phase_l_x.real(), 0.0, 1e-14);
    EXPECT_NEAR(phase_l_x.imag(), -1.0, 1e-14);

    // Check interior is unchanged
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                int loc = i + j * N + k * N * N;
                int ext = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nx_ex * ny_ex;
                EXPECT_EQ(x_ex[ext], x[loc])
                    << "Interior mismatch at (" << i << "," << j << "," << k << ")";
            }

    // Check x-direction left ghost: x_ex[FDn-1, j, k] = x[N-1, j, k] * phase_l_x
    for (int j_test = 0; j_test < 2; ++j_test) {
        for (int k_test = 0; k_test < 2; ++k_test) {
            int loc = (N-1) + j_test * N + k_test * N * N;
            int ext = (FDn - 1) + (j_test + FDn) * nx_ex + (k_test + FDn) * nx_ex * ny_ex;
            Complex expected = x[loc] * phase_l_x;
            EXPECT_NEAR(x_ex[ext].real(), expected.real(), 1e-12)
                << "x-left ghost real at j=" << j_test << ",k=" << k_test;
            EXPECT_NEAR(x_ex[ext].imag(), expected.imag(), 1e-12)
                << "x-left ghost imag at j=" << j_test << ",k=" << k_test;
        }
    }

    // Check x-direction right ghost: x_ex[FDn+N, j, k] = x[0, j, k] * phase_r_x
    for (int j_test = 0; j_test < 2; ++j_test) {
        for (int k_test = 0; k_test < 2; ++k_test) {
            int loc = 0 + j_test * N + k_test * N * N;
            int ext = (FDn + N) + (j_test + FDn) * nx_ex + (k_test + FDn) * nx_ex * ny_ex;
            Complex expected = x[loc] * phase_r_x;
            EXPECT_NEAR(x_ex[ext].real(), expected.real(), 1e-12)
                << "x-right ghost real at j=" << j_test << ",k=" << k_test;
            EXPECT_NEAR(x_ex[ext].imag(), expected.imag(), 1e-12)
                << "x-right ghost imag at j=" << j_test << ",k=" << k_test;
        }
    }

    // Test with all three phase directions non-trivial
    Vec3 kpt2 = {0.25 * 2.0 * PI / L, -0.3333 * 2.0 * PI / L, 0.125 * 2.0 * PI / L};
    std::vector<Complex> x_ex2(nd_ex);
    halo.execute_kpt(x.data(), x_ex2.data(), 1, kpt2, cell_lengths);

    // Check z-direction left ghost: x_ex[i, j, FDn-1] = x[i, j, N-1] * phase_l_z
    double theta_z = kpt2.z * L;
    Complex phase_l_z(std::cos(theta_z), -std::sin(theta_z));
    {
        int i_test = 3, j_test = 2;
        int loc = i_test + j_test * N + (N-1) * N * N;
        int ext = (i_test + FDn) + (j_test + FDn) * nx_ex + (FDn - 1) * nx_ex * ny_ex;
        Complex expected = x[loc] * phase_l_z;
        EXPECT_NEAR(x_ex2[ext].real(), expected.real(), 1e-12) << "z-left ghost";
        EXPECT_NEAR(x_ex2[ext].imag(), expected.imag(), 1e-12) << "z-left ghost";
    }
}

// Test that Gamma-only (k=0) complex halo equals real halo
TEST(HaloExchange, ComplexGammaEqualsReal) {
    using Complex = std::complex<double>;

    Mat3 lv;
    lv(0, 0) = 10.0; lv(1, 1) = 10.0; lv(2, 2) = 10.0;
    Lattice lat(lv, CellType::Orthogonal);
    int N = 8;
    FDGrid grid(N, N, N, lat, BCType::Periodic, BCType::Periodic, BCType::Periodic);
    DomainVertices v{0, N-1, 0, N-1, 0, N-1};
    Domain domain(grid, v);

    int FDn = 6;  // large stencil
    HaloExchange halo(domain, FDn);
    int nd = N * N * N;
    int nd_ex = halo.nd_ex();

    // Fill with real data
    std::vector<double> x_real(nd);
    std::vector<Complex> x_complex(nd);
    for (int i = 0; i < nd; ++i) {
        x_real[i] = std::sin(0.1 * i);
        x_complex[i] = Complex(x_real[i], 0.0);
    }

    // Real halo
    std::vector<double> x_ex_real(nd_ex);
    halo.execute(x_real.data(), x_ex_real.data(), 1);

    // Complex halo at Gamma (k=0)
    std::vector<Complex> x_ex_complex(nd_ex);
    Vec3 kpt_gamma = {0.0, 0.0, 0.0};
    Vec3 cell_lengths = {10.0, 10.0, 10.0};
    halo.execute_kpt(x_complex.data(), x_ex_complex.data(), 1, kpt_gamma, cell_lengths);

    // Should be identical (imaginary parts all zero, real parts match)
    for (int i = 0; i < nd_ex; ++i) {
        EXPECT_NEAR(x_ex_complex[i].real(), x_ex_real[i], 1e-14)
            << "Gamma complex vs real mismatch at " << i;
        EXPECT_NEAR(x_ex_complex[i].imag(), 0.0, 1e-14)
            << "Gamma imaginary nonzero at " << i;
    }
}
