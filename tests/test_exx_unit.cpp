// Unit test for exact exchange: compare LYNX ExactExchange with a reference
// SPARC-style implementation using the same random orbitals.
//
// Tests M matrix, Xi, apply_Vx, and Eexx for gamma-point.

#include <mpi.h>
#include <mkl_dfti.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

#include "core/types.hpp"
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include "core/Domain.hpp"
#include "core/KPoints.hpp"
#include "core/constants.hpp"
#include "xc/ExchangePoissonSolver.hpp"
#include "xc/ExactExchange.hpp"
#include "parallel/MPIComm.hpp"
#include "electronic/Wavefunction.hpp"

extern "C" {
    void dgemm_(const char*, const char*, const int*, const int*, const int*,
                const double*, const double*, const int*, const double*, const int*,
                const double*, double*, const int*);
    void dpotrf_(const char*, const int*, double*, const int*, int*);
    void dtrsm_(const char*, const char*, const char*, const char*,
                const int*, const int*, const double*, const double*, const int*,
                double*, const int*);
    void zgemm_(const char*, const char*, const int*, const int*, const int*,
                const void*, const void*, const int*, const void*, const int*,
                const void*, void*, const int*);
    void zpotrf_(const char*, const int*, void*, const int*, int*);
    void ztrsm_(const char*, const char*, const char*, const char*,
                const int*, const int*, const void*, const void*, const int*,
                void*, const int*);
}

using namespace lynx;

// ===========================================================================
// Reference SPARC-style implementation (standalone, no SPARC dependency)
// ===========================================================================
namespace ref {

// Compute singularity removal constant (spherical cutoff, exx_div_flag=0)
double singularity_const_sph(double G2, double Rc) {
    if (std::fabs(G2) > 1e-4) {
        double x = Rc * std::sqrt(G2);
        return 4.0 * M_PI * (1.0 - std::cos(x)) / G2;
    }
    return 2.0 * M_PI * Rc * Rc;
}

// Compute FFT Poisson constants (gamma, spherical cutoff, orthogonal cell)
void compute_pois_fft_const(int Nx, int Ny, int Nz,
                             double L1, double L2, double L3,
                             double volume, int Nkpts_hf,
                             const double lapcT[9],
                             std::vector<double>& pois_const) {
    int Nxh = Nx / 2 + 1;
    int Ndc = Nz * Ny * Nxh;
    double V = volume * Nkpts_hf;
    double Rc = std::cbrt(3.0 * V / (4.0 * M_PI));

    pois_const.resize(Ndc);
    int count = 0;
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nxh; i++) {
                double G[3], g[3];
                G[0] = (i < Nx/2+1) ? (i * 2*M_PI/L1) : ((i-Nx) * 2*M_PI/L1);
                G[1] = (j < Ny/2+1) ? (j * 2*M_PI/L2) : ((j-Ny) * 2*M_PI/L2);
                G[2] = (k < Nz/2+1) ? (k * 2*M_PI/L3) : ((k-Nz) * 2*M_PI/L3);
                // lapcT * G  (matrixTimesVec_3d in SPARC does lapcT^T * G)
                g[0] = lapcT[0]*G[0] + lapcT[3]*G[1] + lapcT[6]*G[2];
                g[1] = lapcT[1]*G[0] + lapcT[4]*G[1] + lapcT[7]*G[2];
                g[2] = lapcT[2]*G[0] + lapcT[5]*G[1] + lapcT[8]*G[2];
                double G2 = G[0]*g[0] + G[1]*g[1] + G[2]*g[2];
                pois_const[count++] = singularity_const_sph(G2, Rc);
            }
        }
    }
}

// FFT Poisson solve (gamma, real, matching SPARC pois_fft exactly)
void pois_fft(const double* rhs, const double* pois_const,
              int Nx, int Ny, int Nz, double* sol) {
    int Nd = Nx * Ny * Nz;
    int Ndc = Nz * Ny * (Nx/2+1);

    std::vector<std::complex<double>> rhs_bar(Ndc);

    // Forward R2C FFT
    DFTI_DESCRIPTOR_HANDLE desc = nullptr;
    MKL_LONG dims[3] = {Nz, Ny, Nx};
    DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_REAL, 3, dims);
    DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(desc, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    MKL_LONG si[4] = {0, (MKL_LONG)(Ny*Nx), (MKL_LONG)Nx, 1};
    MKL_LONG so[4] = {0, (MKL_LONG)(Ny*(Nx/2+1)), (MKL_LONG)(Nx/2+1), 1};
    DftiSetValue(desc, DFTI_INPUT_STRIDES, si);
    DftiSetValue(desc, DFTI_OUTPUT_STRIDES, so);
    DftiCommitDescriptor(desc);
    // Need a mutable copy for MKL
    std::vector<double> rhs_copy(rhs, rhs + Nd);
    DftiComputeForward(desc, rhs_copy.data(), reinterpret_cast<double*>(rhs_bar.data()));
    DftiFreeDescriptor(&desc);

    // Multiply by pois_const
    for (int i = 0; i < Ndc; i++) {
        rhs_bar[i] = std::complex<double>(
            rhs_bar[i].real() * pois_const[i],
            rhs_bar[i].imag() * pois_const[i]);
    }

    // Inverse C2R FFT
    desc = nullptr;
    DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_REAL, 3, dims);
    DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(desc, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    MKL_LONG si2[4] = {0, (MKL_LONG)(Ny*(Nx/2+1)), (MKL_LONG)(Nx/2+1), 1};
    MKL_LONG so2[4] = {0, (MKL_LONG)(Ny*Nx), (MKL_LONG)Nx, 1};
    DftiSetValue(desc, DFTI_INPUT_STRIDES, si2);
    DftiSetValue(desc, DFTI_OUTPUT_STRIDES, so2);
    DftiCommitDescriptor(desc);
    DftiComputeBackward(desc, reinterpret_cast<double*>(rhs_bar.data()), sol);
    DftiFreeDescriptor(&desc);

    // Normalize
    double inv = 1.0 / Nd;
    for (int i = 0; i < Nd; i++) sol[i] *= inv;
}

// Build Xi matching LYNX convention.
// psi: [Nd x Ns], column-major. occ: [Ns]. Xi: [Nd x Ns], output.
// The coefficient is occ * Nd * sqrt(dV) to make Xi_LYNX = Xi_SPARC
// (absorbs the 1/Nd Poisson normalization and sqrt(dV) psi normalization).
void solve_for_Xi(const double* psi, const double* occ, int Nd, int Ns,
                  double dV, const double* pois_const,
                  int Nx, int Ny, int Nz, double* Xi) {
    std::memset(Xi, 0, sizeof(double) * Nd * Ns);
    double coeff_scale = std::sqrt(dV);

    std::vector<double> rhs(Nd), sol(Nd);

    for (int j = 0; j < Ns; j++) {
        if (occ[j] < 1e-6) continue;
        for (int i = j; i < Ns; i++) {
            for (int k = 0; k < Nd; k++)
                rhs[k] = psi[k + i*Nd] * psi[k + j*Nd];

            pois_fft(rhs.data(), pois_const, Nx, Ny, Nz, sol.data());

            double cj = occ[j] * coeff_scale;
            for (int k = 0; k < Nd; k++)
                Xi[k + i*Nd] -= cj * psi[k + j*Nd] * sol[k];

            if (i != j && occ[i] > 1e-6) {
                double ci = occ[i] * coeff_scale;
                for (int k = 0; k < Nd; k++)
                    Xi[k + j*Nd] -= ci * psi[k + i*Nd] * sol[k];
            }
        }
    }
}

// Compute M = sqrt(dV) * Xi^T * psi, then negate, Cholesky, trsm
// Returns Xi_final in-place in Xi, and M in M_out.
void calculate_ACE(const double* psi, int Nd, int Ns, double dV, double* Xi, double* M_out) {
    // M = sqrt(dV) * Xi^T * psi  (matching LYNX convention)
    char tA = 'T', tB = 'N';
    double alpha = std::sqrt(dV), beta = 0.0;
    dgemm_(&tA, &tB, &Ns, &Ns, &Nd, &alpha, Xi, &Nd, psi, &Nd, &beta, M_out, &Ns);

    // Negate
    for (int i = 0; i < Ns*Ns; i++) M_out[i] = -M_out[i];

    // Cholesky
    char uplo = 'U';
    int info = 0;
    dpotrf_(&uplo, &Ns, M_out, &Ns, &info);
    if (info != 0) {
        printf("REF: dpotrf failed, info=%d\n", info);
        return;
    }

    // Xi_ace = Xi_raw * L^{-T}  via dtrsm (Xi already has Xi_raw)
    char side = 'R', trans = 'N', diag = 'N';
    alpha = 1.0;
    dtrsm_(&side, &uplo, &trans, &diag, &Nd, &Ns, &alpha, M_out, &Ns, Xi, &Nd);
}

} // namespace ref


// ===========================================================================
// Test: compare reference vs LYNX for gamma-point PBE0 (spherical cutoff)
// ===========================================================================
void test_gamma_pbe0() {
    printf("\n=== TEST: Gamma-point PBE0 (spherical cutoff) ===\n");

    // Use an orthogonal cubic cell for simplicity
    double Lx = 10.0, Ly = 10.0, Lz = 10.0;
    Mat3 latvec = {};
    latvec(0,0) = Lx; latvec(1,1) = Ly; latvec(2,2) = Lz;

    Lattice lattice(latvec, CellType::Orthogonal);
    int Nx = 20, Ny = 20, Nz = 20;
    FDGrid grid(Nx, Ny, Nz, lattice, BCType::Periodic, BCType::Periodic, BCType::Periodic);
    int Nd = Nx * Ny * Nz;
    int Ns = 6;  // number of states
    double dV = grid.dV();
    double volume = lattice.jacobian();

    printf("Grid: %dx%dx%d = %d, dV=%.10e, volume=%.6f\n", Nx, Ny, Nz, Nd, dV, volume);

    // lapcT for orthogonal = identity
    double lapcT[9] = {1,0,0, 0,1,0, 0,0,1};

    // Generate deterministic random orbitals: uniform in [-0.5, 0.5]
    std::vector<double> psi(Nd * Ns);
    {
        unsigned seed = 42;
        std::srand(seed);
        for (int i = 0; i < Nd * Ns; i++)
            psi[i] = (std::rand() / (double)RAND_MAX) - 0.5;
    }

    // Occupations: all occupied
    std::vector<double> occ(Ns);
    for (int n = 0; n < Ns; n++) occ[n] = 1.0 - 0.1 * n;

    printf("occ = [");
    for (int n = 0; n < Ns; n++) printf("%.2f%s", occ[n], n<Ns-1?", ":"]\n");

    // ======== REFERENCE (SPARC-style) ========
    printf("\n--- Reference (SPARC-style) ---\n");

    // Compute Poisson constants
    std::vector<double> ref_pois_const;
    ref::compute_pois_fft_const(Nx, Ny, Nz, Lx, Ly, Lz, volume, 1, lapcT, ref_pois_const);
    printf("ref pois_const[0]=%.10e, [1]=%.10e, [2]=%.10e\n",
           ref_pois_const[0], ref_pois_const[1], ref_pois_const[2]);

    // Build Xi
    std::vector<double> ref_Xi(Nd * Ns);
    ref::solve_for_Xi(psi.data(), occ.data(), Nd, Ns, dV, ref_pois_const.data(),
                      Nx, Ny, Nz, ref_Xi.data());

    // Print Xi norms
    for (int n = 0; n < Ns; n++) {
        double norm = 0;
        for (int k = 0; k < Nd; k++) norm += ref_Xi[k + n*Nd] * ref_Xi[k + n*Nd];
        printf("ref Xi[%d] norm = %.10e\n", n, std::sqrt(norm));
    }

    // Compute M and ACE
    std::vector<double> ref_M(Ns * Ns);
    // Save pre-Cholesky M (with sqrt(dV) factor matching LYNX)
    {
        char tA = 'T', tB = 'N';
        double alpha = std::sqrt(dV), beta = 0.0;
        dgemm_(&tA, &tB, &Ns, &Ns, &Nd, &alpha, ref_Xi.data(), &Nd, psi.data(), &Nd, &beta, ref_M.data(), &Ns);
    }
    printf("ref M (before negation):\n");
    for (int i = 0; i < Ns; i++) {
        printf("  ");
        for (int j = 0; j < Ns; j++) printf("%14.6e", ref_M[i + j*Ns]);
        printf("\n");
    }

    // Now do full ACE (modifies ref_Xi and ref_M)
    std::vector<double> ref_Xi_ace(ref_Xi);
    std::vector<double> ref_M_ace(Ns * Ns);
    ref::calculate_ACE(psi.data(), Nd, Ns, dV, ref_Xi_ace.data(), ref_M_ace.data());

    // ======== LYNX ========
    printf("\n--- LYNX ---\n");

    // Create Wavefunction with same data
    Wavefunction wfn;
    wfn.allocate(Nd, Ns, Ns, 1, 1, false, 1);
    std::memcpy(wfn.psi(0, 0).data(), psi.data(), Nd * Ns * sizeof(double));
    for (int n = 0; n < Ns; n++) {
        wfn.occupations(0, 0)(n) = occ[n];
        wfn.eigenvalues(0, 0)(n) = -1.0 + 0.5 * n;
    }

    // Setup EXX with PBE0 spherical cutoff
    EXXParams params;
    params.exx_frac = 0.25;
    params.hyb_range_fock = -1.0;  // PBE0, unscreened
    params.exx_div_flag = 0;       // spherical cutoff

    MPIComm bandcomm(MPI_COMM_SELF);
    MPIComm kpt_bridge(MPI_COMM_SELF);
    MPIComm spin_bridge(MPI_COMM_SELF);

    ExactExchange exx;
    exx.setup(grid, lattice, nullptr, bandcomm, kpt_bridge, spin_bridge,
              params, 1, Ns, Ns, 0, 1, 1, 0, 0, 1, 1, 1);

    exx.build_ACE(wfn);

    // ======== COMPARE Poisson constants ========
    printf("\n--- Compare Poisson constants ---\n");
    // LYNX pois_const is inside ExchangePoissonSolver, not directly accessible.
    // But we can compare the Poisson solve output instead.
    {
        std::vector<double> test_rhs(Nd);
        for (int k = 0; k < Nd; k++) test_rhs[k] = psi[k] * psi[k];  // psi_0^2

        std::vector<double> ref_sol(Nd), lynx_sol(Nd);
        ref::pois_fft(test_rhs.data(), ref_pois_const.data(), Nx, Ny, Nz, ref_sol.data());

        ExchangePoissonSolver lynx_poisson;
        lynx_poisson.setup(grid, lattice, params, nullptr, 1, 1, 1);
        std::vector<double> lynx_rhs(test_rhs);  // copy since solve_batch may modify
        lynx_poisson.solve_batch(lynx_rhs.data(), 1, lynx_sol.data());

        double max_diff = 0, max_ref = 0;
        for (int k = 0; k < Nd; k++) {
            max_diff = std::max(max_diff, std::fabs(ref_sol[k] - lynx_sol[k]));
            max_ref = std::max(max_ref, std::fabs(ref_sol[k]));
        }
        printf("Poisson solve: max_diff=%.6e, max_ref=%.6e, rel_err=%.6e\n",
               max_diff, max_ref, max_diff / (max_ref + 1e-30));
        printf("  ref_sol[0]=%.10e, lynx_sol[0]=%.10e\n", ref_sol[0], lynx_sol[0]);
        printf("  ref_sol[100]=%.10e, lynx_sol[100]=%.10e\n", ref_sol[100], lynx_sol[100]);
    }

    // ======== COMPARE M matrix ========
    // LYNX doesn't expose M directly, but we can reconstruct it from Xi
    // Actually, let's extract Xi from LYNX's ExactExchange... it's private.
    // Instead, compare Eexx and apply_Vx which depend on the full ACE.

    // Compare Eexx
    double ref_Eexx = 0.0;
    {
        // ref ACE energy: Eexx = -exx_frac / Nspin * sum_n occ[n] * |sqrt(dV)*psi_n^T * Xi_j|^2
        std::vector<double> Y(Ns * Ns, 0.0);
        char tA = 'T', tB = 'N';
        double alpha = std::sqrt(dV), beta = 0.0;
        dgemm_(&tA, &tB, &Ns, &Ns, &Nd, &alpha, psi.data(), &Nd, ref_Xi_ace.data(), &Nd, &beta, Y.data(), &Ns);

        for (int n = 0; n < Ns; n++) {
            if (occ[n] < 1e-6) continue;
            double sum = 0;
            for (int j = 0; j < Ns; j++)
                sum += Y[n + j*Ns] * Y[n + j*Ns];
            ref_Eexx += occ[n] * sum;
        }
        ref_Eexx *= -params.exx_frac / 1.0;  // / Nspin=1
    }
    double lynx_Eexx = exx.compute_energy(wfn);
    printf("\n--- Compare Eexx ---\n");
    printf("ref_Eexx  = %.15e\n", ref_Eexx);
    printf("lynx_Eexx = %.15e\n", lynx_Eexx);
    printf("diff      = %.6e\n", std::fabs(ref_Eexx - lynx_Eexx));

    // Compare apply_Vx on first orbital
    printf("\n--- Compare apply_Vx ---\n");
    std::vector<double> ref_Hx(Nd, 0.0), lynx_Hx(Nd, 0.0);
    {
        // ref: Hx -= exx_frac * Xi * (Xi^T * X)
        std::vector<double> Y(Ns, 0.0);
        char tA = 'T', tB = 'N';
        double alpha = 1.0, beta = 0.0;
        int one = 1;
        dgemm_(&tA, &tB, &Ns, &one, &Nd, &alpha, ref_Xi_ace.data(), &Nd, psi.data(), &Nd, &beta, Y.data(), &Ns);
        alpha = -params.exx_frac;
        beta = 0.0;
        tA = 'N';
        dgemm_(&tA, &tB, &Nd, &one, &Ns, &alpha, ref_Xi_ace.data(), &Nd, Y.data(), &Ns, &beta, ref_Hx.data(), &Nd);
    }
    exx.apply_Vx(psi.data(), Nd, 1, Nd, lynx_Hx.data(), Nd, 0);

    double max_diff_hx = 0, max_ref_hx = 0;
    for (int k = 0; k < Nd; k++) {
        max_diff_hx = std::max(max_diff_hx, std::fabs(ref_Hx[k] - lynx_Hx[k]));
        max_ref_hx = std::max(max_ref_hx, std::fabs(ref_Hx[k]));
    }
    printf("apply_Vx: max_diff=%.6e, max_ref=%.6e, rel_err=%.6e\n",
           max_diff_hx, max_ref_hx, max_diff_hx / (max_ref_hx + 1e-30));
    printf("  ref_Hx[0]=%.10e, lynx_Hx[0]=%.10e\n", ref_Hx[0], lynx_Hx[0]);

    // ======== VERDICT ========
    double rel_eexx = std::fabs(ref_Eexx - lynx_Eexx) / (std::fabs(ref_Eexx) + 1e-30);
    double rel_hx = max_diff_hx / (max_ref_hx + 1e-30);
    printf("\n=== VERDICT ===\n");
    printf("Eexx relative error: %.6e %s\n", rel_eexx, rel_eexx < 1e-10 ? "PASS" : "FAIL");
    printf("Hx relative error:   %.6e %s\n", rel_hx, rel_hx < 1e-10 ? "PASS" : "FAIL");
}

// ===========================================================================
// Test: non-orthogonal cell (FCC diamond)
// ===========================================================================
void test_gamma_nonorth() {
    printf("\n=== TEST: Gamma-point PBE0 non-orthogonal (FCC) ===\n");

    Mat3 latvec;
    latvec(0,0) = 3.370070919135085; latvec(0,1) = 3.370070919135085; latvec(0,2) = 0.0;
    latvec(1,0) = 0.0; latvec(1,1) = 3.370070919135085; latvec(1,2) = 3.370070919135085;
    latvec(2,0) = 3.370070919135085; latvec(2,1) = 0.0; latvec(2,2) = 3.370070919135085;

    Lattice lattice(latvec, CellType::NonOrthogonal);
    int Nx = 16, Ny = 16, Nz = 16;
    FDGrid grid(Nx, Ny, Nz, lattice, BCType::Periodic, BCType::Periodic, BCType::Periodic);
    int Nd = Nx * Ny * Nz;
    int Ns = 6;
    double dV = grid.dV();
    double volume = lattice.jacobian();
    double L1 = grid.dx() * Nx, L2 = grid.dy() * Ny, L3 = grid.dz() * Nz;

    printf("Grid: %dx%dx%d = %d, dV=%.10e, volume=%.6f\n", Nx, Ny, Nz, Nd, dV, volume);

    // Get lapcT in row-major (SPARC stores it row-major)
    const Mat3& lt = lattice.lapc_T();
    double lapcT[9] = {lt(0,0), lt(0,1), lt(0,2),
                       lt(1,0), lt(1,1), lt(1,2),
                       lt(2,0), lt(2,1), lt(2,2)};
    printf("lapcT = [[%.4f,%.4f,%.4f],[%.4f,%.4f,%.4f],[%.4f,%.4f,%.4f]]\n",
           lapcT[0],lapcT[1],lapcT[2],lapcT[3],lapcT[4],lapcT[5],lapcT[6],lapcT[7],lapcT[8]);

    // Random orbitals
    std::vector<double> psi(Nd * Ns);
    std::srand(42);
    for (int i = 0; i < Nd * Ns; i++) psi[i] = (std::rand() / (double)RAND_MAX) - 0.5;

    std::vector<double> occ(Ns);
    for (int n = 0; n < Ns; n++) occ[n] = 1.0 - 0.1 * n;

    // Reference
    std::vector<double> ref_pois_const;
    ref::compute_pois_fft_const(Nx, Ny, Nz, L1, L2, L3, volume, 1, lapcT, ref_pois_const);

    std::vector<double> ref_Xi(Nd * Ns);
    ref::solve_for_Xi(psi.data(), occ.data(), Nd, Ns, dV, ref_pois_const.data(),
                      Nx, Ny, Nz, ref_Xi.data());

    std::vector<double> ref_Xi_ace(ref_Xi), ref_M(Ns*Ns);
    ref::calculate_ACE(psi.data(), Nd, Ns, dV, ref_Xi_ace.data(), ref_M.data());

    // LYNX
    Wavefunction wfn;
    wfn.allocate(Nd, Ns, Ns, 1, 1, false, 1);
    std::memcpy(wfn.psi(0, 0).data(), psi.data(), Nd * Ns * sizeof(double));
    for (int n = 0; n < Ns; n++) {
        wfn.occupations(0, 0)(n) = occ[n];
        wfn.eigenvalues(0, 0)(n) = -1.0 + 0.5 * n;
    }

    EXXParams params;
    params.exx_frac = 0.25;
    params.hyb_range_fock = -1.0;
    params.exx_div_flag = 0;

    MPIComm bandcomm(MPI_COMM_SELF);
    MPIComm kpt_bridge(MPI_COMM_SELF);
    MPIComm spin_bridge(MPI_COMM_SELF);

    ExactExchange exx;
    exx.setup(grid, lattice, nullptr, bandcomm, kpt_bridge, spin_bridge,
              params, 1, Ns, Ns, 0, 1, 1, 0, 0, 1, 1, 1);
    exx.build_ACE(wfn);

    // Compare Poisson solve
    {
        std::vector<double> test_rhs(Nd);
        for (int k = 0; k < Nd; k++) test_rhs[k] = psi[k] * psi[k];
        std::vector<double> ref_sol(Nd), lynx_sol(Nd);
        ref::pois_fft(test_rhs.data(), ref_pois_const.data(), Nx, Ny, Nz, ref_sol.data());
        ExchangePoissonSolver lynx_poisson;
        lynx_poisson.setup(grid, lattice, params, nullptr, 1, 1, 1);
        std::vector<double> lynx_rhs(test_rhs);
        lynx_poisson.solve_batch(lynx_rhs.data(), 1, lynx_sol.data());
        double max_diff = 0, max_ref = 0;
        for (int k = 0; k < Nd; k++) {
            max_diff = std::max(max_diff, std::fabs(ref_sol[k] - lynx_sol[k]));
            max_ref = std::max(max_ref, std::fabs(ref_sol[k]));
        }
        printf("Poisson solve: rel_err=%.6e %s\n",
               max_diff/(max_ref+1e-30), max_diff/(max_ref+1e-30) < 1e-10 ? "PASS" : "FAIL");
    }

    // Compare Eexx (using sqrt(dV) in Y computation to match LYNX)
    double ref_Eexx = 0.0;
    {
        std::vector<double> Y(Ns * Ns, 0.0);
        char tA = 'T', tB = 'N';
        double alpha = std::sqrt(dV), beta = 0.0;
        dgemm_(&tA, &tB, &Ns, &Ns, &Nd, &alpha, psi.data(), &Nd, ref_Xi_ace.data(), &Nd, &beta, Y.data(), &Ns);
        for (int n = 0; n < Ns; n++) {
            if (occ[n] < 1e-6) continue;
            double sum = 0;
            for (int j = 0; j < Ns; j++) sum += Y[n + j*Ns] * Y[n + j*Ns];
            ref_Eexx += occ[n] * sum;
        }
        ref_Eexx *= -params.exx_frac;
    }
    double lynx_Eexx = exx.compute_energy(wfn);
    double rel_eexx = std::fabs(ref_Eexx - lynx_Eexx) / (std::fabs(ref_Eexx) + 1e-30);
    printf("Eexx: ref=%.15e, lynx=%.15e, rel_err=%.6e %s\n",
           ref_Eexx, lynx_Eexx, rel_eexx, rel_eexx < 1e-10 ? "PASS" : "FAIL");

    // Compare apply_Vx
    std::vector<double> ref_Hx(Nd, 0.0), lynx_Hx(Nd, 0.0);
    {
        std::vector<double> Y(Ns, 0.0);
        char tA = 'T', tB = 'N';
        double alpha = 1.0, beta = 0.0;
        int one = 1;
        dgemm_(&tA, &tB, &Ns, &one, &Nd, &alpha, ref_Xi_ace.data(), &Nd, psi.data(), &Nd, &beta, Y.data(), &Ns);
        alpha = -params.exx_frac; beta = 0.0; tA = 'N';
        dgemm_(&tA, &tB, &Nd, &one, &Ns, &alpha, ref_Xi_ace.data(), &Nd, Y.data(), &Ns, &beta, ref_Hx.data(), &Nd);
    }
    exx.apply_Vx(psi.data(), Nd, 1, Nd, lynx_Hx.data(), Nd, 0);
    double max_diff_hx = 0, max_ref_hx = 0;
    for (int k = 0; k < Nd; k++) {
        max_diff_hx = std::max(max_diff_hx, std::fabs(ref_Hx[k] - lynx_Hx[k]));
        max_ref_hx = std::max(max_ref_hx, std::fabs(ref_Hx[k]));
    }
    double rel_hx = max_diff_hx / (max_ref_hx + 1e-30);
    printf("Hx: rel_err=%.6e %s\n", rel_hx, rel_hx < 1e-10 ? "PASS" : "FAIL");
}

// ===========================================================================
// Test: k-point case (2x2x2 MP grid, non-orthogonal FCC)
// ===========================================================================
void test_kpoint() {
    printf("\n=== TEST: K-point PBE0 (2x2x2 FCC) ===\n");

    // FCC cell
    Mat3 latvec;
    double a = 5.13;  // ~half equilibrium Si
    latvec(0,0) = a*0.5; latvec(0,1) = a*0.5; latvec(0,2) = 0.0;
    latvec(1,0) = 0.0;   latvec(1,1) = a*0.5; latvec(1,2) = a*0.5;
    latvec(2,0) = a*0.5; latvec(2,1) = 0.0;   latvec(2,2) = a*0.5;

    Lattice lattice(latvec, CellType::NonOrthogonal);
    int Nx = 12, Ny = 12, Nz = 12;
    FDGrid grid(Nx, Ny, Nz, lattice, BCType::Periodic, BCType::Periodic, BCType::Periodic);
    int Nd = Nx * Ny * Nz;
    int Ns = 6;
    double dV = grid.dV();
    double volume = lattice.jacobian();
    double L1 = grid.dx() * Nx, L2 = grid.dy() * Ny, L3 = grid.dz() * Nz;

    printf("Grid: %dx%dx%d = %d, dV=%.10e, volume=%.6f\n", Nx, Ny, Nz, Nd, dV, volume);

    // Generate 2x2x2 k-points
    KPoints kpoints;
    kpoints.generate(2, 2, 2, Vec3(0.5, 0.5, 0.5), lattice);
    int Nkpts = kpoints.Nkpts();
    printf("K-points: Nkpts_sym=%d, Nkpts_full=%d\n", Nkpts, kpoints.Nkpts_full());
    for (int ik = 0; ik < Nkpts; ik++) {
        auto& kr = kpoints.kpts_red()[ik];
        printf("  k[%d]: (%.4f, %.4f, %.4f) wt=%.3f\n", ik, kr.x, kr.y, kr.z, kpoints.weights()[ik]);
    }

    // Create complex wavefunctions for all k-points
    Wavefunction wfn;
    wfn.allocate(Nd, Ns, Ns, 1, Nkpts, true, 1);
    std::srand(42);
    for (int ik = 0; ik < Nkpts; ik++) {
        Complex* psi_k = wfn.psi_kpt(0, ik).data();
        for (int i = 0; i < Nd * Ns; i++)
            psi_k[i] = Complex((std::rand()/(double)RAND_MAX) - 0.5,
                               (std::rand()/(double)RAND_MAX) - 0.5);
        for (int n = 0; n < Ns; n++) {
            wfn.occupations(0, ik)(n) = 1.0 - 0.1 * n;
            wfn.eigenvalues(0, ik)(n) = -1.0 + 0.5 * n;
        }
    }

    // Setup LYNX ExactExchange
    EXXParams params;
    params.exx_frac = 0.25;
    params.hyb_range_fock = -1.0;  // PBE0
    params.exx_div_flag = 0;       // spherical cutoff

    MPIComm bandcomm(MPI_COMM_SELF);
    MPIComm kpt_bridge(MPI_COMM_SELF);
    MPIComm spin_bridge(MPI_COMM_SELF);

    ExactExchange exx;
    exx.setup(grid, lattice, &kpoints, bandcomm, kpt_bridge, spin_bridge,
              params, 1, Ns, Ns, 0, 1, 1, 0, 0, 2, 2, 2);
    exx.build_ACE(wfn);

    double lynx_Eexx = exx.compute_energy(wfn);
    printf("LYNX Eexx = %.15e\n", lynx_Eexx);

    // ===== Reference k-point implementation =====
    // For k-points, Xi at each k is built from pairs (k,q) over all q-points.
    // For each (k, q): rhs = conj(psi_q[j]) * psi_k[i], solve Poisson with (k,q) shift.
    // Then M = Xi^H * psi, Cholesky, trsm.

    // Setup reference Poisson solver (reuse LYNX's for the complex FFT)
    ExchangePoissonSolver ref_poisson;
    ref_poisson.setup(grid, lattice, params, &kpoints, 2, 2, 2);

    // Compute pois_const for k-point (full Nd, not Ndc)
    const Mat3& lt = lattice.lapc_T();
    int Nkpts_shift = ref_poisson.Nkpts_shift();
    printf("Nkpts_shift = %d\n", Nkpts_shift);

    // No inv_dV needed for LYNX normalization

    // Build reference Xi for each k-point using full HF BZ
    kpoints.setup_hf_kpoints();
    int Nkpts_hf = kpoints.Nkpts_hf();
    const auto& kpthf_ind = kpoints.kpthf_ind();
    const auto& kpthf_pn = kpoints.kpthf_pn();
    double kptWts_hf = 1.0 / kpoints.Nkpts_full();

    std::vector<std::vector<Complex>> ref_Xi_kpt(Nkpts);

    for (int ik = 0; ik < Nkpts; ik++) {
        ref_Xi_kpt[ik].resize(Nd * Ns, Complex(0.0));
        const Complex* psi_k = wfn.psi_kpt(0, ik).data();

        for (int q_hf = 0; q_hf < Nkpts_hf; q_hf++) {
            int q_sym = kpthf_ind[q_hf];
            int q_pn = kpthf_pn[q_hf];

            // Get psi_q (conjugate if time-reversed)
            std::vector<Complex> psi_q(Nd * Ns);
            std::memcpy(psi_q.data(), wfn.psi_kpt(0, q_sym).data(), Nd * Ns * sizeof(Complex));
            if (q_pn == 0) {
                for (int m = 0; m < Nd * Ns; m++) psi_q[m] = std::conj(psi_q[m]);
            }
            const double* occ_q = wfn.occupations(0, q_sym).data();

            for (int j = 0; j < Ns; j++) {
                if (occ_q[j] < 1e-6) continue;
                for (int i = 0; i < Ns; i++) {
                    std::vector<Complex> rhs(Nd), sol(Nd);
                    for (int m = 0; m < Nd; m++)
                        rhs[m] = std::conj(psi_q[m + j*Nd]) * psi_k[m + i*Nd];

                    ref_poisson.solve_batch_kpt(rhs.data(), 1, sol.data(), ik, q_hf);

                    double coeff = kptWts_hf * occ_q[j] * std::sqrt(dV);
                    for (int m = 0; m < Nd; m++)
                        ref_Xi_kpt[ik][m + i*Nd] -= coeff * psi_q[m + j*Nd] * sol[m];
                }
            }
        }
    }

    // Debug: compare Xi_raw norms
    double sqrt_dV_dbg = std::sqrt(dV);
    printf("DEBUG dV=%.10e, sqrt(dV)=%.10e\n", dV, sqrt_dV_dbg);
    for (int ik = 0; ik < Nkpts; ik++) {
        double ref_norm = 0, lynx_norm = 0;
        const Complex* lynx_Xi_raw = exx.Xi_kpt_data(0, ik);  // need accessor
        for (int n = 0; n < Ns; n++) {
            double rn = 0, ln = 0;
            for (int k = 0; k < Nd; k++) {
                rn += std::norm(ref_Xi_kpt[ik][k + n*Nd]);
                ln += std::norm(lynx_Xi_raw[k + n*Nd]);
            }
            if (n == 0) printf("DEBUG Xi_raw k=%d n=0: ref_norm=%.6e, lynx_norm=%.6e, ratio=%.6e (expected sqrt(dV)=%.6e)\n",
                               ik, std::sqrt(rn), std::sqrt(ln), std::sqrt(ln/rn), sqrt_dV_dbg);
        }
    }

    // ACE: M = Xi^H * psi, Cholesky(-M), Xi = Xi_raw * L^{-H}
    std::vector<std::vector<Complex>> ref_Xi_ace(Nkpts);
    for (int ik = 0; ik < Nkpts; ik++) {
        ref_Xi_ace[ik] = ref_Xi_kpt[ik];
        const Complex* psi_k = wfn.psi_kpt(0, ik).data();

        std::vector<Complex> M(Ns * Ns, Complex(0.0));
        // M = sqrt(dV) * Xi^H * psi (matching LYNX convention)
        {
            char tA = 'C', tB = 'N';
            Complex alpha(std::sqrt(dV), 0.0), beta(0.0, 0.0);
            zgemm_(&tA, &tB, &Ns, &Ns, &Nd, &alpha, ref_Xi_ace[ik].data(), &Nd,
                   psi_k, &Nd, &beta, M.data(), &Ns);
        }
        // Negate
        for (int i = 0; i < Ns*Ns; i++) M[i] = -M[i];
        // Cholesky
        char uplo = 'U';
        int info = 0;
        zpotrf_(&uplo, &Ns, M.data(), &Ns, &info);
        if (info != 0) { printf("REF kpt zpotrf failed k=%d info=%d\n", ik, info); continue; }
        // Xi_ace = Xi_raw * L^{-H} (Xi already has Xi_raw)
        {
            char side = 'R', trans = 'N', diag = 'N';
            Complex alpha(1.0, 0.0);
            ztrsm_(&side, &uplo, &trans, &diag, &Nd, &Ns, &alpha, M.data(), &Ns,
                   ref_Xi_ace[ik].data(), &Nd);
        }
    }

    // Reference Eexx — SPARC uses kptWts_loc[k] / Nkpts_sym
    double ref_Eexx = 0.0;
    auto sym_wts = kpoints.weights();
    for (int ik = 0; ik < Nkpts; ik++) {
        double wk = sym_wts[ik] / kpoints.Nkpts();
        const Complex* psi_k = wfn.psi_kpt(0, ik).data();
        const double* occ_k = wfn.occupations(0, ik).data();

        std::vector<Complex> Y(Ns * Ns, Complex(0.0));
        {
            char tA = 'C', tB = 'N';
            Complex alpha(std::sqrt(dV), 0.0), beta(0.0, 0.0);
            zgemm_(&tA, &tB, &Ns, &Ns, &Nd, &alpha, ref_Xi_ace[ik].data(), &Nd,
                   psi_k, &Nd, &beta, Y.data(), &Ns);
        }
        for (int n = 0; n < Ns; n++) {
            if (occ_k[n] < 1e-6) continue;
            double sum = 0;
            for (int j = 0; j < Ns; j++) {
                double r = Y[j + n*Ns].real(), im = Y[j + n*Ns].imag();
                sum += r*r + im*im;
            }
            ref_Eexx += wk * occ_k[n] * sum;
        }
    }
    ref_Eexx *= -params.exx_frac;

    printf("REF  Eexx = %.15e\n", ref_Eexx);
    double rel = std::fabs(ref_Eexx - lynx_Eexx) / (std::fabs(ref_Eexx) + 1e-30);
    printf("Eexx rel_err = %.6e %s\n", rel, rel < 1e-10 ? "PASS" : "FAIL");

    // Compare apply_Vx_kpt for first k-point, first orbital
    {
        const Complex* psi_k0 = wfn.psi_kpt(0, 0).data();
        std::vector<Complex> ref_Hx(Nd, Complex(0.0)), lynx_Hx(Nd, Complex(0.0));

        // Reference
        {
            std::vector<Complex> Y(Ns, Complex(0.0));
            char tA = 'C', tB = 'N';
            Complex alpha(1.0, 0.0), beta(0.0, 0.0);
            int one = 1;
            zgemm_(&tA, &tB, &Ns, &one, &Nd, &alpha, ref_Xi_ace[0].data(), &Nd,
                   psi_k0, &Nd, &beta, Y.data(), &Ns);
            Complex malpha(-params.exx_frac, 0.0);
            beta = Complex(0.0, 0.0);
            tA = 'N';
            zgemm_(&tA, &tB, &Nd, &one, &Ns, &malpha, ref_Xi_ace[0].data(), &Nd,
                   Y.data(), &Ns, &beta, ref_Hx.data(), &Nd);
        }

        exx.apply_Vx_kpt(psi_k0, Nd, 1, Nd, lynx_Hx.data(), Nd, 0, 0);

        double max_diff = 0, max_ref = 0;
        for (int k = 0; k < Nd; k++) {
            max_diff = std::max(max_diff, std::abs(ref_Hx[k] - lynx_Hx[k]));
            max_ref = std::max(max_ref, std::abs(ref_Hx[k]));
        }
        double rel_hx = max_diff / (max_ref + 1e-30);
        printf("Hx_kpt: rel_err=%.6e %s\n", rel_hx, rel_hx < 1e-10 ? "PASS" : "FAIL");
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    test_gamma_pbe0();
    test_gamma_nonorth();
    test_kpoint();

    printf("\n=== All unit tests complete ===\n");
    MPI_Finalize();
    return 0;
}
