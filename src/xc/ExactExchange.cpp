#include "xc/ExactExchange.hpp"
#include "core/constants.hpp"
#include "parallel/Parallelization.hpp"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <mpi.h>

// BLAS/LAPACK declarations
extern "C" {
    void dgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
                const double* alpha, const double* a, const int* lda, const double* b, const int* ldb,
                const double* beta, double* c, const int* ldc);
    void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
    void dtrsm_(const char* side, const char* uplo, const char* transa, const char* diag,
                const int* m, const int* n, const double* alpha, const double* a, const int* lda,
                double* b, const int* ldb);

    void zgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
                const void* alpha, const void* a, const int* lda, const void* b, const int* ldb,
                const void* beta, void* c, const int* ldc);
    void zpotrf_(const char* uplo, const int* n, void* a, const int* lda, int* info);
    void ztrsm_(const char* side, const char* uplo, const char* transa, const char* diag,
                const int* m, const int* n, const void* alpha, const void* a, const int* lda,
                void* b, const int* ldb);
}

namespace lynx {

// ---------------------------------------------------------------------------
// Setup (LynxContext overload)
// ---------------------------------------------------------------------------
void ExactExchange::setup(const LynxContext& ctx,
                           const EXXParams& params,
                           int Kx_hf, int Ky_hf, int Kz_hf) {
    ctx_ = &ctx;
    int npband = ctx.scf_bandcomm().size();
    int npkpt = ctx.kpt_bridge().size();
    setup(ctx.grid(), ctx.lattice(), &ctx.kpoints(),
          ctx.scf_bandcomm(), ctx.kpt_bridge(), ctx.spin_bridge(),
          params, ctx.Nspin(), ctx.Nstates(), ctx.Nband_local(), ctx.band_start(),
          npband, npkpt, ctx.kpt_start(), ctx.spin_start(),
          Kx_hf, Ky_hf, Kz_hf);
}

// ---------------------------------------------------------------------------
// Setup (explicit params)
// ---------------------------------------------------------------------------
void ExactExchange::setup(const FDGrid& grid, const Lattice& lattice,
                           const KPoints* kpoints,
                           const MPIComm& bandcomm,
                           const MPIComm& kpt_bridge,
                           const MPIComm& spin_bridge,
                           const EXXParams& params,
                           int Nspin, int Nstates, int Nband_local, int band_start,
                           int npband, int npkpt, int kpt_start, int spin_start,
                           int Kx_hf, int Ky_hf, int Kz_hf) {
    params_ = params;
    exx_frac_ = params.exx_frac;
    Nd_d_ = grid.Nd();
    dV_ = grid.dV();
    grid_ = &grid;
    lattice_ = &lattice;
    bandcomm_ = &bandcomm;
    kpt_bridge_ = &kpt_bridge;
    spin_bridge_ = &spin_bridge;
    Nspin_ = Nspin;
    Nstates_ = Nstates;
    Nband_local_ = Nband_local;
    band_start_ = band_start;
    npband_ = npband;
    npkpt_ = npkpt;
    kpt_start_ = kpt_start;
    spin_start_ = spin_start;
    kpoints_ = kpoints;
    is_gamma_ = !kpoints || kpoints->is_gamma_only();

    // Local k-points
    if (kpoints) {
        // Each kpt_bridge rank handles ceil(Nkpts/npkpt) k-points
        Nkpts_local_ = kpoints->Nkpts();  // Will be distributed if npkpt > 1
        // For now: Nkpts_local is already set by parallelization
    } else {
        Nkpts_local_ = 1;
    }

    // Setup the exchange Poisson solver
    poisson_.setup(grid, lattice, params, kpoints, Kx_hf, Ky_hf, Kz_hf);

    setup_done_ = true;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::printf("EXX setup: exx_frac=%.4f, gamma=%d, Nstates=%d, Nband_local=%d, npband=%d\n",
                    exx_frac_, is_gamma_ ? 1 : 0, Nstates_, Nband_local_, npband_);
    }
}

// ---------------------------------------------------------------------------
// Gather all bands from band-parallel processes (Allgatherv)
// ---------------------------------------------------------------------------
void ExactExchange::gather_bandcomm(double* vec, int Nd, int Ns) const {
    if (npband_ <= 1) return;

    int rank = bandcomm_->rank();
    int size = bandcomm_->size();

    std::vector<int> recvcounts(size), displs(size);
    int NB = (Ns + size - 1) / size;
    for (int i = 0; i < size; i++) {
        int start = i * NB;
        int count = std::min(NB, Ns - start);
        if (count < 0) count = 0;
        recvcounts[i] = count * Nd;
        displs[i] = start * Nd;
    }

    MPI_Allgatherv(MPI_IN_PLACE, recvcounts[rank], MPI_DOUBLE,
                   vec, recvcounts.data(), displs.data(), MPI_DOUBLE,
                   bandcomm_->comm());
}

void ExactExchange::gather_bandcomm_kpt(Complex* vec, int Nd, int Ns) const {
    if (npband_ <= 1) return;

    int rank = bandcomm_->rank();
    int size = bandcomm_->size();

    std::vector<int> recvcounts(size), displs(size);
    int NB = (Ns + size - 1) / size;
    for (int i = 0; i < size; i++) {
        int start = i * NB;
        int count = std::min(NB, Ns - start);
        if (count < 0) count = 0;
        recvcounts[i] = count * Nd;
        displs[i] = start * Nd;
    }

    MPI_Allgatherv(MPI_IN_PLACE, recvcounts[rank], MPI_DOUBLE_COMPLEX,
                   vec, recvcounts.data(), displs.data(), MPI_DOUBLE_COMPLEX,
                   bandcomm_->comm());
}

// ---------------------------------------------------------------------------
// Cyclic orbital rotation across bandcomm
// ---------------------------------------------------------------------------
void ExactExchange::transfer_orbitals_bandcomm(const double* sendbuf, int send_count,
                                                 double* recvbuf, int recv_count, int shift) const {
    int rank = bandcomm_->rank();
    int size = bandcomm_->size();
    int lneighbor = (rank - shift + size) % size;
    int rneighbor = (rank + shift) % size;

    MPI_Request reqs[2];
    MPI_Irecv(recvbuf, recv_count, MPI_DOUBLE, lneighbor, 111, bandcomm_->comm(), &reqs[0]);
    MPI_Isend(sendbuf, send_count, MPI_DOUBLE, rneighbor, 111, bandcomm_->comm(), &reqs[1]);
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
}

void ExactExchange::transfer_orbitals_bandcomm_kpt(const Complex* sendbuf, int send_count,
                                                     Complex* recvbuf, int recv_count, int shift) const {
    int rank = bandcomm_->rank();
    int size = bandcomm_->size();
    int lneighbor = (rank - shift + size) % size;
    int rneighbor = (rank + shift) % size;

    MPI_Request reqs[2];
    MPI_Irecv(recvbuf, recv_count, MPI_DOUBLE_COMPLEX, lneighbor, 111, bandcomm_->comm(), &reqs[0]);
    MPI_Isend(sendbuf, send_count, MPI_DOUBLE_COMPLEX, rneighbor, 111, bandcomm_->comm(), &reqs[1]);
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
}

// ---------------------------------------------------------------------------
// Allocate ACE operator (gamma-point)
// ---------------------------------------------------------------------------
void ExactExchange::allocate_ACE(const Wavefunction& wfn) {
    // Count occupied states (matching SPARC: Nstates_occ = max occ count + ACE_VALENCE_STATES)
    Nstates_occ_ = 0;
    for (int s = 0; s < wfn.Nspin(); s++) {
        for (int k = 0; k < wfn.Nkpts(); k++) {
            const auto& occ = wfn.occupations(s, k);
            for (int n = 0; n < wfn.Nband_global(); n++) {
                if (occ(n) > OCC_THRESHOLD)
                    Nstates_occ_ = std::max(Nstates_occ_, n + 1);
            }
        }
    }
    Nstates_occ_ += 3;  // default EXX_ACE_VALENCE_STATES = 3
    Nstates_occ_ = std::min(Nstates_occ_, Nstates_);

    Xi_.resize(Nspin_);
    for (int s = 0; s < Nspin_; s++) {
        Xi_[s] = NDArray<double>(Nd_d_ * Nstates_occ_);
        Xi_[s].zero();
    }
}

// ---------------------------------------------------------------------------
// Allocate ACE operator (k-point)
// ---------------------------------------------------------------------------
void ExactExchange::allocate_ACE_kpt(const Wavefunction& wfn) {
    Nstates_occ_ = 0;
    for (int s = 0; s < wfn.Nspin(); s++) {
        for (int k = 0; k < wfn.Nkpts(); k++) {
            const auto& occ = wfn.occupations(s, k);
            for (int n = 0; n < wfn.Nband_global(); n++) {
                if (occ(n) > OCC_THRESHOLD)
                    Nstates_occ_ = std::max(Nstates_occ_, n + 1);
            }
        }
    }
    Nstates_occ_ += 3;
    Nstates_occ_ = std::min(Nstates_occ_, Nstates_);

    int Nkpts = wfn.Nkpts();
    Xi_kpt_.resize(Nspin_ * Nkpts);
    for (int s = 0; s < Nspin_; s++) {
        for (int k = 0; k < Nkpts; k++) {
            Xi_kpt_[s * Nkpts + k] = NDArray<Complex>(Nd_d_ * Nstates_occ_);
            Xi_kpt_[s * Nkpts + k].zero();
        }
    }
}

// ---------------------------------------------------------------------------
// Build ACE operator from current orbitals
// ---------------------------------------------------------------------------
void ExactExchange::build_ACE(const Wavefunction& wfn) {
    if (is_gamma_) {
        allocate_ACE(wfn);
        for (int s = 0; s < wfn.Nspin(); s++) {
            solve_for_Xi(wfn, s);

            calculate_ACE_operator(wfn, s);

        }
    } else {
        allocate_ACE_kpt(wfn);
        for (int s = 0; s < wfn.Nspin(); s++) {
            solve_for_Xi_kpt(wfn, s);
            for (int k = 0; k < wfn.Nkpts(); k++) {
                calculate_ACE_operator_kpt(wfn, s, k);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Solve for Xi (gamma-point ACE construction)
//
// For each pair (i,j) with i >= j and occ[i]+occ[j] > threshold:
//   rhs[k] = psi[k,i] * psi[k,j]
//   Solve: Poisson(rhs) = sol
//   Xi[k,i] -= (occ[j]/dV) * psi[k,j] * sol[k]
//   if i != j: Xi[k,j] -= (occ[i]/dV) * psi[k,i] * sol[k]
// ---------------------------------------------------------------------------
void ExactExchange::solve_for_Xi(const Wavefunction& wfn, int spin) {
    const int Nd = Nd_d_;
    const int Ns = Nstates_;
    const int Nocc = Nstates_occ_;

    // LYNX orbitals satisfy psi^T * psi * dV = I  ("lattice" normalization)
    // SPARC orbitals satisfy psi^T * psi = I       ("bare-sum" normalization)
    // SPARC's Poisson solver returns Nd * phi_physical (no 1/Nd normalization)
    // LYNX's Poisson solver returns phi_physical (divides by Nd)
    //
    // To make Xi_LYNX = Xi_SPARC (numerically identical ACE operator),
    // the coefficient must absorb both factors:
    //   coeff = occ * Nd * sqrt(dV)   instead of SPARC's   occ / dV
    //
    // Derivation: Xi_S = occ/dV * psi_S * sol_S
    //   psi_S = sqrt(dV)*psi_L,  sol_L = sol_S/dV  (both SPARC & LYNX normalize iFFT by 1/Nd)
    //   => Xi_S = occ/dV * psi_S * sol_S, want Xi_L = Xi_S
    //   coeff * psi_L * sol_L = occ/dV * psi_S * sol_S
    //   coeff/(sqrt(dV)*dV) = occ/dV => coeff = occ*sqrt(dV)
    double coeff_scale = std::sqrt(dV_);

    // Gather full psi (LYNX lattice orbitals)
    std::vector<double> psi_full(Nd * Ns);
    {
        int ld = wfn.psi(spin, 0).ld();
        const double* src = wfn.psi(spin, 0).data();
        for (int j = 0; j < Nband_local_; ++j)
            std::memcpy(psi_full.data() + (band_start_ + j) * Nd,
                        src + j * ld, Nd * sizeof(double));
    }
    gather_bandcomm(psi_full.data(), Nd, Ns);

    const double* occ = wfn.occupations(spin, 0).data();

    double* Xi = Xi_[spin].data();
    std::memset(Xi, 0, sizeof(double) * Nd * Nocc);

    std::vector<double> rhs(Nd);
    std::vector<double> sol(Nd);

    // Xi -= occ * Nd * sqrt(dV) * psi * sol  (gives Xi_LYNX = Xi_SPARC)
    for (int j = 0; j < Nocc; j++) {
        if (occ[j] < OCC_THRESHOLD) continue;
        const double* psi_j = psi_full.data() + j * Nd;

        for (int i = j; i < Nocc; i++) {
            const double* psi_i = psi_full.data() + i * Nd;

            for (int k = 0; k < Nd; k++)
                rhs[k] = psi_i[k] * psi_j[k];

            poisson_.solve_batch(rhs.data(), 1, sol.data());

            double coeff_j = occ[j] * coeff_scale;
            for (int k = 0; k < Nd; k++)
                Xi[i * Nd + k] -= coeff_j * psi_j[k] * sol[k];

            if (i != j && occ[i] > OCC_THRESHOLD) {
                double coeff_i = occ[i] * coeff_scale;
                for (int k = 0; k < Nd; k++)
                    Xi[j * Nd + k] -= coeff_i * psi_i[k] * sol[k];
            }
        }
    }

    // Allreduce Xi across bandcomm (if band-parallel Poisson pairs are distributed)
    // In serial mode (npband=1), no reduction needed
    // With band parallelism and cyclic rotation, this would be needed
    // For now: all procs compute all pairs (no distribution)
}

// ---------------------------------------------------------------------------
// Calculate ACE operator (gamma-point)
// M = Xi^T * psi, Cholesky(-M), Xi = psi * M^{-T}
// ---------------------------------------------------------------------------
void ExactExchange::calculate_ACE_operator(const Wavefunction& wfn, int spin) {
    const int Nd = Nd_d_;
    const int Ns = Nstates_;
    const int Nocc = Nstates_occ_;

    // Gather full psi (raw lattice orbitals, matching SPARC convention)
    std::vector<double> psi_full(Nd * Ns);
    {
        int ld = wfn.psi(spin, 0).ld();
        const double* src = wfn.psi(spin, 0).data();
        for (int j = 0; j < Nband_local_; ++j)
            std::memcpy(psi_full.data() + (band_start_ + j) * Nd,
                        src + j * ld, Nd * sizeof(double));
    }
    gather_bandcomm(psi_full.data(), Nd, Ns);

    double* Xi = Xi_[spin].data();

    // M = sqrt(dV) * Xi^T * psi
    // Xi is in SPARC convention (Xi_LYNX = Xi_SPARC), psi is LYNX convention.
    // To get M_SPARC = Xi_S^T * psi_S = Xi_S^T * sqrt(dV)*psi_L,
    // we use alpha = sqrt(dV) in the dgemm.
    std::vector<double> M(Nocc * Nocc, 0.0);
    {
        char transA = 'T', transB = 'N';
        double alpha_blas = std::sqrt(dV_);
        double beta_blas = 0.0;
        dgemm_(&transA, &transB, &Nocc, &Nocc, &Nd,
               &alpha_blas, Xi, &Nd, psi_full.data(), &Nd,
               &beta_blas, M.data(), &Nocc);
    }

    // M = -M
    for (int i = 0; i < Nocc * Nocc; i++) M[i] = -M[i];

    // Cholesky factorization: M = L * L^T (upper triangular)
    int info = 0;
    char uplo = 'U';
    dpotrf_(&uplo, &Nocc, M.data(), &Nocc, &info);
    if (info != 0) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            std::fprintf(stderr, "WARNING: dpotrf failed in ACE operator (info=%d)\n", info);
        return;
    }

    // Xi_ace = Xi_raw * L^{-T}: solve Xi * L^T = Xi_raw (in-place)
    // dtrsm('R', 'U', 'N', 'N', Nd, Nocc, 1.0, L, Nocc, Xi, Nd)
    // Xi already contains Xi_raw from solve_for_Xi()
    {
        char side = 'R', uplo_t = 'U', trans = 'N', diag = 'N';
        double alpha_blas = 1.0;
        dtrsm_(&side, &uplo_t, &trans, &diag,
               &Nd, &Nocc, &alpha_blas, M.data(), &Nocc, Xi, &Nd);
    }
}

// ---------------------------------------------------------------------------
// Solve for Xi (k-point ACE construction)
// ---------------------------------------------------------------------------
void ExactExchange::solve_for_Xi_kpt(const Wavefunction& wfn, int spin) {
    const int Nd = Nd_d_;
    const int Ns = Nstates_;
    const int Nocc = Nstates_occ_;
    int Nkpts = wfn.Nkpts();

    // For each k-point (local to this process):
    for (int kpt_loc = 0; kpt_loc < Nkpts; kpt_loc++) {
        int kpt_glob = kpt_start_ + kpt_loc;

        // Gather full psi_kpt (LYNX lattice orbitals)
        double coeff_scale_k = std::sqrt(dV_);
        std::vector<Complex> psi_k(Nd * Ns);
        {
            int ld = wfn.psi_kpt(spin, kpt_loc).ld();
            const Complex* src = wfn.psi_kpt(spin, kpt_loc).data();
            for (int j = 0; j < Nband_local_; ++j)
                std::memcpy(psi_k.data() + (band_start_ + j) * Nd,
                            src + j * ld, Nd * sizeof(Complex));
        }
        gather_bandcomm_kpt(psi_k.data(), Nd, Ns);

        const double* occ = wfn.occupations(spin, kpt_loc).data();
        Complex* Xi = Xi_kpt_[spin * Nkpts + kpt_loc].data();
        std::memset(Xi, 0, sizeof(Complex) * Nd * Nocc);

        // Iterate over FULL unreduced HF BZ (matching SPARC exactly)
        double kptWts_hf = 1.0 / kpoints_->Nkpts_full();  // flat weight
        int Nkpts_hf = kpoints_->Nkpts_hf();
        const auto& kpthf_ind = kpoints_->kpthf_ind();
        const auto& kpthf_pn = kpoints_->kpthf_pn();

        for (int q_hf = 0; q_hf < Nkpts_hf; q_hf++) {
            int q_sym = kpthf_ind[q_hf];    // sym-reduced index for psi/occ
            int q_pn = kpthf_pn[q_hf];      // 1=direct, 0=time-reversed (need conj)

            // Find local q-index (q_sym is global sym index, need local)
            int q_loc = q_sym - kpt_start_;
            if (q_loc < 0 || q_loc >= Nkpts) continue;  // not local

            // Gather psi for this q-point (raw lattice orbitals)
            std::vector<Complex> psi_q(Nd * Ns);
            {
                int ld_q = wfn.psi_kpt(spin, q_loc).ld();
                const Complex* src_q = wfn.psi_kpt(spin, q_loc).data();
                for (int j = 0; j < Nband_local_; ++j)
                    std::memcpy(psi_q.data() + (band_start_ + j) * Nd,
                                src_q + j * ld_q, Nd * sizeof(Complex));
            }
            gather_bandcomm_kpt(psi_q.data(), Nd, Ns);

            // If time-reversed, conjugate psi_q
            if (q_pn == 0) {
                for (int m = 0; m < Nd * Ns; m++)
                    psi_q[m] = std::conj(psi_q[m]);
            }

            const double* occ_q = wfn.occupations(spin, q_loc).data();

            std::vector<Complex> rhs(Nd);
            std::vector<Complex> sol(Nd);

            // Matching SPARC: j outer (occupied q-states), i inner (all k-states)
            for (int j = 0; j < Nocc; j++) {
                if (occ_q[j] < OCC_THRESHOLD) continue;
                const Complex* psi_qj = psi_q.data() + j * Nd;

                for (int i = 0; i < Nocc; i++) {
                    const Complex* psi_ki = psi_k.data() + i * Nd;

                    // rhs = conj(psi_q[j]) * psi_k[i]
                    for (int m = 0; m < Nd; m++)
                        rhs[m] = std::conj(psi_qj[m]) * psi_ki[m];

                    // Solve Poisson — use full HF BZ index for Kptshift_map
                    poisson_.solve_batch_kpt(rhs.data(), 1, sol.data(), kpt_glob, q_hf);

                    // Xi[k,i] -= kptWts_hf * occ_q[j] * sqrt(dV) * psi_q[j] * sol
                    double coeff = kptWts_hf * occ_q[j] * coeff_scale_k;
                    for (int m = 0; m < Nd; m++)
                        Xi[i * Nd + m] -= coeff * psi_qj[m] * sol[m];
                }
            }
        }

        // If npkpt > 1, allreduce Xi across kpt_bridge
        if (npkpt_ > 1 && !kpt_bridge_->is_null() && kpt_bridge_->size() > 1) {
            MPI_Allreduce(MPI_IN_PLACE, Xi, Nd * Nocc, MPI_DOUBLE_COMPLEX,
                          MPI_SUM, kpt_bridge_->comm());
        }

    }
}

// ---------------------------------------------------------------------------
// Calculate ACE operator (k-point)
// M = Xi^H * psi, Cholesky(-M), Xi = psi * M^{-H}
// ---------------------------------------------------------------------------
void ExactExchange::calculate_ACE_operator_kpt(const Wavefunction& wfn, int spin, int kpt) {
    const int Nd = Nd_d_;
    const int Ns = Nstates_;
    const int Nocc = Nstates_occ_;

    // Gather full psi_kpt (raw lattice orbitals, matching SPARC convention)
    std::vector<Complex> psi_full(Nd * Ns);
    {
        int ld = wfn.psi_kpt(spin, kpt).ld();
        const Complex* src = wfn.psi_kpt(spin, kpt).data();
        for (int j = 0; j < Nband_local_; ++j)
            std::memcpy(psi_full.data() + (band_start_ + j) * Nd,
                        src + j * ld, Nd * sizeof(Complex));
    }
    gather_bandcomm_kpt(psi_full.data(), Nd, Ns);

    Complex* Xi = Xi_kpt_[spin * wfn.Nkpts() + kpt].data();

    // M = sqrt(dV) * Xi^H * psi (Xi is SPARC convention, psi is LYNX)
    std::vector<Complex> M(Nocc * Nocc, Complex(0.0));
    {
        char transA = 'C', transB = 'N';
        Complex alpha_blas(std::sqrt(dV_), 0.0);
        Complex beta_blas(0.0, 0.0);
        zgemm_(&transA, &transB, &Nocc, &Nocc, &Nd,
               &alpha_blas, Xi, &Nd, psi_full.data(), &Nd,
               &beta_blas, M.data(), &Nocc);
    }

    // M = -M
    for (int i = 0; i < Nocc * Nocc; i++) M[i] = -M[i];

    // Cholesky
    int info = 0;
    char uplo = 'U';
    zpotrf_(&uplo, &Nocc, M.data(), &Nocc, &info);
    if (info != 0) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            std::fprintf(stderr, "WARNING: zpotrf failed in ACE operator kpt (info=%d, Nocc=%d)\n", info, Nocc);
        return;
    }

    // Xi_ace = Xi_raw * L^{-H}: solve Xi * L^H = Xi_raw (in-place)
    // Xi already contains Xi_raw from solve_for_Xi_kpt()
    {
        char side = 'R', uplo_t = 'U', trans = 'N', diag = 'N';
        Complex alpha_blas(1.0, 0.0);
        ztrsm_(&side, &uplo_t, &trans, &diag,
               &Nd, &Nocc, &alpha_blas, M.data(), &Nocc, Xi, &Nd);
    }

}

// ---------------------------------------------------------------------------
// Apply Vx (gamma-point): Hx -= exx_frac * Xi * (Xi^T * X)
// ---------------------------------------------------------------------------
void ExactExchange::apply_Vx(const double* X, int ldx, int ncol, int DMnd,
                              double* Hx, int ldhx, int spin) const {
    if (!setup_done_ || Nstates_occ_ == 0) return;

    const int Nd = DMnd;
    const int Nocc = Nstates_occ_;
    const double* Xi = Xi_[spin].data();

    // Y = Xi^T * X  (bare sum, matching SPARC convention)
    std::vector<double> Y(Nocc * ncol, 0.0);
    {
        char transA = 'T', transB = 'N';
        double alpha_blas = 1.0;
        double beta_blas = 0.0;
        dgemm_(&transA, &transB, &Nocc, &ncol, &Nd,
               &alpha_blas, Xi, &Nd, X, &ldx,
               &beta_blas, Y.data(), &Nocc);
    }

    // Hx -= exx_frac * Xi * Y
    // Xi_L = Xi_S (same numerical values), so no 1/dV needed.
    // LYNX eigensolver: e = dV * psi_L^T * H*psi_L, and the dV in the projection
    // cancels with dV in the overlap (Mp = dV * psi^T*psi = I), giving correct eigenvalues.
    {
        char transA = 'N', transB = 'N';
        double alpha_blas = -exx_frac_;
        double beta_blas = 1.0;
        dgemm_(&transA, &transB, &Nd, &ncol, &Nocc,
               &alpha_blas, Xi, &Nd, Y.data(), &Nocc,
               &beta_blas, Hx, &ldhx);
    }
}

// ---------------------------------------------------------------------------
// Apply Vx (k-point): Hx -= exx_frac * Xi * (Xi^H * X)
// ---------------------------------------------------------------------------
void ExactExchange::apply_Vx_kpt(const Complex* X, int ldx, int ncol, int DMnd,
                                  Complex* Hx, int ldhx, int spin, int kpt) const {
    if (!setup_done_ || Nstates_occ_ == 0) return;

    const int Nd = DMnd;
    const int Nocc = Nstates_occ_;
    int Nkpts = kpoints_ ? kpoints_->Nkpts() : 1;
    // Adjust kpt index for local storage
    int kpt_loc = kpt - kpt_start_;
    if (kpt_loc < 0 || kpt_loc >= static_cast<int>(Xi_kpt_.size() / std::max(Nspin_, 1))) return;
    const Complex* Xi = Xi_kpt_[spin * Nkpts + kpt_loc].data();

    // Y = Xi^H * X (alpha=1.0, same derivation as gamma)
    std::vector<Complex> Y(Nocc * ncol, Complex(0.0));
    {
        char transA = 'C', transB = 'N';
        Complex alpha_blas(1.0, 0.0);
        Complex beta_blas(0.0, 0.0);
        zgemm_(&transA, &transB, &Nocc, &ncol, &Nd,
               &alpha_blas, Xi, &Nd, X, &ldx,
               &beta_blas, Y.data(), &Nocc);
    }

    // Hx -= exx_frac * Xi * Y  (Xi_L = Xi_S, no 1/dV needed)
    {
        char transA = 'N', transB = 'N';
        Complex alpha_blas(-exx_frac_, 0.0);
        Complex beta_blas(1.0, 0.0);
        zgemm_(&transA, &transB, &Nd, &ncol, &Nocc,
               &alpha_blas, Xi, &Nd, Y.data(), &Nocc,
               &beta_blas, Hx, &ldhx);
    }
}

// ---------------------------------------------------------------------------
// Compute exact exchange energy
// ---------------------------------------------------------------------------
double ExactExchange::compute_energy(const Wavefunction& wfn) {
    Eexx_ = 0.0;

    if (is_gamma_) {
        // ACE energy: Eexx = -exx_frac / Nspin * sum_n occ_n * |psi^T * Xi_ace|^2
        // Use raw psi (lattice orbitals), matching SPARC's ACE energy convention
        for (int s = 0; s < wfn.Nspin(); s++) {
            // Gather psi (no sqrt(dV) scaling — Xi_ace already absorbs the normalization)
            std::vector<double> psi_full(Nd_d_ * Nstates_);
            {
                int ld = wfn.psi(s, 0).ld();
                const double* src = wfn.psi(s, 0).data();
                for (int j = 0; j < Nband_local_; ++j)
                    std::memcpy(psi_full.data() + (band_start_ + j) * Nd_d_,
                                src + j * ld, Nd_d_ * sizeof(double));
            }
            gather_bandcomm(psi_full.data(), Nd_d_, Nstates_);

            const double* occ = wfn.occupations(s, 0).data();
            const double* Xi = Xi_[s].data();

            // Y = sqrt(dV) * psi^T * Xi
            // SPARC: Y_S = psi_S^T * Xi_S. With psi_S = sqrt(dV)*psi_L and Xi_L = Xi_S:
            //   Y_S = sqrt(dV) * psi_L^T * Xi_L, so alpha = sqrt(dV).
            int Nocc = Nstates_occ_;
            std::vector<double> Y(Nstates_ * Nocc, 0.0);
            {
                char transA = 'T', transB = 'N';
                double alpha_blas = std::sqrt(dV_);
                double beta_blas = 0.0;
                dgemm_(&transA, &transB, &Nstates_, &Nocc, &Nd_d_,
                       &alpha_blas, psi_full.data(), &Nd_d_, Xi, &Nd_d_,
                       &beta_blas, Y.data(), &Nstates_);
            }

            for (int n = 0; n < Nstates_; n++) {
                if (occ[n] < OCC_THRESHOLD) continue;
                double sum = 0.0;
                for (int j = 0; j < Nocc; j++) {
                    double yval = Y[n + j * Nstates_];
                    sum += yval * yval;
                }
                Eexx_ += occ[n] * sum;
            }
        }
        // Divide by Nspin, multiply by -exx_frac (matching SPARC line 721-722)
        Eexx_ /= Nspin_;
        Eexx_ *= -exx_frac_;
    } else {
        // K-point ACE energy
        // SPARC uses kptWts_loc[k] / Nkpts where Nkpts = Kx*Ky*Kz (full grid count)
        // and kptWts_loc are raw symmetry weights (sum = Nkpts_full). Must divide by Nkpts_full.
        auto sym_wts = kpoints_ ? kpoints_->weights() : std::vector<double>{1.0};
        int Nkpts_full = kpoints_ ? kpoints_->Nkpts_full() : 1;

        for (int s = 0; s < wfn.Nspin(); s++) {
            for (int kpt_loc = 0; kpt_loc < wfn.Nkpts(); kpt_loc++) {
                int kpt_glob = kpt_start_ + kpt_loc;
                double wk = sym_wts[kpt_glob] / Nkpts_full;

                // Gather psi_kpt (no sqrt(dV) scaling — matching SPARC ACE energy)
                std::vector<Complex> psi_full(Nd_d_ * Nstates_);
                {
                    int ld = wfn.psi_kpt(s, kpt_loc).ld();
                    const Complex* src = wfn.psi_kpt(s, kpt_loc).data();
                    for (int j = 0; j < Nband_local_; ++j)
                        std::memcpy(psi_full.data() + (band_start_ + j) * Nd_d_,
                                    src + j * ld, Nd_d_ * sizeof(Complex));
                }
                gather_bandcomm_kpt(psi_full.data(), Nd_d_, Nstates_);

                const double* occ = wfn.occupations(s, kpt_loc).data();
                const Complex* Xi = Xi_kpt_[s * wfn.Nkpts() + kpt_loc].data();

                // Y = sqrt(dV) * Xi^H * psi (Xi is SPARC convention, psi is LYNX)
                int Nocc = Nstates_occ_;
                std::vector<Complex> Y(Nocc * Nstates_, Complex(0.0));
                {
                    char transA = 'C', transB = 'N';
                    Complex alpha_blas(std::sqrt(dV_), 0.0);
                    Complex beta_blas(0.0, 0.0);
                    zgemm_(&transA, &transB, &Nocc, &Nstates_, &Nd_d_,
                           &alpha_blas, Xi, &Nd_d_, psi_full.data(), &Nd_d_,
                           &beta_blas, Y.data(), &Nocc);
                }

                for (int n = 0; n < Nstates_; n++) {
                    if (occ[n] < OCC_THRESHOLD) continue;
                    double sum = 0.0;
                    for (int j = 0; j < Nocc; j++) {
                        double r = Y[j + n * Nocc].real();
                        double im = Y[j + n * Nocc].imag();
                        sum += r * r + im * im;
                    }
                    Eexx_ += wk * occ[n] * sum;
                }
            }
        }
        // Divide by Nspin, multiply by -exx_frac (matching SPARC)
        Eexx_ /= Nspin_;
        Eexx_ *= -exx_frac_;

        // Allreduce across kpt_bridge and spin_bridge
        if (kpt_bridge_ && !kpt_bridge_->is_null() && kpt_bridge_->size() > 1) {
            MPI_Allreduce(MPI_IN_PLACE, &Eexx_, 1, MPI_DOUBLE, MPI_SUM, kpt_bridge_->comm());
        }
        if (spin_bridge_ && !spin_bridge_->is_null() && spin_bridge_->size() > 1) {
            MPI_Allreduce(MPI_IN_PLACE, &Eexx_, 1, MPI_DOUBLE, MPI_SUM, spin_bridge_->comm());
        }
    }

    return Eexx_;
}

// ---------------------------------------------------------------------------
// Compute EXX stress tensor (k-point case)
// Matching SPARC's Calculate_exact_exchange_stress_kpt + assembly
// ---------------------------------------------------------------------------
std::array<double, 6> ExactExchange::compute_stress(const Wavefunction& wfn) {
    assert(ctx_ && "Must call setup(LynxContext&, ...) before compute_stress(wfn)");
    return compute_stress(wfn, ctx_->gradient(), ctx_->halo(), ctx_->domain());
}

std::array<double, 6> ExactExchange::compute_stress(
    const Wavefunction& wfn,
    const Gradient& gradient,
    const HaloExchange& halo,
    const Domain& domain)
{
    std::array<double, 6> stress_exx = {};
    double stress_exx_sph = 0.0;  // spherical truncation correction (component [6])

    if (!setup_done_) return stress_exx;

    const int Nd = Nd_d_;
    const int Ns = Nstates_;
    const int Nocc = Nstates_occ_;
    int mflag = params_.exx_div_flag;

    if (is_gamma_) {
        // Gamma-point EXX stress: real wavefunctions
        int Nd_ex = halo.nd_ex();
        for (int s = 0; s < wfn.Nspin(); s++) {
            // Gather full real psi across band communicator
            std::vector<double> psi_full(Nd * Ns, 0.0);
            {
                int ld = wfn.psi(s, 0).ld();
                const double* src = wfn.psi(s, 0).data();
                for (int j = 0; j < Nband_local_; ++j)
                    std::memcpy(psi_full.data() + (band_start_ + j) * Nd,
                                src + j * ld, Nd * sizeof(double));
            }
            gather_bandcomm(psi_full.data(), Nd, Ns);

            const double* occ = wfn.occupations(s, 0).data();

            std::vector<double> rhs(Nd), phi_stress(Nd);
            std::vector<double> rhs_ex(Nd_ex), phi_ex(Nd_ex);
            std::vector<double> Drhs(Nd), Dphi(Nd);

            for (int j = 0; j < Nocc; j++) {
                if (occ[j] < OCC_THRESHOLD) continue;
                const double* psi_j = psi_full.data() + j * Nd;

                for (int i = 0; i < Nocc; i++) {
                    if (occ[i] < OCC_THRESHOLD) continue;
                    const double* psi_i = psi_full.data() + i * Nd;
                    double occ_ij = occ[i] * occ[j];

                    // rhs = psi_j * psi_i
                    for (int m = 0; m < Nd; m++)
                        rhs[m] = psi_j[m] * psi_i[m];

                    // Solve Poisson with stress constant
                    poisson_.solve_batch_stress(rhs.data(), 1, phi_stress.data(), 1);

                    // Halo exchange rhs and phi for gradient computation
                    halo.execute(rhs.data(), rhs_ex.data(), 1);
                    halo.execute(phi_stress.data(), phi_ex.data(), 1);

                    // ∂x rhs
                    gradient.apply(rhs_ex.data(), Drhs.data(), 0, 1);
                    // ∂x phi
                    gradient.apply(phi_ex.data(), Dphi.data(), 0, 1);

                    // (0,0) = xx
                    double sum = 0.0;
                    for (int m = 0; m < Nd; m++)
                        sum += Drhs[m] * Dphi[m];
                    stress_exx[0] += occ_ij * sum;

                    // Save ∂x rhs for xy, xz
                    std::vector<double> Drhs_x(Drhs);

                    // ∂y phi
                    gradient.apply(phi_ex.data(), Dphi.data(), 1, 1);

                    // (0,1) = xy
                    sum = 0.0;
                    for (int m = 0; m < Nd; m++)
                        sum += Drhs_x[m] * Dphi[m];
                    stress_exx[1] += occ_ij * sum;

                    // ∂z phi
                    gradient.apply(phi_ex.data(), Dphi.data(), 2, 1);
                    std::vector<double> Dphi_z(Dphi);

                    // (0,2) = xz
                    sum = 0.0;
                    for (int m = 0; m < Nd; m++)
                        sum += Drhs_x[m] * Dphi_z[m];
                    stress_exx[2] += occ_ij * sum;

                    // ∂y rhs
                    gradient.apply(rhs_ex.data(), Drhs.data(), 1, 1);

                    // ∂y phi (recompute)
                    gradient.apply(phi_ex.data(), Dphi.data(), 1, 1);

                    // (1,1) = yy
                    sum = 0.0;
                    for (int m = 0; m < Nd; m++)
                        sum += Drhs[m] * Dphi[m];
                    stress_exx[3] += occ_ij * sum;

                    // (1,2) = yz
                    sum = 0.0;
                    for (int m = 0; m < Nd; m++)
                        sum += Drhs[m] * Dphi_z[m];
                    stress_exx[4] += occ_ij * sum;

                    // ∂z rhs
                    gradient.apply(rhs_ex.data(), Drhs.data(), 2, 1);

                    // (2,2) = zz
                    sum = 0.0;
                    for (int m = 0; m < Nd; m++)
                        sum += Drhs[m] * Dphi_z[m];
                    stress_exx[5] += occ_ij * sum;

                    // Spherical truncation correction
                    if (mflag == 0) {
                        std::vector<double> phi_stress2(Nd);
                        poisson_.solve_batch_stress(rhs.data(), 1, phi_stress2.data(), 2);
                        sum = 0.0;
                        for (int m = 0; m < Nd; m++)
                            sum += rhs[m] * phi_stress2[m];
                        stress_exx_sph += occ_ij * sum;
                    }
                }
            }
        } // end spin loop

        // Allreduce across communicators (bandcomm not needed: all pairs computed locally)
        if (spin_bridge_ && !spin_bridge_->is_null() && spin_bridge_->size() > 1) {
            MPI_Allreduce(MPI_IN_PLACE, stress_exx.data(), 6, MPI_DOUBLE, MPI_SUM, spin_bridge_->comm());
            MPI_Allreduce(MPI_IN_PLACE, &stress_exx_sph, 1, MPI_DOUBLE, MPI_SUM, spin_bridge_->comm());
        }

        // Non-orthogonal cell: transform from lattice to Cartesian coordinates
        bool is_orth = lattice_->is_orthogonal();
        if (!is_orth) {
            Mat3 gradT = lattice_->grad_T();
            double cg[9];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    cg[3*i+j] = gradT(j, i);

            double S_nc[6] = {stress_exx[0], stress_exx[1], stress_exx[2],
                              stress_exx[3], stress_exx[4], stress_exx[5]};
            double Sfull[3][3] = {
                {S_nc[0], S_nc[1], S_nc[2]},
                {S_nc[1], S_nc[3], S_nc[4]},
                {S_nc[2], S_nc[4], S_nc[5]}
            };
            double Scart[3][3] = {};
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    for (int a = 0; a < 3; a++)
                        for (int b = 0; b < 3; b++)
                            Scart[i][j] += cg[3*i+a] * Sfull[a][b] * cg[3*j+b];
            stress_exx[0] = Scart[0][0]; stress_exx[1] = Scart[0][1]; stress_exx[2] = Scart[0][2];
            stress_exx[3] = Scart[1][1]; stress_exx[4] = Scart[1][2]; stress_exx[5] = Scart[2][2];
        }

        // Add spherical truncation correction to diagonal
        if (mflag == 0) {
            stress_exx[0] += stress_exx_sph;
            stress_exx[3] += stress_exx_sph;
            stress_exx[5] += stress_exx_sph;
        }

        // Scale: multiply by (-exx_frac * dV / Nspin)
        // LYNX psi are dV-normalized (sum |psi|^2 * dV = 1, so sum |psi|^2 = 1/dV)
        // rhs = conj(psi_q)*psi_k is 1/dV times SPARC's, phi is 1/dV times SPARC's
        // gradient sums are 1/dV^2 times SPARC's. SPARC uses -exx_frac/dV/Nspin,
        // so we need (-exx_frac/dV/Nspin)*dV^2 = -exx_frac*dV/Nspin
        double scale = -exx_frac_ * dV_ / Nspin_;
        for (int i = 0; i < 6; i++)
            stress_exx[i] *= scale;

        // Final assembly: diagonal gets -2*Eexx correction + divergence correction
        stress_exx[0] = 2.0 * stress_exx[0] - 2.0 * Eexx_ + (mflag == 1 ? Eexx_ / 2.0 : 0.0);
        stress_exx[1] = 2.0 * stress_exx[1];
        stress_exx[2] = 2.0 * stress_exx[2];
        stress_exx[3] = 2.0 * stress_exx[3] - 2.0 * Eexx_ + (mflag == 1 ? Eexx_ / 2.0 : 0.0);
        stress_exx[4] = 2.0 * stress_exx[4];
        stress_exx[5] = 2.0 * stress_exx[5] - 2.0 * Eexx_ + (mflag == 1 ? Eexx_ / 2.0 : 0.0);

        // Divide by cell measure
        const auto& lat = grid_->lattice();
        Vec3 L = lat.lengths();
        double Jacbdet = lat.jacobian() / (L.x * L.y * L.z);
        double cell_measure = Jacbdet;
        if (grid_->bcx() == BCType::Periodic) cell_measure *= L.x;
        if (grid_->bcy() == BCType::Periodic) cell_measure *= L.y;
        if (grid_->bcz() == BCType::Periodic) cell_measure *= L.z;

        for (int i = 0; i < 6; i++)
            stress_exx[i] /= cell_measure;

        return stress_exx;
    }

    // K-point case below
    int Nkpts_loc = wfn.Nkpts();
    int Nkpts_sym = kpoints_->Nkpts();
    int Nkpts_hf = kpoints_->Nkpts_hf();
    int Nkpts_full = kpoints_->Nkpts_full();
    const auto& kpthf_ind = kpoints_->kpthf_ind();
    const auto& kpthf_pn = kpoints_->kpthf_pn();
    const auto& kpts_cart = kpoints_->kpts_cart();
    const auto& kpts_hf_cart = kpoints_->kpts_hf_cart();
    auto sym_wts = kpoints_->weights();
    double kptWts_hf = 1.0 / Nkpts_full;

    // Gather psi and occ from all k-points (same as solve_for_Xi_kpt)
    std::vector<std::vector<Complex>> psi_all(Nkpts_sym);
    std::vector<std::vector<double>> occ_all(Nkpts_sym);

    for (int s = 0; s < wfn.Nspin(); s++) {
        // Gather local psi
        for (int kpt_loc = 0; kpt_loc < Nkpts_loc; kpt_loc++) {
            int kpt_glob = kpt_start_ + kpt_loc;
            psi_all[kpt_glob].resize(Nd * Ns, Complex(0.0));
            {
                int ld = wfn.psi_kpt(s, kpt_loc).ld();
                const Complex* src = wfn.psi_kpt(s, kpt_loc).data();
                for (int j = 0; j < Nband_local_; ++j)
                    std::memcpy(psi_all[kpt_glob].data() + (band_start_ + j) * Nd,
                                src + j * ld, Nd * sizeof(Complex));
            }
            gather_bandcomm_kpt(psi_all[kpt_glob].data(), Nd, Ns);

            occ_all[kpt_glob].resize(Ns);
            std::memcpy(occ_all[kpt_glob].data(), wfn.occupations(s, kpt_loc).data(),
                        Ns * sizeof(double));
        }

        // Broadcast across kpt_bridge
        if (npkpt_ > 1 && kpt_bridge_ && !kpt_bridge_->is_null() && kpt_bridge_->size() > 1) {
            int bridge_size = kpt_bridge_->size();
            for (int r = 0; r < bridge_size; r++) {
                int r_nkpts = Parallelization::block_size(Nkpts_sym, npkpt_, r);
                int r_kstart = Parallelization::block_start(Nkpts_sym, npkpt_, r);
                for (int k = 0; k < r_nkpts; k++) {
                    int kpt_glob = r_kstart + k;
                    psi_all[kpt_glob].resize(Nd * Ns, Complex(0.0));
                    occ_all[kpt_glob].resize(Ns, 0.0);
                    MPI_Bcast(psi_all[kpt_glob].data(), Nd * Ns, MPI_DOUBLE_COMPLEX,
                              r, kpt_bridge_->comm());
                    MPI_Bcast(occ_all[kpt_glob].data(), Ns, MPI_DOUBLE,
                              r, kpt_bridge_->comm());
                }
            }
        }

        // Halo-extended buffer for gradient computation
        int Nd_ex = halo.nd_ex();
        Vec3 cell_lengths = lattice_->lengths();

        std::vector<Complex> rhs(Nd);
        std::vector<Complex> phi_stress(Nd);
        std::vector<Complex> rhs_ex(Nd_ex), phi_ex(Nd_ex);
        std::vector<Complex> Drhs(Nd), Dphi(Nd);
        // Loop over all local k-points and all q-points
        for (int kpt_loc = 0; kpt_loc < Nkpts_loc; kpt_loc++) {
            int kpt_k = kpt_start_ + kpt_loc;
            double wk = sym_wts[kpt_k] / Nkpts_full;

            const Complex* psi_k = psi_all[kpt_k].data();
            const double* occ_k = occ_all[kpt_k].data();

            for (int q_hf = 0; q_hf < Nkpts_hf; q_hf++) {
                int q_sym = kpthf_ind[q_hf];

                const Complex* psi_q_raw = psi_all[q_sym].data();
                const double* occ_q = occ_all[q_sym].data();

                // Handle time-reversal
                std::vector<Complex> psi_q_buf;
                const Complex* psi_q_ptr;
                if (kpthf_pn[q_hf] == 0) {
                    psi_q_buf.resize(Nd * Ns);
                    for (int m = 0; m < Nd * Ns; m++)
                        psi_q_buf[m] = std::conj(psi_q_raw[m]);
                    psi_q_ptr = psi_q_buf.data();
                } else {
                    psi_q_ptr = psi_q_raw;
                }

                // k-q vector for Bloch-periodic halo exchange
                Vec3 dk_vec(
                    kpts_cart[kpt_k].x - kpts_hf_cart[q_hf].x,
                    kpts_cart[kpt_k].y - kpts_hf_cart[q_hf].y,
                    kpts_cart[kpt_k].z - kpts_hf_cart[q_hf].z
                );

                for (int j = 0; j < Nocc; j++) {
                    if (occ_q[j] < OCC_THRESHOLD) continue;
                    const Complex* psi_qj = psi_q_ptr + j * Nd;

                    for (int i = 0; i < Nocc; i++) {
                        if (occ_k[i] < OCC_THRESHOLD) continue;
                        const Complex* psi_ki = psi_k + i * Nd;

                        double occ_ij = occ_k[i] * occ_q[j];

                        // rhs = conj(psi_q[j]) * psi_k[i]
                        for (int m = 0; m < Nd; m++)
                            rhs[m] = std::conj(psi_qj[m]) * psi_ki[m];

                        // Solve Poisson with stress constant
                        poisson_.solve_batch_kpt_stress(rhs.data(), 1, phi_stress.data(), kpt_k, q_hf, 1);

                        // Poisson diagnostic — removed

                        // Halo exchange rhs and phi with (k-q) Bloch phases
                        halo.execute_kpt(rhs.data(), rhs_ex.data(), 1, dk_vec, cell_lengths);
                        halo.execute_kpt(phi_stress.data(), phi_ex.data(), 1, dk_vec, cell_lengths);

                        // Direction 0 (x): grad of rhs
                        // Note: halo exchange already applies Bloch phases to ghost cells,
                        // so the FD gradient gives the full derivative including the i*dk
                        // contribution. No explicit i*dk term needed (matches SPARC c=0.0).
                        gradient.apply(rhs_ex.data(), Drhs.data(), 0, 1);

                        // Direction 0 (x): grad of phi_stress
                        gradient.apply(phi_ex.data(), Dphi.data(), 0, 1);

                        // Component (0,0) = xx
                        double sum = 0.0;
                        for (int m = 0; m < Nd; m++)
                            sum += (std::conj(Drhs[m]) * Dphi[m]).real();

                        stress_exx[0] += kptWts_hf * wk * occ_ij * sum;

                        // Store ∂x rhs for xy, xz components
                        std::vector<Complex> Drhs_x(Drhs);

                        // Direction 1 (y): grad of phi_stress
                        gradient.apply(phi_ex.data(), Dphi.data(), 1, 1);

                        // Component (0,1) = xy
                        sum = 0.0;
                        for (int m = 0; m < Nd; m++)
                            sum += (std::conj(Drhs_x[m]) * Dphi[m]).real();
                        stress_exx[1] += kptWts_hf * wk * occ_ij * sum;

                        // Direction 2 (z): grad of phi_stress
                        gradient.apply(phi_ex.data(), Dphi.data(), 2, 1);

                        // Store ∂z phi for later
                        std::vector<Complex> Dphi_z(Dphi);

                        // Component (0,2) = xz
                        sum = 0.0;
                        for (int m = 0; m < Nd; m++)
                            sum += (std::conj(Drhs_x[m]) * Dphi_z[m]).real();
                        stress_exx[2] += kptWts_hf * wk * occ_ij * sum;

                        // Direction 1 (y): grad of rhs (rhs_ex already computed)
                        gradient.apply(rhs_ex.data(), Drhs.data(), 1, 1);

                        // Recompute ∂y phi (phi_ex already computed)
                        gradient.apply(phi_ex.data(), Dphi.data(), 1, 1);

                        // Component (1,1) = yy
                        sum = 0.0;
                        for (int m = 0; m < Nd; m++)
                            sum += (std::conj(Drhs[m]) * Dphi[m]).real();
                        stress_exx[3] += kptWts_hf * wk * occ_ij * sum;

                        // Component (1,2) = yz
                        sum = 0.0;
                        for (int m = 0; m < Nd; m++)
                            sum += (std::conj(Drhs[m]) * Dphi_z[m]).real();
                        stress_exx[4] += kptWts_hf * wk * occ_ij * sum;

                        // Direction 2 (z): grad of rhs (rhs_ex already computed)
                        gradient.apply(rhs_ex.data(), Drhs.data(), 2, 1);

                        // Component (2,2) = zz
                        sum = 0.0;
                        for (int m = 0; m < Nd; m++)
                            sum += (std::conj(Drhs[m]) * Dphi_z[m]).real();
                        stress_exx[5] += kptWts_hf * wk * occ_ij * sum;

                        // Spherical truncation correction
                        if (mflag == 0) {
                            std::vector<Complex> phi_stress2(Nd);
                            poisson_.solve_batch_kpt_stress(rhs.data(), 1, phi_stress2.data(), kpt_k, q_hf, 2);
                            sum = 0.0;
                            for (int m = 0; m < Nd; m++)
                                sum += (std::conj(rhs[m]) * phi_stress2[m]).real();
                            stress_exx_sph += kptWts_hf * wk * occ_ij * sum;
                        }
                    }
                }
            }
        }
    } // end spin loop

    // Allreduce across communicators
    if (kpt_bridge_ && !kpt_bridge_->is_null() && kpt_bridge_->size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, stress_exx.data(), 6, MPI_DOUBLE, MPI_SUM, kpt_bridge_->comm());
        MPI_Allreduce(MPI_IN_PLACE, &stress_exx_sph, 1, MPI_DOUBLE, MPI_SUM, kpt_bridge_->comm());
    }
    if (spin_bridge_ && !spin_bridge_->is_null() && spin_bridge_->size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, stress_exx.data(), 6, MPI_DOUBLE, MPI_SUM, spin_bridge_->comm());
        MPI_Allreduce(MPI_IN_PLACE, &stress_exx_sph, 1, MPI_DOUBLE, MPI_SUM, spin_bridge_->comm());
    }

    // Non-orthogonal cell: transform from lattice to Cartesian coordinates
    bool is_orth = lattice_->is_orthogonal();
    if (!is_orth) {
        // cg = inv(LatUVec), matching SPARC's c_g[3*i+j] = gradT[3*j+i] = inv(LatUVec)[i][j]
        Mat3 uvec_inv = lattice_->lat_uvec_inv();
        double cg[9];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                cg[3*i+j] = uvec_inv(i, j);

        // Transform symmetric tensor from non-Cart to Cartesian
        // S_cart[i,j] = Σ_{a,b} cg[i,a] * S_nc[a,b] * cg[j,b]
        double S_nc[6] = {stress_exx[0], stress_exx[1], stress_exx[2],
                          stress_exx[3], stress_exx[4], stress_exx[5]};
        // Build full 3x3 from Voigt
        double Sfull[3][3] = {
            {S_nc[0], S_nc[1], S_nc[2]},
            {S_nc[1], S_nc[3], S_nc[4]},
            {S_nc[2], S_nc[4], S_nc[5]}
        };
        double Scart[3][3] = {};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++)
                        Scart[i][j] += cg[3*i+a] * Sfull[a][b] * cg[3*j+b];
        stress_exx[0] = Scart[0][0];
        stress_exx[1] = Scart[0][1];
        stress_exx[2] = Scart[0][2];
        stress_exx[3] = Scart[1][1];
        stress_exx[4] = Scart[1][2];
        stress_exx[5] = Scart[2][2];
    }

    // Add spherical truncation correction to diagonal
    if (mflag == 0) {
        stress_exx[0] += stress_exx_sph;
        stress_exx[3] += stress_exx_sph;
        stress_exx[5] += stress_exx_sph;
    }

    // Scale: multiply by (-exx_frac * dV / Nspin)
    // LYNX psi are dV-normalized, so gradient sums are 1/dV^2 times SPARC's.
    // SPARC uses -exx_frac/dV/Nspin; we need (-exx_frac/dV/Nspin)*dV^2 = -exx_frac*dV/Nspin
    double scale = -exx_frac_ * dV_ / Nspin_;
    for (int i = 0; i < 6; i++)
        stress_exx[i] *= scale;

    // Final assembly (matching SPARC):
    // diagonal: 2*σ - 2*Eexx + (mflag==1)*Eexx/2
    // off-diagonal: 2*σ
    stress_exx[0] = 2.0 * stress_exx[0] - 2.0 * Eexx_ + (mflag == 1 ? Eexx_ / 2.0 : 0.0);
    stress_exx[1] = 2.0 * stress_exx[1];
    stress_exx[2] = 2.0 * stress_exx[2];
    stress_exx[3] = 2.0 * stress_exx[3] - 2.0 * Eexx_ + (mflag == 1 ? Eexx_ / 2.0 : 0.0);
    stress_exx[4] = 2.0 * stress_exx[4];
    stress_exx[5] = 2.0 * stress_exx[5] - 2.0 * Eexx_ + (mflag == 1 ? Eexx_ / 2.0 : 0.0);

    // Divide by cell measure (volume for 3D periodic)
    const auto& lat = grid_->lattice();
    Vec3 L = lat.lengths();
    double Jacbdet = lat.jacobian() / (L.x * L.y * L.z);
    double cell_measure = Jacbdet;
    if (grid_->bcx() == BCType::Periodic) cell_measure *= L.x;
    if (grid_->bcy() == BCType::Periodic) cell_measure *= L.y;
    if (grid_->bcz() == BCType::Periodic) cell_measure *= L.z;

    for (int i = 0; i < 6; i++)
        stress_exx[i] /= cell_measure;

    return stress_exx;
}

} // namespace lynx
