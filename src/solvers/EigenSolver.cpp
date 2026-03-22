#include "solvers/EigenSolver.hpp"
#include "solvers/LinearSolver.hpp"
#include "parallel/Parallelization.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <mpi.h>

#ifdef USE_SCALAPACK
#include "solvers/ScaLAPACK.hpp"
#endif

// LAPACK/BLAS declarations
extern "C" {
    // Real
    void dsyev_(const char* jobz, const char* uplo, const int* n,
                double* a, const int* lda, double* w,
                double* work, const int* lwork, int* info);
    void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
    void dtrsm_(const char* side, const char* uplo, const char* transa, const char* diag,
                const int* m, const int* n, const double* alpha,
                const double* a, const int* lda, double* b, const int* ldb);
    void dgemm_(const char* transa, const char* transb,
                const int* m, const int* n, const int* k,
                const double* alpha, const double* a, const int* lda,
                const double* b, const int* ldb,
                const double* beta, double* c, const int* ldc);
    void dsymm_(const char* side, const char* uplo,
                const int* m, const int* n,
                const double* alpha, const double* a, const int* lda,
                const double* b, const int* ldb,
                const double* beta, double* c, const int* ldc);
    void dsterf_(const int* n, double* d, double* e, int* info);

    // Complex
    void zgemm_(const char* transa, const char* transb,
                const int* m, const int* n, const int* k,
                const void* alpha, const void* a, const int* lda,
                const void* b, const int* ldb,
                const void* beta, void* c, const int* ldc);
    void zpotrf_(const char* uplo, const int* n, void* a, const int* lda, int* info);
    void ztrsm_(const char* side, const char* uplo, const char* transa, const char* diag,
                const int* m, const int* n, const void* alpha,
                const void* a, const int* lda, void* b, const int* ldb);
    void zheev_(const char* jobz, const char* uplo, const int* n,
                void* a, const int* lda, double* w,
                void* work, const int* lwork, double* rwork, int* info);
    void zhegvd_(const int* itype, const char* jobz, const char* uplo, const int* n,
                 void* a, const int* lda, void* b, const int* ldb, double* w,
                 void* work, const int* lwork, double* rwork, const int* lrwork,
                 int* iwork, const int* liwork, int* info);
}

namespace lynx {

EigenSolver::~EigenSolver() {
#ifdef USE_SCALAPACK
    cleanup_blacs();
#endif
}

void EigenSolver::setup(const Hamiltonian& H,
                         const HaloExchange& halo,
                         const Domain& domain,
                         const MPIComm& bandcomm,
                         int Nband_global) {
    H_ = &H;
    halo_ = &halo;
    domain_ = &domain;
    bandcomm_ = &bandcomm;

    npband_ = bandcomm.is_null() ? 1 : bandcomm.size();
    band_rank_ = bandcomm.is_null() ? 0 : bandcomm.rank();
    Nband_global_ = (Nband_global > 0) ? Nband_global : 0;

#ifdef USE_SCALAPACK
    if (npband_ > 1 && Nband_global_ > 0) {
        setup_blacs();
    }
#endif
}

#ifdef USE_SCALAPACK
void EigenSolver::setup_blacs() {
    if (blacs_setup_) return;
    // Currently using Allgather + redundant serial LAPACK approach
    // (no ScaLAPACK distributed matrix operations needed).
    // BLACS context setup deferred until true pdgemm/pdsyev needed.
    blacs_setup_ = true;
}

void EigenSolver::cleanup_blacs() {
    blacs_setup_ = false;
}

void EigenSolver::orthogonalize_scalapack(double* X, int Nd_d, int Nband_loc, double dV) {
    // Cholesky QR with ScaLAPACK:
    // 1. S = X^T * X * dV (distributed: each proc has Nd_d x Nband_loc block)
    //    Use local dgemm then Allreduce into distributed matrix
    // 2. pdpotrf: Cholesky S = R^T * R
    // 3. pdtrsm: X <- X * R^{-1}

    int N = Nband_global_;

    // Strategy: Allgather X columns to get full X, then compute S and factorize redundantly.
    // This is communication-optimal for moderate N (typical DFT band counts).

    // Allgather all band columns to get full X (Nd_d x Nband_global)
    std::vector<double> X_full(Nd_d * N);
    {
        std::vector<int> recvcounts(npband_), displs(npband_);
        for (int p = 0; p < npband_; ++p) {
            int nb_p = Parallelization::block_size(N, npband_, p);
            int bs_p = Parallelization::block_start(N, npband_, p);
            recvcounts[p] = Nd_d * nb_p;
            displs[p] = Nd_d * bs_p;
        }
        MPI_Allgatherv(X, Nd_d * Nband_loc, MPI_DOUBLE,
                        X_full.data(), recvcounts.data(), displs.data(),
                        MPI_DOUBLE, bandcomm_->comm());
    }

    // Compute full S = X_full^T * X_full * dV (N x N)
    std::vector<double> S_full(N * N, 0.0);
    {
        char transT = 'T', transN = 'N';
        double alpha = dV, beta = 0.0;
        dgemm_(&transT, &transN, &N, &N, &Nd_d,
               &alpha, X_full.data(), &Nd_d, X_full.data(), &Nd_d,
               &beta, S_full.data(), &N);
    }

    // Cholesky on full S (all procs do the same — redundant but simple and correct)
    {
        char uplo = 'U';
        int info;
        dpotrf_(&uplo, &N, S_full.data(), &N, &info);
        if (info != 0)
            throw std::runtime_error("Cholesky factorization failed in band-parallel orthogonalize");
    }

    // X_full <- X_full * R^{-1}
    {
        char side = 'R', uplo = 'U', transN = 'N', diag = 'N';
        double one = 1.0;
        dtrsm_(&side, &uplo, &transN, &diag, &Nd_d, &N, &one,
               S_full.data(), &N, X_full.data(), &Nd_d);
    }

    // Extract my local columns back
    int band_start = Parallelization::block_start(N, npband_, band_rank_);
    std::memcpy(X, X_full.data() + band_start * Nd_d,
                Nd_d * Nband_loc * sizeof(double));
}

void EigenSolver::project_hamiltonian_scalapack(const double* X, const double* Veff,
                                                 double* eigvals, int Nd_d, int Nband_loc, double dV) {
    // 1. HX = H * X (each proc applies H to its local bands)
    std::vector<double> HX(Nd_d * Nband_loc);
    int nd_ex = halo_->nd_ex();
    std::vector<double> x_ex(nd_ex * Nband_loc);
    halo_->execute(X, x_ex.data(), Nband_loc);
    H_->apply(X, Veff, HX.data(), Nband_loc);

    int N = Nband_global_;

    // 2. Allgather X and HX to get full matrices
    std::vector<double> X_full(Nd_d * N), HX_full(Nd_d * N);
    {
        std::vector<int> recvcounts(npband_), displs(npband_);
        for (int p = 0; p < npband_; ++p) {
            int nb_p = Parallelization::block_size(N, npband_, p);
            int bs_p = Parallelization::block_start(N, npband_, p);
            recvcounts[p] = Nd_d * nb_p;
            displs[p] = Nd_d * bs_p;
        }
        MPI_Allgatherv(X, Nd_d * Nband_loc, MPI_DOUBLE,
                        X_full.data(), recvcounts.data(), displs.data(),
                        MPI_DOUBLE, bandcomm_->comm());
        MPI_Allgatherv(HX.data(), Nd_d * Nband_loc, MPI_DOUBLE,
                        HX_full.data(), recvcounts.data(), displs.data(),
                        MPI_DOUBLE, bandcomm_->comm());
    }

    // 3. Hs = X^T * HX * dV (all procs compute redundantly)
    std::vector<double> Hs(N * N, 0.0);
    {
        char transT = 'T', transN = 'N';
        double alpha = dV, beta = 0.0;
        dgemm_(&transT, &transN, &N, &N, &Nd_d,
               &alpha, X_full.data(), &Nd_d, HX_full.data(), &Nd_d,
               &beta, Hs.data(), &N);
    }

    // 4. Symmetrize
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double avg = 0.5 * (Hs[i + j * N] + Hs[j + i * N]);
            Hs[i + j * N] = avg;
            Hs[j + i * N] = avg;
        }
    }

    // 5. Diagonalize (redundant on all procs — use ScaLAPACK for very large N)
    {
        char jobz = 'V', uplo = 'U';
        int lwork = -1, info;
        double work_query;
        dsyev_(&jobz, &uplo, &N, Hs.data(), &N, eigvals, &work_query, &lwork, &info);
        lwork = static_cast<int>(work_query);
        std::vector<double> work(lwork);
        dsyev_(&jobz, &uplo, &N, Hs.data(), &N, eigvals, work.data(), &lwork, &info);
        if (info != 0)
            throw std::runtime_error("dsyev failed in band-parallel project_hamiltonian");
    }

    // Store eigenvectors for rotation step
    Q_dist_ = std::move(Hs);
}

void EigenSolver::rotate_orbitals_scalapack(double* X, int Nd_d, int Nband_loc) {
    int N = Nband_global_;

    // Allgather X columns
    std::vector<double> X_full(Nd_d * N);
    {
        std::vector<int> recvcounts(npband_), displs(npband_);
        for (int p = 0; p < npband_; ++p) {
            int nb_p = Parallelization::block_size(N, npband_, p);
            int bs_p = Parallelization::block_start(N, npband_, p);
            recvcounts[p] = Nd_d * nb_p;
            displs[p] = Nd_d * bs_p;
        }
        MPI_Allgatherv(X, Nd_d * Nband_loc, MPI_DOUBLE,
                        X_full.data(), recvcounts.data(), displs.data(),
                        MPI_DOUBLE, bandcomm_->comm());
    }

    // X_new = X_full * Q (full rotation, then extract local columns)
    // Q_dist_ has the eigenvectors from diag step (N x N column-major)
    std::vector<double> X_new(Nd_d * N);
    {
        char transN = 'N';
        double one = 1.0, zero = 0.0;
        dgemm_(&transN, &transN, &Nd_d, &N, &N,
               &one, X_full.data(), &Nd_d, Q_dist_.data(), &N,
               &zero, X_new.data(), &Nd_d);
    }

    // Extract my local columns
    int band_start = Parallelization::block_start(N, npband_, band_rank_);
    std::memcpy(X, X_new.data() + band_start * Nd_d,
                Nd_d * Nband_loc * sizeof(double));
}

void EigenSolver::orthogonalize_kpt_scalapack(Complex* X, int Nd_d, int Nband_loc, double dV) {
    int N = Nband_global_;

    // Allgather complex X columns
    std::vector<Complex> X_full(Nd_d * N);
    {
        std::vector<int> recvcounts(npband_), displs(npband_);
        for (int p = 0; p < npband_; ++p) {
            int nb_p = Parallelization::block_size(N, npband_, p);
            int bs_p = Parallelization::block_start(N, npband_, p);
            recvcounts[p] = Nd_d * nb_p;
            displs[p] = Nd_d * bs_p;
        }
        MPI_Allgatherv(X, Nd_d * Nband_loc, MPI_C_DOUBLE_COMPLEX,
                        X_full.data(), recvcounts.data(), displs.data(),
                        MPI_C_DOUBLE_COMPLEX, bandcomm_->comm());
    }

    // S = X^H * X * dV
    std::vector<Complex> S(N * N, Complex(0.0));
    {
        char transC = 'C', transN = 'N';
        Complex alpha_z(dV, 0.0), beta_z(0.0, 0.0);
        zgemm_(&transC, &transN, &N, &N, &Nd_d,
               &alpha_z, X_full.data(), &Nd_d, X_full.data(), &Nd_d,
               &beta_z, S.data(), &N);
    }

    // Cholesky
    {
        char uplo = 'U';
        int info;
        zpotrf_(&uplo, &N, S.data(), &N, &info);
        if (info != 0)
            throw std::runtime_error("zpotrf failed in band-parallel orthogonalize_kpt");
    }

    // X <- X * R^{-1}
    {
        char side = 'R', uplo = 'U', transN = 'N', diag = 'N';
        Complex one_z(1.0, 0.0);
        ztrsm_(&side, &uplo, &transN, &diag, &Nd_d, &N, &one_z,
               S.data(), &N, X_full.data(), &Nd_d);
    }

    // Extract local columns
    int band_start = Parallelization::block_start(N, npband_, band_rank_);
    std::memcpy(X, X_full.data() + band_start * Nd_d,
                Nd_d * Nband_loc * sizeof(Complex));
}

void EigenSolver::project_hamiltonian_kpt_scalapack(const Complex* X, const double* Veff,
                                                     double* eigvals, int Nd_d, int Nband_loc, double dV,
                                                     const Vec3& kpt_cart, const Vec3& cell_lengths) {
    int N = Nband_global_;

    // Apply H to local bands
    std::vector<Complex> HX(Nd_d * Nband_loc);
    H_->apply_kpt(X, Veff, HX.data(), Nband_loc, kpt_cart, cell_lengths);

    // Allgather X and HX
    std::vector<Complex> X_full(Nd_d * N), HX_full(Nd_d * N);
    {
        std::vector<int> recvcounts(npband_), displs(npband_);
        for (int p = 0; p < npband_; ++p) {
            int nb_p = Parallelization::block_size(N, npband_, p);
            int bs_p = Parallelization::block_start(N, npband_, p);
            recvcounts[p] = Nd_d * nb_p;
            displs[p] = Nd_d * bs_p;
        }
        MPI_Allgatherv(X, Nd_d * Nband_loc, MPI_C_DOUBLE_COMPLEX,
                        X_full.data(), recvcounts.data(), displs.data(),
                        MPI_C_DOUBLE_COMPLEX, bandcomm_->comm());
        MPI_Allgatherv(HX.data(), Nd_d * Nband_loc, MPI_C_DOUBLE_COMPLEX,
                        HX_full.data(), recvcounts.data(), displs.data(),
                        MPI_C_DOUBLE_COMPLEX, bandcomm_->comm());
    }

    // Hs = X^H * HX * dV
    std::vector<Complex> Hs(N * N, Complex(0.0));
    {
        char transC = 'C', transN = 'N';
        Complex alpha_z(dV, 0.0), beta_z(0.0, 0.0);
        zgemm_(&transC, &transN, &N, &N, &Nd_d,
               &alpha_z, X_full.data(), &Nd_d, HX_full.data(), &Nd_d,
               &beta_z, Hs.data(), &N);
    }

    // Hermitianize
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            Complex avg = 0.5 * (Hs[i + j * N] + std::conj(Hs[j + i * N]));
            Hs[i + j * N] = avg;
            Hs[j + i * N] = std::conj(avg);
        }
        Hs[i + i * N] = Complex(Hs[i + i * N].real(), 0.0);
    }

    // Diagonalize
    {
        char jobz = 'V', uplo = 'U';
        int lwork = -1, info;
        Complex work_query;
        std::vector<double> rwork(std::max(1, 3 * N - 2));
        zheev_(&jobz, &uplo, &N, Hs.data(), &N, eigvals, &work_query, &lwork, rwork.data(), &info);
        lwork = static_cast<int>(work_query.real());
        std::vector<Complex> work(lwork);
        zheev_(&jobz, &uplo, &N, Hs.data(), &N, eigvals, work.data(), &lwork, rwork.data(), &info);
        if (info != 0)
            throw std::runtime_error("zheev failed in band-parallel project_hamiltonian_kpt");
    }

    Q_dist_z_ = std::move(Hs);
}

void EigenSolver::rotate_orbitals_kpt_scalapack(Complex* X, int Nd_d, int Nband_loc) {
    int N = Nband_global_;

    std::vector<Complex> X_full(Nd_d * N);
    {
        std::vector<int> recvcounts(npband_), displs(npband_);
        for (int p = 0; p < npband_; ++p) {
            int nb_p = Parallelization::block_size(N, npband_, p);
            int bs_p = Parallelization::block_start(N, npband_, p);
            recvcounts[p] = Nd_d * nb_p;
            displs[p] = Nd_d * bs_p;
        }
        MPI_Allgatherv(X, Nd_d * Nband_loc, MPI_C_DOUBLE_COMPLEX,
                        X_full.data(), recvcounts.data(), displs.data(),
                        MPI_C_DOUBLE_COMPLEX, bandcomm_->comm());
    }

    std::vector<Complex> X_new(Nd_d * N);
    {
        char transN = 'N';
        Complex one_z(1.0, 0.0), zero_z(0.0, 0.0);
        zgemm_(&transN, &transN, &Nd_d, &N, &N,
               &one_z, X_full.data(), &Nd_d, Q_dist_z_.data(), &N,
               &zero_z, X_new.data(), &Nd_d);
    }

    int band_start = Parallelization::block_start(N, npband_, band_rank_);
    std::memcpy(X, X_new.data() + band_start * Nd_d,
                Nd_d * Nband_loc * sizeof(Complex));
}
#endif // USE_SCALAPACK

void EigenSolver::chebyshev_filter(const double* X, double* Y, const double* Veff,
                                    int Nd_d, int Nband,
                                    double lambda_cutoff, double eigval_min, double eigval_max,
                                    int degree) {
    double e = (eigval_max - lambda_cutoff) / 2.0;
    double c = (eigval_max + lambda_cutoff) / 2.0;
    double sigma_1 = e / (eigval_min - c);  // reference: sigma1 = e / (a0 - c)
    double sigma = sigma_1;
    double sigma_new;

    int nd_ex = halo_->nd_ex();

    // Work arrays
    std::vector<double> Xold(Nd_d * Nband);
    std::vector<double> Xnew(Nd_d * Nband);
    std::vector<double> HX(Nd_d * Nband);
    std::vector<double> x_ex(nd_ex * Nband);

    // Y = (H*X - c*X) * (sigma/e)
    halo_->execute(X, x_ex.data(), Nband);
    H_->apply(X, Veff, HX.data(), Nband);

    double scale = sigma / e;
    for (int j = 0; j < Nband; ++j) {
        for (int i = 0; i < Nd_d; ++i) {
            int idx = i + j * Nd_d;
            Y[idx] = scale * (HX[idx] - c * X[idx]);
        }
    }

    // Save X as Xold
    std::memcpy(Xold.data(), X, Nd_d * Nband * sizeof(double));

    for (int k = 2; k <= degree; ++k) {
        sigma_new = 1.0 / (2.0 / sigma_1 - sigma);
        double gamma = 2.0 * sigma_new / e;

        // Xnew = gamma * (H*Y - c*Y) - sigma*sigma_new * Xold
        halo_->execute(Y, x_ex.data(), Nband);
        H_->apply(Y, Veff, HX.data(), Nband);

        double ss = sigma * sigma_new;
        for (int j = 0; j < Nband; ++j) {
            for (int i = 0; i < Nd_d; ++i) {
                int idx = i + j * Nd_d;
                Xnew[idx] = gamma * (HX[idx] - c * Y[idx]) - ss * Xold[idx];
            }
        }

        // Rotate: Xold <- Y, Y <- Xnew
        std::memcpy(Xold.data(), Y, Nd_d * Nband * sizeof(double));
        std::memcpy(Y, Xnew.data(), Nd_d * Nband * sizeof(double));
        sigma = sigma_new;
    }
}

void EigenSolver::orthogonalize(double* X, int Nd_d, int Nband, double dV) {
    // Cholesky QR: X^T * X = R^T * R, then X <- X * R^{-1}
    // Compute overlap: S = X^T * X * dV
    std::vector<double> S(Nband * Nband, 0.0);

    // S = X^T * X (local contribution)
    char transT = 'T', transN = 'N';
    double alpha = dV, beta = 0.0;
    dgemm_(&transT, &transN, &Nband, &Nband, &Nd_d,
           &alpha, X, &Nd_d, X, &Nd_d, &beta, S.data(), &Nband);

    // Cholesky: S = R^T * R (upper triangular)
    char uplo = 'U';
    int info;
    dpotrf_(&uplo, &Nband, S.data(), &Nband, &info);
    if (info != 0) {
        throw std::runtime_error("Cholesky factorization failed in orthogonalize");
    }

    // X <- X * R^{-1}  (solve X * R = X_new => X_new = X * inv(R))
    char side = 'R', diag = 'N';
    double one = 1.0;
    dtrsm_(&side, &uplo, &transN, &diag, &Nd_d, &Nband, &one,
           S.data(), &Nband, X, &Nd_d);
}

void EigenSolver::project_hamiltonian(const double* X, const double* Veff,
                                       double* Hs, int Nd_d, int Nband, double dV) {
    // HX = H * X
    std::vector<double> HX(Nd_d * Nband);
    int nd_ex = halo_->nd_ex();
    std::vector<double> x_ex(nd_ex * Nband);
    halo_->execute(X, x_ex.data(), Nband);
    H_->apply(X, Veff, HX.data(), Nband);

    // Hs = X^T * HX * dV
    char transT = 'T', transN = 'N';
    double alpha = dV, beta_v = 0.0;
    dgemm_(&transT, &transN, &Nband, &Nband, &Nd_d,
           &alpha, X, &Nd_d, HX.data(), &Nd_d, &beta_v, Hs, &Nband);

    // Symmetrize
    for (int i = 0; i < Nband; ++i) {
        for (int j = i + 1; j < Nband; ++j) {
            double avg = 0.5 * (Hs[i + j * Nband] + Hs[j + i * Nband]);
            Hs[i + j * Nband] = avg;
            Hs[j + i * Nband] = avg;
        }
    }
}

void EigenSolver::diag_subspace(double* Hs, double* eigvals, int N) {
    // dsyev: diagonalize symmetric matrix
    char jobz = 'V', uplo = 'U';
    int lwork = -1, info;
    double work_query;
    dsyev_(&jobz, &uplo, &N, Hs, &N, eigvals, &work_query, &lwork, &info);

    lwork = static_cast<int>(work_query);
    std::vector<double> work(lwork);
    dsyev_(&jobz, &uplo, &N, Hs, &N, eigvals, work.data(), &lwork, &info);

    if (info != 0) {
        // Diagnostic: check for NaN/Inf in input
        std::fprintf(stderr, "dsyev failed with info=%d, N=%d\n", info, N);
        throw std::runtime_error("dsyev failed in diag_subspace");
    }
}

void EigenSolver::rotate_orbitals(double* X, const double* Q, int Nd_d, int Nband) {
    // X_new = X * Q
    std::vector<double> X_new(Nd_d * Nband);
    char transN = 'N';
    double one = 1.0, zero = 0.0;
    dgemm_(&transN, &transN, &Nd_d, &Nband, &Nband,
           &one, X, &Nd_d, Q, &Nband, &zero, X_new.data(), &Nd_d);

    std::memcpy(X, X_new.data(), Nd_d * Nband * sizeof(double));
}

void EigenSolver::lanczos_bounds(const double* Veff, int Nd_d,
                                  double& eigval_min, double& eigval_max,
                                  double tol_lanczos, int max_iter) {
    // Reference: Lanczos() in eigenSolver.c
    // Tolerance-based stopping: converges when both eigmin and eigmax errors < tol
    int nd_ex = halo_->nd_ex();

    std::vector<double> V_j(Nd_d), V_jm1(Nd_d), V_jp1(Nd_d);
    std::vector<double> a(max_iter + 1, 0.0), b(max_iter + 1, 0.0);

    // Local comm for dot products (no domain decomposition)
    MPIComm self_comm(MPI_COMM_SELF);

    // Initial vector: match reference srand(rank*100+1), range [0,1]
    int rank = 0;  // single domain process
    std::srand(rank * 100 + 1);
    for (int i = 0; i < Nd_d; ++i)
        V_jm1[i] = (double)std::rand() / RAND_MAX;

    // Normalize
    double vscal = std::sqrt(LinearSolver::dot(V_jm1.data(), V_jm1.data(), Nd_d, self_comm));
    vscal = 1.0 / vscal;
    for (int i = 0; i < Nd_d; ++i) V_jm1[i] *= vscal;

    // First H*v
    std::vector<double> v_ex(nd_ex);
    halo_->execute(V_jm1.data(), v_ex.data(), 1);
    H_->apply(V_jm1.data(), Veff, V_j.data(), 1);

    // a[0] = <V_jm1, V_j>
    a[0] = LinearSolver::dot(V_jm1.data(), V_j.data(), Nd_d, self_comm);

    // Orthogonalize: V_j = V_j - a[0]*V_jm1
    for (int i = 0; i < Nd_d; ++i)
        V_j[i] -= a[0] * V_jm1[i];

    // b[0] = ||V_j||
    b[0] = std::sqrt(LinearSolver::dot(V_j.data(), V_j.data(), Nd_d, self_comm));

    if (b[0] == 0.0) {
        // Invariant subspace; pick random vector orthogonal to V_jm1
        for (int i = 0; i < Nd_d; ++i)
            V_j[i] = -1.0 + 2.0 * ((double)std::rand() / RAND_MAX);
        double dot_val = LinearSolver::dot(V_j.data(), V_jm1.data(), Nd_d, self_comm);
        for (int i = 0; i < Nd_d; ++i) V_j[i] -= dot_val * V_jm1[i];
        b[0] = std::sqrt(LinearSolver::dot(V_j.data(), V_j.data(), Nd_d, self_comm));
    }

    // Scale V_j
    vscal = (b[0] == 0.0) ? 1.0 : (1.0 / b[0]);
    for (int i = 0; i < Nd_d; ++i) V_j[i] *= vscal;

    eigval_min = 0.0;
    eigval_max = 0.0;
    double eigmin_pre = 0.0, eigmax_pre = 0.0;
    double err_eigmin = tol_lanczos + 1.0;
    double err_eigmax = tol_lanczos + 1.0;

    int j = 0;
    while ((err_eigmin > tol_lanczos || err_eigmax > tol_lanczos) && j < max_iter) {
        // V_{j+1} = H * V_j
        halo_->execute(V_j.data(), v_ex.data(), 1);
        H_->apply(V_j.data(), Veff, V_jp1.data(), 1);

        // a[j+1] = <V_j, V_{j+1}>
        a[j + 1] = LinearSolver::dot(V_j.data(), V_jp1.data(), Nd_d, self_comm);

        // V_{j+1} = V_{j+1} - a[j+1]*V_j - b[j]*V_{j-1}
        for (int i = 0; i < Nd_d; ++i) {
            V_jp1[i] -= (a[j + 1] * V_j[i] + b[j] * V_jm1[i]);
            V_jm1[i] = V_j[i];
        }

        b[j + 1] = std::sqrt(LinearSolver::dot(V_jp1.data(), V_jp1.data(), Nd_d, self_comm));
        if (b[j + 1] == 0.0) break;

        vscal = 1.0 / b[j + 1];
        for (int i = 0; i < Nd_d; ++i) V_j[i] = V_jp1[i] * vscal;

        // Solve tridiagonal eigenvalue problem using dsterf_
        // Copy a[0..j+1] into d, b[0..j+1] into e
        int n = j + 2;
        std::vector<double> d(n), e(n);
        for (int k = 0; k < n; ++k) { d[k] = a[k]; e[k] = b[k]; }

        int info;
        dsterf_(&n, d.data(), e.data(), &info);
        if (info == 0) {
            eigval_min = d[0];
            eigval_max = d[n - 1];
        } else {
            break;
        }

        err_eigmin = std::abs(eigval_min - eigmin_pre);
        err_eigmax = std::abs(eigval_max - eigmax_pre);
        eigmin_pre = eigval_min;
        eigmax_pre = eigval_max;

        j++;
    }

    // Apply safety margins (reference: eigmax *= 1.01, eigmin -= 0.1)
    eigval_max *= 1.01;
    eigval_min -= 0.1;
}

void EigenSolver::solve(double* psi, double* eigvals, const double* Veff,
                        int Nd_d, int Nband,
                        double lambda_cutoff, double eigval_min, double eigval_max,
                        int cheb_degree, int ld) {
    if (ld == 0) ld = Nd_d;
    double dV = domain_->global_grid().dV();

    // Nband here is local band count. For serial case, Nband = Nband_global.
    int Nband_loc = Nband;

    // Pack from NDArray layout (stride=ld) to packed layout (stride=Nd_d)
    std::vector<double> psi_packed;
    double* psi_work = psi;
    if (ld != Nd_d) {
        psi_packed.resize(Nd_d * Nband_loc);
        for (int j = 0; j < Nband_loc; ++j)
            std::memcpy(psi_packed.data() + j * Nd_d, psi + j * ld, Nd_d * sizeof(double));
        psi_work = psi_packed.data();
    }

    // Step 1: Chebyshev filter (embarrassingly parallel — each proc filters its local bands)
    std::vector<double> Y(Nd_d * Nband_loc);
    chebyshev_filter(psi_work, Y.data(), Veff, Nd_d, Nband_loc,
                     lambda_cutoff, eigval_min, eigval_max, cheb_degree);

#ifdef USE_SCALAPACK
    if (npband_ > 1) {
        // Band-parallel path: distributed subspace operations
        // Step 2: Orthogonalize (Allgather + Cholesky QR)
        orthogonalize_scalapack(Y.data(), Nd_d, Nband_loc, dV);

        // Step 3+4: Project Hamiltonian + diagonalize (returns ALL eigenvalues)
        int N = Nband_global_;
        std::vector<double> eigs_all(N);
        project_hamiltonian_scalapack(Y.data(), Veff, eigs_all.data(), Nd_d, Nband_loc, dV);

        // Step 5: Rotate orbitals (Allgather + dgemm + extract local)
        rotate_orbitals_scalapack(Y.data(), Nd_d, Nband_loc);

        // Copy ALL eigenvalues (caller needs them all for Fermi level)
        // eigvals buffer must be large enough for Nband_global
        std::memcpy(eigvals, eigs_all.data(), N * sizeof(double));
    } else
#endif
    {
        // Serial path: standard LAPACK
        // Step 2: Orthogonalize filtered vectors
        orthogonalize(Y.data(), Nd_d, Nband_loc, dV);

        // Step 3: Project Hamiltonian onto subspace
        std::vector<double> Hs(Nband_loc * Nband_loc);
        project_hamiltonian(Y.data(), Veff, Hs.data(), Nd_d, Nband_loc, dV);

        // Step 4: Diagonalize subspace Hamiltonian
        std::vector<double> eigs(Nband_loc);
        diag_subspace(Hs.data(), eigs.data(), Nband_loc);

        // Step 5: Rotate orbitals
        rotate_orbitals(Y.data(), Hs.data(), Nd_d, Nband_loc);

        std::memcpy(eigvals, eigs.data(), Nband_loc * sizeof(double));
    }

    // Unpack from packed layout (stride=Nd_d) back to NDArray layout (stride=ld)
    if (ld != Nd_d) {
        for (int j = 0; j < Nband_loc; ++j)
            std::memcpy(psi + j * ld, Y.data() + j * Nd_d, Nd_d * sizeof(double));
    } else {
        std::memcpy(psi, Y.data(), Nd_d * Nband_loc * sizeof(double));
    }

    // Update lambda_cutoff: use last global eigenvalue
    int N_eig = (npband_ > 1) ? Nband_global_ : Nband_loc;
    lambda_cutoff_ = eigvals[N_eig - 1] + 0.1;
}

// ===========================================================================
// Complex (k-point) implementations
// ===========================================================================

void EigenSolver::chebyshev_filter_kpt(const Complex* X, Complex* Y, const double* Veff,
                                        int Nd_d, int Nband,
                                        double lambda_cutoff, double eigval_min, double eigval_max,
                                        const Vec3& kpt_cart, const Vec3& cell_lengths,
                                        int degree) {
    double e = (eigval_max - lambda_cutoff) / 2.0;
    double c = (eigval_max + lambda_cutoff) / 2.0;
    double sigma_1 = e / (eigval_min - c);
    double sigma = sigma_1;
    double sigma_new;

    std::vector<Complex> Xold(Nd_d * Nband);
    std::vector<Complex> Xnew(Nd_d * Nband);
    std::vector<Complex> HX(Nd_d * Nband);

    // Y = (H*X - c*X) * (sigma/e)
    H_->apply_kpt(X, Veff, HX.data(), Nband, kpt_cart, cell_lengths);

    double scale = sigma / e;
    for (int j = 0; j < Nband; ++j) {
        for (int i = 0; i < Nd_d; ++i) {
            int idx = i + j * Nd_d;
            Y[idx] = scale * (HX[idx] - c * X[idx]);
        }
    }

    std::memcpy(Xold.data(), X, Nd_d * Nband * sizeof(Complex));

    for (int k = 2; k <= degree; ++k) {
        sigma_new = 1.0 / (2.0 / sigma_1 - sigma);
        double gamma = 2.0 * sigma_new / e;

        H_->apply_kpt(Y, Veff, HX.data(), Nband, kpt_cart, cell_lengths);

        double ss = sigma * sigma_new;
        for (int j = 0; j < Nband; ++j) {
            for (int i = 0; i < Nd_d; ++i) {
                int idx = i + j * Nd_d;
                Xnew[idx] = gamma * (HX[idx] - c * Y[idx]) - ss * Xold[idx];
            }
        }

        std::memcpy(Xold.data(), Y, Nd_d * Nband * sizeof(Complex));
        std::memcpy(Y, Xnew.data(), Nd_d * Nband * sizeof(Complex));
        sigma = sigma_new;
    }
}

void EigenSolver::orthogonalize_kpt(Complex* X, int Nd_d, int Nband, double dV) {
    // Cholesky QR: S = X^H * X * dV, S = R^H * R, X <- X * R^{-1}
    std::vector<Complex> S(Nband * Nband, Complex(0.0));

    // S = X^H * X * dV
    char transC = 'C', transN = 'N';
    Complex alpha_z(dV, 0.0), beta_z(0.0, 0.0);
    zgemm_(&transC, &transN, &Nband, &Nband, &Nd_d,
           &alpha_z, X, &Nd_d, X, &Nd_d, &beta_z, S.data(), &Nband);

    // Cholesky
    char uplo = 'U';
    int info;
    zpotrf_(&uplo, &Nband, S.data(), &Nband, &info);
    if (info != 0) {
        throw std::runtime_error("zpotrf failed in orthogonalize_kpt (info=" + std::to_string(info) + ", Nd_d=" + std::to_string(Nd_d) + ", Nband=" + std::to_string(Nband) + ")");
    }

    // X <- X * R^{-1}
    char side = 'R', diag = 'N';
    Complex one_z(1.0, 0.0);
    ztrsm_(&side, &uplo, &transN, &diag, &Nd_d, &Nband, &one_z,
           S.data(), &Nband, X, &Nd_d);
}

void EigenSolver::project_hamiltonian_kpt(const Complex* X, const double* Veff,
                                           Complex* Hs, int Nd_d, int Nband, double dV,
                                           const Vec3& kpt_cart, const Vec3& cell_lengths) {
    std::vector<Complex> HX(Nd_d * Nband);
    H_->apply_kpt(X, Veff, HX.data(), Nband, kpt_cart, cell_lengths);

    // Hs = X^H * HX * dV
    char transC = 'C', transN = 'N';
    Complex alpha_z(dV, 0.0), beta_z(0.0, 0.0);
    zgemm_(&transC, &transN, &Nband, &Nband, &Nd_d,
           &alpha_z, X, &Nd_d, HX.data(), &Nd_d, &beta_z, Hs, &Nband);

    // Hermitianize
    for (int i = 0; i < Nband; ++i) {
        for (int j = i + 1; j < Nband; ++j) {
            Complex avg = 0.5 * (Hs[i + j * Nband] + std::conj(Hs[j + i * Nband]));
            Hs[i + j * Nband] = avg;
            Hs[j + i * Nband] = std::conj(avg);
        }
        // Diagonal must be real
        Hs[i + i * Nband] = Complex(Hs[i + i * Nband].real(), 0.0);
    }
}

void EigenSolver::diag_subspace_kpt(Complex* Hs, double* eigvals, int N) {
    char jobz = 'V', uplo = 'U';
    int lwork = -1, info;
    Complex work_query;
    std::vector<double> rwork(std::max(1, 3 * N - 2));

    zheev_(&jobz, &uplo, &N, Hs, &N, eigvals, &work_query, &lwork, rwork.data(), &info);

    lwork = static_cast<int>(work_query.real());
    std::vector<Complex> work(lwork);
    zheev_(&jobz, &uplo, &N, Hs, &N, eigvals, work.data(), &lwork, rwork.data(), &info);

    if (info != 0) {
        std::fprintf(stderr, "zheev failed with info=%d, N=%d\n", info, N);
        throw std::runtime_error("zheev failed in diag_subspace_kpt");
    }
}

void EigenSolver::rotate_orbitals_kpt(Complex* X, const Complex* Q, int Nd_d, int Nband) {
    std::vector<Complex> X_new(Nd_d * Nband);
    char transN = 'N';
    Complex one_z(1.0, 0.0), zero_z(0.0, 0.0);
    zgemm_(&transN, &transN, &Nd_d, &Nband, &Nband,
           &one_z, X, &Nd_d, Q, &Nband, &zero_z, X_new.data(), &Nd_d);
    std::memcpy(X, X_new.data(), Nd_d * Nband * sizeof(Complex));
}

void EigenSolver::solve_kpt(Complex* psi, double* eigvals, const double* Veff,
                             int Nd_d, int Nband,
                             double lambda_cutoff, double eigval_min, double eigval_max,
                             const Vec3& kpt_cart, const Vec3& cell_lengths,
                             int cheb_degree, int ld) {
    if (ld == 0) ld = Nd_d;
    double dV = domain_->global_grid().dV();
    int Nband_loc = Nband;

    // Pack from NDArray layout (stride=ld) to packed layout (stride=Nd_d)
    std::vector<Complex> psi_packed;
    Complex* psi_work = psi;
    if (ld != Nd_d) {
        psi_packed.resize(Nd_d * Nband_loc);
        for (int j = 0; j < Nband_loc; ++j)
            std::memcpy(psi_packed.data() + j * Nd_d, psi + j * ld, Nd_d * sizeof(Complex));
        psi_work = psi_packed.data();
    }

    // Step 1: Chebyshev filter (each proc filters its local bands)
    std::vector<Complex> Y(Nd_d * Nband_loc);
    chebyshev_filter_kpt(psi_work, Y.data(), Veff, Nd_d, Nband_loc,
                         lambda_cutoff, eigval_min, eigval_max,
                         kpt_cart, cell_lengths, cheb_degree);

#ifdef USE_SCALAPACK
    if (npband_ > 1) {
        // Band-parallel path
        orthogonalize_kpt_scalapack(Y.data(), Nd_d, Nband_loc, dV);

        int N = Nband_global_;
        std::vector<double> eigs_all(N);
        project_hamiltonian_kpt_scalapack(Y.data(), Veff, eigs_all.data(), Nd_d, Nband_loc, dV,
                                          kpt_cart, cell_lengths);

        rotate_orbitals_kpt_scalapack(Y.data(), Nd_d, Nband_loc);

        std::memcpy(eigvals, eigs_all.data(), N * sizeof(double));
    } else
#endif
    {
        // Serial path: generalized eigenvalue problem Hp * Q = Mp * Q * Lambda
        // (matching SPARC's zhegvd approach, which handles ill-conditioned overlap
        // matrices much better than explicit Cholesky orthogonalization)

        // Compute overlap matrix Mp = Y^H * Y * dV
        int N = Nband_loc;
        std::vector<Complex> Mp(N * N, Complex(0.0));
        {
            char transC = 'C', transN = 'N';
            Complex alpha_z(dV, 0.0), beta_z(0.0, 0.0);
            zgemm_(&transC, &transN, &N, &N, &Nd_d,
                   &alpha_z, Y.data(), &Nd_d, Y.data(), &Nd_d, &beta_z, Mp.data(), &N);
        }

        // Compute projected Hamiltonian Hp = Y^H * H * Y * dV
        std::vector<Complex> HY(Nd_d * N);
        H_->apply_kpt(Y.data(), Veff, HY.data(), N, kpt_cart, cell_lengths);
        std::vector<Complex> Hp(N * N, Complex(0.0));
        {
            char transC = 'C', transN = 'N';
            Complex alpha_z(dV, 0.0), beta_z(0.0, 0.0);
            zgemm_(&transC, &transN, &N, &N, &Nd_d,
                   &alpha_z, Y.data(), &Nd_d, HY.data(), &Nd_d, &beta_z, Hp.data(), &N);
        }

        // Hermitianize both matrices
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                Complex avg_h = 0.5 * (Hp[i + j * N] + std::conj(Hp[j + i * N]));
                Hp[i + j * N] = avg_h;
                Hp[j + i * N] = std::conj(avg_h);
                Complex avg_m = 0.5 * (Mp[i + j * N] + std::conj(Mp[j + i * N]));
                Mp[i + j * N] = avg_m;
                Mp[j + i * N] = std::conj(avg_m);
            }
            Hp[i + i * N] = Complex(Hp[i + i * N].real(), 0.0);
            Mp[i + i * N] = Complex(Mp[i + i * N].real(), 0.0);
        }

        // Solve generalized eigenvalue problem: Hp * Q = Mp * Q * Lambda
        std::vector<double> eigs(N);
        int itype = 1;
        char jobz = 'V', uplo = 'U';
        int lwork = -1, lrwork = -1, liwork = -1, info;
        Complex work_query;
        double rwork_query;
        int iwork_query;

        zhegvd_(&itype, &jobz, &uplo, &N,
                Hp.data(), &N, Mp.data(), &N, eigs.data(),
                &work_query, &lwork, &rwork_query, &lrwork,
                &iwork_query, &liwork, &info);

        lwork = static_cast<int>(work_query.real());
        lrwork = static_cast<int>(rwork_query);
        liwork = iwork_query;
        std::vector<Complex> work(lwork);
        std::vector<double> rwork(lrwork);
        std::vector<int> iwork(liwork);

        zhegvd_(&itype, &jobz, &uplo, &N,
                Hp.data(), &N, Mp.data(), &N, eigs.data(),
                work.data(), &lwork, rwork.data(), &lrwork,
                iwork.data(), &liwork, &info);

        if (info != 0) {
            // SPARC silently continues on zhegvd failure (info > N means overlap
            // matrix not positive definite). The partially-valid results may still
            // be useful for SCF convergence. Only warn, don't throw.
            std::fprintf(stderr, "WARNING: zhegvd info=%d, N=%d (overlap matrix ill-conditioned)\n", info, N);
        }

        // Rotate orbitals: Y <- Y * Q (eigenvectors are in Hp after zhegvd)
        rotate_orbitals_kpt(Y.data(), Hp.data(), Nd_d, N);

        std::memcpy(eigvals, eigs.data(), N * sizeof(double));
    }

    // Unpack
    if (ld != Nd_d) {
        for (int j = 0; j < Nband_loc; ++j)
            std::memcpy(psi + j * ld, Y.data() + j * Nd_d, Nd_d * sizeof(Complex));
    } else {
        std::memcpy(psi, Y.data(), Nd_d * Nband_loc * sizeof(Complex));
    }

    int N_eig = (npband_ > 1) ? Nband_global_ : Nband_loc;
    lambda_cutoff_ = eigvals[N_eig - 1] + 0.1;
}

void EigenSolver::lanczos_bounds_kpt(const double* Veff, int Nd_d,
                                      const Vec3& kpt_cart, const Vec3& cell_lengths,
                                      double& eigval_min, double& eigval_max,
                                      double tol_lanczos, int max_iter) {
    MPIComm self_comm(MPI_COMM_SELF);

    std::vector<Complex> V_j(Nd_d), V_jm1(Nd_d), V_jp1(Nd_d);
    std::vector<double> a(max_iter + 1, 0.0), b(max_iter + 1, 0.0);

    // Initial random vector
    int rank = 0;
    std::srand(rank * 100 + 1);
    for (int i = 0; i < Nd_d; ++i)
        V_jm1[i] = Complex((double)std::rand() / RAND_MAX, (double)std::rand() / RAND_MAX);

    // Normalize: ||v|| = sqrt(Re(<v,v>))
    double norm2 = 0.0;
    for (int i = 0; i < Nd_d; ++i)
        norm2 += std::norm(V_jm1[i]);  // |z|^2
    double vscal = 1.0 / std::sqrt(norm2);
    for (int i = 0; i < Nd_d; ++i) V_jm1[i] *= vscal;

    // H * v
    H_->apply_kpt(V_jm1.data(), Veff, V_j.data(), 1, kpt_cart, cell_lengths);

    // a[0] = Re(<V_jm1, V_j>)
    Complex dot_val(0.0, 0.0);
    for (int i = 0; i < Nd_d; ++i)
        dot_val += std::conj(V_jm1[i]) * V_j[i];
    a[0] = dot_val.real();

    // V_j -= a[0] * V_jm1
    for (int i = 0; i < Nd_d; ++i)
        V_j[i] -= a[0] * V_jm1[i];

    // b[0] = ||V_j||
    norm2 = 0.0;
    for (int i = 0; i < Nd_d; ++i)
        norm2 += std::norm(V_j[i]);
    b[0] = std::sqrt(norm2);

    if (b[0] == 0.0) {
        for (int i = 0; i < Nd_d; ++i)
            V_j[i] = Complex(-1.0 + 2.0 * ((double)std::rand() / RAND_MAX),
                             -1.0 + 2.0 * ((double)std::rand() / RAND_MAX));
        dot_val = Complex(0.0, 0.0);
        for (int i = 0; i < Nd_d; ++i)
            dot_val += std::conj(V_j[i]) * V_jm1[i];
        for (int i = 0; i < Nd_d; ++i)
            V_j[i] -= dot_val * V_jm1[i];
        norm2 = 0.0;
        for (int i = 0; i < Nd_d; ++i)
            norm2 += std::norm(V_j[i]);
        b[0] = std::sqrt(norm2);
    }

    vscal = (b[0] == 0.0) ? 1.0 : (1.0 / b[0]);
    for (int i = 0; i < Nd_d; ++i) V_j[i] *= vscal;

    eigval_min = 0.0;
    eigval_max = 0.0;
    double eigmin_pre = 0.0, eigmax_pre = 0.0;
    double err_eigmin = tol_lanczos + 1.0;
    double err_eigmax = tol_lanczos + 1.0;

    int j = 0;
    while ((err_eigmin > tol_lanczos || err_eigmax > tol_lanczos) && j < max_iter) {
        H_->apply_kpt(V_j.data(), Veff, V_jp1.data(), 1, kpt_cart, cell_lengths);

        dot_val = Complex(0.0, 0.0);
        for (int i = 0; i < Nd_d; ++i)
            dot_val += std::conj(V_j[i]) * V_jp1[i];
        a[j + 1] = dot_val.real();

        for (int i = 0; i < Nd_d; ++i) {
            V_jp1[i] -= (a[j + 1] * V_j[i] + b[j] * V_jm1[i]);
            V_jm1[i] = V_j[i];
        }

        norm2 = 0.0;
        for (int i = 0; i < Nd_d; ++i)
            norm2 += std::norm(V_jp1[i]);
        b[j + 1] = std::sqrt(norm2);
        if (b[j + 1] == 0.0) break;

        vscal = 1.0 / b[j + 1];
        for (int i = 0; i < Nd_d; ++i) V_j[i] = V_jp1[i] * vscal;

        // Solve tridiagonal eigenvalue problem
        int n = j + 2;
        std::vector<double> d(n), e_vec(n);
        for (int kk = 0; kk < n; ++kk) { d[kk] = a[kk]; e_vec[kk] = b[kk]; }

        int info;
        dsterf_(&n, d.data(), e_vec.data(), &info);
        if (info == 0) {
            eigval_min = d[0];
            eigval_max = d[n - 1];
        } else {
            break;
        }

        err_eigmin = std::abs(eigval_min - eigmin_pre);
        err_eigmax = std::abs(eigval_max - eigmax_pre);
        eigmin_pre = eigval_min;
        eigmax_pre = eigval_max;

        j++;
    }

    eigval_max *= 1.01;
    eigval_min -= 0.1;
}

// ===========================================================================
// Spinor (SOC) implementations
// ===========================================================================

void EigenSolver::solve_spinor_kpt(Complex* psi, double* eigvals, const double* Veff_spinor,
                                    int Nd_d, int Nband,
                                    double lambda_cutoff, double eigval_min, double eigval_max,
                                    const Vec3& kpt_cart, const Vec3& cell_lengths,
                                    int cheb_degree, int ld) {
    int Nd_d_spinor = 2 * Nd_d;
    if (ld == 0) ld = Nd_d_spinor;
    double dV = domain_->global_grid().dV();
    int Nband_loc = Nband;

    // Pack from NDArray layout (stride=ld) to packed layout (stride=Nd_d_spinor)
    std::vector<Complex> psi_packed;
    Complex* psi_work = psi;
    if (ld != Nd_d_spinor) {
        psi_packed.resize(Nd_d_spinor * Nband_loc);
        for (int j = 0; j < Nband_loc; ++j)
            std::memcpy(psi_packed.data() + j * Nd_d_spinor, psi + j * ld, Nd_d_spinor * sizeof(Complex));
        psi_work = psi_packed.data();
    }

    // Step 1: Chebyshev filter using spinor H*psi
    // Use the existing chebyshev_filter_kpt logic but with spinor Hamiltonian
    std::vector<Complex> Y(Nd_d_spinor * Nband_loc);
    {
        // Chebyshev filter for spinor: same algorithm, different H*psi callback
        double e = (eigval_max - lambda_cutoff) / 2.0;
        double c_cheb = (eigval_max + lambda_cutoff) / 2.0;
        double sigma_1 = e / (eigval_min - c_cheb);
        double sigma = sigma_1;

        std::vector<Complex> Xold(Nd_d_spinor * Nband_loc);
        std::vector<Complex> Xnew(Nd_d_spinor * Nband_loc);
        std::vector<Complex> HX(Nd_d_spinor * Nband_loc);

        // Y = (H*X - c*X) * (sigma/e)
        H_->apply_spinor_kpt(psi_work, Veff_spinor, HX.data(), Nband_loc, Nd_d,
                              kpt_cart, cell_lengths);

        double scale = sigma / e;
        for (int j = 0; j < Nband_loc; ++j) {
            for (int i = 0; i < Nd_d_spinor; ++i) {
                int idx = i + j * Nd_d_spinor;
                Y[idx] = scale * (HX[idx] - c_cheb * psi_work[idx]);
            }
        }

        std::memcpy(Xold.data(), psi_work, Nd_d_spinor * Nband_loc * sizeof(Complex));

        for (int k = 2; k <= cheb_degree; ++k) {
            double sigma_new = 1.0 / (2.0 / sigma_1 - sigma);
            double gamma = 2.0 * sigma_new / e;

            H_->apply_spinor_kpt(Y.data(), Veff_spinor, HX.data(), Nband_loc, Nd_d,
                                  kpt_cart, cell_lengths);

            double ss = sigma * sigma_new;
            for (int j = 0; j < Nband_loc; ++j) {
                for (int i = 0; i < Nd_d_spinor; ++i) {
                    int idx = i + j * Nd_d_spinor;
                    Xnew[idx] = gamma * (HX[idx] - c_cheb * Y[idx]) - ss * Xold[idx];
                }
            }

            std::memcpy(Xold.data(), Y.data(), Nd_d_spinor * Nband_loc * sizeof(Complex));
            std::memcpy(Y.data(), Xnew.data(), Nd_d_spinor * Nband_loc * sizeof(Complex));
            sigma = sigma_new;

        }
    }

    // Normalize columns of Y before orthogonalization to prevent overflow
    // in the overlap matrix (Chebyshev filter can amplify norms by 10^30+)
    for (int j = 0; j < Nband_loc; ++j) {
        Complex* col = Y.data() + j * Nd_d_spinor;
        double norm2 = 0.0;
        for (int i = 0; i < Nd_d_spinor; ++i) norm2 += std::norm(col[i]);
        if (norm2 > 0.0) {
            double inv_norm = 1.0 / std::sqrt(norm2);
            for (int i = 0; i < Nd_d_spinor; ++i) col[i] *= inv_norm;
        }
    }

    // Steps 2-5: Orthogonalize, project, diag, rotate — using Nd_d_spinor as row dim
    // The ScaLAPACK band-parallel functions accept row dimension as parameter,
    // so passing Nd_d_spinor instead of Nd_d makes them work for spinors.
#ifdef USE_SCALAPACK
    if (npband_ > 1) {
        orthogonalize_kpt_scalapack(Y.data(), Nd_d_spinor, Nband_loc, dV);

        int N = Nband_global_;
        std::vector<double> eigs_all(N);
        // For spinor project_hamiltonian: need spinor H*psi callback
        // Inline the ScaLAPACK projection with spinor H application
        {
            std::vector<Complex> HX(Nd_d_spinor * Nband_loc);
            H_->apply_spinor_kpt(Y.data(), Veff_spinor, HX.data(), Nband_loc, Nd_d,
                                  kpt_cart, cell_lengths);

            // Allgather X and HX
            std::vector<Complex> X_full(Nd_d_spinor * N), HX_full(Nd_d_spinor * N);
            {
                std::vector<int> recvcounts(npband_), displs(npband_);
                for (int p = 0; p < npband_; ++p) {
                    int nb_p = Parallelization::block_size(N, npband_, p);
                    int bs_p = Parallelization::block_start(N, npband_, p);
                    recvcounts[p] = Nd_d_spinor * nb_p;
                    displs[p] = Nd_d_spinor * bs_p;
                }
                MPI_Allgatherv(Y.data(), Nd_d_spinor * Nband_loc, MPI_C_DOUBLE_COMPLEX,
                               X_full.data(), recvcounts.data(), displs.data(),
                               MPI_C_DOUBLE_COMPLEX, bandcomm_->comm());
                MPI_Allgatherv(HX.data(), Nd_d_spinor * Nband_loc, MPI_C_DOUBLE_COMPLEX,
                               HX_full.data(), recvcounts.data(), displs.data(),
                               MPI_C_DOUBLE_COMPLEX, bandcomm_->comm());
            }

            // Hs = X^H * HX * dV
            std::vector<Complex> Hs(N * N, Complex(0.0));
            {
                char transC = 'C', transN = 'N';
                Complex alpha_z(dV, 0.0), beta_z(0.0, 0.0);
                zgemm_(&transC, &transN, &N, &N, &Nd_d_spinor,
                       &alpha_z, X_full.data(), &Nd_d_spinor, HX_full.data(), &Nd_d_spinor,
                       &beta_z, Hs.data(), &N);
            }

            // Hermitianize
            for (int i = 0; i < N; ++i) {
                for (int j = i + 1; j < N; ++j) {
                    Complex avg = 0.5 * (Hs[i + j * N] + std::conj(Hs[j + i * N]));
                    Hs[i + j * N] = avg;
                    Hs[j + i * N] = std::conj(avg);
                }
                Hs[i + i * N] = Complex(Hs[i + i * N].real(), 0.0);
            }

            // Diagonalize
            {
                char jobz = 'V', uplo = 'U';
                int lwork = -1, info;
                Complex work_query;
                std::vector<double> rwork(std::max(1, 3 * N - 2));
                zheev_(&jobz, &uplo, &N, Hs.data(), &N, eigs_all.data(),
                       &work_query, &lwork, rwork.data(), &info);
                lwork = static_cast<int>(work_query.real());
                std::vector<Complex> work(lwork);
                zheev_(&jobz, &uplo, &N, Hs.data(), &N, eigs_all.data(),
                       work.data(), &lwork, rwork.data(), &info);
            }

            Q_dist_z_ = std::move(Hs);
        }

        // Rotate orbitals
        rotate_orbitals_kpt_scalapack(Y.data(), Nd_d_spinor, Nband_loc);

        std::memcpy(eigvals, eigs_all.data(), N * sizeof(double));
    } else
#endif
    {
        // Serial path
        orthogonalize_kpt(Y.data(), Nd_d_spinor, Nband_loc, dV);

        std::vector<Complex> Hs(Nband_loc * Nband_loc);
        {
            std::vector<Complex> HX(Nd_d_spinor * Nband_loc);
            H_->apply_spinor_kpt(Y.data(), Veff_spinor, HX.data(), Nband_loc, Nd_d,
                                  kpt_cart, cell_lengths);

            char transC = 'C', transN = 'N';
            Complex alpha_z(dV, 0.0), beta_z(0.0, 0.0);
            zgemm_(&transC, &transN, &Nband_loc, &Nband_loc, &Nd_d_spinor,
                   &alpha_z, Y.data(), &Nd_d_spinor, HX.data(), &Nd_d_spinor,
                   &beta_z, Hs.data(), &Nband_loc);

            for (int i = 0; i < Nband_loc; ++i) {
                for (int j = i + 1; j < Nband_loc; ++j) {
                    Complex avg = 0.5 * (Hs[i + j * Nband_loc] + std::conj(Hs[j + i * Nband_loc]));
                    Hs[i + j * Nband_loc] = avg;
                    Hs[j + i * Nband_loc] = std::conj(avg);
                }
                Hs[i + i * Nband_loc] = Complex(Hs[i + i * Nband_loc].real(), 0.0);
            }
        }

        std::vector<double> eigs(Nband_loc);
        diag_subspace_kpt(Hs.data(), eigs.data(), Nband_loc);
        rotate_orbitals_kpt(Y.data(), Hs.data(), Nd_d_spinor, Nband_loc);

        std::memcpy(eigvals, eigs.data(), Nband_loc * sizeof(double));
    }

    // Unpack
    if (ld != Nd_d_spinor) {
        for (int j = 0; j < Nband_loc; ++j)
            std::memcpy(psi + j * ld, Y.data() + j * Nd_d_spinor, Nd_d_spinor * sizeof(Complex));
    } else {
        std::memcpy(psi, Y.data(), Nd_d_spinor * Nband_loc * sizeof(Complex));
    }

    int N_eig = Nband_loc;
    lambda_cutoff_ = eigvals[N_eig - 1] + 0.1;
}

void EigenSolver::lanczos_bounds_spinor_kpt(const double* Veff_spinor, int Nd_d,
                                              const Vec3& kpt_cart, const Vec3& cell_lengths,
                                              double& eigval_min, double& eigval_max,
                                              double tol_lanczos, int max_iter) {
    int Nd_d_spinor = 2 * Nd_d;
    MPIComm self_comm(MPI_COMM_SELF);

    std::vector<Complex> V_j(Nd_d_spinor), V_jm1(Nd_d_spinor), V_jp1(Nd_d_spinor);
    std::vector<double> a(max_iter + 1, 0.0), b(max_iter + 1, 0.0);

    // Initial random vector
    std::srand(1);
    for (int i = 0; i < Nd_d_spinor; ++i)
        V_jm1[i] = Complex((double)std::rand() / RAND_MAX, (double)std::rand() / RAND_MAX);

    // Normalize
    double norm2 = 0.0;
    for (int i = 0; i < Nd_d_spinor; ++i)
        norm2 += std::norm(V_jm1[i]);
    double vscal = 1.0 / std::sqrt(norm2);
    for (int i = 0; i < Nd_d_spinor; ++i) V_jm1[i] *= vscal;

    // H * v
    H_->apply_spinor_kpt(V_jm1.data(), Veff_spinor, V_j.data(), 1, Nd_d,
                          kpt_cart, cell_lengths);

    // a[0] = Re(<V_jm1, V_j>)
    Complex dot_val(0.0);
    for (int i = 0; i < Nd_d_spinor; ++i)
        dot_val += std::conj(V_jm1[i]) * V_j[i];
    a[0] = dot_val.real();

    for (int i = 0; i < Nd_d_spinor; ++i)
        V_j[i] -= a[0] * V_jm1[i];

    norm2 = 0.0;
    for (int i = 0; i < Nd_d_spinor; ++i)
        norm2 += std::norm(V_j[i]);
    b[0] = std::sqrt(norm2);

    if (b[0] == 0.0) {
        for (int i = 0; i < Nd_d_spinor; ++i)
            V_j[i] = Complex(-1.0 + 2.0 * ((double)std::rand() / RAND_MAX),
                             -1.0 + 2.0 * ((double)std::rand() / RAND_MAX));
        dot_val = Complex(0.0);
        for (int i = 0; i < Nd_d_spinor; ++i)
            dot_val += std::conj(V_j[i]) * V_jm1[i];
        for (int i = 0; i < Nd_d_spinor; ++i)
            V_j[i] -= dot_val * V_jm1[i];
        norm2 = 0.0;
        for (int i = 0; i < Nd_d_spinor; ++i)
            norm2 += std::norm(V_j[i]);
        b[0] = std::sqrt(norm2);
    }

    vscal = (b[0] == 0.0) ? 1.0 : (1.0 / b[0]);
    for (int i = 0; i < Nd_d_spinor; ++i) V_j[i] *= vscal;

    eigval_min = 0.0;
    eigval_max = 0.0;
    double eigmin_pre = 0.0, eigmax_pre = 0.0;
    double err_eigmin = tol_lanczos + 1.0;
    double err_eigmax = tol_lanczos + 1.0;

    int j = 0;
    while ((err_eigmin > tol_lanczos || err_eigmax > tol_lanczos) && j < max_iter) {
        H_->apply_spinor_kpt(V_j.data(), Veff_spinor, V_jp1.data(), 1, Nd_d,
                              kpt_cart, cell_lengths);

        dot_val = Complex(0.0);
        for (int i = 0; i < Nd_d_spinor; ++i)
            dot_val += std::conj(V_j[i]) * V_jp1[i];
        a[j + 1] = dot_val.real();

        for (int i = 0; i < Nd_d_spinor; ++i) {
            V_jp1[i] -= (a[j + 1] * V_j[i] + b[j] * V_jm1[i]);
            V_jm1[i] = V_j[i];
        }

        norm2 = 0.0;
        for (int i = 0; i < Nd_d_spinor; ++i)
            norm2 += std::norm(V_jp1[i]);
        b[j + 1] = std::sqrt(norm2);
        if (b[j + 1] == 0.0) break;

        vscal = 1.0 / b[j + 1];
        for (int i = 0; i < Nd_d_spinor; ++i) V_j[i] = V_jp1[i] * vscal;

        int n = j + 2;
        std::vector<double> d(n), e_vec(n);
        for (int kk = 0; kk < n; ++kk) { d[kk] = a[kk]; e_vec[kk] = b[kk]; }

        int info;
        dsterf_(&n, d.data(), e_vec.data(), &info);
        if (info == 0) {
            eigval_min = d[0];
            eigval_max = d[n - 1];
        } else {
            break;
        }

        err_eigmin = std::abs(eigval_min - eigmin_pre);
        err_eigmax = std::abs(eigval_max - eigmax_pre);
        eigmin_pre = eigval_min;
        eigmax_pre = eigval_max;

        j++;
    }

    eigval_max *= 1.01;
    eigval_min -= 0.1;
}

} // namespace lynx
