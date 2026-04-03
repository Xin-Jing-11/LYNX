#include "solvers/EigenSolver.hpp"
#include "solvers/Lanczos.hpp"
#include "solvers/LinearSolver.hpp"
#include "parallel/Parallelization.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <mpi.h>
#include <omp.h>

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
}

namespace lynx {

// ===== BLAS/LAPACK type-dispatch helpers =====
namespace blas {

// gemm: C = alpha * op(A) * op(B) + beta * C
inline void gemm(const char* ta, const char* tb, const int* m, const int* n, const int* k,
                 const double* alpha, const double* A, const int* lda,
                 const double* B, const int* ldb,
                 const double* beta, double* C, const int* ldc) {
    dgemm_(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline void gemm(const char* ta, const char* tb, const int* m, const int* n, const int* k,
                 const Complex* alpha, const Complex* A, const int* lda,
                 const Complex* B, const int* ldb,
                 const Complex* beta, Complex* C, const int* ldc) {
    zgemm_(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// potrf: Cholesky factorization
inline void potrf(const char* uplo, const int* n, double* A, const int* lda, int* info) {
    dpotrf_(uplo, n, A, lda, info);
}
inline void potrf(const char* uplo, const int* n, Complex* A, const int* lda, int* info) {
    zpotrf_(uplo, n, A, lda, info);
}

// trsm: triangular solve
inline void trsm(const char* side, const char* uplo, const char* ta, const char* diag,
                 const int* m, const int* n, const double* alpha,
                 const double* A, const int* lda, double* B, const int* ldb) {
    dtrsm_(side, uplo, ta, diag, m, n, alpha, A, lda, B, ldb);
}
inline void trsm(const char* side, const char* uplo, const char* ta, const char* diag,
                 const int* m, const int* n, const Complex* alpha,
                 const Complex* A, const int* lda, Complex* B, const int* ldb) {
    ztrsm_(side, uplo, ta, diag, m, n, alpha, A, lda, B, ldb);
}

// Eigen-decomposition
inline void syev(const char* jobz, const char* uplo, const int* n,
                 double* A, const int* lda, double* w, int* info) {
    int lwork = -1;
    double work_query;
    dsyev_(jobz, uplo, n, A, lda, w, &work_query, &lwork, info);
    lwork = static_cast<int>(work_query);
    std::vector<double> work(lwork);
    dsyev_(jobz, uplo, n, A, lda, w, work.data(), &lwork, info);
}
inline void syev(const char* jobz, const char* uplo, const int* n,
                 Complex* A, const int* lda, double* w, int* info) {
    int lwork = -1;
    Complex work_query;
    std::vector<double> rwork(std::max(1, 3 * (*n) - 2));
    zheev_(jobz, uplo, n, A, lda, w, &work_query, &lwork, rwork.data(), info);
    lwork = static_cast<int>(work_query.real());
    std::vector<Complex> work(lwork);
    zheev_(jobz, uplo, n, A, lda, w, work.data(), &lwork, rwork.data(), info);
}

// Transpose character for gemm: 'T' for real, 'C' for complex
template<typename T> constexpr char trans_char() { return 'T'; }
template<> constexpr char trans_char<Complex>() { return 'C'; }

// Scalar constants
template<typename T> inline T one() { return T(1.0); }
template<> inline Complex one<Complex>() { return Complex(1.0, 0.0); }
template<typename T> inline T zero() { return T(0.0); }
template<> inline Complex zero<Complex>() { return Complex(0.0, 0.0); }
template<typename T> inline T make_scalar(double v) { return T(v); }
template<> inline Complex make_scalar<Complex>(double v) { return Complex(v, 0.0); }

// Symmetrize: real -> average off-diag, complex -> hermitianize
inline void symmetrize(double* Hs, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) {
            double avg = 0.5 * (Hs[i + j * N] + Hs[j + i * N]);
            Hs[i + j * N] = avg;
            Hs[j + i * N] = avg;
        }
}
inline void symmetrize(Complex* Hs, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            Complex avg = 0.5 * (Hs[i + j * N] + std::conj(Hs[j + i * N]));
            Hs[i + j * N] = avg;
            Hs[j + i * N] = std::conj(avg);
        }
        Hs[i + i * N] = Complex(Hs[i + i * N].real(), 0.0);
    }
}

// MPI type dispatch
inline MPI_Datatype mpi_type(const double*) { return MPI_DOUBLE; }
inline MPI_Datatype mpi_type(const Complex*) { return MPI_C_DOUBLE_COMPLEX; }

} // namespace blas

EigenSolver::~EigenSolver() {
#ifdef USE_SCALAPACK
    cleanup_blacs();
#endif
}

void EigenSolver::setup(const LynxContext& ctx, const Hamiltonian& H) {
    H_ = &H;
    halo_ = &ctx.halo();
    domain_ = &ctx.domain();
    bandcomm_ = &ctx.scf_bandcomm();

    npband_ = bandcomm_->is_null() ? 1 : bandcomm_->size();
    band_rank_ = bandcomm_->is_null() ? 0 : bandcomm_->rank();
    int Nband_global = ctx.Nstates();
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

// ===== Template-unified implementations =====

template<typename T>
void EigenSolver::chebyshev_filter_impl(const T* X, T* Y, const double* Veff,
                                         int Nd_d, int Nband,
                                         double lambda_cutoff, double eigval_min, double eigval_max,
                                         int degree,
                                         const Vec3& kpt_cart, const Vec3& cell_lengths) {
    double e = (eigval_max - lambda_cutoff) / 2.0;
    double c = (eigval_max + lambda_cutoff) / 2.0;
    double sigma_1 = e / (eigval_min - c);
    double sigma = sigma_1;

    std::vector<T> Xold(Nd_d * Nband);
    std::vector<T> Xnew(Nd_d * Nband);
    std::vector<T> HX(Nd_d * Nband);

    // Y = (H*X - c*X) * (sigma/e)
    if constexpr (std::is_same_v<T, Complex>) {
        H_->apply_kpt(X, Veff, HX.data(), Nband, kpt_cart, cell_lengths);
    } else {
        H_->apply(X, Veff, HX.data(), Nband);
    }

    int total = Nd_d * Nband;
    double scale = sigma / e;
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total; ++idx) {
        Y[idx] = scale * (HX[idx] - c * X[idx]);
    }

    std::memcpy(Xold.data(), X, Nd_d * Nband * sizeof(T));

    T* pY = Y;
    T* pXold = Xold.data();
    T* pXnew = Xnew.data();

    for (int k = 2; k <= degree; ++k) {
        double sigma_new = 1.0 / (2.0 / sigma_1 - sigma);
        double gamma = 2.0 * sigma_new / e;

        if constexpr (std::is_same_v<T, Complex>) {
            H_->apply_kpt(pY, Veff, HX.data(), Nband, kpt_cart, cell_lengths);
        } else {
            H_->apply(pY, Veff, HX.data(), Nband);
        }

        double ss = sigma * sigma_new;
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < total; ++idx) {
            pXnew[idx] = gamma * (HX[idx] - c * pY[idx]) - ss * pXold[idx];
        }

        T* tmp = pXold;
        pXold = pY;
        pY = pXnew;
        pXnew = tmp;
        sigma = sigma_new;
    }

    if (pY != Y) {
        std::memcpy(Y, pY, Nd_d * Nband * sizeof(T));
    }
}

template<typename T>
void EigenSolver::orthogonalize_impl(T* X, int Nd_d, int Nband, double dV) {
    std::vector<T> S(Nband * Nband, blas::zero<T>());

    char transH = blas::trans_char<T>(), transN = 'N';
    T alpha = blas::make_scalar<T>(dV), beta = blas::zero<T>();
    blas::gemm(&transH, &transN, &Nband, &Nband, &Nd_d,
               &alpha, X, &Nd_d, X, &Nd_d, &beta, S.data(), &Nband);

    char uplo = 'U';
    int info;
    blas::potrf(&uplo, &Nband, S.data(), &Nband, &info);
    if (info != 0) {
        throw std::runtime_error("Cholesky factorization failed in orthogonalize (info=" + std::to_string(info) + ")");
    }

    char side = 'R', diag = 'N';
    T one = blas::one<T>();
    blas::trsm(&side, &uplo, &transN, &diag, &Nd_d, &Nband, &one,
               S.data(), &Nband, X, &Nd_d);
}

template<typename T>
void EigenSolver::project_hamiltonian_impl(const T* X, const double* Veff,
                                            T* Hs, int Nd_d, int Nband, double dV,
                                            const Vec3& kpt_cart, const Vec3& cell_lengths) {
    std::vector<T> HX(Nd_d * Nband);
    if constexpr (std::is_same_v<T, Complex>) {
        H_->apply_kpt(X, Veff, HX.data(), Nband, kpt_cart, cell_lengths);
    } else {
        H_->apply(X, Veff, HX.data(), Nband);
    }

    char transH = blas::trans_char<T>(), transN = 'N';
    T alpha = blas::make_scalar<T>(dV), beta = blas::zero<T>();
    blas::gemm(&transH, &transN, &Nband, &Nband, &Nd_d,
               &alpha, X, &Nd_d, HX.data(), &Nd_d, &beta, Hs, &Nband);

    blas::symmetrize(Hs, Nband);
}

template<typename T>
void EigenSolver::diag_subspace_impl(T* Hs, double* eigvals, int N) {
    char jobz = 'V', uplo = 'U';
    int info;
    blas::syev(&jobz, &uplo, &N, Hs, &N, eigvals, &info);
    if (info != 0) {
        std::fprintf(stderr, "Eigen-decomposition failed with info=%d, N=%d\n", info, N);
        throw std::runtime_error("Eigen-decomposition failed in diag_subspace");
    }
}

template<typename T>
void EigenSolver::rotate_orbitals_impl(T* X, const T* Q, int Nd_d, int Nband) {
    std::vector<T> X_new(Nd_d * Nband);
    char transN = 'N';
    T one = blas::one<T>(), zero = blas::zero<T>();
    blas::gemm(&transN, &transN, &Nd_d, &Nband, &Nband,
               &one, X, &Nd_d, Q, &Nband, &zero, X_new.data(), &Nd_d);
    std::memcpy(X, X_new.data(), Nd_d * Nband * sizeof(T));
}

// Explicit instantiations
template void EigenSolver::chebyshev_filter_impl<double>(const double*, double*, const double*,
    int, int, double, double, double, int, const Vec3&, const Vec3&);
template void EigenSolver::chebyshev_filter_impl<Complex>(const Complex*, Complex*, const double*,
    int, int, double, double, double, int, const Vec3&, const Vec3&);
template void EigenSolver::orthogonalize_impl<double>(double*, int, int, double);
template void EigenSolver::orthogonalize_impl<Complex>(Complex*, int, int, double);
template void EigenSolver::project_hamiltonian_impl<double>(const double*, const double*, double*, int, int, double, const Vec3&, const Vec3&);
template void EigenSolver::project_hamiltonian_impl<Complex>(const Complex*, const double*, Complex*, int, int, double, const Vec3&, const Vec3&);
template void EigenSolver::diag_subspace_impl<double>(double*, double*, int);
template void EigenSolver::diag_subspace_impl<Complex>(Complex*, double*, int);
template void EigenSolver::rotate_orbitals_impl<double>(double*, const double*, int, int);
template void EigenSolver::rotate_orbitals_impl<Complex>(Complex*, const Complex*, int, int);

// ===== Original methods now delegate to template implementations =====

void EigenSolver::chebyshev_filter(const double* X, double* Y, const double* Veff,
                                    int Nd_d, int Nband,
                                    double lambda_cutoff, double eigval_min, double eigval_max,
                                    int degree) {
    chebyshev_filter_impl<double>(X, Y, Veff, Nd_d, Nband,
                                  lambda_cutoff, eigval_min, eigval_max, degree);
}

void EigenSolver::orthogonalize(double* X, int Nd_d, int Nband, double dV) {
    orthogonalize_impl<double>(X, Nd_d, Nband, dV);
}

void EigenSolver::project_hamiltonian(const double* X, const double* Veff,
                                       double* Hs, int Nd_d, int Nband, double dV) {
    project_hamiltonian_impl<double>(X, Veff, Hs, Nd_d, Nband, dV);
}

void EigenSolver::diag_subspace(double* Hs, double* eigvals, int N) {
    diag_subspace_impl<double>(Hs, eigvals, N);
}

void EigenSolver::rotate_orbitals(double* X, const double* Q, int Nd_d, int Nband) {
    rotate_orbitals_impl<double>(X, Q, Nd_d, Nband);
}

void EigenSolver::lanczos_bounds(const double* Veff, int Nd_d,
                                  double& eigval_min, double& eigval_max,
                                  double tol_lanczos, int max_iter) {
    auto matvec = [&](const double* x, double* y) {
        H_->apply(x, Veff, y, 1);
    };
    lynx::lanczos_bounds<double>(matvec, Nd_d, eigval_min, eigval_max,
                                  tol_lanczos, max_iter);
}

// ===========================================================================
// Template-unified solve
// ===========================================================================

template<typename T>
void EigenSolver::solve_impl(T* psi, double* eigvals, const double* Veff,
                              int Nd_d, int Nband,
                              double lambda_cutoff, double eigval_min, double eigval_max,
                              int cheb_degree, int ld,
                              const Vec3& kpt_cart, const Vec3& cell_lengths) {
    if (ld == 0) ld = Nd_d;
    double dV = domain_->global_grid().dV();
    int Nband_loc = Nband;

    // Pack from NDArray layout (stride=ld) to packed layout (stride=Nd_d)
    std::vector<T> psi_packed;
    T* psi_work = psi;
    if (ld != Nd_d) {
        psi_packed.resize(Nd_d * Nband_loc);
        for (int j = 0; j < Nband_loc; ++j)
            std::memcpy(psi_packed.data() + j * Nd_d, psi + j * ld, Nd_d * sizeof(T));
        psi_work = psi_packed.data();
    }

    // Step 1: Chebyshev filter
    std::vector<T> Y(Nd_d * Nband_loc);
    chebyshev_filter_impl<T>(psi_work, Y.data(), Veff, Nd_d, Nband_loc,
                              lambda_cutoff, eigval_min, eigval_max, cheb_degree,
                              kpt_cart, cell_lengths);

#ifdef USE_SCALAPACK
    if (npband_ > 1) {
        // Band-parallel path
        if constexpr (std::is_same_v<T, Complex>) {
            orthogonalize_kpt_scalapack(Y.data(), Nd_d, Nband_loc, dV);

            int N = Nband_global_;
            std::vector<double> eigs_all(N);
            project_hamiltonian_kpt_scalapack(Y.data(), Veff, eigs_all.data(), Nd_d, Nband_loc, dV,
                                              kpt_cart, cell_lengths);

            rotate_orbitals_kpt_scalapack(Y.data(), Nd_d, Nband_loc);

            std::memcpy(eigvals, eigs_all.data(), N * sizeof(double));
        } else {
            orthogonalize_scalapack(Y.data(), Nd_d, Nband_loc, dV);

            int N = Nband_global_;
            std::vector<double> eigs_all(N);
            project_hamiltonian_scalapack(Y.data(), Veff, eigs_all.data(), Nd_d, Nband_loc, dV);

            rotate_orbitals_scalapack(Y.data(), Nd_d, Nband_loc);

            std::memcpy(eigvals, eigs_all.data(), N * sizeof(double));
        }
    } else
#endif
    {
        // Serial path
        orthogonalize_impl<T>(Y.data(), Nd_d, Nband_loc, dV);

        std::vector<T> Hs(Nband_loc * Nband_loc);
        project_hamiltonian_impl<T>(Y.data(), Veff, Hs.data(), Nd_d, Nband_loc, dV,
                                     kpt_cart, cell_lengths);

        std::vector<double> eigs(Nband_loc);
        diag_subspace_impl<T>(Hs.data(), eigs.data(), Nband_loc);

        rotate_orbitals_impl<T>(Y.data(), Hs.data(), Nd_d, Nband_loc);

        std::memcpy(eigvals, eigs.data(), Nband_loc * sizeof(double));
    }

    // Unpack from packed layout (stride=Nd_d) back to NDArray layout (stride=ld)
    if (ld != Nd_d) {
        for (int j = 0; j < Nband_loc; ++j)
            std::memcpy(psi + j * ld, Y.data() + j * Nd_d, Nd_d * sizeof(T));
    } else {
        std::memcpy(psi, Y.data(), Nd_d * Nband_loc * sizeof(T));
    }

    // Update lambda_cutoff: use last global eigenvalue
    int N_eig = (npband_ > 1) ? Nband_global_ : Nband_loc;
    lambda_cutoff_ = eigvals[N_eig - 1] + 0.1;
}

// Explicit instantiations for solve_impl
template void EigenSolver::solve_impl<double>(double*, double*, const double*,
    int, int, double, double, double, int, int, const Vec3&, const Vec3&);
template void EigenSolver::solve_impl<Complex>(Complex*, double*, const double*,
    int, int, double, double, double, int, int, const Vec3&, const Vec3&);

void EigenSolver::solve(double* psi, double* eigvals, const double* Veff,
                        int Nd_d, int Nband,
                        double lambda_cutoff, double eigval_min, double eigval_max,
                        int cheb_degree, int ld) {
    solve_impl<double>(psi, eigvals, Veff, Nd_d, Nband,
                        lambda_cutoff, eigval_min, eigval_max, cheb_degree, ld);
}

void EigenSolver::solve_kpt(Complex* psi, double* eigvals, const double* Veff,
                             int Nd_d, int Nband,
                             double lambda_cutoff, double eigval_min, double eigval_max,
                             const Vec3& kpt_cart, const Vec3& cell_lengths,
                             int cheb_degree, int ld) {
    solve_impl<Complex>(psi, eigvals, Veff, Nd_d, Nband,
                         lambda_cutoff, eigval_min, eigval_max, cheb_degree, ld,
                         kpt_cart, cell_lengths);
}

// ===========================================================================
// Complex (k-point) thin wrappers (used by spinor solver)
// ===========================================================================

void EigenSolver::chebyshev_filter_kpt(const Complex* X, Complex* Y, const double* Veff,
                                        int Nd_d, int Nband,
                                        double lambda_cutoff, double eigval_min, double eigval_max,
                                        const Vec3& kpt_cart, const Vec3& cell_lengths,
                                        int degree) {
    chebyshev_filter_impl<Complex>(X, Y, Veff, Nd_d, Nband,
                                   lambda_cutoff, eigval_min, eigval_max, degree,
                                   kpt_cart, cell_lengths);
}

void EigenSolver::orthogonalize_kpt(Complex* X, int Nd_d, int Nband, double dV) {
    orthogonalize_impl<Complex>(X, Nd_d, Nband, dV);
}

void EigenSolver::project_hamiltonian_kpt(const Complex* X, const double* Veff,
                                           Complex* Hs, int Nd_d, int Nband, double dV,
                                           const Vec3& kpt_cart, const Vec3& cell_lengths) {
    project_hamiltonian_impl<Complex>(X, Veff, Hs, Nd_d, Nband, dV, kpt_cart, cell_lengths);
}

void EigenSolver::diag_subspace_kpt(Complex* Hs, double* eigvals, int N) {
    diag_subspace_impl<Complex>(Hs, eigvals, N);
}

void EigenSolver::rotate_orbitals_kpt(Complex* X, const Complex* Q, int Nd_d, int Nband) {
    rotate_orbitals_impl<Complex>(X, Q, Nd_d, Nband);
}

void EigenSolver::lanczos_bounds_kpt(const double* Veff, int Nd_d,
                                      const Vec3& kpt_cart, const Vec3& cell_lengths,
                                      double& eigval_min, double& eigval_max,
                                      double tol_lanczos, int max_iter) {
    auto matvec = [&](const Complex* x, Complex* y) {
        H_->apply_kpt(x, Veff, y, 1, kpt_cart, cell_lengths);
    };
    lynx::lanczos_bounds<Complex>(matvec, Nd_d, eigval_min, eigval_max,
                                   tol_lanczos, max_iter);
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

        int total_spinor = Nd_d_spinor * Nband_loc;
        double scale = sigma / e;
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < total_spinor; ++idx) {
            Y[idx] = scale * (HX[idx] - c_cheb * psi_work[idx]);
        }

        std::memcpy(Xold.data(), psi_work, Nd_d_spinor * Nband_loc * sizeof(Complex));

        // Use pointer swap instead of memcpy in iteration loop
        Complex* pY_s = Y.data();
        Complex* pXold_s = Xold.data();
        Complex* pXnew_s = Xnew.data();

        for (int k = 2; k <= cheb_degree; ++k) {
            double sigma_new = 1.0 / (2.0 / sigma_1 - sigma);
            double gamma = 2.0 * sigma_new / e;

            H_->apply_spinor_kpt(pY_s, Veff_spinor, HX.data(), Nband_loc, Nd_d,
                                  kpt_cart, cell_lengths);

            double ss = sigma * sigma_new;
            #pragma omp parallel for schedule(static)
            for (int idx = 0; idx < total_spinor; ++idx) {
                pXnew_s[idx] = gamma * (HX[idx] - c_cheb * pY_s[idx]) - ss * pXold_s[idx];
            }

            // Pointer swap
            Complex* tmp_s = pXold_s;
            pXold_s = pY_s;
            pY_s = pXnew_s;
            pXnew_s = tmp_s;
            sigma = sigma_new;

        }

        // Copy result back to Y vector if needed
        if (pY_s != Y.data()) {
            std::memcpy(Y.data(), pY_s, Nd_d_spinor * Nband_loc * sizeof(Complex));
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
    int Nd_spinor = 2 * Nd_d;
    auto matvec = [&](const Complex* x, Complex* y) {
        H_->apply_spinor_kpt(x, Veff_spinor, y, 1, Nd_d, kpt_cart, cell_lengths);
    };
    lynx::lanczos_bounds<Complex>(matvec, Nd_spinor, eigval_min, eigval_max,
                                   tol_lanczos, max_iter);
}

} // namespace lynx
