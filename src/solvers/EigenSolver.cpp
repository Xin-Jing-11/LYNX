#include "solvers/EigenSolver.hpp"
#include "solvers/LinearSolver.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

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
}

namespace sparc {

void EigenSolver::setup(const Hamiltonian& H,
                         const HaloExchange& halo,
                         const Domain& domain,
                         const MPIComm& bandcomm) {
    H_ = &H;
    halo_ = &halo;
    domain_ = &domain;
    bandcomm_ = &bandcomm;
}

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

    // Pack from NDArray layout (stride=ld) to packed layout (stride=Nd_d)
    std::vector<double> psi_packed;
    double* psi_work = psi;
    if (ld != Nd_d) {
        psi_packed.resize(Nd_d * Nband);
        for (int j = 0; j < Nband; ++j)
            std::memcpy(psi_packed.data() + j * Nd_d, psi + j * ld, Nd_d * sizeof(double));
        psi_work = psi_packed.data();
    }

    // Step 1: Chebyshev filter
    std::vector<double> Y(Nd_d * Nband);
    chebyshev_filter(psi_work, Y.data(), Veff, Nd_d, Nband,
                     lambda_cutoff, eigval_min, eigval_max, cheb_degree);

    // Step 2: Orthogonalize filtered vectors
    orthogonalize(Y.data(), Nd_d, Nband, dV);

    // Step 3: Project Hamiltonian onto subspace
    std::vector<double> Hs(Nband * Nband);
    project_hamiltonian(Y.data(), Veff, Hs.data(), Nd_d, Nband, dV);

    // Step 4: Diagonalize subspace Hamiltonian
    std::vector<double> eigs(Nband);
    diag_subspace(Hs.data(), eigs.data(), Nband);

    // Step 5: Rotate orbitals
    rotate_orbitals(Y.data(), Hs.data(), Nd_d, Nband);

    // Unpack from packed layout (stride=Nd_d) back to NDArray layout (stride=ld)
    if (ld != Nd_d) {
        for (int j = 0; j < Nband; ++j)
            std::memcpy(psi + j * ld, Y.data() + j * Nd_d, Nd_d * sizeof(double));
    } else {
        std::memcpy(psi, Y.data(), Nd_d * Nband * sizeof(double));
    }
    std::memcpy(eigvals, eigs.data(), Nband * sizeof(double));

    // Update lambda_cutoff for next call
    lambda_cutoff_ = eigvals[Nband - 1] + 0.1;
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
        throw std::runtime_error("zpotrf failed in orthogonalize_kpt");
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

    // Pack from NDArray layout (stride=ld) to packed layout (stride=Nd_d)
    std::vector<Complex> psi_packed;
    Complex* psi_work = psi;
    if (ld != Nd_d) {
        psi_packed.resize(Nd_d * Nband);
        for (int j = 0; j < Nband; ++j)
            std::memcpy(psi_packed.data() + j * Nd_d, psi + j * ld, Nd_d * sizeof(Complex));
        psi_work = psi_packed.data();
    }

    // Step 1: Chebyshev filter
    std::vector<Complex> Y(Nd_d * Nband);
    chebyshev_filter_kpt(psi_work, Y.data(), Veff, Nd_d, Nband,
                         lambda_cutoff, eigval_min, eigval_max,
                         kpt_cart, cell_lengths, cheb_degree);

    // Step 2: Orthogonalize
    orthogonalize_kpt(Y.data(), Nd_d, Nband, dV);

    // Step 3: Project Hamiltonian
    std::vector<Complex> Hs(Nband * Nband);
    project_hamiltonian_kpt(Y.data(), Veff, Hs.data(), Nd_d, Nband, dV,
                            kpt_cart, cell_lengths);

    // Step 4: Diagonalize
    std::vector<double> eigs(Nband);
    diag_subspace_kpt(Hs.data(), eigs.data(), Nband);

    // Step 5: Rotate
    rotate_orbitals_kpt(Y.data(), Hs.data(), Nd_d, Nband);

    // Unpack from packed layout (stride=Nd_d) back to NDArray layout (stride=ld)
    if (ld != Nd_d) {
        for (int j = 0; j < Nband; ++j)
            std::memcpy(psi + j * ld, Y.data() + j * Nd_d, Nd_d * sizeof(Complex));
    } else {
        std::memcpy(psi, Y.data(), Nd_d * Nband * sizeof(Complex));
    }
    std::memcpy(eigvals, eigs.data(), Nband * sizeof(double));

    lambda_cutoff_ = eigvals[Nband - 1] + 0.1;
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

} // namespace sparc
