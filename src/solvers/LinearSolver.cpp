#include "solvers/LinearSolver.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

namespace sparc {

double LinearSolver::dot(const double* a, const double* b, int N, const MPIComm& comm) {
    double local_dot = 0.0;
    for (int i = 0; i < N; ++i) {
        local_dot += a[i] * b[i];
    }
    if (!comm.is_null() && comm.size() > 1) {
        return comm.allreduce_sum(local_dot);
    }
    return local_dot;
}

int LinearSolver::aar(const OpFunc& op,
                       const double* b,
                       double* x,
                       int N,
                       const AARParams& params,
                       const MPIComm& comm,
                       const PrecondFunc* precond) {
    // Reference: linearSolver.c AAR()
    // Uses Jacobi preconditioner inside AAR loop.
    // f = M^{-1} * r (preconditioned residual)
    // History stores differences of preconditioned residuals (F) and iterates (X)
    // Richardson step: x = x_old + omega * f
    // Anderson step: uses x_old, f, X, F

    int m = params.m;
    int p = params.p;

    std::vector<double> r(N);           // residual r = b - A*x
    std::vector<double> f(N, 0.0);      // preconditioned residual f = M^{-1} * r
    std::vector<double> f_old(N, 0.0);
    std::vector<double> x_old(N, 0.0);

    // History matrices (column-major N x m)
    std::vector<double> X(N * m, 0.0);  // X(:,j) = x_k - x_{k-1}
    std::vector<double> F(N * m, 0.0);  // F(:,j) = f_k - f_{k-1}

    // Initialize x_old
    std::memcpy(x_old.data(), x, N * sizeof(double));

    double b_2norm = std::sqrt(dot(b, b, N, comm));

    // Compute initial residual using the res_fun convention:
    // Reference: res_fun computes r = b + (Lap+c)*x, but our op computes A*x = -(Lap+c)*x
    // So r = b - op(x)
    {
        std::vector<double> Ax(N);
        op(x, Ax.data());
        for (int i = 0; i < N; ++i)
            r[i] = b[i] - Ax[i];
    }

    // Replace absolute tol: tol * ||b|| (matching reference)
    double abs_tol = params.tol * b_2norm;
    double r_2norm = abs_tol + 1.0; // skip initial norm check (reference does this)

    int iter_count = 0;
    while (r_2norm > abs_tol && iter_count < params.max_iter) {
        // Apply preconditioner: f = M^{-1} * r
        if (precond) {
            (*precond)(r.data(), f.data());
        } else {
            std::memcpy(f.data(), r.data(), N * sizeof(double));
        }

        // Store history: X(:,i_hist) = x - x_old, F(:,i_hist) = f - f_old
        if (iter_count > 0) {
            int i_hist = (iter_count - 1) % m;
            for (int i = 0; i < N; ++i) {
                X[i_hist * N + i] = x[i] - x_old[i];
                F[i_hist * N + i] = f[i] - f_old[i];
            }
        }

        // Save current state
        std::memcpy(x_old.data(), x, N * sizeof(double));
        std::memcpy(f_old.data(), f.data(), N * sizeof(double));

        if ((iter_count + 1) % p == 0 && iter_count > 0) {
            // Anderson extrapolation: x = x_old + beta*f - sum gamma_j*(DX_j + beta*DF_j)
            int cols = std::min(iter_count, m);

            // Solve: (F^T F) gamma = F^T f
            std::vector<double> FTF(cols * cols, 0.0);
            std::vector<double> gamma(cols, 0.0);

            for (int ii = 0; ii < cols; ++ii) {
                double* Fi = F.data() + ii * N;
                gamma[ii] = dot(Fi, f.data(), N, comm); // F^T * f
                for (int jj = 0; jj <= ii; ++jj) {
                    double* Fj = F.data() + jj * N;
                    FTF[ii * cols + jj] = dot(Fi, Fj, N, comm);
                    FTF[jj * cols + ii] = FTF[ii * cols + jj];
                }
            }

            // Solve via Gaussian elimination
            {
                std::vector<double> A(FTF);
                for (int k = 0; k < cols; ++k) {
                    int pivot = k;
                    for (int ii = k + 1; ii < cols; ++ii)
                        if (std::abs(A[ii * cols + k]) > std::abs(A[pivot * cols + k]))
                            pivot = ii;
                    if (pivot != k) {
                        for (int j = 0; j < cols; ++j)
                            std::swap(A[k * cols + j], A[pivot * cols + j]);
                        std::swap(gamma[k], gamma[pivot]);
                    }
                    double d = A[k * cols + k];
                    if (std::abs(d) < 1e-14) continue;
                    for (int ii = k + 1; ii < cols; ++ii) {
                        double factor = A[ii * cols + k] / d;
                        for (int j = k + 1; j < cols; ++j)
                            A[ii * cols + j] -= factor * A[k * cols + j];
                        gamma[ii] -= factor * gamma[k];
                    }
                }
                for (int k = cols - 1; k >= 0; --k) {
                    if (std::abs(A[k * cols + k]) < 1e-14) continue;
                    for (int j = k + 1; j < cols; ++j)
                        gamma[k] -= A[k * cols + j] * gamma[j];
                    gamma[k] /= A[k * cols + k];
                }
            }

            // Reference: AndersonExtrapolation(N, m, x, x_old, f, X, F, beta, comm)
            // x = x_old + beta*f - sum gamma_j*(X_j + beta*F_j)
            // (Using x_old, not current x)
            for (int i = 0; i < N; ++i)
                x[i] = x_old[i] + params.beta * f[i];
            for (int j = 0; j < cols; ++j) {
                double* Xj = X.data() + j * N;
                double* Fj = F.data() + j * N;
                double gj = gamma[j];
                for (int i = 0; i < N; ++i)
                    x[i] -= gj * (Xj[i] + params.beta * Fj[i]);
            }

            // Recompute residual
            {
                std::vector<double> Ax(N);
                op(x, Ax.data());
                for (int i = 0; i < N; ++i)
                    r[i] = b[i] - Ax[i];
            }
            r_2norm = std::sqrt(dot(r.data(), r.data(), N, comm));
        } else {
            // Richardson update: x = x_old + omega * f
            for (int i = 0; i < N; ++i)
                x[i] = x_old[i] + params.omega * f[i];

            // Recompute residual
            {
                std::vector<double> Ax(N);
                op(x, Ax.data());
                for (int i = 0; i < N; ++i)
                    r[i] = b[i] - Ax[i];
            }
        }

        iter_count++;
    }

    return iter_count;
}

int LinearSolver::cg(const OpFunc& op,
                      const PrecondFunc* precond,
                      const double* b,
                      double* x,
                      int N,
                      double tol,
                      int max_iter,
                      const MPIComm& comm) {
    std::vector<double> r(N), p(N), Ap(N), z(N);

    // r = b - A*x
    op(x, Ap.data());
    for (int i = 0; i < N; ++i) {
        r[i] = b[i] - Ap[i];
    }

    double b_norm = std::sqrt(dot(b, b, N, comm));
    if (b_norm < 1e-30) b_norm = 1.0;

    // z = M^{-1} r (or z = r if no preconditioner)
    if (precond) {
        (*precond)(r.data(), z.data());
    } else {
        std::memcpy(z.data(), r.data(), N * sizeof(double));
    }

    std::memcpy(p.data(), z.data(), N * sizeof(double));
    double rz = dot(r.data(), z.data(), N, comm);

    for (int iter = 0; iter < max_iter; ++iter) {
        double r_norm = std::sqrt(dot(r.data(), r.data(), N, comm));
        if (r_norm / b_norm < tol) {
            return iter;
        }

        op(p.data(), Ap.data());
        double pAp = dot(p.data(), Ap.data(), N, comm);
        if (std::abs(pAp) < 1e-30) return -(iter + 1);

        double alpha = rz / pAp;

        for (int i = 0; i < N; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        if (precond) {
            (*precond)(r.data(), z.data());
        } else {
            std::memcpy(z.data(), r.data(), N * sizeof(double));
        }

        double rz_new = dot(r.data(), z.data(), N, comm);
        double beta = rz_new / rz;
        rz = rz_new;

        for (int i = 0; i < N; ++i) {
            p[i] = z[i] + beta * p[i];
        }
    }

    return -max_iter;
}

} // namespace sparc
