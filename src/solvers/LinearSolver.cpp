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
                       const MPIComm& comm) {
    std::vector<double> Ax(N), r(N), x_new(N);

    // Anderson history storage
    int m = params.m;
    int p = params.p;
    std::vector<std::vector<double>> DX(m, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> DF(m, std::vector<double>(N, 0.0));
    std::vector<double> f_old(N, 0.0);
    std::vector<double> x_old(N, 0.0);

    // Compute initial residual
    op(x, Ax.data());
    for (int i = 0; i < N; ++i) {
        r[i] = b[i] - Ax[i];
    }

    double b_norm = std::sqrt(dot(b, b, N, comm));
    if (b_norm < 1e-30) b_norm = 1.0;

    int anderson_count = 0;

    for (int iter = 0; iter < params.max_iter; ++iter) {
        double r_norm = std::sqrt(dot(r.data(), r.data(), N, comm));
        if (r_norm / b_norm < params.tol) {
            return iter;
        }

        if ((iter + 1) % p == 0 && anderson_count > 0) {
            // Anderson update step
            int cols = std::min(anderson_count, m);

            // Solve least-squares: DF^T * DF * gamma = DF^T * r
            // Small system: cols x cols
            std::vector<double> FTF(cols * cols, 0.0);
            std::vector<double> FTr(cols, 0.0);

            for (int i = 0; i < cols; ++i) {
                FTr[i] = dot(DF[i].data(), r.data(), N, comm);
                for (int j = 0; j <= i; ++j) {
                    FTF[i * cols + j] = dot(DF[i].data(), DF[j].data(), N, comm);
                    FTF[j * cols + i] = FTF[i * cols + j];
                }
            }

            // Solve small system via Cholesky or direct for small cols
            std::vector<double> gamma(cols, 0.0);
            // Simple Gaussian elimination for the small system
            std::vector<double> A_sys(FTF);
            std::vector<double> b_sys(FTr);
            for (int k = 0; k < cols; ++k) {
                // Partial pivoting
                int pivot = k;
                for (int i = k + 1; i < cols; ++i) {
                    if (std::abs(A_sys[i * cols + k]) > std::abs(A_sys[pivot * cols + k]))
                        pivot = i;
                }
                if (pivot != k) {
                    for (int j = 0; j < cols; ++j)
                        std::swap(A_sys[k * cols + j], A_sys[pivot * cols + j]);
                    std::swap(b_sys[k], b_sys[pivot]);
                }
                double diag = A_sys[k * cols + k];
                if (std::abs(diag) < 1e-14) continue;
                for (int i = k + 1; i < cols; ++i) {
                    double factor = A_sys[i * cols + k] / diag;
                    for (int j = k + 1; j < cols; ++j)
                        A_sys[i * cols + j] -= factor * A_sys[k * cols + j];
                    b_sys[i] -= factor * b_sys[k];
                }
            }
            for (int k = cols - 1; k >= 0; --k) {
                if (std::abs(A_sys[k * cols + k]) < 1e-14) continue;
                gamma[k] = b_sys[k];
                for (int j = k + 1; j < cols; ++j)
                    gamma[k] -= A_sys[k * cols + j] * gamma[j];
                gamma[k] /= A_sys[k * cols + k];
            }

            // Anderson update: x_new = x + beta*r - sum gamma_i * (DX_i + beta*DF_i)
            for (int i = 0; i < N; ++i) {
                x_new[i] = x[i] + params.beta * r[i];
            }
            for (int j = 0; j < cols; ++j) {
                for (int i = 0; i < N; ++i) {
                    x_new[i] -= gamma[j] * (DX[j][i] + params.beta * DF[j][i]);
                }
            }
            std::memcpy(x, x_new.data(), N * sizeof(double));
        } else {
            // Richardson relaxation: x = x + omega * r
            for (int i = 0; i < N; ++i) {
                x[i] += params.omega * r[i];
            }
        }

        // Store history for Anderson
        std::memcpy(x_old.data(), x, N * sizeof(double));

        // Recompute residual
        op(x, Ax.data());
        for (int i = 0; i < N; ++i) {
            r[i] = b[i] - Ax[i];
        }

        // Update DX and DF history (ring buffer)
        if (anderson_count > 0 || iter > 0) {
            int idx = (anderson_count < m) ? anderson_count : (anderson_count % m);
            if (anderson_count >= m) idx = anderson_count % m;
            for (int i = 0; i < N; ++i) {
                DX[idx][i] = x[i] - x_old[i];  // This is approximate; proper storage below
                DF[idx][i] = r[i] - f_old[i];
            }
            if (anderson_count < m) anderson_count++;
        }

        std::memcpy(f_old.data(), r.data(), N * sizeof(double));
        std::memcpy(x_old.data(), x, N * sizeof(double));
    }

    return -params.max_iter;  // not converged
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
