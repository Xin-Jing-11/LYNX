#include <gtest/gtest.h>
#include "solvers/LinearSolver.hpp"
#include "parallel/MPIComm.hpp"
#include <cmath>
#include <vector>

using namespace sparc;

TEST(LinearSolver, CGSolveDiagonal) {
    // Solve D*x = b where D is a diagonal matrix
    int N = 50;
    std::vector<double> diag(N);
    std::vector<double> b(N), x(N, 0.0);

    for (int i = 0; i < N; ++i) {
        diag[i] = 2.0 + i * 0.1;
        b[i] = diag[i] * (i + 1.0);  // exact solution: x = i+1
    }

    auto op = [&](const double* xin, double* Axout) {
        for (int i = 0; i < N; ++i) {
            Axout[i] = diag[i] * xin[i];
        }
    };

    MPIComm null_comm;
    int iters = LinearSolver::cg(op, nullptr, b.data(), x.data(), N, 1e-10, 200, null_comm);

    EXPECT_GE(iters, 0);  // converged
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(x[i], i + 1.0, 1e-7);
    }
}

TEST(LinearSolver, CGSolveTridiagonal) {
    // Solve a tridiagonal system: -x_{i-1} + 2*x_i - x_{i+1} = b_i
    int N = 100;
    std::vector<double> b(N), x(N, 0.0);

    // Known solution: x_i = sin(pi * i / N)
    std::vector<double> x_exact(N);
    for (int i = 0; i < N; ++i) {
        x_exact[i] = std::sin(M_PI * (i + 1) / (N + 1));
    }

    auto op = [N](const double* xin, double* Axout) {
        for (int i = 0; i < N; ++i) {
            Axout[i] = 2.0 * xin[i];
            if (i > 0) Axout[i] -= xin[i - 1];
            if (i < N - 1) Axout[i] -= xin[i + 1];
        }
    };

    // Compute RHS
    op(x_exact.data(), b.data());

    MPIComm null_comm;
    int iters = LinearSolver::cg(op, nullptr, b.data(), x.data(), N, 1e-10, 500, null_comm);

    EXPECT_GE(iters, 0);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(x[i], x_exact[i], 1e-6);
    }
}

TEST(LinearSolver, AARSolveDiagonal) {
    int N = 30;
    std::vector<double> diag(N);
    std::vector<double> b(N), x(N, 0.0);

    for (int i = 0; i < N; ++i) {
        diag[i] = 3.0 + i * 0.05;
        b[i] = diag[i] * 1.0;  // exact solution: x = 1
    }

    auto op = [&](const double* xin, double* Axout) {
        for (int i = 0; i < N; ++i) {
            Axout[i] = diag[i] * xin[i];
        }
    };

    AARParams params;
    params.omega = 0.3;
    params.tol = 1e-8;
    params.max_iter = 500;

    MPIComm null_comm;
    int iters = LinearSolver::aar(op, b.data(), x.data(), N, params, null_comm);

    EXPECT_GE(iters, 0);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(x[i], 1.0, 1e-4);
    }
}

TEST(LinearSolver, DotProduct) {
    int N = 100;
    std::vector<double> a(N, 1.0), b(N, 2.0);
    MPIComm null_comm;
    double result = LinearSolver::dot(a.data(), b.data(), N, null_comm);
    EXPECT_NEAR(result, 200.0, 1e-10);
}
