#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "operators/Laplacian.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"

#include <functional>

namespace sparc {

// Alternating Anderson-Richardson (AAR) iterative solver.
// Solves: A*x = b  where A is typically the Laplacian operator.
//
// The operator is provided as a callback: op(x, Ax)
// Preconditioner is also a callback: precond(r, z)
struct AARParams {
    double omega = 0.6;     // Richardson relaxation parameter
    double beta = 0.6;      // Anderson mixing parameter
    int m = 7;              // Anderson history depth
    int p = 6;              // Anderson update frequency (every p iterations)
    double tol = 1e-8;      // Convergence tolerance (relative residual)
    int max_iter = 3000;
};

class LinearSolver {
public:
    using OpFunc = std::function<void(const double* x, double* Ax)>;
    using PrecondFunc = std::function<void(const double* r, double* z)>;

    // AAR solver: solves A*x = b
    // Returns number of iterations (negative if not converged)
    // Optional preconditioner: precond(r, z) applies z = M^{-1} * r
    static int aar(const OpFunc& op,
                   const double* b,
                   double* x,
                   int N,
                   const AARParams& params,
                   const MPIComm& comm,
                   const PrecondFunc* precond = nullptr);

    // CG solver: solves A*x = b with optional preconditioner
    static int cg(const OpFunc& op,
                  const PrecondFunc* precond,
                  const double* b,
                  double* x,
                  int N,
                  double tol,
                  int max_iter,
                  const MPIComm& comm);

    // Dot product with MPI allreduce
    static double dot(const double* a, const double* b, int N, const MPIComm& comm);
};

} // namespace sparc
