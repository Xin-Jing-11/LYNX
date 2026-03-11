#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "operators/Hamiltonian.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"

namespace sparc {

// Chebyshev-filtered subspace iteration eigensolver.
// 1. Chebyshev filter to expand subspace
// 2. Orthogonalization (Cholesky QR)
// 3. Subspace Hamiltonian (Rayleigh-Ritz)
// 4. Diagonalize subspace Hamiltonian
// 5. Rotate orbitals
class EigenSolver {
public:
    EigenSolver() = default;

    void setup(const Hamiltonian& H,
               const HaloExchange& halo,
               const Domain& domain,
               const MPIComm& dmcomm,
               const MPIComm& bandcomm);

    // Chebyshev-filtered subspace iteration.
    // psi: (Nd_d, Nband) orbitals — updated in place
    // eigvals: (Nband,) eigenvalues — updated
    // Veff: effective potential on local domain
    // lambda_cutoff: upper bound of unwanted spectrum
    // eigval_min: lower bound of wanted spectrum
    // eigval_max: upper bound of full spectrum
    // cheb_degree: polynomial degree for filter
    void solve(double* psi, double* eigvals, const double* Veff,
               int Nd_d, int Nband,
               double lambda_cutoff, double eigval_min, double eigval_max,
               int cheb_degree = 20);

    // Estimate spectral bounds using Lanczos (tolerance-based stopping)
    // Reference: Lanczos() in eigenSolver.c
    void lanczos_bounds(const double* Veff, int Nd_d,
                        double& eigval_min, double& eigval_max,
                        double tol_lanczos = 1e-2, int max_iter = 1000);

    // Get lambda_cutoff (upper bound of wanted eigenvalues)
    // Typically set from eigenvalues of previous SCF iteration
    double lambda_cutoff() const { return lambda_cutoff_; }
    void set_lambda_cutoff(double lc) { lambda_cutoff_ = lc; }

private:
    const Hamiltonian* H_ = nullptr;
    const HaloExchange* halo_ = nullptr;
    const Domain* domain_ = nullptr;
    const MPIComm* dmcomm_ = nullptr;
    const MPIComm* bandcomm_ = nullptr;
    double lambda_cutoff_ = 0.0;

    // Chebyshev filter: Y = T_m((H - c)/e) * X
    // where c = (lambda_cutoff + eigval_min)/2, e = (lambda_cutoff - eigval_min)/2
    void chebyshev_filter(const double* X, double* Y, const double* Veff,
                          int Nd_d, int Nband,
                          double lambda_cutoff, double eigval_min, double eigval_max,
                          int degree);

    // Orthogonalize columns of X via Cholesky QR: X = Q*R
    void orthogonalize(double* X, int Nd_d, int Nband, double dV);

    // Project: Hs = X^T * H * X (subspace Hamiltonian)
    void project_hamiltonian(const double* X, const double* Veff,
                             double* Hs, int Nd_d, int Nband, double dV);

    // Diagonalize dense symmetric matrix Hs -> eigvecs, eigvals
    void diag_subspace(double* Hs, double* eigvals, int N);

    // Rotate: X_new = X * Q where Q are eigenvectors of Hs
    void rotate_orbitals(double* X, const double* Q, int Nd_d, int Nband);
};

} // namespace sparc
