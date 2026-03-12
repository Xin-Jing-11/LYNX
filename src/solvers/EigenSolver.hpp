#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "operators/Hamiltonian.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"
#include <complex>

namespace sparc {

using Complex = std::complex<double>;

// Chebyshev-filtered subspace iteration eigensolver.
// Supports both real (Gamma-point) and complex (k-point) wavefunctions.
class EigenSolver {
public:
    EigenSolver() = default;

    void setup(const Hamiltonian& H,
               const HaloExchange& halo,
               const Domain& domain,
               const MPIComm& bandcomm);

    // --- Real (Gamma-point) interface ---

    void solve(double* psi, double* eigvals, const double* Veff,
               int Nd_d, int Nband,
               double lambda_cutoff, double eigval_min, double eigval_max,
               int cheb_degree = 20, int ld = 0);

    void lanczos_bounds(const double* Veff, int Nd_d,
                        double& eigval_min, double& eigval_max,
                        double tol_lanczos = 1e-2, int max_iter = 1000);

    // --- Complex (k-point) interface ---

    void solve_kpt(Complex* psi, double* eigvals, const double* Veff,
                   int Nd_d, int Nband,
                   double lambda_cutoff, double eigval_min, double eigval_max,
                   const Vec3& kpt_cart, const Vec3& cell_lengths,
                   int cheb_degree = 20, int ld = 0);

    void lanczos_bounds_kpt(const double* Veff, int Nd_d,
                            const Vec3& kpt_cart, const Vec3& cell_lengths,
                            double& eigval_min, double& eigval_max,
                            double tol_lanczos = 1e-2, int max_iter = 1000);

    double lambda_cutoff() const { return lambda_cutoff_; }
    void set_lambda_cutoff(double lc) { lambda_cutoff_ = lc; }

private:
    const Hamiltonian* H_ = nullptr;
    const HaloExchange* halo_ = nullptr;
    const Domain* domain_ = nullptr;
    const MPIComm* bandcomm_ = nullptr;
    double lambda_cutoff_ = 0.0;

    // --- Real private methods ---
    void chebyshev_filter(const double* X, double* Y, const double* Veff,
                          int Nd_d, int Nband,
                          double lambda_cutoff, double eigval_min, double eigval_max,
                          int degree);
    void orthogonalize(double* X, int Nd_d, int Nband, double dV);
    void project_hamiltonian(const double* X, const double* Veff,
                             double* Hs, int Nd_d, int Nband, double dV);
    void diag_subspace(double* Hs, double* eigvals, int N);
    void rotate_orbitals(double* X, const double* Q, int Nd_d, int Nband);

    // --- Complex private methods ---
    void chebyshev_filter_kpt(const Complex* X, Complex* Y, const double* Veff,
                              int Nd_d, int Nband,
                              double lambda_cutoff, double eigval_min, double eigval_max,
                              const Vec3& kpt_cart, const Vec3& cell_lengths,
                              int degree);
    void orthogonalize_kpt(Complex* X, int Nd_d, int Nband, double dV);
    void project_hamiltonian_kpt(const Complex* X, const double* Veff,
                                 Complex* Hs, int Nd_d, int Nband, double dV,
                                 const Vec3& kpt_cart, const Vec3& cell_lengths);
    void diag_subspace_kpt(Complex* Hs, double* eigvals, int N);
    void rotate_orbitals_kpt(Complex* X, const Complex* Q, int Nd_d, int Nband);
};

} // namespace sparc
