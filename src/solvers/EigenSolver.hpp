#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "operators/Hamiltonian.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"
#include <complex>

namespace lynx {

using Complex = std::complex<double>;

// Chebyshev-filtered subspace iteration eigensolver.
// Supports both real (Gamma-point) and complex (k-point) wavefunctions.
// When bandcomm has size > 1 and ScaLAPACK is available, uses distributed
// subspace operations for band parallelism.
class EigenSolver {
public:
    EigenSolver() = default;
    ~EigenSolver();

    // Setup with Nband_global for band-parallel mode.
    // bandcomm groups processes that share bands for the same (spin,kpt).
    // Nband_global: total number of bands across all band-parallel processes.
    void setup(const Hamiltonian& H,
               const HaloExchange& halo,
               const Domain& domain,
               const MPIComm& bandcomm,
               int Nband_global = 0);

    // --- Real (Gamma-point) interface ---
    // Nband here is the LOCAL band count (Nband_local).
    // In band-parallel mode, each process holds Nband_local columns of psi.

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

    // --- Spinor (SOC) interface ---
    // Solve for spinor wavefunctions (2*Nd_d rows per band)
    void solve_spinor_kpt(Complex* psi, double* eigvals, const double* Veff_spinor,
                          int Nd_d, int Nband,
                          double lambda_cutoff, double eigval_min, double eigval_max,
                          const Vec3& kpt_cart, const Vec3& cell_lengths,
                          int cheb_degree = 20, int ld = 0);

    void lanczos_bounds_spinor_kpt(const double* Veff_spinor, int Nd_d,
                                    const Vec3& kpt_cart, const Vec3& cell_lengths,
                                    double& eigval_min, double& eigval_max,
                                    double tol_lanczos = 1e-2, int max_iter = 1000);

    double lambda_cutoff() const { return lambda_cutoff_; }
    void set_lambda_cutoff(double lc) { lambda_cutoff_ = lc; }

    int Nband_global() const { return Nband_global_; }
    bool is_band_parallel() const { return npband_ > 1; }

private:
    const Hamiltonian* H_ = nullptr;
    const HaloExchange* halo_ = nullptr;
    const Domain* domain_ = nullptr;
    const MPIComm* bandcomm_ = nullptr;
    double lambda_cutoff_ = 0.0;

    // Band parallelism state
    int Nband_global_ = 0;   // total bands across all band procs
    int npband_ = 1;         // number of band-parallel processes
    int band_rank_ = 0;      // rank within bandcomm

#ifdef USE_SCALAPACK
    // BLACS context for distributed subspace matrices
    int blacs_ctxt_ = -1;
    int blacs_nprow_ = 0, blacs_npcol_ = 0;
    int blacs_myrow_ = -1, blacs_mycol_ = -1;
    int scalapack_nb_ = 64;  // block size for block-cyclic distribution
    bool blacs_setup_ = false;

    void setup_blacs();
    void cleanup_blacs();

    // ScaLAPACK-based distributed subspace operations (real)
    void orthogonalize_scalapack(double* X, int Nd_d, int Nband_loc, double dV);
    void project_hamiltonian_scalapack(const double* X, const double* Veff,
                                       double* eigvals, int Nd_d, int Nband_loc, double dV);
    void rotate_orbitals_scalapack(double* X, int Nd_d, int Nband_loc);

    // ScaLAPACK-based distributed subspace operations (complex)
    void orthogonalize_kpt_scalapack(Complex* X, int Nd_d, int Nband_loc, double dV);
    void project_hamiltonian_kpt_scalapack(const Complex* X, const double* Veff,
                                           double* eigvals, int Nd_d, int Nband_loc, double dV,
                                           const Vec3& kpt_cart, const Vec3& cell_lengths);
    void rotate_orbitals_kpt_scalapack(Complex* X, int Nd_d, int Nband_loc);

    // Distributed subspace matrix storage (allocated in project_hamiltonian)
    std::vector<double> Hs_dist_;      // local portion of distributed Hs
    std::vector<double> Q_dist_;       // local portion of eigenvectors
    std::vector<Complex> Hs_dist_z_;   // complex versions
    std::vector<Complex> Q_dist_z_;
    int desc_Hs_[9] = {};             // ScaLAPACK descriptor for Hs
    int desc_Q_[9] = {};              // ScaLAPACK descriptor for Q (eigvecs)
    int local_rows_Hs_ = 0, local_cols_Hs_ = 0;
#endif

    // --- Template-unified solve ---
    template<typename T>
    void solve_impl(T* psi, double* eigvals, const double* Veff,
                    int Nd_d, int Nband,
                    double lambda_cutoff, double eigval_min, double eigval_max,
                    int cheb_degree, int ld,
                    const Vec3& kpt_cart = {0,0,0},
                    const Vec3& cell_lengths = {0,0,0});

    // --- Template-unified implementations ---
    template<typename T>
    void chebyshev_filter_impl(const T* X, T* Y, const double* Veff,
                               int Nd_d, int Nband,
                               double lambda_cutoff, double eigval_min, double eigval_max,
                               int degree,
                               const Vec3& kpt_cart = {0,0,0},
                               const Vec3& cell_lengths = {0,0,0});

    template<typename T>
    void orthogonalize_impl(T* X, int Nd_d, int Nband, double dV);

    template<typename T>
    void project_hamiltonian_impl(const T* X, const double* Veff,
                                  T* Hs, int Nd_d, int Nband, double dV,
                                  const Vec3& kpt_cart = {0,0,0},
                                  const Vec3& cell_lengths = {0,0,0});

    template<typename T>
    void diag_subspace_impl(T* Hs, double* eigvals, int N);

    template<typename T>
    void rotate_orbitals_impl(T* X, const T* Q, int Nd_d, int Nband);

    // --- Real private methods (serial LAPACK) ---
    void chebyshev_filter(const double* X, double* Y, const double* Veff,
                          int Nd_d, int Nband,
                          double lambda_cutoff, double eigval_min, double eigval_max,
                          int degree);
    void orthogonalize(double* X, int Nd_d, int Nband, double dV);
    void project_hamiltonian(const double* X, const double* Veff,
                             double* Hs, int Nd_d, int Nband, double dV);
    void diag_subspace(double* Hs, double* eigvals, int N);
    void rotate_orbitals(double* X, const double* Q, int Nd_d, int Nband);

    // --- Complex private methods (serial LAPACK) ---
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

} // namespace lynx
