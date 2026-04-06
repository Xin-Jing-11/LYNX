#pragma once

#include "core/types.hpp"
#include "core/DeviceArray.hpp"
#include "core/Domain.hpp"
#include "core/DeviceTag.hpp"
#include "operators/Hamiltonian.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"
#include "core/LynxContext.hpp"
#include "core/GPUStatePtr.hpp"
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
    EigenSolver(EigenSolver&&) noexcept = default;
    EigenSolver& operator=(EigenSolver&&) noexcept = default;
    EigenSolver(const EigenSolver&) = delete;
    EigenSolver& operator=(const EigenSolver&) = delete;

    /// Setup using LynxContext for all infrastructure.
    void setup(const LynxContext& ctx, const Hamiltonian& H);

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

#ifdef USE_CUDA
    GPUStatePtr gpu_state_;  // RAII-managed GPUEigenState (defined in .cu)

    void setup_gpu(const LynxContext& ctx, int Nband, int Nband_global,
                         bool is_kpt, bool is_soc);
    void cleanup_gpu();

    // --- GPU-resident data accessors ---
    double* gpu_psi();
    const double* gpu_psi() const;
    double* gpu_eigvals();
    double* gpu_Veff();

    // Download eigenvalues from device
    void download_eigvals(double* h_eigvals, int Nband);

    // Legacy psi transfer methods — for testing only.
    // Production code must use solve_resident / compute_from_device_ptrs.
    // Psi must stay GPU-resident in the SCF loop (see GPU Data Residency Rules).
    void upload_psi_to_device(const double* h_psi, int Nd, int Nband);
    void upload_psi_z_to_device(const Complex* h_psi, int Nd, int Nband);
    void download_psi(double* h_psi, int Nd, int Nband);
    void download_psi_z(Complex* h_psi, int Nd, int Nband);
    void upload_Veff(const double* h_Veff, int Nd);

    // GPU-resident solve: psi is already on device, only upload Veff and
    // download eigvals. psi stays on device after the call.
    // Algorithm lives in EigenSolver.cpp, calls _gpu() sub-steps.
    void solve_resident(double* h_eigvals, const double* h_Veff,
                        int Nd_d, int Nband,
                        double lambda_cutoff, double eigval_min, double eigval_max,
                        int cheb_degree);

    // GPU-resident solve for k-point (complex): psi_z already on device.
    void solve_kpt_resident(double* h_eigvals, const double* h_Veff,
                            int Nd_d, int Nband,
                            double lambda_cutoff, double eigval_min, double eigval_max,
                            int cheb_degree);

    // GPU transfer helpers (defined in EigenSolver.cu, called from .cpp algorithm)
    void upload_Veff_sync(const double* h_Veff, int Nd);
    void download_eigvals_sync(double* h_eigvals, int Nband);

    // --- GPU sub-step methods (defined in EigenSolver.cu) ---
    // Real (gamma-point) GPU sub-steps
    void chebyshev_filter_gpu(int Nd_d, int Nband,
                               double lambda_cutoff, double eigval_min, double eigval_max,
                               int cheb_degree);
    void orthogonalize_gpu(int Nd_d, int Nband);
    void project_and_diag_gpu(int Nd_d, int Nband);
    void subspace_rotation_gpu(int Nd_d, int Nband);

    // Complex (k-point) GPU sub-steps
    void chebyshev_filter_kpt_gpu(int Nd_d, int Nband,
                                   double lambda_cutoff, double eigval_min, double eigval_max,
                                   int cheb_degree);
    void orthogonalize_kpt_gpu(int Nd_d, int Nband);
    void project_and_diag_kpt_gpu(int Nd_d, int Nband);
    void subspace_rotation_kpt_gpu(int Nd_d, int Nband);

    // GPU workspace pointer accessors (for use in .cpp algorithm)
    double* gpu_Y();
    double* gpu_Hs();

    // Per-(spin,kpt) device psi buffer management.
    // Allocates one psi buffer per (spin,kpt) so all wavefunctions stay
    // GPU-resident simultaneously — no upload/download between solves.
    void allocate_psi_buffers(int Nspin_local, int Nkpts);
    void set_active_psi(int spin, int kpt);

    // Randomize all per-(spin,kpt) psi buffers directly on GPU via cuRAND.
    // Psi is born on GPU — no CPU randomization or H2D transfer needed.
    void randomize_psi_gpu(int Nspin_local, int spin_start, int Nkpts);

    // Device psi pointer for a specific (spin, kpt)
    double* device_psi_real(int spin, int kpt);
    const double* device_psi_real(int spin, int kpt) const;
    void* device_psi_z(int spin, int kpt);
    const void* device_psi_z(int spin, int kpt) const;
#endif

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
