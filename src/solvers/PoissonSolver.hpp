#pragma once

#include "core/types.hpp"
#include "core/DeviceArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "core/DeviceTag.hpp"
#include "operators/Laplacian.hpp"
#include "operators/FDStencil.hpp"
#include "parallel/HaloExchange.hpp"
#include "solvers/LinearSolver.hpp"

namespace lynx {

// Solves the Poisson equation: -Lap(phi) = 4*pi*(rho + b)
// where b is the pseudocharge density.
// Uses AAR iterative solver with Jacobi preconditioner.
//
// Dispatch: dev_ is set at setup. solve() contains the AAR loop once.
// Sub-operations dispatch to _cpu() / _gpu() methods internally.
class PoissonSolver {
public:
    PoissonSolver() = default;
    ~PoissonSolver();
    PoissonSolver(PoissonSolver&&) noexcept = default;
    PoissonSolver& operator=(PoissonSolver&&) noexcept = default;
    PoissonSolver(const PoissonSolver&) = delete;
    PoissonSolver& operator=(const PoissonSolver&) = delete;

    void setup(const Laplacian& laplacian,
               const FDStencil& stencil,
               const Domain& domain,
               const FDGrid& grid,
               const HaloExchange& halo);

    /// Set device for dispatch. Default is CPU.
    void set_device(Device dev) { dev_ = dev; }
    Device device() const { return dev_; }

    // Solve: -Lap(phi) = rhs
    // rhs: (Nd_d,) right-hand side = 4*pi*(rho + b)
    // phi: (Nd_d,) output electrostatic potential
    // Returns number of iterations
    int solve(const double* rhs, double* phi, double tol = 1e-8) const;

    // Set AAR parameters
    void set_aar_params(const AARParams& params) { aar_params_ = params; }

#ifdef USE_CUDA
    void setup_gpu(const class LynxContext& ctx);
    void cleanup_gpu();
#endif

private:
    const Laplacian* laplacian_ = nullptr;
    const FDStencil* stencil_ = nullptr;
    const Domain* domain_ = nullptr;
    const FDGrid* grid_ = nullptr;
    const HaloExchange* halo_ = nullptr;

    Device dev_ = Device::CPU;
    AARParams aar_params_;
    double jacobi_weight_ = 0.0;  // Jacobi preconditioner weight

    // --- Sub-operation dispatchers (called from solve loop) ---

    // Apply operator: Ax = -Lap(x)
    void apply_laplacian(const double* x, double* Ax) const;
    void apply_laplacian_cpu(const double* x, double* Ax) const;

    // Apply Jacobi preconditioner: f = m_inv * r
    void apply_preconditioner(const double* r, double* f, int N) const;
    void apply_preconditioner_cpu(const double* r, double* f, int N) const;

    // --- GPU kernel dispatch targets (defined in PoissonSolver.cu) ---
#ifdef USE_CUDA
    // GPU solve path (defined in PoissonSolver.cu)
    int solve_gpu(const double* rhs, double* phi, double tol) const;

    void apply_laplacian_gpu(const double* d_x, double* d_Ax) const;
    void apply_preconditioner_gpu(const double* d_r, double* d_f, int N) const;

    // GPU AAR sub-operations (thin kernel wrappers in .cu)
    void aar_residual_gpu(const double* d_b, const double* d_Ax, double* d_r, int N) const;
    void aar_richardson_gpu(const double* d_x_old, const double* d_f, double* d_x, double omega, int N) const;
    void aar_store_history_gpu(const double* d_x, const double* d_x_old,
                               const double* d_f, const double* d_f_old,
                               double* d_X_hist, double* d_F_hist, int col, int N) const;
    void aar_anderson_gpu(const double* d_x_old, const double* d_f,
                          const double* d_X_hist, const double* d_F_hist,
                          const double* d_gamma, double* d_x,
                          double beta, int cols, int N) const;
    void aar_fused_gram_gpu(const double* d_F_hist, const double* d_f,
                            double* d_gram_out, const int* d_pair_i,
                            const int* d_pair_j, int N, int cols, int n_jobs) const;
    double aar_norm2_gpu(const double* d_r, int N) const;
    void aar_copy_gpu(double* dst, const double* src, int N) const;
    void mean_subtract_gpu(double* d_x, double mean, int N) const;

    // Opaque GPU state (grid parameters for laplacian dispatch)
    void* gpu_state_raw_ = nullptr;
#endif
};

} // namespace lynx
