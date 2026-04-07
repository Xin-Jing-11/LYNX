#pragma once

#include "core/types.hpp"
#include "core/DeviceArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "core/DeviceTag.hpp"
#include "core/GPUStatePtr.hpp"
#include "operators/Laplacian.hpp"
#include "operators/FDStencil.hpp"
#include "parallel/HaloExchange.hpp"
#include "solvers/LinearSolver.hpp"

namespace lynx {

// Workspace pointers used during a single AAR solve call.
// On CPU these point into std::vector storage; on GPU into device memory.
struct AARWorkspace {
    double* r      = nullptr;  // (Nd) residual
    double* f      = nullptr;  // (Nd) preconditioned residual
    double* Ax     = nullptr;  // (Nd) operator output
    double* X_hist = nullptr;  // (Nd, m) Anderson iterate history
    double* F_hist = nullptr;  // (Nd, m) Anderson residual history
    double* x_old  = nullptr;  // (Nd) previous iterate
    double* f_old  = nullptr;  // (Nd) previous preconditioned residual
    double* rhs_ms = nullptr;  // (Nd) mean-subtracted RHS (GPU only)
};

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

    // Compute residual: r = b - Ax
    void compute_residual(const double* b, const double* Ax, double* r, int N) const;
    void compute_residual_cpu(const double* b, const double* Ax, double* r, int N) const;

    // Richardson update: x = x_old + omega * f
    void richardson_update(const double* x_old, const double* f, double* x, double omega, int N) const;
    void richardson_update_cpu(const double* x_old, const double* f, double* x, double omega, int N) const;

    // Store Anderson history: X(:,col) = x - x_old, F(:,col) = f - f_old
    void store_history(const double* x, const double* x_old,
                       const double* f, const double* f_old,
                       double* X_hist, double* F_hist, int col, int N) const;
    void store_history_cpu(const double* x, const double* x_old,
                           const double* f, const double* f_old,
                           double* X_hist, double* F_hist, int col, int N) const;

    // Full Anderson step: compute gamma, then extrapolate
    // x = x_old + beta*f - sum gamma_j*(X_j + beta*F_j)
    // Ax_scratch is reused as temp storage on GPU for Gram pair indices/gamma upload
    void anderson_step(const double* x_old, const double* f,
                       const double* X_hist, const double* F_hist,
                       double* Ax_scratch, double* x,
                       double beta, int cols, int N) const;
    void anderson_step_cpu(const double* x_old, const double* f,
                           const double* X_hist, const double* F_hist,
                           double* x, double beta, int cols, int N) const;

    // Copy: dst = src
    void vec_copy(double* dst, const double* src, int N) const;
    void vec_copy_cpu(double* dst, const double* src, int N) const;

    // Compute squared norm: returns ||r||^2
    double compute_norm2(const double* r, int N) const;
    double compute_norm2_cpu(const double* r, int N) const;

    // Subtract mean: x[i] -= mean
    void subtract_mean(double* x, double mean, int N) const;
    void subtract_mean_cpu(double* x, double mean, int N) const;

    // Compute mean of a vector (requires D2H on GPU)
    double compute_mean(const double* x, int N) const;
    double compute_mean_cpu(const double* x, int N) const;

    // --- GPU kernel dispatch targets (defined in PoissonSolver.cu) ---
#ifdef USE_CUDA
    void apply_laplacian_gpu(const double* d_x, double* d_Ax) const;
    void apply_preconditioner_gpu(const double* d_r, double* d_f, int N) const;
    void compute_residual_gpu(const double* d_b, const double* d_Ax, double* d_r, int N) const;
    void richardson_update_gpu(const double* d_x_old, const double* d_f, double* d_x, double omega, int N) const;
    void store_history_gpu(const double* d_x, const double* d_x_old,
                           const double* d_f, const double* d_f_old,
                           double* d_X_hist, double* d_F_hist, int col, int N) const;
    // Full Anderson step on GPU: Gram matrix, gauss_solve, upload gamma, extrapolate
    void anderson_step_gpu(const double* d_x_old, const double* d_f,
                           const double* d_X_hist, const double* d_F_hist,
                           double* d_Ax_scratch, double* d_x,
                           double beta, int cols, int N) const;
    double compute_norm2_gpu(const double* d_r, int N) const;
    void vec_copy_gpu(double* dst, const double* src, int N) const;
    void subtract_mean_gpu(double* d_x, double mean, int N) const;
    double compute_mean_gpu(const double* d_x, int N) const;

    // GPU workspace allocation helpers
    void alloc_gpu_scratch(AARWorkspace& ws, int Nd) const;
    void free_gpu_scratch() const;

    // Opaque GPU state (grid parameters for laplacian dispatch)
    GPUStatePtr gpu_state_;
#endif
};

} // namespace lynx
