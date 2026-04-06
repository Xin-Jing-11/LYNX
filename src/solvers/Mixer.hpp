#pragma once

#include "core/types.hpp"
#include "core/DeviceArray.hpp"
#include "core/DeviceTag.hpp"
#include "core/GPUStatePtr.hpp"
#include "solvers/Preconditioner.hpp"

#include <vector>
#include <functional>

namespace lynx {

// Anderson/Pulay mixing with pluggable preconditioner.
// Matches reference LYNX Mixing_periodic_pulay exactly.
//
// Dispatch: dev_ is set at setup. mix() dispatches internally.
// Algorithm lives in .cpp; .cu has only kernel wrappers.
class Mixer {
public:
    Mixer() = default;
    ~Mixer();
    Mixer(Mixer&&) noexcept = default;
    Mixer& operator=(Mixer&&) noexcept = default;
    Mixer(const Mixer&) = delete;
    Mixer& operator=(const Mixer&) = delete;

    void setup(int Nd_d,
               MixingVariable var,
               MixingPrecond precond_type,
               int history_depth,
               double mixing_param,
               Preconditioner* preconditioner = nullptr);

    /// Set device for dispatch. Default is CPU.
    void set_device(Device dev) { dev_ = dev; }
    Device device() const { return dev_; }

    // Set density constraint: after mixing, clamp negative densities and
    // renormalize to Nelectron.  Only used when mixing_var == Density.
    void set_density_constraint(int Nelectron, int Nd_global, double dV);

    // Set potential mean-shift mode: remove per-spin mean before mixing,
    // restore after.  Only used when mixing_var == Potential.
    void set_potential_mean_shift(int Nd_global);

    // Mix density: x_k is the current input, g_k is the output from SCF.
    // After mixing, x_k is updated in-place with the new mixed density.
    // ncol: number of density columns (1 for non-spin, 3 for collinear spin [total,up,down])
    void mix(double* x_k_inout, const double* g_k, int Nd_d, int ncol = 1);

#ifdef USE_CUDA
    GPUStatePtr gpu_state_;  // Opaque pointer to GPUMixerState (defined in .cu)
    double* d_fkm1_ = nullptr;       // Persistent GPU buffer for previous residual
    int gpu_mix_iter_ = 0;            // GPU-side iteration counter
    double gpu_precond_tol_ = 1e-3;   // Kerker preconditioner tolerance
public:
    void setup_gpu(int Nd_d, int ncol, int m_depth, double beta_mix);
    void cleanup_gpu();
#endif

    // Reset history (e.g., at start of new SCF)
    void reset();

    // Access saved Veff mean (per spin) for potential mixing.
    const std::vector<double>& veff_mean() const { return veff_mean_; }

private:
    int Nd_d_ = 0;
    int Nd_ = 0;           // global grid size (for renormalization / mean)
    Device dev_ = Device::CPU;
    MixingVariable var_ = MixingVariable::Density;
    MixingPrecond precond_type_ = MixingPrecond::None;
    int m_ = 7;            // history depth
    double beta_ = 0.3;    // mixing parameter (Pulay)
    int iter_ = 0;

    // State from previous iteration
    std::vector<double> x_km1_;    // x_{k-1}
    std::vector<double> f_k_;      // current residual f_k = g_k - x_k
    std::vector<double> f_km1_;    // previous residual f_{k-1}

    // History matrices (column-major, Nd_d x m)
    std::vector<double> R_;    // [x_k - x_{k-1}] differences
    std::vector<double> F_;    // [f_k - f_{k-1}] differences

    // Pluggable preconditioner (externally owned, may be null)
    Preconditioner* preconditioner_ = nullptr;

    // Density post-processing
    bool density_constraint_ = false;
    int Nelectron_ = 0;
    double dV_ = 0.0;

    // Potential mean-shift
    bool potential_mean_shift_ = false;
    std::vector<double> veff_mean_;  // per-spin mean values

    void apply_density_constraint(double* x, int ncol);
    void remove_mean(double* x, int ncol);
    void restore_mean(double* x, int ncol);

    // --- GPU dispatch: algorithm in .cpp, kernel wrappers in .cu ---
#ifdef USE_CUDA
    // Main GPU mixing algorithm (Pulay) — lives in .cpp
    void mix_gpu(double* d_x_k_inout, const double* d_g_k, int Nd_d, int ncol);

    // Kerker inner AAR solve — loop lives in .cpp
    void kerker_aar_solve_gpu(const double* d_Lf, double* d_Pf, int Nd_kerker);

    // --- GPU kernel wrappers (defined in Mixer.cu) ---

    // Mixer-specific kernels
    void mixer_residual_gpu(const double* d_g, const double* d_x, double* d_f, int N);
    void mixer_store_history_gpu(const double* d_x, const double* d_x_old,
                                  const double* d_f, const double* d_f_old,
                                  double* d_X_hist, double* d_F_hist, int col, int N);

    // Kerker operator: apply (-Lap + kTF^2) to x
    void kerker_apply_op_gpu(const double* d_x, double* d_Ax, int Nd);

    // Kerker RHS: apply (Lap - idiemac*kTF^2) to f
    void kerker_apply_rhs_gpu(const double* d_f, double* d_Lf, int Nd);

    // Kerker Jacobi preconditioner: f = m_inv * r
    void kerker_precondition_gpu(const double* d_r, double* d_f, int N);

    // AAR sub-operations (thin kernel wrappers)
    void aar_residual_gpu(const double* d_b, const double* d_Ax, double* d_r, int N);
    void aar_richardson_gpu(const double* d_x_old, const double* d_f, double* d_x, double omega, int N);
    void aar_store_history_gpu(const double* d_x, const double* d_x_old,
                               const double* d_f, const double* d_f_old,
                               double* d_X_hist, double* d_F_hist, int col, int N);
    void aar_anderson_gpu(const double* d_x_old, const double* d_f,
                          const double* d_X_hist, const double* d_F_hist,
                          const double* d_gamma, double* d_x,
                          double beta, int cols, int N);
    void aar_fused_gram_gpu(const double* d_F_hist, const double* d_f,
                            double* d_gram_out, const int* d_pair_i,
                            const int* d_pair_j, int N, int cols, int n_jobs);
    double aar_norm2_gpu(const double* d_r, int N);
    void aar_copy_gpu(double* dst, const double* src, int N);
#endif
};

} // namespace lynx
