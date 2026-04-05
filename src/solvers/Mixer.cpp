#include "solvers/Mixer.hpp"
#include "core/NumericalMethods.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cstdio>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "core/GPUContext.cuh"
#endif

namespace lynx {

#ifndef USE_CUDA
Mixer::~Mixer() = default;
#endif

void Mixer::setup(int Nd_d,
                   MixingVariable var,
                   MixingPrecond precond_type,
                   int history_depth,
                   double mixing_param,
                   Preconditioner* preconditioner) {
    Nd_d_ = Nd_d;
    var_ = var;
    precond_type_ = precond_type;
    m_ = history_depth;
    beta_ = mixing_param;
    preconditioner_ = preconditioner;

    reset();
}

void Mixer::set_density_constraint(int Nelectron, int Nd_global, double dV) {
    density_constraint_ = true;
    Nelectron_ = Nelectron;
    Nd_ = Nd_global;
    dV_ = dV;
}

void Mixer::set_potential_mean_shift(int Nd_global) {
    potential_mean_shift_ = true;
    Nd_ = Nd_global;
}

void Mixer::reset() {
    iter_ = 0;
    x_km1_.clear();
    f_k_.clear();
    f_km1_.clear();
    R_.clear();
    F_.clear();
}

void Mixer::remove_mean(double* x, int ncol) {
    veff_mean_.resize(ncol, 0.0);
    for (int s = 0; s < ncol; ++s) {
        double mean = 0;
        for (int i = 0; i < Nd_d_; ++i) mean += x[s * Nd_d_ + i];
        mean /= Nd_;
        veff_mean_[s] = mean;
        for (int i = 0; i < Nd_d_; ++i) x[s * Nd_d_ + i] -= mean;
    }
}

void Mixer::restore_mean(double* x, int ncol) {
    for (int s = 0; s < ncol; ++s) {
        for (int i = 0; i < Nd_d_; ++i) {
            x[s * Nd_d_ + i] += veff_mean_[s];
        }
    }
}

void Mixer::apply_density_constraint(double* x, int ncol) {
    if (!density_constraint_) return;

    if (ncol == 2) {
        // Spin-polarized: x = [total | magnetization]
        for (int i = 0; i < Nd_d_; ++i) {
            double rho_tot = x[i];
            double mag = x[Nd_d_ + i];
            double rho_up = 0.5 * (rho_tot + mag);
            double rho_dn = 0.5 * (rho_tot - mag);
            if (rho_up < 0.0) rho_up = 0.0;
            if (rho_dn < 0.0) rho_dn = 0.0;
            x[i] = rho_up + rho_dn;
            x[Nd_d_ + i] = rho_up - rho_dn;
        }
        double rho_sum = 0.0;
        for (int i = 0; i < Nd_d_; ++i) rho_sum += x[i];
        double Ne_current = rho_sum * dV_;
        if (Ne_current > 1e-10) {
            double scale = static_cast<double>(Nelectron_) / Ne_current;
            for (int i = 0; i < Nd_d_; ++i) {
                double rho_tot = x[i];
                double mag = x[Nd_d_ + i];
                double rho_up = 0.5 * (rho_tot + mag);
                double rho_dn = 0.5 * (rho_tot - mag);
                rho_up *= scale;
                rho_dn *= scale;
                x[i] = rho_up + rho_dn;
                x[Nd_d_ + i] = rho_up - rho_dn;
            }
        }
    } else {
        for (int i = 0; i < Nd_d_; ++i) {
            if (x[i] < 0.0) x[i] = 0.0;
        }
        double rho_sum = 0.0;
        for (int i = 0; i < Nd_d_; ++i) rho_sum += x[i];
        double Ne_current = rho_sum * dV_;
        if (Ne_current > 1e-10) {
            double scale = static_cast<double>(Nelectron_) / Ne_current;
            for (int i = 0; i < Nd_d_; ++i) x[i] *= scale;
        }
    }
}

void Mixer::mix(double* x_k, const double* g_k, int Nd_d, int ncol) {
#ifdef USE_CUDA
    if (dev_ == Device::GPU && gpu_state_raw_) {
        mix_gpu(x_k, g_k, Nd_d, ncol);
        return;
    }
#endif

    // ---- CPU path: Pulay mixing algorithm ----

    int N = Nd_d * ncol;

    // For potential mixing with mean-shift
    std::vector<double> g_k_shifted;
    std::vector<double> xk_mean;
    if (potential_mean_shift_ && var_ == MixingVariable::Potential) {
        remove_mean(x_k, ncol);
        g_k_shifted.resize(N);
        std::memcpy(g_k_shifted.data(), g_k, N * sizeof(double));
        remove_mean(g_k_shifted.data(), ncol);
        g_k = g_k_shifted.data();
    }

    // Allocate state on first call
    if (f_k_.empty()) {
        f_k_.resize(N, 0.0);
        f_km1_.resize(N, 0.0);
        x_km1_.resize(N, 0.0);
        R_.resize(N * m_, 0.0);
        F_.resize(N * m_, 0.0);
    }

    // Save old residual
    if (iter_ > 0) {
        std::memcpy(f_km1_.data(), f_k_.data(), N * sizeof(double));
    }

    // Compute current residual: f_k = g_k - x_k
    for (int i = 0; i < N; ++i) {
        f_k_[i] = g_k[i] - x_k[i];
    }

    // Store history
    if (iter_ > 0) {
        int i_hist = (iter_ - 1) % m_;
        for (int i = 0; i < N; ++i) {
            R_[i_hist * N + i] = x_k[i] - x_km1_[i];
            F_[i_hist * N + i] = f_k_[i] - f_km1_[i];
        }
    }

    bool pulay_flag = (iter_ > 0);
    double amix = beta_;

    std::vector<double> x_wavg(N);
    std::vector<double> f_wavg(N);

    if (pulay_flag) {
        int cols = std::min(iter_, m_);

        std::vector<double> FtF(cols * cols, 0.0);
        std::vector<double> Ftf(cols, 0.0);

        for (int i = 0; i < cols; ++i) {
            double* Fi = F_.data() + i * N;
            double dot_ff = 0.0;
            for (int j = 0; j < N; ++j)
                dot_ff += Fi[j] * f_k_[j];
            Ftf[i] = dot_ff;

            for (int k = 0; k <= i; ++k) {
                double* Fk = F_.data() + k * N;
                double dot_FiFk = 0.0;
                for (int j = 0; j < N; ++j)
                    dot_FiFk += Fi[j] * Fk[j];
                FtF[i * cols + k] = dot_FiFk;
                FtF[k * cols + i] = dot_FiFk;
            }
        }

        std::vector<double> Gamma(cols, 0.0);
        gauss_solve(FtF.data(), Ftf.data(), Gamma.data(), cols);

        std::memcpy(x_wavg.data(), x_k, N * sizeof(double));
        for (int j = 0; j < cols; ++j) {
            double* Rj = R_.data() + j * N;
            double gj = Gamma[j];
            for (int i = 0; i < N; ++i) {
                x_wavg[i] -= gj * Rj[i];
            }
        }

        std::memcpy(f_wavg.data(), f_k_.data(), N * sizeof(double));
        for (int j = 0; j < cols; ++j) {
            double* Fj = F_.data() + j * N;
            double gj = Gamma[j];
            for (int i = 0; i < N; ++i) {
                f_wavg[i] -= gj * Fj[i];
            }
        }
    } else {
        std::memcpy(x_wavg.data(), x_k, N * sizeof(double));
        std::memcpy(f_wavg.data(), f_k_.data(), N * sizeof(double));
    }

    // Apply preconditioner
    std::vector<double> Pf(N);
    if (precond_type_ == MixingPrecond::Kerker && preconditioner_) {
        if (var_ == MixingVariable::Potential) {
            for (int c = 0; c < ncol; ++c) {
                preconditioner_->apply(f_wavg.data() + c * Nd_d_, Pf.data() + c * Nd_d_, Nd_d_, amix);
            }
        } else {
            preconditioner_->apply(f_wavg.data(), Pf.data(), Nd_d_, amix);
            for (int c = 1; c < ncol; ++c) {
                for (int i = 0; i < Nd_d_; ++i)
                    Pf[c * Nd_d_ + i] = amix * f_wavg[c * Nd_d_ + i];
            }
        }
    } else {
        for (int i = 0; i < N; ++i)
            Pf[i] = amix * f_wavg[i];
    }

    // x_{k+1} = x_wavg + Pf
    std::memcpy(x_km1_.data(), x_k, N * sizeof(double));

    for (int i = 0; i < N; ++i) {
        x_k[i] = x_wavg[i] + Pf[i];
    }

    // Post-processing
    if (potential_mean_shift_ && var_ == MixingVariable::Potential) {
        restore_mean(x_k, ncol);
    }

    if (density_constraint_ && var_ != MixingVariable::Potential) {
        apply_density_constraint(x_k, ncol);
    }

    iter_++;
}

// ============================================================
// GPU Pulay mixing algorithm — lives in .cpp
// Kernel launches via _gpu() wrappers in .cu
// ============================================================

#ifdef USE_CUDA

// Kerker inner AAR solve: loop in .cpp, kernel launches via _gpu() methods
// Solves (-Lap + kTF^2)*Pf = Lf
void Mixer::kerker_aar_solve_gpu(const double* d_Lf, double* d_Pf, int Nd_kerker) {
    auto& ctx = gpu::GPUContext::instance();
    auto& sp = ctx.scratch_pool;
    size_t sp_cp = sp.checkpoint();
    cudaStream_t stream = ctx.compute_stream;

    double* d_kr    = sp.alloc<double>(Nd_kerker);
    double* d_kf    = sp.alloc<double>(Nd_kerker);
    double* d_kAx   = sp.alloc<double>(Nd_kerker);
    double* d_kX    = sp.alloc<double>(Nd_kerker * 7);
    double* d_kF    = sp.alloc<double>(Nd_kerker * 7);
    double* d_kxold = sp.alloc<double>(Nd_kerker);
    double* d_kfold = sp.alloc<double>(Nd_kerker);

    int m = 7, p = 6;
    double omega = 0.6, beta_aar = 0.6;
    double tol = gpu_precond_tol_;
    int max_iter = 1000;

    // x_old = x (Pf)
    aar_copy_gpu(d_kxold, d_Pf, Nd_kerker);

    // ||b||
    double b_norm2 = aar_norm2_gpu(d_Lf, Nd_kerker);
    double abs_tol = tol * std::sqrt(b_norm2);

    // Initial residual
    kerker_apply_op_gpu(d_Pf, d_kAx, Nd_kerker);
    aar_residual_gpu(d_Lf, d_kAx, d_kr, Nd_kerker);

    double r_2norm = abs_tol + 1.0;
    int iter_count = 0;

    // Host buffers for Gram solve
    int max_jobs = m * (m + 1) / 2 + m;
    std::vector<double> h_FTF(m * m);
    std::vector<double> h_gamma(m);
    std::vector<int> h_pair_i(max_jobs), h_pair_j(max_jobs);
    std::vector<double> h_gram_out(max_jobs);

    while (r_2norm > abs_tol && iter_count < max_iter) {
        // Precondition: f = M^{-1} * r
        kerker_precondition_gpu(d_kr, d_kf, Nd_kerker);

        // Store history
        if (iter_count > 0) {
            int i_hist = (iter_count - 1) % m;
            aar_store_history_gpu(d_Pf, d_kxold, d_kf, d_kfold,
                                  d_kX, d_kF, i_hist, Nd_kerker);
        }

        aar_copy_gpu(d_kxold, d_Pf, Nd_kerker);
        aar_copy_gpu(d_kfold, d_kf, Nd_kerker);

        if ((iter_count + 1) % p == 0 && iter_count > 0) {
            int cols = std::min(iter_count, m);

            // Reuse d_kAx memory for pair indices and gram output
            int* d_pi = reinterpret_cast<int*>(d_kAx);
            int* d_pj = d_pi + max_jobs;
            double* d_go = reinterpret_cast<double*>(d_pj + max_jobs);

            int n_jobs = 0;
            for (int ii = 0; ii < cols; ++ii)
                for (int jj = 0; jj <= ii; ++jj) {
                    h_pair_i[n_jobs] = ii; h_pair_j[n_jobs] = jj; n_jobs++;
                }
            int ftf_pairs = n_jobs;
            for (int ii = 0; ii < cols; ++ii) {
                h_pair_i[n_jobs] = ii; h_pair_j[n_jobs] = -1; n_jobs++;
            }

            CUDA_CHECK(cudaMemcpyAsync(d_pi, h_pair_i.data(), n_jobs * sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_pj, h_pair_j.data(), n_jobs * sizeof(int), cudaMemcpyHostToDevice, stream));

            aar_fused_gram_gpu(d_kF, d_kf, d_go, d_pi, d_pj, Nd_kerker, cols, n_jobs);

            CUDA_CHECK(cudaMemcpyAsync(h_gram_out.data(), d_go, n_jobs * sizeof(double), cudaMemcpyDeviceToHost, stream));

            // Unpack and solve on CPU (tiny matrix)
            int k = 0;
            for (int ii = 0; ii < cols; ++ii)
                for (int jj = 0; jj <= ii; ++jj) {
                    h_FTF[ii * cols + jj] = h_gram_out[k];
                    h_FTF[jj * cols + ii] = h_gram_out[k];
                    k++;
                }
            for (int ii = 0; ii < cols; ++ii)
                h_gamma[ii] = h_gram_out[ftf_pairs + ii];

            gauss_solve(h_FTF.data(), h_gamma.data(), h_gamma.data(), cols);

            double* d_gamma = d_kAx;
            CUDA_CHECK(cudaMemcpyAsync(d_gamma, h_gamma.data(), cols * sizeof(double), cudaMemcpyHostToDevice, stream));
            aar_anderson_gpu(d_kxold, d_kf, d_kX, d_kF, d_gamma, d_Pf, beta_aar, cols, Nd_kerker);

            // Recompute residual + convergence check
            kerker_apply_op_gpu(d_Pf, d_kAx, Nd_kerker);
            aar_residual_gpu(d_Lf, d_kAx, d_kr, Nd_kerker);
            r_2norm = std::sqrt(aar_norm2_gpu(d_kr, Nd_kerker));
        } else {
            aar_richardson_gpu(d_kxold, d_kf, d_Pf, omega, Nd_kerker);
            kerker_apply_op_gpu(d_Pf, d_kAx, Nd_kerker);
            aar_residual_gpu(d_Lf, d_kAx, d_kr, Nd_kerker);
        }
        iter_count++;
    }

    sp.restore(sp_cp);
}

// GPU Pulay mixing — algorithm in .cpp, kernel launches via _gpu() wrappers
void Mixer::mix_gpu(double* x_k_inout, const double* g_k, int Nd_d, int ncol) {
    auto& ctx = gpu::GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;

    int Nd = Nd_d * ncol;
    int m_depth = m_;
    double beta_mix = beta_;
    int Nd_kerker = Nd_d;
    double beta_mag = beta_mix;

    double* d_fk   = ctx.buf.mix_fk;
    double* d_xkm1 = ctx.buf.mix_xkm1;
    double* d_R    = ctx.buf.mix_R;
    double* d_F    = ctx.buf.mix_F;

    // Ensure fkm1 buffer exists
    if (!d_fkm1_) {
        CUDA_CHECK(cudaMallocAsync(&d_fkm1_, Nd * sizeof(double), stream));
        CUDA_CHECK(cudaMemsetAsync(d_fkm1_, 0, Nd * sizeof(double), stream));
    }

    // Save old f_k -> f_km1
    if (gpu_mix_iter_ > 0) {
        aar_copy_gpu(d_fkm1_, d_fk, Nd);
    }

    // f_k = g - x
    mixer_residual_gpu(g_k, x_k_inout, d_fk, Nd);

    // Store history
    if (gpu_mix_iter_ > 0) {
        int i_hist = (gpu_mix_iter_ - 1) % m_depth;
        mixer_store_history_gpu(x_k_inout, d_xkm1, d_fk, d_fkm1_, d_R, d_F, i_hist, Nd);
    }

    // Workspace
    auto& sp = ctx.scratch_pool;
    size_t sp_cp = sp.checkpoint();
    double* d_x_wavg = sp.alloc<double>(Nd);
    double* d_f_wavg = sp.alloc<double>(Nd);

    if (gpu_mix_iter_ > 0) {
        int cols = std::min(gpu_mix_iter_, m_depth);

        // Build F^T*F and F^T*f_k via cuBLAS
        std::vector<double> h_FtF(cols * cols);
        std::vector<double> h_Ftf(cols);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int ii = 0; ii < cols; ii++) {
            double* Fi = d_F + ii * Nd;
            cublasDdot(ctx.cublas, Nd, Fi, 1, d_fk, 1, &h_Ftf[ii]);
            for (int jj = 0; jj <= ii; jj++) {
                double* Fj = d_F + jj * Nd;
                cublasDdot(ctx.cublas, Nd, Fi, 1, Fj, 1, &h_FtF[ii * cols + jj]);
                h_FtF[jj * cols + ii] = h_FtF[ii * cols + jj];
            }
        }

        // Solve Gamma on CPU
        std::vector<double> Gamma(cols, 0.0);
        gauss_solve(h_FtF.data(), h_Ftf.data(), Gamma.data(), cols);

        // x_wavg = x - R * Gamma, f_wavg = f_k - F * Gamma
        aar_copy_gpu(d_x_wavg, x_k_inout, Nd);
        aar_copy_gpu(d_f_wavg, d_fk, Nd);

        for (int j = 0; j < cols; j++) {
            double neg_gj = -Gamma[j];
            cublasDaxpy(ctx.cublas, Nd, &neg_gj, d_R + j * Nd, 1, d_x_wavg, 1);
            cublasDaxpy(ctx.cublas, Nd, &neg_gj, d_F + j * Nd, 1, d_f_wavg, 1);
        }
    } else {
        aar_copy_gpu(d_x_wavg, x_k_inout, Nd);
        aar_copy_gpu(d_f_wavg, d_fk, Nd);
    }

    // Kerker preconditioner
    double* d_Pf = sp.alloc<double>(Nd);
    CUDA_CHECK(cudaMemsetAsync(d_Pf, 0, Nd * sizeof(double), stream));

    // Apply Kerker to first Nd_kerker elements
    {
        double* d_f_col = d_f_wavg;
        double* d_Pf_col = d_Pf;

        // Step 1: Lf = (Lap - idiemac*kTF^2) * f_wavg
        double* d_Lf = sp.alloc<double>(Nd_kerker);
        kerker_apply_rhs_gpu(d_f_col, d_Lf, Nd_kerker);

        // Step 2: Solve (-Lap + kTF^2)*Pf = Lf via AAR
        kerker_aar_solve_gpu(d_Lf, d_Pf_col, Nd_kerker);

        // Step 3: Pf *= -beta_mix
        {
            double neg_beta = -beta_mix;
            cublasDscal(ctx.cublas, Nd_kerker, &neg_beta, d_Pf_col, 1);
        }
    }

    // Simple mixing for magnetization part (if ncol > 1)
    if (Nd_kerker < Nd) {
        int Nd_mag = Nd - Nd_kerker;
        aar_copy_gpu(d_Pf + Nd_kerker, d_f_wavg + Nd_kerker, Nd_mag);
        cublasDscal(ctx.cublas, Nd_mag, &beta_mag, d_Pf + Nd_kerker, 1);
    }

    // Save x_km1
    aar_copy_gpu(d_xkm1, x_k_inout, Nd);

    // x_{k+1} = x_wavg + Pf
    aar_copy_gpu(x_k_inout, d_x_wavg, Nd);
    {
        double one = 1.0;
        cublasDaxpy(ctx.cublas, Nd, &one, d_Pf, 1, x_k_inout, 1);
    }

    sp.restore(sp_cp);
    gpu_mix_iter_++;
}

#endif // USE_CUDA

} // namespace lynx
