#include "solvers/Mixer.hpp"
#include "core/NumericalMethods.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cstdio>

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

} // namespace lynx
