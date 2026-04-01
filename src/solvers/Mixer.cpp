#include "solvers/Mixer.hpp"
#include "core/NumericalMethods.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cstdio>

namespace lynx {

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
        // Unpack to up/dn, clamp, repack
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
        // Renormalize total density (column 0)
        double rho_sum = 0.0;
        for (int i = 0; i < Nd_d_; ++i) rho_sum += x[i];
        double Ne_current = rho_sum * dV_;
        if (Ne_current > 1e-10) {
            double scale = static_cast<double>(Nelectron_) / Ne_current;
            for (int i = 0; i < Nd_d_; ++i) {
                // Scale both total and magnetization consistently
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
        // ncol==1 (non-spin) or ncol==4 (SOC): clamp and renormalize column 0
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
    // Reference: Mixing_periodic_pulay
    // x_k = current input density (x^{in}_k)
    // g_k = output density from SCF (g(x_k) = x^{out}_k)
    // After this call, x_k is updated to x_{k+1}
    // ncol: number of density columns (1 for non-spin, 3 for spin [total,up,down])

    int N = Nd_d * ncol;  // total vector length

    // For potential mixing with mean-shift: work with zero-mean x_k and g_k,
    // then restore g_k's mean to the result. Matches SPARC's flow exactly.
    std::vector<double> g_k_shifted;
    std::vector<double> xk_mean;
    if (potential_mean_shift_ && var_ == MixingVariable::Potential) {
        // Remove mean from x_k (saves to veff_mean_ temporarily)
        remove_mean(x_k, ncol);
        // Make a zero-mean copy of g_k, saving g_k's mean as the authoritative mean
        g_k_shifted.resize(N);
        std::memcpy(g_k_shifted.data(), g_k, N * sizeof(double));
        remove_mean(g_k_shifted.data(), ncol);  // veff_mean_ now holds g_k's mean
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

    // Save old residual: f_{k-1} = f_k
    if (iter_ > 0) {
        std::memcpy(f_km1_.data(), f_k_.data(), N * sizeof(double));
    }

    // Compute current residual: f_k = g_k - x_k
    for (int i = 0; i < N; ++i) {
        f_k_[i] = g_k[i] - x_k[i];
    }

    // Store history: R(:,i_hist) = x_k - x_{k-1}, F(:,i_hist) = f_k - f_{k-1}
    if (iter_ > 0) {
        int i_hist = (iter_ - 1) % m_;
        for (int i = 0; i < N; ++i) {
            R_[i_hist * N + i] = x_k[i] - x_km1_[i];
            F_[i_hist * N + i] = f_k_[i] - f_km1_[i];
        }
    }

    // Pulay mixing flag: reference uses PulayFrequency=1
    // Pulay_mixing_flag = ((iter_count+1) % p == 0 && iter_count > 0)
    // With p=1: true for all iter > 0
    bool pulay_flag = (iter_ > 0);

    // amix = beta for Pulay, omega for simple
    // Reference: omega defaults to same as beta
    double amix = beta_;

    std::vector<double> x_wavg(N);
    std::vector<double> f_wavg(N);

    if (pulay_flag) {
        // Anderson extrapolation: find Gamma = inv(F^T * F) * F^T * f_k
        int cols = std::min(iter_, m_);

        // Build F^T * F and F^T * f_k
        std::vector<double> FtF(cols * cols, 0.0);
        std::vector<double> Ftf(cols, 0.0);

        for (int i = 0; i < cols; ++i) {
            double* Fi = F_.data() + i * N;
            // F^T * f_k
            double dot_ff = 0.0;
            for (int j = 0; j < N; ++j)
                dot_ff += Fi[j] * f_k_[j];
            Ftf[i] = dot_ff;

            // F^T * F (lower triangle, then symmetrize)
            for (int k = 0; k <= i; ++k) {
                double* Fk = F_.data() + k * N;
                double dot_FiFk = 0.0;
                for (int j = 0; j < N; ++j)
                    dot_FiFk += Fi[j] * Fk[j];
                FtF[i * cols + k] = dot_FiFk;
                FtF[k * cols + i] = dot_FiFk;
            }
        }

        // Solve FtF * Gamma = Ftf via Gaussian elimination with partial pivoting
        std::vector<double> Gamma(cols, 0.0);
        gauss_solve(FtF.data(), Ftf.data(), Gamma.data(), cols);

        // x_wavg = x_k - R * Gamma
        std::memcpy(x_wavg.data(), x_k, N * sizeof(double));
        for (int j = 0; j < cols; ++j) {
            double* Rj = R_.data() + j * N;
            double gj = Gamma[j];
            for (int i = 0; i < N; ++i) {
                x_wavg[i] -= gj * Rj[i];
            }
        }

        // f_wavg = f_k - F * Gamma
        std::memcpy(f_wavg.data(), f_k_.data(), N * sizeof(double));
        for (int j = 0; j < cols; ++j) {
            double* Fj = F_.data() + j * N;
            double gj = Gamma[j];
            for (int i = 0; i < N; ++i) {
                f_wavg[i] -= gj * Fj[i];
            }
        }
    } else {
        // Simple mixing: x_wavg = x_k, f_wavg = f_k
        std::memcpy(x_wavg.data(), x_k, N * sizeof(double));
        std::memcpy(f_wavg.data(), f_k_.data(), N * sizeof(double));
    }

    // Apply preconditioner to f_wavg -> Pf
    // For potential mixing: precondition ALL columns (each spin channel independently)
    // For density mixing: precondition column 0 (total density), identity on magnetization
    std::vector<double> Pf(N);
    if (precond_type_ == MixingPrecond::Kerker && preconditioner_) {
        if (var_ == MixingVariable::Potential) {
            // Potential mixing: apply Kerker to each spin channel independently
            for (int c = 0; c < ncol; ++c) {
                preconditioner_->apply(f_wavg.data() + c * Nd_d_, Pf.data() + c * Nd_d_, Nd_d_, amix);
            }
        } else {
            // Density mixing: Kerker on column 0 (total density) only
            preconditioner_->apply(f_wavg.data(), Pf.data(), Nd_d_, amix);
            // Columns 1+: no preconditioner (MixingPrecondMag default = none)
            for (int c = 1; c < ncol; ++c) {
                for (int i = 0; i < Nd_d_; ++i)
                    Pf[c * Nd_d_ + i] = amix * f_wavg[c * Nd_d_ + i];
            }
        }
    } else {
        // No preconditioner: Pf = amix * f_wavg
        for (int i = 0; i < N; ++i)
            Pf[i] = amix * f_wavg[i];
    }

    // x_{k+1} = x_wavg + Pf (amix is already in Pf)
    // Save x_km1 = x_k before overwriting
    std::memcpy(x_km1_.data(), x_k, N * sizeof(double));

    for (int i = 0; i < N; ++i) {
        x_k[i] = x_wavg[i] + Pf[i];
    }

    // For potential mixing with mean-shift: restore mean to x_k
    if (potential_mean_shift_ && var_ == MixingVariable::Potential) {
        restore_mean(x_k, ncol);
    }

    // For density mixing: apply clamping + renormalization
    if (density_constraint_ && var_ != MixingVariable::Potential) {
        apply_density_constraint(x_k, ncol);
    }

    iter_++;
}

} // namespace lynx
