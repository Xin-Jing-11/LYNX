#include "solvers/Mixer.hpp"
#include "parallel/MPIComm.hpp"
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
                   const Laplacian* laplacian,
                   const HaloExchange* halo,
                   const FDGrid* grid) {
    Nd_d_ = Nd_d;
    var_ = var;
    precond_type_ = precond_type;
    m_ = history_depth;
    beta_ = mixing_param;
    laplacian_ = laplacian;
    halo_ = halo;
    grid_ = grid;
    if (grid) Nd_ = grid->Nd();

    // Compute TOL_PRECOND = h_eff^2 * 1e-3 (reference: initialization.c:2655-2664)
    if (grid) {
        double dx = grid->dx(), dy = grid->dy(), dz = grid->dz();
        double h_eff;
        if (std::abs(dx - dy) < 1e-12 && std::abs(dy - dz) < 1e-12) {
            h_eff = dx;
        } else {
            double dx2_inv = 1.0 / (dx * dx);
            double dy2_inv = 1.0 / (dy * dy);
            double dz2_inv = 1.0 / (dz * dz);
            h_eff = std::sqrt(3.0 / (dx2_inv + dy2_inv + dz2_inv));
        }
        precond_tol_ = h_eff * h_eff * 1e-3;
    }

    reset();
}

void Mixer::reset() {
    iter_ = 0;
    x_km1_.clear();
    f_k_.clear();
    f_km1_.clear();
    R_.clear();
    F_.clear();
}

void Mixer::apply_kerker(const double* f, double amix, double* Pf) const {
    if (!laplacian_ || !halo_) {
        // No preconditioner available, just scale
        for (int i = 0; i < Nd_d_; ++i)
            Pf[i] = amix * f[i];
        return;
    }

    int Nd_d = Nd_d_;
    int nd_ex = halo_->nd_ex();

    // Reference: Kerker_precond
    // Step 1: Compute Lf = (Lap - idiemac*kTF²) * f
    // where Lap_vec_mult with c = -idiemac*kTF² does: Lap*f + c*f
    constexpr double kTF = 1.0;        // PRECOND_KERKER_KTF
    constexpr double idiemac = 0.1;    // PRECOND_KERKER_THRESH

    std::vector<double> f_ex(nd_ex, 0.0);
    halo_->execute(f, f_ex.data(), 1);

    std::vector<double> Lf(Nd_d);
    // Lf = Lap*f + (-idiemac*kTF²)*f = (Lap - idiemac*kTF²)*f
    laplacian_->apply(f_ex.data(), Lf.data(), 1.0, -idiemac * kTF * kTF, 1);

    // Step 2: Solve -(Lap - kTF²) * Pf = Lf
    // i.e., (-Lap + kTF²) * Pf = -Lf  ... wait, reference uses:
    //   AAR(pLYNX, res_fun, precond_fun, -lambda_TF^2, DMnd, Pf, Lf, ...)
    // where res_fun computes: b + (Lap + c)*x = Lf + (Lap - kTF²)*x
    // So residual = Lf - (-Lap + kTF²)*Pf, and we solve (-Lap + kTF²)*Pf = Lf
    // Actually: res_fun = b + (Lap + c)*x where c = -kTF², so operator is (Lap - kTF²)
    // We solve (Lap - kTF²)*Pf = -Lf  ... no.
    // Let me re-check: AAR solves A*x = b where A*x = -(Lap + c)*x = -Lap*x - c*x
    // with c = -kTF², A*x = -Lap*x + kTF²*x, and b = Lf
    // Actually reference AAR: res = b - A*x, and the operator does b + (Lap+c)*x
    // So it's solving: -(Lap + c)*x = b, i.e., (-Lap + kTF²)*x = Lf
    // That means: (-Lap + kTF²)*Pf = Lf

    auto op = [this, Nd_d](const double* x, double* Ax) {
        int nd_ex = halo_->nd_ex();
        std::vector<double> x_ex(nd_ex, 0.0);
        halo_->execute(x, x_ex.data(), 1);
        // (-Lap + kTF²)*x = -Lap*x + kTF²*x
        laplacian_->apply(x_ex.data(), Ax, -1.0, kTF * kTF, 1);
    };

    // Jacobi preconditioner for the operator -(Lap + c) where c = -kTF²
    // Reference: m_inv = (D2_coeff_x[0] + D2_coeff_y[0] + D2_coeff_z[0] + c)
    //            m_inv = -1.0 / m_inv
    //            f[i] = m_inv * r[i]
    const auto& stencil = laplacian_->stencil();
    double m_diag = stencil.D2_coeff_x()[0] + stencil.D2_coeff_y()[0]
                  + stencil.D2_coeff_z()[0] + (-kTF * kTF);
    double m_inv = (std::abs(m_diag) < 1e-14) ? 1.0 : (-1.0 / m_diag);

    auto jacobi_precond = [m_inv, Nd_d](const double* r, double* z) {
        for (int i = 0; i < Nd_d; ++i)
            z[i] = m_inv * r[i];
    };

    // Initial guess: Pf = 0
    std::memset(Pf, 0, Nd_d * sizeof(double));

    // Reference uses omega=0.6, beta=0.6, m=7, p=6
    AARParams params;
    params.omega = 0.6;
    params.beta = 0.6;
    params.m = 7;
    params.p = 6;
    params.tol = precond_tol_;  // TOL_PRECOND = h_eff^2 * 1e-3
    params.max_iter = 1000;

    LinearSolver::PrecondFunc precond_fn = jacobi_precond;
    MPIComm self_comm(MPI_COMM_SELF);
    LinearSolver::aar(op, Lf.data(), Pf, Nd_d, params, self_comm, &precond_fn);

    // Step 3: Scale by -amix (reference: Pf[i] *= -a)
    for (int i = 0; i < Nd_d; ++i) {
        Pf[i] *= -amix;
    }
}

void Mixer::mix(double* x_k, const double* g_k, int Nd_d, int ncol) {
    // Reference: Mixing_periodic_pulay
    // x_k = current input density (x^{in}_k)
    // g_k = output density from SCF (g(x_k) = x^{out}_k)
    // After this call, x_k is updated to x_{k+1}
    // ncol: number of density columns (1 for non-spin, 3 for spin [total,up,down])

    int N = Nd_d * ncol;  // total vector length

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

        // Solve FtF * Gamma = Ftf via least squares (Gaussian elimination)
        std::vector<double> Gamma(cols, 0.0);
        {
            std::vector<double> A(FtF);
            std::vector<double> b(Ftf);
            for (int k = 0; k < cols; ++k) {
                int pivot = k;
                for (int i = k + 1; i < cols; ++i) {
                    if (std::abs(A[i * cols + k]) > std::abs(A[pivot * cols + k]))
                        pivot = i;
                }
                if (pivot != k) {
                    for (int j = 0; j < cols; ++j)
                        std::swap(A[k * cols + j], A[pivot * cols + j]);
                    std::swap(b[k], b[pivot]);
                }
                double diag = A[k * cols + k];
                if (std::abs(diag) < 1e-14) continue;
                for (int i = k + 1; i < cols; ++i) {
                    double factor = A[i * cols + k] / diag;
                    for (int j = k + 1; j < cols; ++j)
                        A[i * cols + j] -= factor * A[k * cols + j];
                    b[i] -= factor * b[k];
                }
            }
            for (int k = cols - 1; k >= 0; --k) {
                if (std::abs(A[k * cols + k]) < 1e-14) continue;
                Gamma[k] = b[k];
                for (int j = k + 1; j < cols; ++j)
                    Gamma[k] -= A[k * cols + j] * Gamma[j];
                Gamma[k] /= A[k * cols + k];
            }
        }

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
    // For potential mixing: Kerker on ALL columns (each spin channel independently)
    // For density mixing: Kerker on column 0 (total density), none on magnetization
    std::vector<double> Pf(N);
    if (precond_type_ == MixingPrecond::Kerker && laplacian_ && halo_) {
        if (var_ == MixingVariable::Potential) {
            // Potential mixing: apply Kerker to each spin channel independently
            for (int c = 0; c < ncol; ++c) {
                apply_kerker(f_wavg.data() + c * Nd_d_, amix, Pf.data() + c * Nd_d_);
            }
        } else {
            // Density mixing: Kerker on column 0 (total density) only
            apply_kerker(f_wavg.data(), amix, Pf.data());
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

    iter_++;
}

} // namespace lynx
