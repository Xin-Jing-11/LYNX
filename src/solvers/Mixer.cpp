#include "solvers/Mixer.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

namespace sparc {

void Mixer::setup(int Nd_d,
                   MixingVariable var,
                   MixingPrecond precond_type,
                   int history_depth,
                   double mixing_param,
                   const Laplacian* laplacian,
                   const HaloExchange* halo,
                   const FDGrid* grid,
                   const MPIComm* dmcomm) {
    Nd_d_ = Nd_d;
    var_ = var;
    precond_type_ = precond_type;
    m_ = history_depth;
    beta_ = mixing_param;
    laplacian_ = laplacian;
    halo_ = halo;
    grid_ = grid;
    dmcomm_ = dmcomm;

    reset();
}

void Mixer::reset() {
    iter_ = 0;
    dF_.clear();
    dX_.clear();
    dF_.reserve(m_);
    dX_.reserve(m_);
    f_prev_ = NDArray<double>();
    x_prev_ = NDArray<double>();
}

void Mixer::apply_kerker(const double* r, double* Pr) const {
    if (!laplacian_ || !halo_ || !dmcomm_) {
        // No preconditioner available, just copy
        std::memcpy(Pr, r, Nd_d_ * sizeof(double));
        return;
    }

    // Kerker: P = -Lap / (-Lap + k_TF^2)
    // Approximate: just apply a damping factor
    // k_TF^2 ≈ some constant (Thomas-Fermi screening length)
    // For simplicity, use the reference code's approach:
    // Solve (-Lap + kTF^2)*Pr = -Lap*r via a few Richardson iterations

    // Simple approximation: Pr = r (identity preconditioner for now)
    // The full Kerker requires solving a modified Poisson equation.
    // A practical approach: apply in Fourier-like manner using the Laplacian

    int Nd_d = Nd_d_;
    int nd_ex = halo_->nd_ex();

    // Compute -Lap*r
    std::vector<double> r_ex(nd_ex, 0.0);
    halo_->execute(r, r_ex.data(), 1);

    std::vector<double> Lapr(Nd_d);
    laplacian_->apply(r_ex.data(), Lapr.data(), -1.0, 0.0, 1);

    // Solve (-Lap + kTF^2) * Pr = -Lap * r using AAR
    constexpr double kTF2 = 1.0;  // Thomas-Fermi screening constant

    auto op = [this, Nd_d, kTF2](const double* x, double* Ax) {
        int nd_ex = halo_->nd_ex();
        std::vector<double> x_ex(nd_ex, 0.0);
        halo_->execute(x, x_ex.data(), 1);
        laplacian_->apply(x_ex.data(), Ax, -1.0, 0.0, 1);
        for (int i = 0; i < Nd_d; ++i) {
            Ax[i] += kTF2 * x[i];
        }
    };

    // Initial guess: Pr = r
    std::memcpy(Pr, r, Nd_d * sizeof(double));

    AARParams params;
    params.omega = 0.6;
    params.beta = 0.6;
    params.m = 4;
    params.p = 4;
    params.tol = 1e-4;  // Don't need high accuracy for preconditioner
    params.max_iter = 50;

    LinearSolver::aar(op, Lapr.data(), Pr, Nd_d, params, *dmcomm_);
}

void Mixer::mix(double* x_in, const double* x_out, int Nd_d) {
    // Compute residual: f = x_out - x_in
    std::vector<double> f(Nd_d);
    for (int i = 0; i < Nd_d; ++i) {
        f[i] = x_out[i] - x_in[i];
    }

    // Apply preconditioner to residual
    std::vector<double> Pf(Nd_d);
    if (precond_type_ == MixingPrecond::Kerker) {
        apply_kerker(f.data(), Pf.data());
    } else {
        std::memcpy(Pf.data(), f.data(), Nd_d * sizeof(double));
    }

    if (iter_ == 0) {
        // First iteration: simple mixing
        for (int i = 0; i < Nd_d; ++i) {
            x_in[i] = x_in[i] + beta_ * Pf[i];
        }
    } else {
        // Store history differences
        int idx = (iter_ - 1) % m_;
        if (static_cast<int>(dF_.size()) <= idx) {
            dF_.emplace_back(Nd_d);
            dX_.emplace_back(Nd_d);
        }

        for (int i = 0; i < Nd_d; ++i) {
            dF_[idx](i) = f[i] - f_prev_(i);
            dX_[idx](i) = x_in[i] - x_prev_(i);
        }

        int cols = std::min(iter_, m_);

        // Build and solve normal equations: (dF^T * dF) * gamma = dF^T * f
        std::vector<double> FTF(cols * cols, 0.0);
        std::vector<double> FTf(cols, 0.0);

        // Use a null comm wrapper for serial case
        MPIComm null_comm;
        const MPIComm& comm_ref = dmcomm_ ? *dmcomm_ : null_comm;

        for (int i = 0; i < cols; ++i) {
            FTf[i] = LinearSolver::dot(dF_[i].data(), f.data(), Nd_d, comm_ref);
            for (int j = 0; j <= i; ++j) {
                FTF[i * cols + j] = LinearSolver::dot(dF_[i].data(), dF_[j].data(), Nd_d, comm_ref);
                FTF[j * cols + i] = FTF[i * cols + j];
            }
        }

        // Solve small system via Gaussian elimination
        std::vector<double> gamma(cols, 0.0);
        std::vector<double> A_sys(FTF);
        std::vector<double> b_sys(FTf);

        for (int k = 0; k < cols; ++k) {
            int pivot = k;
            for (int i = k + 1; i < cols; ++i) {
                if (std::abs(A_sys[i * cols + k]) > std::abs(A_sys[pivot * cols + k]))
                    pivot = i;
            }
            if (pivot != k) {
                for (int j = 0; j < cols; ++j)
                    std::swap(A_sys[k * cols + j], A_sys[pivot * cols + j]);
                std::swap(b_sys[k], b_sys[pivot]);
            }
            double diag = A_sys[k * cols + k];
            if (std::abs(diag) < 1e-14) continue;
            for (int i = k + 1; i < cols; ++i) {
                double factor = A_sys[i * cols + k] / diag;
                for (int j = k + 1; j < cols; ++j)
                    A_sys[i * cols + j] -= factor * A_sys[k * cols + j];
                b_sys[i] -= factor * b_sys[k];
            }
        }
        for (int k = cols - 1; k >= 0; --k) {
            if (std::abs(A_sys[k * cols + k]) < 1e-14) continue;
            gamma[k] = b_sys[k];
            for (int j = k + 1; j < cols; ++j)
                gamma[k] -= A_sys[k * cols + j] * gamma[j];
            gamma[k] /= A_sys[k * cols + k];
        }

        // Anderson mixing:
        // x_new = (x_in + beta*Pf) - sum_j gamma_j * (dX_j + beta*PdF_j)
        // Simplified: use dF directly instead of PdF
        for (int i = 0; i < Nd_d; ++i) {
            x_in[i] = x_in[i] + beta_ * Pf[i];
        }
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < Nd_d; ++i) {
                x_in[i] -= gamma[j] * (dX_[j](i) + beta_ * dF_[j](i));
            }
        }
    }

    // Save current state for next iteration
    if (f_prev_.empty()) {
        f_prev_ = NDArray<double>(Nd_d);
        x_prev_ = NDArray<double>(Nd_d);
    }
    std::memcpy(f_prev_.data(), f.data(), Nd_d * sizeof(double));
    std::memcpy(x_prev_.data(), x_in, Nd_d * sizeof(double));

    iter_++;
}

} // namespace sparc
