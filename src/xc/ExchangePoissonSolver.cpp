#include "xc/ExchangePoissonSolver.hpp"
#include "core/constants.hpp"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <mpi.h>

#include <mkl_dfti.h>

namespace lynx {

// ---------------------------------------------------------------------------
// Singularity removal constant for a given G^2
// ---------------------------------------------------------------------------
double ExchangePoissonSolver::singularity_removal_const(double G2) const {
    double omega = hyb_range_fock_;
    double omega2 = omega * omega;

    if (exx_div_flag_ == 0) {
        // Spherical cutoff (Spencer & Alavi)
        // V = cell_volume * Nkpts_hf (matches SPARC: L1*L2*L3 * Jacbdet * Nkpts_hf)
        double V = Jacbdet_ * Nkpts_hf_;
        double R_c = std::cbrt(3.0 * V / (4.0 * constants::PI));
        if (std::fabs(G2) > 1e-4) {
            double x = R_c * std::sqrt(G2);
            return 4.0 * constants::PI * (1.0 - std::cos(x)) / G2;
        } else {
            return 2.0 * constants::PI * R_c * R_c;
        }
    } else if (exx_div_flag_ == 1) {
        // Auxiliary function (Gygi & Baldereschi)
        double x = -0.25 / omega2 * G2;
        if (std::fabs(G2) > 1e-4) {
            if (omega > 0) {
                return 4.0 * constants::PI / G2 * (1.0 - std::exp(x));
            } else {
                return 4.0 * constants::PI / G2;
            }
        } else {
            if (omega > 0) {
                return 4.0 * constants::PI * (const_aux_ + 0.25 / omega2);
            } else {
                return 4.0 * constants::PI * const_aux_;
            }
        }
    } else if (exx_div_flag_ == 2) {
        // ERFC short-range screened
        double x = -0.25 / omega2 * G2;
        if (std::fabs(G2) > 1e-4) {
            return 4.0 * constants::PI * (1.0 - std::exp(x)) / G2;
        } else {
            return constants::PI / omega2;
        }
    }
    return 0.0;
}

// ---------------------------------------------------------------------------
// Singularity removal constants for stress computation
// Matches SPARC singularity_remooval_const for stress/stress2
// ---------------------------------------------------------------------------
void ExchangePoissonSolver::singularity_removal_const_stress(double G2, double& stress, double& stress2) const {
    double omega = hyb_range_fock_;
    double omega2 = omega * omega;

    if (exx_div_flag_ == 0) {
        // Spherical cutoff
        double V = Jacbdet_ * Nkpts_hf_;
        double R_c = std::cbrt(3.0 * V / (4.0 * constants::PI));
        double G4 = G2 * G2;
        if (std::fabs(G2) > 1e-4) {
            double x = R_c * std::sqrt(G2);
            stress  = 4.0 * constants::PI * (1.0 - std::cos(x) - x * 0.5 * std::sin(x)) / G4;
            stress2 = 4.0 * constants::PI * (x * 0.5 * std::sin(x)) / G2 / 3.0;
        } else {
            stress  = 4.0 * constants::PI * std::pow(R_c, 4) / 24.0;
            stress2 = 4.0 * constants::PI * (R_c * R_c * 0.5) / 3.0;
        }
    } else if (exx_div_flag_ == 1) {
        // Auxiliary function (matching SPARC: note the /4 factor)
        double G4 = G2 * G2;
        if (std::fabs(G2) > 1e-4) {
            double x = -0.25 / omega2 * G2;
            if (omega > 0) {
                stress = 4.0 * constants::PI * (1.0 - std::exp(x) * (1.0 - x)) / G4 / 4.0;
            } else {
                stress = 4.0 * constants::PI / G4 / 4.0;
            }
        } else {
            stress = 0.0;  // matches SPARC: G=0 stress constant is zero for auxiliary
        }
        stress2 = 0.0;
    } else if (exx_div_flag_ == 2) {
        // ERFC short-ranged screened (matching SPARC)
        double G4 = G2 * G2;
        double x = -0.25 / omega2 * G2;
        if (std::fabs(G2) > 1e-4) {
            stress = 4.0 * constants::PI * (1.0 - std::exp(x) * (1.0 - x)) / G4;
        } else {
            stress = 0.0;
        }
        stress2 = 0.0;
    } else {
        stress = 0.0;
        stress2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Compute auxiliary constant for G=0 term (Gygi & Baldereschi method)
// ---------------------------------------------------------------------------
void ExchangePoissonSolver::compute_auxiliary_constant() {
    if (exx_div_flag_ != 1) return;

    double tpiblx = 2.0 * constants::PI / L1_;
    double tpibly = 2.0 * constants::PI / L2_;
    double tpiblz = 2.0 * constants::PI / L3_;

    // Estimate energy cutoff (matching SPARC ecut_estimate)
    double dx2i = 1.0 / (dx_ * dx_), dy2i = 1.0 / (dy_ * dy_), dz2i = 1.0 / (dz_ * dz_);
    double h_eff = std::sqrt(3.0 / (dx2i + dy2i + dz2i));
    double ecut = std::exp(-2.0 * std::log(h_eff) + 0.848379709041268);
    double alpha = 10.0 / (2.0 * ecut);

    double V = Jacbdet_;  // Cell volume (Jacbdet_ = lattice.jacobian() = full cell volume in Bohr^3)
    double omega = hyb_range_fock_;
    double omega2 = omega * omega;

    // Iterate over full HF k-grid uniformly (matching SPARC exactly)
    // q[d] = ihf * 2*pi / (L_d * K_hf_d) for ihf = 0..K_hf_d-1
    double sumfGq = 0.0;
    for (int khf = 0; khf < Kz_hf_; khf++) {
        for (int jhf = 0; jhf < Ky_hf_; jhf++) {
            for (int ihf = 0; ihf < Kx_hf_; ihf++) {
                double q0 = ihf * tpiblx / Kx_hf_;
                double q1 = jhf * tpibly / Ky_hf_;
                double q2 = khf * tpiblz / Kz_hf_;

                for (int k = 0; k < Nz_; k++) {
                    for (int j = 0; j < Ny_; j++) {
                        for (int i = 0; i < Nx_; i++) {
                            double G[3], g[3];
                            G[0] = (i < Nx_/2+1) ? (i * tpiblx) : ((i - Nx_) * tpiblx);
                            G[1] = (j < Ny_/2+1) ? (j * tpibly) : ((j - Ny_) * tpibly);
                            G[2] = (k < Nz_/2+1) ? (k * tpiblz) : ((k - Nz_) * tpiblz);

                            G[0] += q0; G[1] += q1; G[2] += q2;

                            g[0] = lapcT_(0,0)*G[0] + lapcT_(0,1)*G[1] + lapcT_(0,2)*G[2];
                            g[1] = lapcT_(1,0)*G[0] + lapcT_(1,1)*G[1] + lapcT_(1,2)*G[2];
                            g[2] = lapcT_(2,0)*G[0] + lapcT_(2,1)*G[1] + lapcT_(2,2)*G[2];
                            double modGq = G[0]*g[0] + G[1]*g[1] + G[2]*g[2];

                            if (modGq > 1e-8) {
                                if (omega < 0)
                                    sumfGq += std::exp(-alpha * modGq) / modGq;
                                else
                                    sumfGq += std::exp(-alpha * modGq) / modGq * (1.0 - std::exp(-modGq / (4.0 * omega2)));
                            }
                        }
                    }
                }
            }
        }
    }

    if (omega < 0) {
        double scaled_intf = V * Nkpts_hf_ / (4.0 * constants::PI * std::sqrt(constants::PI * alpha));
        const_aux_ = scaled_intf + alpha - sumfGq;
    } else {
        sumfGq += 0.25 / omega2;
        int nqq = 100000;
        double dq = 5.0 / std::sqrt(alpha) / nqq;
        double aa = 0.0;
        for (int iq = 0; iq < nqq; iq++) {
            double q_ = dq * (iq + 0.5);
            double qq = q_ * q_;
            aa -= std::exp(-alpha * qq) * std::exp(-0.25 * qq / omega2) * dq;
        }
        aa = 2.0 * aa / constants::PI + 1.0 / std::sqrt(constants::PI * alpha);
        double scaled_intf = V * Nkpts_hf_ / (4.0 * constants::PI) * aa;
        const_aux_ = scaled_intf - sumfGq;
    }
}

// ---------------------------------------------------------------------------
// Compute FFT Poisson constants
// ---------------------------------------------------------------------------
void ExchangePoissonSolver::compute_pois_fft_const() {
    int Nx_len = is_gamma_ ? (Nx_ / 2 + 1) : Nx_;
    int Nd_per_shift = is_gamma_ ? Ndc_ : Nd_;

    double tpiblx = 2.0 * constants::PI / L1_;
    double tpibly = 2.0 * constants::PI / L2_;
    double tpiblz = 2.0 * constants::PI / L3_;

    pois_const_.resize(Nd_per_shift * Nkpts_shift_);
    pois_const_stress_.resize(Nd_per_shift * Nkpts_shift_);
    pois_const_stress2_.resize(Nd_per_shift * Nkpts_shift_);

    for (int l = 0; l < Nkpts_shift_; l++) {
        int count = 0;
        for (int k = 0; k < Nz_; k++) {
            for (int j = 0; j < Ny_; j++) {
                for (int i = 0; i < Nx_len; i++) {
                    double G[3], g[3] = {0, 0, 0};
                    G[0] = (i < Nx_/2+1) ? (i * tpiblx) : ((i - Nx_) * tpiblx);
                    G[1] = (j < Ny_/2+1) ? (j * tpibly) : ((j - Ny_) * tpibly);
                    G[2] = (k < Nz_/2+1) ? (k * tpiblz) : ((k - Nz_) * tpiblz);

                    // Add k-shift (last shift is zero = identity)
                    if (l < Nkpts_shift_ - 1) {
                        G[0] += k1_shift_[l];
                        G[1] += k2_shift_[l];
                        G[2] += k3_shift_[l];
                    }

                    // lapcT * G (metric transformation for non-orthogonal cells)
                    g[0] = lapcT_(0,0)*G[0] + lapcT_(0,1)*G[1] + lapcT_(0,2)*G[2];
                    g[1] = lapcT_(1,0)*G[0] + lapcT_(1,1)*G[1] + lapcT_(1,2)*G[2];
                    g[2] = lapcT_(2,0)*G[0] + lapcT_(2,1)*G[1] + lapcT_(2,2)*G[2];
                    double G2 = G[0]*g[0] + G[1]*g[1] + G[2]*g[2];

                    int idx = count + l * Nd_per_shift;
                    pois_const_[idx] = singularity_removal_const(G2);
                    singularity_removal_const_stress(G2,
                        pois_const_stress_[idx], pois_const_stress2_[idx]);
                    count++;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Find unique k-point shifts (k-q)
// ---------------------------------------------------------------------------
void ExchangePoissonSolver::find_k_shift(const KPoints* kpoints) {
    if (!kpoints || kpoints->is_gamma_only()) {
        Nkpts_shift_ = 1;
        Nkpts_sym_ = 1;
        Nkpts_hf_ = 1;
        Kptshift_map_.assign(1, 0);
        return;
    }

    Nkpts_sym_ = kpoints->Nkpts();
    Nkpts_hf_ = kpoints->Nkpts_hf();  // full HF BZ (includes time-reversed)

    const auto& kpts = kpoints->kpts_cart();
    const auto& kpthf_ind = kpoints->kpthf_ind();
    const auto& kpthf_pn = kpoints->kpthf_pn();
    constexpr double TOL = 1e-12;

    // Temporary storage for shifts
    std::vector<double> tmp_k1(Nkpts_sym_ * Nkpts_hf_ + 1);
    std::vector<double> tmp_k2(Nkpts_sym_ * Nkpts_hf_ + 1);
    std::vector<double> tmp_k3(Nkpts_sym_ * Nkpts_hf_ + 1);

    // Map indexed by [k_sym + q_hf * Nkpts_sym] (q over FULL HF BZ)
    Kptshift_map_.assign(Nkpts_sym_ * Nkpts_hf_, 0);

    // First shift is always {0,0,0}
    tmp_k1[0] = tmp_k2[0] = tmp_k3[0] = 0.0;
    int count = 1;

    for (int q_hf = 0; q_hf < Nkpts_hf_; q_hf++) {
        int q_sym = kpthf_ind[q_hf];
        // For time-reversed q, use -kpts[q_sym]; for direct, use +kpts[q_sym]
        double sign = (kpthf_pn[q_hf] == 1) ? 1.0 : -1.0;
        double qx = sign * kpts[q_sym].x;
        double qy = sign * kpts[q_sym].y;
        double qz = sign * kpts[q_sym].z;

        for (int k_ind = 0; k_ind < Nkpts_sym_; k_ind++) {
            double dk1 = kpts[k_ind].x - qx;
            double dk2 = kpts[k_ind].y - qy;
            double dk3 = kpts[k_ind].z - qz;

            if (std::fabs(dk1) < TOL && std::fabs(dk2) < TOL && std::fabs(dk3) < TOL) {
                Kptshift_map_[k_ind + q_hf * Nkpts_sym_] = 0;
                continue;
            }

            // Check if this shift already exists
            int flag = 0;
            for (int i = 1; i < count; i++) {
                if (std::fabs(tmp_k1[i] - dk1) < TOL &&
                    std::fabs(tmp_k2[i] - dk2) < TOL &&
                    std::fabs(tmp_k3[i] - dk3) < TOL) {
                    flag = i;
                    break;
                }
            }

            if (flag == 0) {
                tmp_k1[count] = dk1;
                tmp_k2[count] = dk2;
                tmp_k3[count] = dk3;
                Kptshift_map_[k_ind + q_hf * Nkpts_sym_] = count;
                count++;
            } else {
                Kptshift_map_[k_ind + q_hf * Nkpts_sym_] = flag;
            }
        }
    }

    Nkpts_shift_ = count;

    // Store shifts (excluding the zero shift which is implicit as last)
    k1_shift_.resize(Nkpts_shift_ - 1);
    k2_shift_.resize(Nkpts_shift_ - 1);
    k3_shift_.resize(Nkpts_shift_ - 1);
    for (int i = 0; i < Nkpts_shift_ - 1; i++) {
        k1_shift_[i] = tmp_k1[i + 1];
        k2_shift_[i] = tmp_k2[i + 1];
        k3_shift_[i] = tmp_k3[i + 1];
    }
}

// ---------------------------------------------------------------------------
// Precompute phase factors for k-point shifts
// ---------------------------------------------------------------------------
void ExchangePoissonSolver::kshift_phasefactor() {
    if (Nkpts_shift_ <= 1) return;

    int Nxy = Nx_ * Ny_;
    neg_phase_.resize(Nd_ * (Nkpts_shift_ - 1));
    pos_phase_.resize(Nd_ * (Nkpts_shift_ - 1));

    for (int l = 0; l < Nkpts_shift_ - 1; l++) {
        for (int k = 0; k < Nz_; k++) {
            for (int j = 0; j < Ny_; j++) {
                for (int i = 0; i < Nx_; i++) {
                    double rx = i * L1_ / Nx_;
                    double ry = j * L2_ / Ny_;
                    double rz = k * L3_ / Nz_;
                    double dot = rx * k1_shift_[l] + ry * k2_shift_[l] + rz * k3_shift_[l];
                    int idx = l * Nd_ + k * Nxy + j * Nx_ + i;
                    neg_phase_[idx] = Complex(std::cos(dot), -std::sin(dot));
                    pos_phase_[idx] = Complex(std::cos(dot),  std::sin(dot));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Apply phase factor to complex data
// ---------------------------------------------------------------------------
void ExchangePoissonSolver::apply_phase_factor(Complex* data, int ncol, bool positive,
                                                int kpt_k, int kpt_q) const {
    int l = Kptshift_map_[kpt_k + kpt_q * Nkpts_sym_];
    if (l == 0) return;  // zero shift, no phase factor needed

    const Complex* phase = positive ? (pos_phase_.data() + (l - 1) * Nd_)
                                     : (neg_phase_.data() + (l - 1) * Nd_);

    for (int n = 0; n < ncol; n++) {
        Complex* dn = data + n * Nd_;
        for (int i = 0; i < Nd_; i++) {
            dn[i] *= phase[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------
void ExchangePoissonSolver::setup(const FDGrid& grid, const Lattice& lattice,
                                   const EXXParams& params, const KPoints* kpoints,
                                   int Kx_hf, int Ky_hf, int Kz_hf) {
    Nx_ = grid.Nx(); Ny_ = grid.Ny(); Nz_ = grid.Nz();
    Nd_ = Nx_ * Ny_ * Nz_;
    Ndc_ = Nz_ * Ny_ * (Nx_ / 2 + 1);
    dx_ = grid.dx(); dy_ = grid.dy(); dz_ = grid.dz();
    L1_ = dx_ * Nx_; L2_ = dy_ * Ny_; L3_ = dz_ * Nz_;
    lapcT_ = lattice.lapc_T();
    Jacbdet_ = lattice.jacobian();
    exx_div_flag_ = params.exx_div_flag;
    hyb_range_fock_ = params.hyb_range_fock;

    is_gamma_ = !kpoints || kpoints->is_gamma_only();

    // Store HF k-grid dimensions
    Kx_hf_ = Kx_hf; Ky_hf_ = Ky_hf; Kz_hf_ = Kz_hf;
    Nkpts_hf_ = Kx_hf * Ky_hf * Kz_hf;

    // Find unique k-shifts and precompute phase factors
    find_k_shift(kpoints);
    if (!is_gamma_) {
        kshift_phasefactor();
    }

    // Compute auxiliary constant for G=0 if needed
    if (exx_div_flag_ == 1) {
        compute_auxiliary_constant();
    }

    // Compute FFT Poisson constants
    compute_pois_fft_const();

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::printf("EXX Poisson solver: Nkpts_shift=%d, div_flag=%d, omega=%.4f\n",
                    Nkpts_shift_, exx_div_flag_, hyb_range_fock_);
    }
}

// ---------------------------------------------------------------------------
// Solve batch — gamma-point (real FFT)
// ---------------------------------------------------------------------------
void ExchangePoissonSolver::solve_batch(double* rhs, int ncol, double* sol) {
    if (ncol == 0) return;

    std::vector<Complex> rhs_bar(Ndc_ * ncol);

    // Forward FFT (real to complex) via MKL DFTI
    {
        DFTI_DESCRIPTOR_HANDLE desc = nullptr;
        MKL_LONG dims[3] = {Nz_, Ny_, Nx_};
        DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_REAL, 3, dims);
        DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(desc, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        MKL_LONG si[4] = {0, (MKL_LONG)(Ny_*Nx_), (MKL_LONG)Nx_, 1};
        MKL_LONG so[4] = {0, (MKL_LONG)(Ny_*(Nx_/2+1)), (MKL_LONG)(Nx_/2+1), 1};
        DftiSetValue(desc, DFTI_INPUT_STRIDES, si);
        DftiSetValue(desc, DFTI_OUTPUT_STRIDES, so);
        if (ncol > 1) {
            DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)ncol);
            DftiSetValue(desc, DFTI_INPUT_DISTANCE, (MKL_LONG)Nd_);
            DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, (MKL_LONG)Ndc_);
        }
        DftiCommitDescriptor(desc);
        DftiComputeForward(desc, rhs, reinterpret_cast<double*>(rhs_bar.data()));
        DftiFreeDescriptor(&desc);
    }

    // Multiply by Poisson constants (last shift = zero shift for gamma)
    const double* alpha = pois_const_.data() + (Nkpts_shift_ - 1) * Ndc_;
    for (int n = 0; n < ncol; n++) {
        Complex* bar = rhs_bar.data() + n * Ndc_;
        for (int i = 0; i < Ndc_; i++)
            bar[i] = Complex(bar[i].real() * alpha[i], bar[i].imag() * alpha[i]);
    }

    // Inverse FFT (complex to real) via MKL DFTI
    {
        DFTI_DESCRIPTOR_HANDLE desc = nullptr;
        MKL_LONG dims[3] = {Nz_, Ny_, Nx_};
        DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_REAL, 3, dims);
        DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(desc, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        MKL_LONG si[4] = {0, (MKL_LONG)(Ny_*(Nx_/2+1)), (MKL_LONG)(Nx_/2+1), 1};
        MKL_LONG so[4] = {0, (MKL_LONG)(Ny_*Nx_), (MKL_LONG)Nx_, 1};
        DftiSetValue(desc, DFTI_INPUT_STRIDES, si);
        DftiSetValue(desc, DFTI_OUTPUT_STRIDES, so);
        if (ncol > 1) {
            DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)ncol);
            DftiSetValue(desc, DFTI_INPUT_DISTANCE, (MKL_LONG)Ndc_);
            DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, (MKL_LONG)Nd_);
        }
        DftiCommitDescriptor(desc);
        DftiComputeBackward(desc, reinterpret_cast<double*>(rhs_bar.data()), sol);
        DftiFreeDescriptor(&desc);
    }

    // Normalize (MKL produces unnormalized output)
    double inv_Nd = 1.0 / Nd_;
    for (int i = 0; i < Nd_ * ncol; i++)
        sol[i] *= inv_Nd;
}

// ---------------------------------------------------------------------------
// Solve batch — gamma stress (real FFT, stress Poisson constants)
// ---------------------------------------------------------------------------
void ExchangePoissonSolver::solve_batch_stress(double* rhs, int ncol, double* sol, int option) {
    if (ncol == 0) return;

    std::vector<Complex> rhs_bar(Ndc_ * ncol);

    // Forward FFT (real to complex)
    {
        DFTI_DESCRIPTOR_HANDLE desc = nullptr;
        MKL_LONG dims[3] = {Nz_, Ny_, Nx_};
        DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_REAL, 3, dims);
        DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(desc, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        MKL_LONG si[4] = {0, (MKL_LONG)(Ny_*Nx_), (MKL_LONG)Nx_, 1};
        MKL_LONG so[4] = {0, (MKL_LONG)(Ny_*(Nx_/2+1)), (MKL_LONG)(Nx_/2+1), 1};
        DftiSetValue(desc, DFTI_INPUT_STRIDES, si);
        DftiSetValue(desc, DFTI_OUTPUT_STRIDES, so);
        if (ncol > 1) {
            DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)ncol);
            DftiSetValue(desc, DFTI_INPUT_DISTANCE, (MKL_LONG)Nd_);
            DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, (MKL_LONG)Ndc_);
        }
        DftiCommitDescriptor(desc);
        DftiComputeForward(desc, rhs, reinterpret_cast<double*>(rhs_bar.data()));
        DftiFreeDescriptor(&desc);
    }

    // Multiply by stress Poisson constants (last shift = zero shift for gamma)
    const auto& pconst = (option == 2) ? pois_const_stress2_ : pois_const_stress_;
    const double* alpha = pconst.data() + (Nkpts_shift_ - 1) * Ndc_;
    for (int n = 0; n < ncol; n++) {
        Complex* bar = rhs_bar.data() + n * Ndc_;
        for (int i = 0; i < Ndc_; i++)
            bar[i] = Complex(bar[i].real() * alpha[i], bar[i].imag() * alpha[i]);
    }

    // Inverse FFT (complex to real)
    {
        DFTI_DESCRIPTOR_HANDLE desc = nullptr;
        MKL_LONG dims[3] = {Nz_, Ny_, Nx_};
        DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_REAL, 3, dims);
        DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(desc, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        MKL_LONG si[4] = {0, (MKL_LONG)(Ny_*(Nx_/2+1)), (MKL_LONG)(Nx_/2+1), 1};
        MKL_LONG so[4] = {0, (MKL_LONG)(Ny_*Nx_), (MKL_LONG)Nx_, 1};
        DftiSetValue(desc, DFTI_INPUT_STRIDES, si);
        DftiSetValue(desc, DFTI_OUTPUT_STRIDES, so);
        if (ncol > 1) {
            DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)ncol);
            DftiSetValue(desc, DFTI_INPUT_DISTANCE, (MKL_LONG)Ndc_);
            DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, (MKL_LONG)Nd_);
        }
        DftiCommitDescriptor(desc);
        DftiComputeBackward(desc, reinterpret_cast<double*>(rhs_bar.data()), sol);
        DftiFreeDescriptor(&desc);
    }

    double inv_Nd = 1.0 / Nd_;
    for (int i = 0; i < Nd_ * ncol; i++)
        sol[i] *= inv_Nd;
}

// ---------------------------------------------------------------------------
// Solve batch — k-point (complex FFT)
// ---------------------------------------------------------------------------
void ExchangePoissonSolver::solve_batch_kpt(const Complex* rhs, int ncol, Complex* sol,
                                             int kpt_k, int kpt_q) {
    if (ncol == 0) return;

    // Work copy of rhs (apply_phase_factor modifies in-place)
    std::vector<Complex> rhs_copy(Nd_ * ncol);
    std::memcpy(rhs_copy.data(), rhs, sizeof(Complex) * Nd_ * ncol);

    // Apply negative phase factor: rhs *= exp(-i*(k-q)*r)
    apply_phase_factor(rhs_copy.data(), ncol, false, kpt_k, kpt_q);

    // Forward FFT (complex to complex) via MKL DFTI
    std::vector<Complex> rhs_bar(Nd_ * ncol);
    {
        DFTI_DESCRIPTOR_HANDLE desc = nullptr;
        MKL_LONG dims[3] = {Nz_, Ny_, Nx_};
        DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 3, dims);
        DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        if (ncol > 1) {
            DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)ncol);
            DftiSetValue(desc, DFTI_INPUT_DISTANCE, (MKL_LONG)Nd_);
            DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, (MKL_LONG)Nd_);
        }
        DftiCommitDescriptor(desc);
        DftiComputeForward(desc, rhs_copy.data(), rhs_bar.data());
        DftiFreeDescriptor(&desc);
    }

    // Multiply by Poisson constants
    int l = Kptshift_map_[kpt_k + kpt_q * Nkpts_sym_];
    const double* alpha;
    if (l == 0) {
        alpha = pois_const_.data() + Nd_ * (Nkpts_shift_ - 1);
    } else {
        alpha = pois_const_.data() + Nd_ * (l - 1);
    }

    for (int n = 0; n < ncol; n++) {
        Complex* bar = rhs_bar.data() + n * Nd_;
        for (int i = 0; i < Nd_; i++) {
            bar[i] = Complex(bar[i].real() * alpha[i], bar[i].imag() * alpha[i]);
        }
    }

    // Inverse FFT (complex to complex) via MKL DFTI
    {
        DFTI_DESCRIPTOR_HANDLE desc = nullptr;
        MKL_LONG dims[3] = {Nz_, Ny_, Nx_};
        DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 3, dims);
        DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        if (ncol > 1) {
            DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)ncol);
            DftiSetValue(desc, DFTI_INPUT_DISTANCE, (MKL_LONG)Nd_);
            DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, (MKL_LONG)Nd_);
        }
        DftiCommitDescriptor(desc);
        DftiComputeBackward(desc, rhs_bar.data(), sol);
        DftiFreeDescriptor(&desc);
    }

    // Normalize
    double inv_Nd = 1.0 / Nd_;
    for (int i = 0; i < Nd_ * ncol; i++) {
        sol[i] *= inv_Nd;
    }

    // Apply positive phase factor: sol *= exp(+i*(k-q)*r)
    apply_phase_factor(sol, ncol, true, kpt_k, kpt_q);
}

// ---------------------------------------------------------------------------
// Solve batch — k-point with stress Poisson constants
// option=1: pois_const_stress_, option=2: pois_const_stress2_
// ---------------------------------------------------------------------------
void ExchangePoissonSolver::solve_batch_kpt_stress(const Complex* rhs, int ncol, Complex* sol,
                                                    int kpt_k, int kpt_q, int option) {
    if (ncol == 0) return;

    std::vector<Complex> rhs_copy(Nd_ * ncol);
    std::memcpy(rhs_copy.data(), rhs, sizeof(Complex) * Nd_ * ncol);

    apply_phase_factor(rhs_copy.data(), ncol, false, kpt_k, kpt_q);

    std::vector<Complex> rhs_bar(Nd_ * ncol);
    {
        DFTI_DESCRIPTOR_HANDLE desc = nullptr;
        MKL_LONG dims[3] = {Nz_, Ny_, Nx_};
        DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 3, dims);
        DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        if (ncol > 1) {
            DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)ncol);
            DftiSetValue(desc, DFTI_INPUT_DISTANCE, (MKL_LONG)Nd_);
            DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, (MKL_LONG)Nd_);
        }
        DftiCommitDescriptor(desc);
        DftiComputeForward(desc, rhs_copy.data(), rhs_bar.data());
        DftiFreeDescriptor(&desc);
    }

    // Select appropriate stress constants
    int l = Kptshift_map_[kpt_k + kpt_q * Nkpts_sym_];
    const double* alpha;
    const auto& pconst = (option == 2) ? pois_const_stress2_ : pois_const_stress_;
    if (l == 0) {
        alpha = pconst.data() + Nd_ * (Nkpts_shift_ - 1);
    } else {
        alpha = pconst.data() + Nd_ * (l - 1);
    }

    for (int n = 0; n < ncol; n++) {
        Complex* bar = rhs_bar.data() + n * Nd_;
        for (int i = 0; i < Nd_; i++) {
            bar[i] = Complex(bar[i].real() * alpha[i], bar[i].imag() * alpha[i]);
        }
    }

    {
        DFTI_DESCRIPTOR_HANDLE desc = nullptr;
        MKL_LONG dims[3] = {Nz_, Ny_, Nx_};
        DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 3, dims);
        DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        if (ncol > 1) {
            DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)ncol);
            DftiSetValue(desc, DFTI_INPUT_DISTANCE, (MKL_LONG)Nd_);
            DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, (MKL_LONG)Nd_);
        }
        DftiCommitDescriptor(desc);
        DftiComputeBackward(desc, rhs_bar.data(), sol);
        DftiFreeDescriptor(&desc);
    }

    double inv_Nd = 1.0 / Nd_;
    for (int i = 0; i < Nd_ * ncol; i++)
        sol[i] *= inv_Nd;

    apply_phase_factor(sol, ncol, true, kpt_k, kpt_q);
}

} // namespace lynx
