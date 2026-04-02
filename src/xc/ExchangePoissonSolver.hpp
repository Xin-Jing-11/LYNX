#pragma once

#include "core/types.hpp"
#include "core/FDGrid.hpp"
#include "core/Lattice.hpp"
#include "core/KPoints.hpp"

#include <vector>
#include <complex>

#ifdef USE_MKL
#ifndef USE_CUDA
#include <mkl_dfti.h>
#endif
#endif

namespace lynx {

// FFT-based Poisson solver for exact exchange.
//
// Solves: -Lap(sol) = rhs in Fourier space, with singularity removal
// appropriate for the exchange integral (spherical cutoff or auxiliary function).
//
// Since LYNX has NO domain decomposition, every call operates on the full grid.
// No MPI communication is needed within the solver.
class ExchangePoissonSolver {
public:
    ExchangePoissonSolver() = default;
    ~ExchangePoissonSolver();

    // Non-copyable (owns DFTI descriptors)
    ExchangePoissonSolver(const ExchangePoissonSolver&) = delete;
    ExchangePoissonSolver& operator=(const ExchangePoissonSolver&) = delete;
    ExchangePoissonSolver(ExchangePoissonSolver&&) noexcept;
    ExchangePoissonSolver& operator=(ExchangePoissonSolver&&) noexcept;

    // Setup the solver and precompute FFT Poisson constants.
    // For gamma-point: Nkpts_shift=1, no phase factors.
    // For k-points: computes unique (k-q) shifts and phase factors.
    // Kx_hf, Ky_hf, Kz_hf: HF k-point grid dimensions (= Monkhorst-Pack grid)
    void setup(const FDGrid& grid, const Lattice& lattice,
               const EXXParams& params, const KPoints* kpoints,
               int Kx_hf = 1, int Ky_hf = 1, int Kz_hf = 1);

    // Solve batch of Poisson equations (gamma-point, real).
    // rhs: [Nd * ncol] input (destroyed)
    // sol: [Nd * ncol] output
    void solve_batch(double* rhs, int ncol, double* sol);
    // Gamma-point stress solve: option=1 uses pois_const_stress_, option=2 uses pois_const_stress2_
    void solve_batch_stress(double* rhs, int ncol, double* sol, int option);

    // Solve batch of Poisson equations (k-point, complex).
    // Applies phase factors exp(±i*(k-q)*r) internally.
    // rhs: [Nd * ncol] complex input (NOT destroyed — internal copy made)
    // sol: [Nd * ncol] complex output
    void solve_batch_kpt(const Complex* rhs, int ncol, Complex* sol,
                         int kpt_k, int kpt_q);

    // Solve with stress Poisson constants (for EXX stress computation).
    // option=1: pois_const_stress, option=2: pois_const_stress2
    void solve_batch_kpt_stress(const Complex* rhs, int ncol, Complex* sol,
                                int kpt_k, int kpt_q, int option = 1);

    int Nkpts_shift() const { return Nkpts_shift_; }
    const std::vector<int>& Kptshift_map() const { return Kptshift_map_; }
    bool has_stress_constants() const { return !pois_const_stress_.empty(); }

    // Accessors for GPU solver setup
    int Nx() const { return Nx_; }
    int Ny() const { return Ny_; }
    int Nz() const { return Nz_; }
    int Nd() const { return Nd_; }
    int Ndc() const { return Ndc_; }
    const double* pois_const_data() const { return pois_const_.data(); }
    const double* pois_const_stress_data() const { return pois_const_stress_.data(); }
    const double* pois_const_stress2_data() const { return pois_const_stress2_.data(); }
    // K-point phase factor accessors for GPU upload
    const Complex* neg_phase_data() const { return neg_phase_.data(); }
    const Complex* pos_phase_data() const { return pos_phase_.data(); }
    int Nkpts_sym() const { return Nkpts_sym_; }
    int Nkpts_hf() const { return Nkpts_hf_; }

    // Public wrapper for phase factor application (for diagnostics)
    void apply_phase_factor_public(Complex* data, int ncol, bool positive, int kpt_k, int kpt_q) const {
        apply_phase_factor(data, ncol, positive, kpt_k, kpt_q);
    }

    // Get k-point shift index for a given (k, q) pair
    int get_kpt_shift(int kpt_k, int kpt_q) const {
        return Kptshift_map_[kpt_k + kpt_q * Nkpts_sym_];
    }

private:
    int Nx_ = 0, Ny_ = 0, Nz_ = 0;
    int Nd_ = 0;       // Nx*Ny*Nz
    int Ndc_ = 0;      // Nz*Ny*(Nx/2+1) for real FFT output
    bool is_gamma_ = true;

    // Precomputed Poisson constants in Fourier space
    // For gamma: [Ndc * Nkpts_shift] (real FFT output size per shift)
    // For k-point: [Nd * Nkpts_shift]
    std::vector<double> pois_const_;
    // Stress Poisson constants (only allocated when stress is needed)
    std::vector<double> pois_const_stress_;   // d/dG2 of pois_const
    std::vector<double> pois_const_stress2_;  // spherical truncation correction

    // K-point shift infrastructure
    int Nkpts_shift_ = 1;
    int Nkpts_sym_ = 0;
    int Nkpts_hf_ = 0;
    std::vector<double> k1_shift_, k2_shift_, k3_shift_;
    std::vector<int> Kptshift_map_;  // [Nkpts_sym * Nkpts_hf]

    // Phase factors for k-point shifts
    std::vector<Complex> neg_phase_;  // [Nd * (Nkpts_shift-1)]
    std::vector<Complex> pos_phase_;  // [Nd * (Nkpts_shift-1)]

    // Auxiliary function constant for G=0 (HSE)
    double const_aux_ = 0.0;

    // Grid parameters
    double dx_ = 0, dy_ = 0, dz_ = 0;
    double L1_ = 0, L2_ = 0, L3_ = 0;
    Mat3 lapcT_;
    double Jacbdet_ = 1.0;

    // EXX params
    int exx_div_flag_ = 0;
    double hyb_range_fock_ = -1.0;

    // Internal methods
    int Kx_hf_ = 1, Ky_hf_ = 1, Kz_hf_ = 1;

    void compute_auxiliary_constant();
    double singularity_removal_const(double G2) const;
    void singularity_removal_const_stress(double G2, double& stress, double& stress2) const;
    void compute_pois_fft_const();
    void find_k_shift(const KPoints* kpoints);
    void kshift_phasefactor();

    // Apply phase factor to complex array
    void apply_phase_factor(Complex* data, int ncol, bool positive, int kpt_k, int kpt_q) const;

#if defined(USE_MKL) && !defined(USE_CUDA)
    // Persistent MKL DFTI descriptors (cached by ncol to avoid per-call creation)
    DFTI_DESCRIPTOR_HANDLE desc_r2c_ = nullptr;  // gamma forward (real-to-complex)
    DFTI_DESCRIPTOR_HANDLE desc_c2r_ = nullptr;  // gamma inverse (complex-to-real)
    DFTI_DESCRIPTOR_HANDLE desc_fwd_ = nullptr;  // k-point forward (complex-to-complex)
    DFTI_DESCRIPTOR_HANDLE desc_inv_ = nullptr;  // k-point inverse (complex-to-complex)
    int cached_ncol_r2c_ = 0;  // ncol for which R2C/C2R descriptors are valid
    int cached_ncol_c2c_ = 0;  // ncol for which C2C descriptors are valid

    void ensure_r2c_descriptors(int ncol);
    void ensure_c2c_descriptors(int ncol);
    void free_descriptors();
#endif
};

} // namespace lynx
