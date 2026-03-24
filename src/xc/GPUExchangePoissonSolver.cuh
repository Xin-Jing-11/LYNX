#pragma once
#ifdef USE_CUDA

#include <cufft.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <vector>

namespace lynx {
namespace gpu {

// GPU FFT-based Poisson solver for exact exchange.
//
// Gamma-point: cuFFT R2C/C2R with real Poisson constants [Ndc].
// K-point: cuFFT Z2Z with phase factors and per-shift Poisson constants [Nd * Nkpts_shift].
class GPUExchangePoissonSolver {
public:
    GPUExchangePoissonSolver() = default;
    ~GPUExchangePoissonSolver();

    GPUExchangePoissonSolver(const GPUExchangePoissonSolver&) = delete;
    GPUExchangePoissonSolver& operator=(const GPUExchangePoissonSolver&) = delete;

    // Setup for gamma-point: cuFFT R2C/C2R plans, Poisson constants [Ndc].
    void setup(int Nx, int Ny, int Nz, const double* pois_const, int max_ncol);

    // Setup for k-points: cuFFT Z2Z plans, per-shift Poisson constants [Nd * Nkpts_shift],
    // phase factors [Nd * (Nkpts_shift-1)], and Kptshift_map [Nkpts_sym * Nkpts_hf].
    void setup_kpt(int Nx, int Ny, int Nz,
                   const double* pois_const, int Nkpts_shift,
                   const void* neg_phase, const void* pos_phase,
                   const int* Kptshift_map, int Nkpts_sym, int Nkpts_hf,
                   int max_ncol);

    // Solve batch of Poisson equations on GPU (gamma-point, real).
    void solve_batch(double* d_rhs, int ncol, double* d_sol, cublasHandle_t cublas);

    // Solve batch of Poisson equations on GPU (k-point, complex Z2Z).
    // d_rhs: [Nd * ncol] complex input (NOT destroyed -- internal copy made)
    // d_sol: [Nd * ncol] complex output
    void solve_batch_kpt(const cuDoubleComplex* d_rhs, int ncol, cuDoubleComplex* d_sol,
                         cublasHandle_t cublas, int kpt_k, int kpt_q);

    // Solve with stress Poisson constants.
    void solve_batch_stress(double* d_rhs, int ncol, double* d_sol,
                            cublasHandle_t cublas, int option);

    // Upload stress Poisson constants (called separately when stress is needed).
    void upload_stress_constants(const double* pois_const_stress,
                                 const double* pois_const_stress2);

    int Nd() const { return Nd_; }
    int Ndc() const { return Ndc_; }
    bool is_setup() const { return is_setup_; }
    bool is_kpt() const { return is_kpt_; }

private:
    bool is_setup_ = false;
    bool is_kpt_ = false;
    int Nx_ = 0, Ny_ = 0, Nz_ = 0;
    int Nd_ = 0;    // Nx*Ny*Nz
    int Ndc_ = 0;   // Nz*Ny*(Nx/2+1)
    int max_ncol_ = 0;

    // cuFFT plans -- gamma: R2C/C2R, k-point: Z2Z forward/backward
    cufftHandle plan_r2c_ = 0;
    cufftHandle plan_c2r_ = 0;
    cufftHandle plan_z2z_fwd_ = 0;
    cufftHandle plan_z2z_inv_ = 0;
    bool plans_created_ = false;

    // Device Poisson constants
    // Gamma: [Ndc], K-point: [Nd * Nkpts_shift]
    double* d_pois_const_ = nullptr;
    double* d_pois_const_stress_ = nullptr;
    double* d_pois_const_stress2_ = nullptr;

    // Workspace for complex FFT output
    // Gamma: [Ndc * max_ncol], K-point: [Nd * max_ncol]
    cufftDoubleComplex* d_fft_work_ = nullptr;

    // K-point phase factor data (on device)
    int Nkpts_shift_ = 1;
    int Nkpts_sym_ = 0;
    int Nkpts_hf_ = 0;
    cuDoubleComplex* d_neg_phase_ = nullptr;  // [Nd * (Nkpts_shift-1)]
    cuDoubleComplex* d_pos_phase_ = nullptr;  // [Nd * (Nkpts_shift-1)]
    int* d_Kptshift_map_ = nullptr;           // [Nkpts_sym * Nkpts_hf]
    // Host copy for indexing
    std::vector<int> h_Kptshift_map_;

    // K-point scratch: copy of input for in-place phase factor application
    cuDoubleComplex* d_kpt_scratch_ = nullptr;

    void cleanup();
    void create_plans(int ncol);
    void create_plans_z2z(int ncol);

    // Internal solve using given device Poisson constants (gamma)
    void solve_batch_impl(double* d_rhs, int ncol, double* d_sol,
                          cublasHandle_t cublas, const double* d_pconst);
};

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
