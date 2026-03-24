#pragma once
#ifdef USE_CUDA

#include <cufft.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

namespace lynx {
namespace gpu {

// GPU FFT-based Poisson solver for exact exchange (gamma-point, real).
//
// Mirrors ExchangePoissonSolver::solve_batch() but uses cuFFT R2C/C2R.
// The cufftHandle and device Poisson constants are created once and reused.
class GPUExchangePoissonSolver {
public:
    GPUExchangePoissonSolver() = default;
    ~GPUExchangePoissonSolver();

    GPUExchangePoissonSolver(const GPUExchangePoissonSolver&) = delete;
    GPUExchangePoissonSolver& operator=(const GPUExchangePoissonSolver&) = delete;

    // Setup: create cuFFT plans and upload Poisson constants to device.
    // pois_const: host array [Ndc] (last shift slice for gamma-point)
    // max_ncol: max batch size (plan is created for this batch size)
    void setup(int Nx, int Ny, int Nz, const double* pois_const, int max_ncol);

    // Solve batch of Poisson equations on GPU.
    // d_rhs: [Nd * ncol] device input (destroyed by forward FFT)
    // d_sol: [Nd * ncol] device output
    void solve_batch(double* d_rhs, int ncol, double* d_sol, cublasHandle_t cublas);

    // Solve with stress Poisson constants.
    // option=1: uses d_pois_const_stress_, option=2: uses d_pois_const_stress2_
    void solve_batch_stress(double* d_rhs, int ncol, double* d_sol,
                            cublasHandle_t cublas, int option);

    // Upload stress Poisson constants (called separately when stress is needed).
    void upload_stress_constants(const double* pois_const_stress,
                                 const double* pois_const_stress2);

    int Nd() const { return Nd_; }
    int Ndc() const { return Ndc_; }
    bool is_setup() const { return is_setup_; }

private:
    bool is_setup_ = false;
    int Nx_ = 0, Ny_ = 0, Nz_ = 0;
    int Nd_ = 0;    // Nx*Ny*Nz
    int Ndc_ = 0;   // Nz*Ny*(Nx/2+1)
    int max_ncol_ = 0;

    // cuFFT plans (batched R2C and C2R)
    cufftHandle plan_r2c_ = 0;
    cufftHandle plan_c2r_ = 0;
    bool plans_created_ = false;

    // Device Poisson constants [Ndc]
    double* d_pois_const_ = nullptr;
    double* d_pois_const_stress_ = nullptr;
    double* d_pois_const_stress2_ = nullptr;

    // Workspace for complex FFT output [Ndc * max_ncol]
    cufftDoubleComplex* d_fft_work_ = nullptr;

    void cleanup();
    void create_plans(int ncol);

    // Internal solve using given device Poisson constants
    void solve_batch_impl(double* d_rhs, int ncol, double* d_sol,
                          cublasHandle_t cublas, const double* d_pconst);
};

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
