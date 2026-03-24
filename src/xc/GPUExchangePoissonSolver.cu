#ifdef USE_CUDA
#include "xc/GPUExchangePoissonSolver.cuh"
#include "core/gpu_common.cuh"
#include <cstdio>
#include <cassert>

namespace lynx {
namespace gpu {

// ---------------------------------------------------------------------------
// Pointwise multiply kernel: d_out[i] *= d_alpha[i % Ndc]
// Each complex element (re, im) is scaled by the same real constant.
// ---------------------------------------------------------------------------
__global__ void pointwise_multiply_kernel(cufftDoubleComplex* d_data,
                                           const double* d_alpha,
                                           int Ndc, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    double a = d_alpha[idx % Ndc];
    d_data[idx].x *= a;
    d_data[idx].y *= a;
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
GPUExchangePoissonSolver::~GPUExchangePoissonSolver() {
    cleanup();
}

void GPUExchangePoissonSolver::cleanup() {
    if (plans_created_) {
        cufftDestroy(plan_r2c_);
        cufftDestroy(plan_c2r_);
        plans_created_ = false;
    }
    if (d_pois_const_) { cudaFree(d_pois_const_); d_pois_const_ = nullptr; }
    if (d_pois_const_stress_) { cudaFree(d_pois_const_stress_); d_pois_const_stress_ = nullptr; }
    if (d_pois_const_stress2_) { cudaFree(d_pois_const_stress2_); d_pois_const_stress2_ = nullptr; }
    if (d_fft_work_) { cudaFree(d_fft_work_); d_fft_work_ = nullptr; }
    is_setup_ = false;
}

// ---------------------------------------------------------------------------
// Create batched cuFFT plans for R2C and C2R
// ---------------------------------------------------------------------------
void GPUExchangePoissonSolver::create_plans(int ncol) {
    // cuFFT uses row-major C order: dims = {Nz, Ny, Nx}
    int rank = 3;
    int n[3] = {Nz_, Ny_, Nx_};

    // Input (real): stride 1, distance Nd between batches
    int inembed_r2c[3] = {Nz_, Ny_, Nx_};
    int istride_r2c = 1;
    int idist_r2c = Nd_;

    // Output (complex): stride 1, distance Ndc between batches
    int onembed_r2c[3] = {Nz_, Ny_, Nx_ / 2 + 1};
    int ostride_r2c = 1;
    int odist_r2c = Ndc_;

    cufftResult res;
    res = cufftPlanMany(&plan_r2c_, rank, n,
                        inembed_r2c, istride_r2c, idist_r2c,
                        onembed_r2c, ostride_r2c, odist_r2c,
                        CUFFT_D2Z, ncol);
    if (res != CUFFT_SUCCESS) {
        std::fprintf(stderr, "GPUExchangePoissonSolver: cufftPlanMany R2C failed (%d)\n", res);
        assert(false);
    }

    // Inverse: input is complex, output is real
    int inembed_c2r[3] = {Nz_, Ny_, Nx_ / 2 + 1};
    int onembed_c2r[3] = {Nz_, Ny_, Nx_};

    res = cufftPlanMany(&plan_c2r_, rank, n,
                        inembed_c2r, ostride_r2c, odist_r2c,   // complex input
                        onembed_c2r, istride_r2c, idist_r2c,   // real output
                        CUFFT_Z2D, ncol);
    if (res != CUFFT_SUCCESS) {
        std::fprintf(stderr, "GPUExchangePoissonSolver: cufftPlanMany C2R failed (%d)\n", res);
        assert(false);
    }

    plans_created_ = true;
    max_ncol_ = ncol;
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------
void GPUExchangePoissonSolver::setup(int Nx, int Ny, int Nz,
                                      const double* pois_const,
                                      int max_ncol) {
    cleanup();

    Nx_ = Nx; Ny_ = Ny; Nz_ = Nz;
    Nd_ = Nx * Ny * Nz;
    Ndc_ = Nz * Ny * (Nx / 2 + 1);
    max_ncol_ = max_ncol;

    // Upload Poisson constants to device
    size_t pois_bytes = Ndc_ * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_pois_const_, pois_bytes));
    CUDA_CHECK(cudaMemcpy(d_pois_const_, pois_const, pois_bytes, cudaMemcpyHostToDevice));

    // Allocate FFT workspace (complex output buffer)
    size_t work_bytes = (size_t)Ndc_ * max_ncol * sizeof(cufftDoubleComplex);
    CUDA_CHECK(cudaMalloc(&d_fft_work_, work_bytes));

    // Create batched FFT plans
    create_plans(max_ncol);

    is_setup_ = true;
    std::printf("GPUExchangePoissonSolver: setup Nx=%d Ny=%d Nz=%d Nd=%d Ndc=%d max_ncol=%d\n",
                Nx_, Ny_, Nz_, Nd_, Ndc_, max_ncol_);
}

// ---------------------------------------------------------------------------
// Upload stress Poisson constants
// ---------------------------------------------------------------------------
void GPUExchangePoissonSolver::upload_stress_constants(const double* pois_const_stress,
                                                        const double* pois_const_stress2) {
    size_t bytes = Ndc_ * sizeof(double);
    if (pois_const_stress) {
        if (!d_pois_const_stress_) CUDA_CHECK(cudaMalloc(&d_pois_const_stress_, bytes));
        CUDA_CHECK(cudaMemcpy(d_pois_const_stress_, pois_const_stress, bytes, cudaMemcpyHostToDevice));
    }
    if (pois_const_stress2) {
        if (!d_pois_const_stress2_) CUDA_CHECK(cudaMalloc(&d_pois_const_stress2_, bytes));
        CUDA_CHECK(cudaMemcpy(d_pois_const_stress2_, pois_const_stress2, bytes, cudaMemcpyHostToDevice));
    }
}

// ---------------------------------------------------------------------------
// Solve batch (public API)
// ---------------------------------------------------------------------------
void GPUExchangePoissonSolver::solve_batch(double* d_rhs, int ncol, double* d_sol,
                                            cublasHandle_t cublas) {
    solve_batch_impl(d_rhs, ncol, d_sol, cublas, d_pois_const_);
}

void GPUExchangePoissonSolver::solve_batch_stress(double* d_rhs, int ncol, double* d_sol,
                                                    cublasHandle_t cublas, int option) {
    const double* pconst = (option == 2) ? d_pois_const_stress2_ : d_pois_const_stress_;
    assert(pconst && "Stress Poisson constants not uploaded");
    solve_batch_impl(d_rhs, ncol, d_sol, cublas, pconst);
}

// ---------------------------------------------------------------------------
// Internal solve implementation
// ---------------------------------------------------------------------------
void GPUExchangePoissonSolver::solve_batch_impl(double* d_rhs, int ncol, double* d_sol,
                                                  cublasHandle_t cublas,
                                                  const double* d_pconst) {
    assert(is_setup_);
    if (ncol == 0) return;

    // If ncol differs from planned batch size, recreate plans
    if (ncol != max_ncol_) {
        if (plans_created_) {
            cufftDestroy(plan_r2c_);
            cufftDestroy(plan_c2r_);
            plans_created_ = false;
        }
        // Reallocate workspace if needed
        if (ncol > max_ncol_) {
            if (d_fft_work_) cudaFree(d_fft_work_);
            CUDA_CHECK(cudaMalloc(&d_fft_work_, (size_t)Ndc_ * ncol * sizeof(cufftDoubleComplex)));
        }
        create_plans(ncol);
    }

    // 1. Forward R2C FFT: d_rhs [Nd*ncol real] → d_fft_work [Ndc*ncol complex]
    cufftResult res = cufftExecD2Z(plan_r2c_, d_rhs, d_fft_work_);
    if (res != CUFFT_SUCCESS) {
        std::fprintf(stderr, "GPUExchangePoissonSolver: cufftExecD2Z failed (%d)\n", res);
        assert(false);
    }

    // 2. Pointwise multiply by Poisson constants
    int total = Ndc_ * ncol;
    int block = 256;
    int grid = (total + block - 1) / block;
    pointwise_multiply_kernel<<<grid, block>>>(d_fft_work_, d_pconst, Ndc_, total);
    CUDA_CHECK(cudaGetLastError());

    // 3. Inverse C2R FFT: d_fft_work [Ndc*ncol complex] → d_sol [Nd*ncol real]
    res = cufftExecZ2D(plan_c2r_, d_fft_work_, d_sol);
    if (res != CUFFT_SUCCESS) {
        std::fprintf(stderr, "GPUExchangePoissonSolver: cufftExecZ2D failed (%d)\n", res);
        assert(false);
    }

    // 4. Normalize: sol *= 1/Nd (cuFFT produces unnormalized output)
    double inv_Nd = 1.0 / Nd_;
    int n_total = Nd_ * ncol;
    cublasDscal(cublas, n_total, &inv_Nd, d_sol, 1);
}

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
