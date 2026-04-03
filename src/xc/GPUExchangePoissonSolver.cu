#ifdef USE_CUDA
#include "xc/GPUExchangePoissonSolver.cuh"
#include "core/GPUContext.cuh"
#include "core/gpu_common.cuh"
#include <cstdio>
#include <cassert>

namespace lynx {
namespace gpu {

// ---------------------------------------------------------------------------
// Pointwise multiply kernel (gamma): d_data[i] *= d_alpha[i % Ndc]
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
// Pointwise multiply kernel (k-point Z2Z): d_data[i] *= d_alpha[i % Nd]
// Same as gamma but Nd-sized constants (full FFT, not half-complex).
// ---------------------------------------------------------------------------
__global__ void pointwise_multiply_z2z_kernel(cufftDoubleComplex* d_data,
                                               const double* d_alpha,
                                               int Nd, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    double a = d_alpha[idx % Nd];
    d_data[idx].x *= a;
    d_data[idx].y *= a;
}

// ---------------------------------------------------------------------------
// Phase factor kernel: d_data[i] *= d_phase[i % Nd]  (complex * complex)
// ---------------------------------------------------------------------------
__global__ void apply_phase_kernel(cuDoubleComplex* d_data,
                                    const cuDoubleComplex* d_phase,
                                    int Nd, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    cuDoubleComplex z = d_data[idx];
    cuDoubleComplex p = d_phase[idx % Nd];
    d_data[idx] = make_cuDoubleComplex(
        z.x * p.x - z.y * p.y,
        z.x * p.y + z.y * p.x);
}

// ---------------------------------------------------------------------------
// Scale kernel: d_data[i] *= scalar  (complex)
// ---------------------------------------------------------------------------
__global__ void scale_z_kernel(cuDoubleComplex* d_data, double scalar, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    d_data[idx].x *= scalar;
    d_data[idx].y *= scalar;
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
GPUExchangePoissonSolver::~GPUExchangePoissonSolver() {
    cleanup();
}

void GPUExchangePoissonSolver::cleanup() {
    if (plans_created_) {
        if (!is_kpt_) {
            cufftDestroy(plan_r2c_);
            cufftDestroy(plan_c2r_);
        } else {
            cufftDestroy(plan_z2z_fwd_);
            cufftDestroy(plan_z2z_inv_);
        }
        plans_created_ = false;
    }
    if (d_pois_const_) { cudaFree(d_pois_const_); d_pois_const_ = nullptr; }
    if (d_pois_const_stress_) { cudaFree(d_pois_const_stress_); d_pois_const_stress_ = nullptr; }
    if (d_pois_const_stress2_) { cudaFree(d_pois_const_stress2_); d_pois_const_stress2_ = nullptr; }
    if (d_fft_work_) { cudaFree(d_fft_work_); d_fft_work_ = nullptr; }
    if (d_neg_phase_) { cudaFree(d_neg_phase_); d_neg_phase_ = nullptr; }
    if (d_pos_phase_) { cudaFree(d_pos_phase_); d_pos_phase_ = nullptr; }
    if (d_Kptshift_map_) { cudaFree(d_Kptshift_map_); d_Kptshift_map_ = nullptr; }
    if (d_kpt_scratch_) { cudaFree(d_kpt_scratch_); d_kpt_scratch_ = nullptr; }
    is_setup_ = false;
    is_kpt_ = false;
}

// ---------------------------------------------------------------------------
// Create batched cuFFT plans for R2C and C2R (gamma-point)
// ---------------------------------------------------------------------------
void GPUExchangePoissonSolver::create_plans(int ncol) {
    int rank = 3;
    int n[3] = {Nz_, Ny_, Nx_};

    int inembed_r2c[3] = {Nz_, Ny_, Nx_};
    int istride_r2c = 1;
    int idist_r2c = Nd_;

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

    int inembed_c2r[3] = {Nz_, Ny_, Nx_ / 2 + 1};
    int onembed_c2r[3] = {Nz_, Ny_, Nx_};

    res = cufftPlanMany(&plan_c2r_, rank, n,
                        inembed_c2r, ostride_r2c, odist_r2c,
                        onembed_c2r, istride_r2c, idist_r2c,
                        CUFFT_Z2D, ncol);
    if (res != CUFFT_SUCCESS) {
        std::fprintf(stderr, "GPUExchangePoissonSolver: cufftPlanMany C2R failed (%d)\n", res);
        assert(false);
    }

    plans_created_ = true;
    max_ncol_ = ncol;
}

// ---------------------------------------------------------------------------
// Create batched cuFFT plans for Z2Z (k-point)
// ---------------------------------------------------------------------------
void GPUExchangePoissonSolver::create_plans_z2z(int ncol) {
    int rank = 3;
    int n[3] = {Nz_, Ny_, Nx_};

    // Z2Z: both input and output are full complex [Nd] per batch element
    int embed[3] = {Nz_, Ny_, Nx_};
    int stride = 1;
    int dist = Nd_;

    cufftResult res;
    res = cufftPlanMany(&plan_z2z_fwd_, rank, n,
                        embed, stride, dist,
                        embed, stride, dist,
                        CUFFT_Z2Z, ncol);
    if (res != CUFFT_SUCCESS) {
        std::fprintf(stderr, "GPUExchangePoissonSolver: cufftPlanMany Z2Z forward failed (%d)\n", res);
        assert(false);
    }

    res = cufftPlanMany(&plan_z2z_inv_, rank, n,
                        embed, stride, dist,
                        embed, stride, dist,
                        CUFFT_Z2Z, ncol);
    if (res != CUFFT_SUCCESS) {
        std::fprintf(stderr, "GPUExchangePoissonSolver: cufftPlanMany Z2Z inverse failed (%d)\n", res);
        assert(false);
    }

    plans_created_ = true;
    max_ncol_ = ncol;
}

// ---------------------------------------------------------------------------
// Setup (gamma-point)
// ---------------------------------------------------------------------------
void GPUExchangePoissonSolver::setup(int Nx, int Ny, int Nz,
                                      const double* pois_const,
                                      int max_ncol) {
    cleanup();

    Nx_ = Nx; Ny_ = Ny; Nz_ = Nz;
    Nd_ = Nx * Ny * Nz;
    Ndc_ = Nz * Ny * (Nx / 2 + 1);
    max_ncol_ = max_ncol;
    is_kpt_ = false;

    size_t pois_bytes = Ndc_ * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_pois_const_, pois_bytes));
    CUDA_CHECK(cudaMemcpy(d_pois_const_, pois_const, pois_bytes, cudaMemcpyHostToDevice));

    size_t work_bytes = (size_t)Ndc_ * max_ncol * sizeof(cufftDoubleComplex);
    CUDA_CHECK(cudaMalloc(&d_fft_work_, work_bytes));

    create_plans(max_ncol);

    is_setup_ = true;
    std::printf("GPUExchangePoissonSolver: setup gamma Nx=%d Ny=%d Nz=%d Nd=%d Ndc=%d max_ncol=%d\n",
                Nx_, Ny_, Nz_, Nd_, Ndc_, max_ncol_);
}

// ---------------------------------------------------------------------------
// Setup (k-point Z2Z)
// ---------------------------------------------------------------------------
void GPUExchangePoissonSolver::setup_kpt(int Nx, int Ny, int Nz,
                                          const double* pois_const, int Nkpts_shift,
                                          const void* neg_phase, const void* pos_phase,
                                          const int* Kptshift_map, int Nkpts_sym, int Nkpts_hf,
                                          int max_ncol) {
    cleanup();

    Nx_ = Nx; Ny_ = Ny; Nz_ = Nz;
    Nd_ = Nx * Ny * Nz;
    Ndc_ = Nz * Ny * (Nx / 2 + 1);
    max_ncol_ = max_ncol;
    is_kpt_ = true;
    Nkpts_shift_ = Nkpts_shift;
    Nkpts_sym_ = Nkpts_sym;
    Nkpts_hf_ = Nkpts_hf;

    // Upload all Poisson constants [Nd * Nkpts_shift]
    size_t pois_bytes = (size_t)Nd_ * Nkpts_shift * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_pois_const_, pois_bytes));
    CUDA_CHECK(cudaMemcpy(d_pois_const_, pois_const, pois_bytes, cudaMemcpyHostToDevice));

    // Upload phase factors [Nd * (Nkpts_shift - 1)] complex each
    if (Nkpts_shift > 1) {
        size_t phase_bytes = (size_t)Nd_ * (Nkpts_shift - 1) * sizeof(cuDoubleComplex);
        CUDA_CHECK(cudaMalloc(&d_neg_phase_, phase_bytes));
        CUDA_CHECK(cudaMemcpy(d_neg_phase_, neg_phase, phase_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_pos_phase_, phase_bytes));
        CUDA_CHECK(cudaMemcpy(d_pos_phase_, pos_phase, phase_bytes, cudaMemcpyHostToDevice));
    }

    // Upload Kptshift_map [Nkpts_sym * Nkpts_hf]
    int map_size = Nkpts_sym * Nkpts_hf;
    h_Kptshift_map_.assign(Kptshift_map, Kptshift_map + map_size);
    CUDA_CHECK(cudaMalloc(&d_Kptshift_map_, map_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_Kptshift_map_, Kptshift_map, map_size * sizeof(int), cudaMemcpyHostToDevice));

    // Workspace: Z2Z uses [Nd * max_ncol] complex
    size_t work_bytes = (size_t)Nd_ * max_ncol * sizeof(cufftDoubleComplex);
    CUDA_CHECK(cudaMalloc(&d_fft_work_, work_bytes));

    // Scratch for input copy (Z2Z modifies in-place for phase factors)
    CUDA_CHECK(cudaMalloc(&d_kpt_scratch_, work_bytes));

    create_plans_z2z(max_ncol);

    is_setup_ = true;
    std::printf("GPUExchangePoissonSolver: setup kpt Nx=%d Ny=%d Nz=%d Nd=%d Nkpts_shift=%d max_ncol=%d\n",
                Nx_, Ny_, Nz_, Nd_, Nkpts_shift_, max_ncol_);
}

// ---------------------------------------------------------------------------
// Upload stress Poisson constants
// ---------------------------------------------------------------------------
void GPUExchangePoissonSolver::upload_stress_constants(const double* pois_const_stress,
                                                        const double* pois_const_stress2) {
    int Nd_per_shift = is_kpt_ ? Nd_ : Ndc_;
    size_t bytes = (size_t)Nd_per_shift * (is_kpt_ ? Nkpts_shift_ : 1) * sizeof(double);
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
// Solve batch (gamma, public API)
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
// Internal solve implementation (gamma R2C/C2R)
// ---------------------------------------------------------------------------
void GPUExchangePoissonSolver::solve_batch_impl(double* d_rhs, int ncol, double* d_sol,
                                                  cublasHandle_t cublas,
                                                  const double* d_pconst) {
    cudaStream_t stream = GPUContext::instance().compute_stream;
    assert(is_setup_ && !is_kpt_);
    if (ncol == 0) return;

    // Recreate plans if batch size changed
    if (ncol != max_ncol_) {
        if (plans_created_) {
            cufftDestroy(plan_r2c_);
            cufftDestroy(plan_c2r_);
            plans_created_ = false;
        }
        if (ncol > max_ncol_) {
            if (d_fft_work_) cudaFree(d_fft_work_);
            CUDA_CHECK(cudaMalloc(&d_fft_work_, (size_t)Ndc_ * ncol * sizeof(cufftDoubleComplex)));
        }
        create_plans(ncol);
    }

    // 1. Forward R2C FFT
    cufftResult res = cufftExecD2Z(plan_r2c_, d_rhs, d_fft_work_);
    if (res != CUFFT_SUCCESS) {
        std::fprintf(stderr, "GPUExchangePoissonSolver: cufftExecD2Z failed (%d)\n", res);
        assert(false);
    }

    // 2. Pointwise multiply by Poisson constants
    int total = Ndc_ * ncol;
    int block = 256;
    int grid = (total + block - 1) / block;
    pointwise_multiply_kernel<<<grid, block, 0, stream>>>(d_fft_work_, d_pconst, Ndc_, total);
    CUDA_CHECK(cudaGetLastError());

    // 3. Inverse C2R FFT
    res = cufftExecZ2D(plan_c2r_, d_fft_work_, d_sol);
    if (res != CUFFT_SUCCESS) {
        std::fprintf(stderr, "GPUExchangePoissonSolver: cufftExecZ2D failed (%d)\n", res);
        assert(false);
    }

    // 4. Normalize
    double inv_Nd = 1.0 / Nd_;
    int n_total = Nd_ * ncol;
    cublasDscal(cublas, n_total, &inv_Nd, d_sol, 1);
}

// ---------------------------------------------------------------------------
// Solve batch (k-point Z2Z)
//
// Algorithm (matching CPU ExchangePoissonSolver::solve_batch_kpt):
//   1. Copy rhs to scratch
//   2. Apply neg phase factor: scratch *= exp(-i*(k-q)*r)
//   3. Forward Z2Z FFT
//   4. Multiply by Poisson constants for this (k,q) shift
//   5. Inverse Z2Z FFT
//   6. Normalize by 1/Nd
//   7. Apply pos phase factor: sol *= exp(+i*(k-q)*r)
// ---------------------------------------------------------------------------
void GPUExchangePoissonSolver::solve_batch_kpt(const cuDoubleComplex* d_rhs, int ncol,
                                                cuDoubleComplex* d_sol,
                                                cublasHandle_t cublas,
                                                int kpt_k, int kpt_q) {
    cudaStream_t stream = GPUContext::instance().compute_stream;
    assert(is_setup_ && is_kpt_);
    if (ncol == 0) return;

    // Recreate plans if batch size changed
    if (ncol != max_ncol_) {
        if (plans_created_) {
            cufftDestroy(plan_z2z_fwd_);
            cufftDestroy(plan_z2z_inv_);
            plans_created_ = false;
        }
        size_t work_bytes = (size_t)Nd_ * ncol * sizeof(cufftDoubleComplex);
        if (ncol > max_ncol_) {
            if (d_fft_work_) cudaFree(d_fft_work_);
            CUDA_CHECK(cudaMalloc(&d_fft_work_, work_bytes));
            if (d_kpt_scratch_) cudaFree(d_kpt_scratch_);
            CUDA_CHECK(cudaMalloc(&d_kpt_scratch_, work_bytes));
        }
        create_plans_z2z(ncol);
    }

    int total = Nd_ * ncol;
    int block = 256;
    int grid = (total + block - 1) / block;

    // Look up shift index from Kptshift_map (on host)
    int l = h_Kptshift_map_[kpt_k + kpt_q * Nkpts_sym_];

    // 1. Copy rhs to scratch
    CUDA_CHECK(cudaMemcpy(d_kpt_scratch_, d_rhs, total * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

    // 2. Apply negative phase factor
    if (l != 0) {
        const cuDoubleComplex* d_phase = d_neg_phase_ + (size_t)(l - 1) * Nd_;
        apply_phase_kernel<<<grid, block, 0, stream>>>(d_kpt_scratch_, d_phase, Nd_, total);
        CUDA_CHECK(cudaGetLastError());
    }

    // 3. Forward Z2Z FFT: scratch -> fft_work
    cufftResult res = cufftExecZ2Z(plan_z2z_fwd_, d_kpt_scratch_, d_fft_work_, CUFFT_FORWARD);
    if (res != CUFFT_SUCCESS) {
        std::fprintf(stderr, "GPUExchangePoissonSolver: cufftExecZ2Z forward failed (%d)\n", res);
        assert(false);
    }

    // 4. Multiply by Poisson constants for this shift
    const double* d_alpha;
    if (l == 0) {
        d_alpha = d_pois_const_ + (size_t)Nd_ * (Nkpts_shift_ - 1);
    } else {
        d_alpha = d_pois_const_ + (size_t)Nd_ * (l - 1);
    }
    pointwise_multiply_z2z_kernel<<<grid, block, 0, stream>>>(d_fft_work_, d_alpha, Nd_, total);
    CUDA_CHECK(cudaGetLastError());

    // 5. Inverse Z2Z FFT: fft_work -> sol
    res = cufftExecZ2Z(plan_z2z_inv_, d_fft_work_, d_sol, CUFFT_INVERSE);
    if (res != CUFFT_SUCCESS) {
        std::fprintf(stderr, "GPUExchangePoissonSolver: cufftExecZ2Z inverse failed (%d)\n", res);
        assert(false);
    }

    // 6. Normalize by 1/Nd
    double inv_Nd = 1.0 / Nd_;
    scale_z_kernel<<<grid, block, 0, stream>>>(d_sol, inv_Nd, total);
    CUDA_CHECK(cudaGetLastError());

    // 7. Apply positive phase factor
    if (l != 0) {
        const cuDoubleComplex* d_phase = d_pos_phase_ + (size_t)(l - 1) * Nd_;
        apply_phase_kernel<<<grid, block, 0, stream>>>(d_sol, d_phase, Nd_, total);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
