#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include "core/GPUContext.cuh"
#include "core/gpu_common.cuh"
#include "solvers/LinearSolver.cuh"

namespace lynx {
namespace gpu {

// ============================================================
// GPU kernels for AAR (Alternating Anderson-Richardson) solver
// ============================================================

// r[i] = b[i] - Ax[i]
__global__ void residual_kernel(
    const double* __restrict__ b,
    const double* __restrict__ Ax,
    double* __restrict__ r,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) r[idx] = b[idx] - Ax[idx];
}

// x[i] = x_old[i] + omega * f[i]
__global__ void richardson_kernel(
    const double* __restrict__ x_old,
    const double* __restrict__ f,
    double* __restrict__ x,
    double omega, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) x[idx] = x_old[idx] + omega * f[idx];
}

// Store history: X(:,col) = x - x_old, F(:,col) = f - f_old
__global__ void store_history_kernel(
    const double* __restrict__ x,
    const double* __restrict__ x_old,
    const double* __restrict__ f,
    const double* __restrict__ f_old,
    double* __restrict__ X_hist,
    double* __restrict__ F_hist,
    int col, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        X_hist[col * N + idx] = x[idx] - x_old[idx];
        F_hist[col * N + idx] = f[idx] - f_old[idx];
    }
}

// Anderson extrapolation:
// x[i] = x_old[i] + beta*f[i] - sum_j gamma[j]*(X(i,j) + beta*F(i,j))
__global__ void anderson_kernel(
    const double* __restrict__ x_old,
    const double* __restrict__ f,
    const double* __restrict__ X_hist,
    const double* __restrict__ F_hist,
    const double* __restrict__ gamma,
    double* __restrict__ x,
    double beta, int cols, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double val = x_old[idx] + beta * f[idx];
        for (int j = 0; j < cols; ++j) {
            val -= gamma[j] * (X_hist[j * N + idx] + beta * F_hist[j * N + idx]);
        }
        x[idx] = val;
    }
}

// ============================================================
// Fused Gram matrix kernel: computes F^T*F (upper triangle) and F^T*f
// in a single launch. One block per (i,j) pair in the upper triangle
// + one block per F^T*f entry. Avoids per-dot-product host sync.
//
// Output: d_out[0..n_pairs-1] = upper triangle of F^T*F (row-major)
//         d_out[n_pairs..n_pairs+cols-1] = F^T*f
// where n_pairs = cols*(cols+1)/2
// ============================================================
__global__ void fused_gram_kernel(
    const double* __restrict__ F_hist,  // (N, cols) column-major
    const double* __restrict__ f,       // (N)
    double* __restrict__ d_out,         // output: [n_pairs + cols] dot products
    const int* __restrict__ d_pair_i,   // [n_jobs] first index
    const int* __restrict__ d_pair_j,   // [n_jobs] second index (-1 = use f)
    int N, int cols, int n_jobs)
{
    int job = blockIdx.x;
    if (job >= n_jobs) return;

    int ci = d_pair_i[job];
    int cj = d_pair_j[job];

    const double* a = F_hist + ci * N;
    const double* b = (cj >= 0) ? (F_hist + cj * N) : f;

    // Block reduction over N elements
    extern __shared__ double sdata[];
    double sum = 0.0;
    for (int idx = threadIdx.x; idx < N; idx += blockDim.x)
        sum += a[idx] * b[idx];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Warp reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) d_out[job] = sdata[0];
}

// Fused norm² kernel: computes ||r||² in a single value on device
__global__ void norm2_kernel(
    const double* __restrict__ r,
    double* __restrict__ d_norm2,  // single output value
    int N)
{
    extern __shared__ double sdata[];
    double sum = 0.0;
    for (int idx = threadIdx.x; idx < N; idx += blockDim.x)
        sum += r[idx] * r[idx];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) *d_norm2 = sdata[0];
}

// ============================================================
// GPU AAR solver
// ============================================================
// op_gpu: function that computes A*x on GPU (input/output both device pointers)
// precond_gpu: optional preconditioner on GPU

int aar_gpu(
    void (*op_gpu)(const double* d_x, double* d_Ax),
    void (*precond_gpu)(const double* d_r, double* d_f),
    const double* d_b,
    double* d_x,
    int N,
    double omega, double beta, int m, int p,
    double tol, int max_iter,
    // Workspace (all pre-allocated from SCFBuffers):
    double* d_r,
    double* d_f,
    double* d_Ax,
    double* d_X_hist,  // (N, m)
    double* d_F_hist,  // (N, m)
    double* d_x_old,   // reuse via caller
    double* d_f_old,
    cudaStream_t stream)   // reuse via caller
{
    auto& ctx = GPUContext::instance();
    int bs = 256;
    int grid = ceildiv(N, bs);

    // x_old = x
    CUDA_CHECK(cudaMemcpy(d_x_old, d_x, N * sizeof(double), cudaMemcpyDeviceToDevice));

    // Small device buffers for fused Gram and norm kernels.
    // Reuse d_Ax (size N doubles) which is idle during Gram/norm phases.
    // Layout: [pair_i (max_jobs ints) | pair_j (max_jobs ints) | gram_out (max_jobs doubles) | norm2 (1 double)]
    int max_jobs = m * (m + 1) / 2 + m;  // max 35 for m=7
    // Verify d_Ax is large enough: need max_jobs*2*4 + (max_jobs+1)*8 = 35*8 + 36*8 = 568 bytes << N*8
    int* d_pair_i     = reinterpret_cast<int*>(d_Ax);
    int* d_pair_j     = d_pair_i + max_jobs;
    double* d_gram_out = reinterpret_cast<double*>(d_pair_j + max_jobs);
    double* d_norm2_buf = d_gram_out + max_jobs;

    // ||b|| via fused norm kernel (1 launch + 1 D2H instead of cublasDnrm2)
    int norm_bs = 256;
    norm2_kernel<<<1, norm_bs, norm_bs * sizeof(double), stream>>>(d_b, d_norm2_buf, N);
    double b_norm2;
    CUDA_CHECK(cudaMemcpy(&b_norm2, d_norm2_buf, sizeof(double), cudaMemcpyDeviceToHost));
    double abs_tol = tol * std::sqrt(b_norm2);

    // Initial residual: r = b - A*x
    op_gpu(d_x, d_Ax);
    residual_kernel<<<grid, bs, 0, stream>>>(d_b, d_Ax, d_r, N);

    double r_2norm = abs_tol + 1.0;
    int iter_count = 0;

    // Small host-side buffers for Gram matrix solve
    std::vector<double> h_FTF(m * m);
    std::vector<double> h_gamma(m);
    std::vector<int> h_pair_i(max_jobs), h_pair_j(max_jobs);
    std::vector<double> h_gram_out(max_jobs);

    while (r_2norm > abs_tol && iter_count < max_iter) {
        // Precondition: f = M^{-1} * r
        if (precond_gpu) {
            precond_gpu(d_r, d_f);
        } else {
            CUDA_CHECK(cudaMemcpy(d_f, d_r, N * sizeof(double), cudaMemcpyDeviceToDevice));
        }

        // Store history
        if (iter_count > 0) {
            int i_hist = (iter_count - 1) % m;
            store_history_kernel<<<grid, bs, 0, stream>>>(d_x, d_x_old, d_f, d_f_old,
                                                d_X_hist, d_F_hist, i_hist, N);
        }

        // Save current state
        CUDA_CHECK(cudaMemcpy(d_x_old, d_x, N * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_f_old, d_f, N * sizeof(double), cudaMemcpyDeviceToDevice));

        if ((iter_count + 1) % p == 0 && iter_count > 0) {
            // Anderson extrapolation
            int cols = std::min(iter_count, m);

            // Build Gram matrix F^T*F and F^T*f via fused kernel (1 launch)
            {
                int n_jobs = 0;
                // Upper triangle of F^T*F
                for (int ii = 0; ii < cols; ++ii)
                    for (int jj = 0; jj <= ii; ++jj) {
                        h_pair_i[n_jobs] = ii;
                        h_pair_j[n_jobs] = jj;
                        n_jobs++;
                    }
                // F^T*f entries
                int ftf_pairs = n_jobs;
                for (int ii = 0; ii < cols; ++ii) {
                    h_pair_i[n_jobs] = ii;
                    h_pair_j[n_jobs] = -1;  // -1 = use f vector
                    n_jobs++;
                }

                CUDA_CHECK(cudaMemcpy(d_pair_i, h_pair_i.data(), n_jobs * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_pair_j, h_pair_j.data(), n_jobs * sizeof(int), cudaMemcpyHostToDevice));

                int gram_bs = std::min(256, N);
                fused_gram_kernel<<<n_jobs, gram_bs, gram_bs * sizeof(double), stream>>>(
                    d_F_hist, d_f, d_gram_out, d_pair_i, d_pair_j, N, cols, n_jobs);

                // Single D2H: download all dot products at once
                CUDA_CHECK(cudaMemcpy(h_gram_out.data(), d_gram_out, n_jobs * sizeof(double), cudaMemcpyDeviceToHost));

                // Unpack into h_FTF (symmetric) and h_gamma
                int k = 0;
                for (int ii = 0; ii < cols; ++ii)
                    for (int jj = 0; jj <= ii; ++jj) {
                        h_FTF[ii * cols + jj] = h_gram_out[k];
                        h_FTF[jj * cols + ii] = h_gram_out[k];
                        k++;
                    }
                for (int ii = 0; ii < cols; ++ii)
                    h_gamma[ii] = h_gram_out[ftf_pairs + ii];
            }

            // Solve (F^T*F) * gamma = F^T*f via Gaussian elimination (tiny matrix, on CPU)
            {
                std::vector<double> A(h_FTF.begin(), h_FTF.begin() + cols * cols);
                for (int k = 0; k < cols; ++k) {
                    int pivot = k;
                    for (int ii = k + 1; ii < cols; ++ii)
                        if (std::abs(A[ii * cols + k]) > std::abs(A[pivot * cols + k]))
                            pivot = ii;
                    if (pivot != k) {
                        for (int j = 0; j < cols; ++j)
                            std::swap(A[k * cols + j], A[pivot * cols + j]);
                        std::swap(h_gamma[k], h_gamma[pivot]);
                    }
                    double d = A[k * cols + k];
                    if (std::abs(d) < 1e-14) continue;
                    for (int ii = k + 1; ii < cols; ++ii) {
                        double factor = A[ii * cols + k] / d;
                        for (int j = k + 1; j < cols; ++j)
                            A[ii * cols + j] -= factor * A[k * cols + j];
                        h_gamma[ii] -= factor * h_gamma[k];
                    }
                }
                for (int k = cols - 1; k >= 0; --k) {
                    if (std::abs(A[k * cols + k]) < 1e-14) continue;
                    for (int j = k + 1; j < cols; ++j)
                        h_gamma[k] -= A[k * cols + j] * h_gamma[j];
                    h_gamma[k] /= A[k * cols + k];
                }
            }

            // Upload gamma to first 'cols' elements of d_Ax (safe: Anderson
            // kernel reads gamma before the next op_gpu overwrites d_Ax)
            double* d_gamma = d_Ax;
            CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), cols * sizeof(double),
                                   cudaMemcpyHostToDevice));

            anderson_kernel<<<grid, bs, 0, stream>>>(d_x_old, d_f, d_X_hist, d_F_hist,
                                           d_gamma, d_x, beta, cols, N);

            // Recompute residual + check convergence
            op_gpu(d_x, d_Ax);
            residual_kernel<<<grid, bs, 0, stream>>>(d_b, d_Ax, d_r, N);
            {
                norm2_kernel<<<1, norm_bs, norm_bs * sizeof(double), stream>>>(d_r, d_norm2_buf, N);
                double r_norm2;
                CUDA_CHECK(cudaMemcpy(&r_norm2, d_norm2_buf, sizeof(double), cudaMemcpyDeviceToHost));
                r_2norm = std::sqrt(r_norm2);
            }
        } else {
            // Richardson update
            richardson_kernel<<<grid, bs, 0, stream>>>(d_x_old, d_f, d_x, omega, N);

            // Recompute residual
            op_gpu(d_x, d_Ax);
            residual_kernel<<<grid, bs, 0, stream>>>(d_b, d_Ax, d_r, N);
        }

        iter_count++;
    }

    return iter_count;
}

// ============================================================
// GPU Density Computation
// rho[i] += weight * occ[n] * |psi[i + n*Nd]|^2
// ============================================================

__global__ void compute_density_kernel(
    const double* __restrict__ psi,
    const double* __restrict__ occ,
    double* __restrict__ rho,
    int Nd, int Ns, double weight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Nd) {
        double sum = 0.0;
        for (int n = 0; n < Ns; ++n) {
            double val = psi[idx + n * Nd];
            sum += occ[n] * val * val;
        }
        rho[idx] += weight * sum;
    }
}

void compute_density_gpu(
    const double* d_psi,
    const double* d_occ,
    double* d_rho,
    int Nd, int Ns, double weight,
    cudaStream_t stream)
{
    int bs = 256;
    int grid = ceildiv(Nd, bs);
    compute_density_kernel<<<grid, bs, 0, stream>>>(d_psi, d_occ, d_rho, Nd, Ns, weight);
}

// ============================================================
// GPU Mixing kernels (Pulay)
// ============================================================

// f[i] = rho_new[i] - rho_old[i]  (mixing residual)
__global__ void mix_residual_kernel(
    const double* __restrict__ rho_new,
    const double* __restrict__ rho_old,
    double* __restrict__ f,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) f[idx] = rho_new[idx] - rho_old[idx];
}

// x[i] = (1-beta)*x_old[i] + beta*x_new[i] + correction
// Pulay mixing: rho_out = rho_in + beta*residual - sum gamma_j*(DX_j + beta*DF_j)
// (Same structure as Anderson in AAR — reuse anderson_kernel)

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
