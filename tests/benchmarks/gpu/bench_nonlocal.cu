// Standalone benchmark for nonlocal projector kernels:
//   fused_gather_chitpsi_kernel + gamma_scale_kernel + fused_chialpha_scatter_kernel
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

__host__ __device__ int ceildiv(int a, int b) { return (a + b - 1) / b; }

// ============================================================
// Warp/block reductions (copied from NonlocalProjector.cu)
// ============================================================
__device__ __forceinline__ double warpReduceSum_d(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ double blockReduceSum_d(double val, double* smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    val = warpReduceSum_d(val);
    if (lane == 0) smem[warp] = val;
    __syncthreads();
    if (warp == 0) {
        val = (lane < nwarps) ? smem[lane] : 0.0;
        val = warpReduceSum_d(val);
    }
    __syncthreads();
    return val;
}

// ============================================================
// Kernels (copied from NonlocalProjector.cu)
// ============================================================
static constexpr int NL_TILE = 256;
static constexpr int NL_MAX_NP = 32;

__global__ void fused_gather_chitpsi_kernel(
    const double* __restrict__ psi,
    const double* __restrict__ Chi_flat,
    const int* __restrict__ gpos_flat,
    const int* __restrict__ gpos_offsets,
    const int* __restrict__ chi_offsets,
    const int* __restrict__ ndc_arr,
    const int* __restrict__ nproj_arr,
    const int* __restrict__ IP_displ,
    double* __restrict__ alpha,
    int Nd, int ncol_this, int ncol_stride, int col_start,
    double dV, int n_atoms)
{
    int iat = blockIdx.x;
    if (iat >= n_atoms) return;
    int ndc = ndc_arr[iat];
    int np = nproj_arr[iat];
    if (ndc == 0 || np == 0) return;
    int goff = gpos_offsets[iat];
    int coff = chi_offsets[iat];
    int abase = IP_displ[iat];

    extern __shared__ double smem[];
    double* psi_tile = smem;
    double* reduce_buf = smem + NL_TILE;

    for (int n = 0; n < ncol_this; n++) {
        double dots[NL_MAX_NP];
        #pragma unroll 4
        for (int jp = 0; jp < NL_MAX_NP; jp++) dots[jp] = 0.0;

        for (int tile = 0; tile < ndc; tile += NL_TILE) {
            int tile_len = min(NL_TILE, ndc - tile);
            for (int i = threadIdx.x; i < tile_len; i += blockDim.x)
                psi_tile[i] = psi[gpos_flat[goff + tile + i] + (col_start + n) * Nd];
            __syncthreads();
            for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
                double pv = psi_tile[i];
                const double* chi_base = Chi_flat + coff + tile + i;
                for (int jp = 0; jp < np; jp++)
                    dots[jp] += chi_base[jp * ndc] * pv;
            }
            __syncthreads();
        }

        for (int jp = 0; jp < np; jp++) {
            double val = blockReduceSum_d(dots[jp], reduce_buf);
            if (threadIdx.x == 0)
                atomicAdd(&alpha[(abase + jp) * ncol_stride + (col_start + n)], val * dV);
        }
    }
}

__global__ void gamma_scale_kernel(
    double* __restrict__ alpha,
    const double* __restrict__ Gamma,
    int total_nproj, int ncol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_nproj * ncol) return;
    int ip = idx / ncol;
    alpha[idx] *= Gamma[ip];
}

__global__ void fused_chialpha_scatter_kernel(
    double* __restrict__ Hpsi,
    const double* __restrict__ Chi_flat,
    const int* __restrict__ gpos_flat,
    const int* __restrict__ gpos_offsets,
    const int* __restrict__ chi_offsets,
    const int* __restrict__ ndc_arr,
    const int* __restrict__ nproj_arr,
    const int* __restrict__ IP_displ,
    const double* __restrict__ alpha,
    int Nd, int ncol_this, int ncol_stride, int col_start,
    int n_atoms)
{
    int iat = blockIdx.x;
    if (iat >= n_atoms) return;
    int ndc = ndc_arr[iat];
    int np = nproj_arr[iat];
    if (ndc == 0 || np == 0) return;
    int goff = gpos_offsets[iat];
    int coff = chi_offsets[iat];
    int abase = IP_displ[iat];

    extern __shared__ double alpha_sh[];
    int alpha_size = np * ncol_this;
    for (int idx = threadIdx.x; idx < alpha_size; idx += blockDim.x) {
        int jp = idx / ncol_this;
        int n = idx % ncol_this;
        alpha_sh[idx] = alpha[(abase + jp) * ncol_stride + (col_start + n)];
    }
    __syncthreads();

    int total_out = ndc * ncol_this;
    for (int idx = threadIdx.x; idx < total_out; idx += blockDim.x) {
        int ig = idx % ndc;
        int n = idx / ndc;
        double val = 0.0;
        const double* chi_row_base = Chi_flat + coff + ig;
        const double* alpha_col = alpha_sh + n;
        for (int jp = 0; jp < np; ++jp)
            val += chi_row_base[jp * ndc] * alpha_col[jp * ncol_this];
        atomicAdd(&Hpsi[gpos_flat[goff + ig] + (col_start + n) * Nd], val);
    }
}

int main() {
    // Typical DFT system: Si4 with ~4 atoms, ~8 projectors each, ~800 ndc points
    const int n_atoms = 4;
    const int nproj_per_atom = 8;
    const int ndc_per_atom = 800;
    const int total_nproj = n_atoms * nproj_per_atom;
    const int Nd = 25 * 26 * 27;  // ~17550
    const int ncol = 20;
    const int NREPS = 200;
    const double dV = 0.01;

    // Prepare host metadata
    int h_ndc_arr[4], h_nproj_arr[4], h_gpos_offsets[5], h_chi_offsets[5], h_IP_displ[4];
    for (int i = 0; i < n_atoms; i++) {
        h_ndc_arr[i] = ndc_per_atom;
        h_nproj_arr[i] = nproj_per_atom;
        h_gpos_offsets[i] = i * ndc_per_atom;
        h_chi_offsets[i] = i * ndc_per_atom * nproj_per_atom;
        h_IP_displ[i] = i * nproj_per_atom;
    }
    h_gpos_offsets[n_atoms] = n_atoms * ndc_per_atom;
    h_chi_offsets[n_atoms] = n_atoms * ndc_per_atom * nproj_per_atom;

    // Generate grid positions (scattered indices within [0, Nd))
    int total_ndc = n_atoms * ndc_per_atom;
    int* h_gpos = (int*)malloc(total_ndc * sizeof(int));
    srand(42);
    for (int i = 0; i < total_ndc; i++)
        h_gpos[i] = rand() % Nd;

    // Allocate device arrays
    double *d_psi, *d_Chi_flat, *d_alpha, *d_Hpsi, *d_Gamma;
    int *d_gpos_flat, *d_gpos_offsets, *d_chi_offsets, *d_ndc_arr, *d_nproj_arr, *d_IP_displ;

    CUDA_CHECK(cudaMalloc(&d_psi, (size_t)Nd * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Chi_flat, (size_t)total_ndc * nproj_per_atom * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_alpha, (size_t)total_nproj * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Hpsi, (size_t)Nd * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Gamma, (size_t)total_nproj * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gpos_flat, total_ndc * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gpos_offsets, (n_atoms + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_chi_offsets, (n_atoms + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ndc_arr, n_atoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nproj_arr, n_atoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_IP_displ, n_atoms * sizeof(int)));

    // Fill with random data
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniformDouble(gen, d_psi, Nd * ncol);
    curandGenerateUniformDouble(gen, d_Chi_flat, total_ndc * nproj_per_atom);
    curandGenerateUniformDouble(gen, d_Gamma, total_nproj);
    curandDestroyGenerator(gen);

    CUDA_CHECK(cudaMemcpy(d_gpos_flat, h_gpos, total_ndc * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gpos_offsets, h_gpos_offsets, (n_atoms + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chi_offsets, h_chi_offsets, (n_atoms + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ndc_arr, h_ndc_arr, n_atoms * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nproj_arr, h_nproj_arr, n_atoms * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_IP_displ, h_IP_displ, n_atoms * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int block_size = 256;
    size_t smem1 = (NL_TILE + block_size / 32) * sizeof(double);
    size_t smem3 = (size_t)nproj_per_atom * ncol * sizeof(double);

    printf("=== Nonlocal Projector Benchmark: atoms=%d ndc=%d nproj=%d ncol=%d (NREPS=%d) ===\n",
           n_atoms, ndc_per_atom, nproj_per_atom, ncol, NREPS);

    // Benchmark gather (Chi^T * psi)
    {
        for (int i = 0; i < 5; i++) {
            CUDA_CHECK(cudaMemset(d_alpha, 0, total_nproj * ncol * sizeof(double)));
            fused_gather_chitpsi_kernel<<<n_atoms, block_size, smem1>>>(
                d_psi, d_Chi_flat, d_gpos_flat, d_gpos_offsets, d_chi_offsets,
                d_ndc_arr, d_nproj_arr, d_IP_displ, d_alpha,
                Nd, ncol, ncol, 0, dV, n_atoms);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++) {
            CUDA_CHECK(cudaMemset(d_alpha, 0, total_nproj * ncol * sizeof(double)));
            fused_gather_chitpsi_kernel<<<n_atoms, block_size, smem1>>>(
                d_psi, d_Chi_flat, d_gpos_flat, d_gpos_offsets, d_chi_offsets,
                d_ndc_arr, d_nproj_arr, d_IP_displ, d_alpha,
                Nd, ncol, ncol, 0, dV, n_atoms);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        printf("  gather:  %8.4f ms\n", ms / NREPS);
    }

    // Benchmark gamma scale
    {
        int total = total_nproj * ncol;
        for (int i = 0; i < 5; i++)
            gamma_scale_kernel<<<ceildiv(total, 256), 256>>>(d_alpha, d_Gamma, total_nproj, ncol);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++)
            gamma_scale_kernel<<<ceildiv(total, 256), 256>>>(d_alpha, d_Gamma, total_nproj, ncol);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        printf("  gamma:   %8.4f ms\n", ms / NREPS);
    }

    // Benchmark scatter (Chi * alpha)
    {
        for (int i = 0; i < 5; i++) {
            CUDA_CHECK(cudaMemset(d_Hpsi, 0, Nd * ncol * sizeof(double)));
            fused_chialpha_scatter_kernel<<<n_atoms, block_size, smem3>>>(
                d_Hpsi, d_Chi_flat, d_gpos_flat, d_gpos_offsets, d_chi_offsets,
                d_ndc_arr, d_nproj_arr, d_IP_displ, d_alpha,
                Nd, ncol, ncol, 0, n_atoms);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++) {
            CUDA_CHECK(cudaMemset(d_Hpsi, 0, Nd * ncol * sizeof(double)));
            fused_chialpha_scatter_kernel<<<n_atoms, block_size, smem3>>>(
                d_Hpsi, d_Chi_flat, d_gpos_flat, d_gpos_offsets, d_chi_offsets,
                d_ndc_arr, d_nproj_arr, d_IP_displ, d_alpha,
                Nd, ncol, ncol, 0, n_atoms);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        printf("  scatter: %8.4f ms\n", ms / NREPS);
    }

    printf("\n");
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_psi); cudaFree(d_Chi_flat); cudaFree(d_alpha);
    cudaFree(d_Hpsi); cudaFree(d_Gamma);
    cudaFree(d_gpos_flat); cudaFree(d_gpos_offsets); cudaFree(d_chi_offsets);
    cudaFree(d_ndc_arr); cudaFree(d_nproj_arr); cudaFree(d_IP_displ);
    free(h_gpos);
    return 0;
}
