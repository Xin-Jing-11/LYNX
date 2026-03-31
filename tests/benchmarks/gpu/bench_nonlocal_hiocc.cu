// High-occupancy nonlocal projector kernel benchmark
// Problem: Production kernels use 1 block per atom (only 4 blocks!)
// V1/V2 fix: 2D grid (atom × column) → 80 blocks
// V3 NEW: 2D grid + __launch_bounds__(256, 6) to guarantee ≤40 regs
// V4 NEW: Smaller block (128) + 2D grid → 2x more blocks for small systems
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

static constexpr int NL_MAX_NP = 32;

// ============================================================
// V2 baseline: 2D grid, register accumulation, no shared psi tile
// (already the best existing version from bench_nonlocal_opt.cu)
// ============================================================
__global__ void fused_gather_chitpsi_kernel_v2(
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
    int n = blockIdx.y;
    if (iat >= n_atoms || n >= ncol_this) return;

    int ndc = ndc_arr[iat];
    int np = nproj_arr[iat];
    if (ndc == 0 || np == 0) return;
    int goff = gpos_offsets[iat];
    int coff = chi_offsets[iat];
    int abase = IP_displ[iat];

    double dots[NL_MAX_NP];
    for (int jp = 0; jp < np; jp++) dots[jp] = 0.0;

    for (int i = threadIdx.x; i < ndc; i += blockDim.x) {
        double pv = psi[gpos_flat[goff + i] + (col_start + n) * Nd];
        const double* chi_base = Chi_flat + coff + i;
        #pragma unroll 4
        for (int jp = 0; jp < np; jp++)
            dots[jp] = fma(chi_base[jp * ndc], pv, dots[jp]);
    }

    extern __shared__ double smem[];
    for (int jp = 0; jp < np; jp++) {
        double val = blockReduceSum_d(dots[jp], smem);
        if (threadIdx.x == 0)
            atomicAdd(&alpha[(abase + jp) * ncol_stride + (col_start + n)], val * dV);
    }
}

// ============================================================
// V3: V2 + __launch_bounds__(256, 6)
// Forces compiler to ≤40 regs (may spill some to local mem)
// ============================================================
__global__ __launch_bounds__(256, 6)
void fused_gather_chitpsi_kernel_v3(
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
    int n = blockIdx.y;
    if (iat >= n_atoms || n >= ncol_this) return;

    int ndc = ndc_arr[iat];
    int np = nproj_arr[iat];
    if (ndc == 0 || np == 0) return;
    int goff = gpos_offsets[iat];
    int coff = chi_offsets[iat];
    int abase = IP_displ[iat];

    double dots[NL_MAX_NP];
    #pragma unroll 1
    for (int jp = 0; jp < np; jp++) dots[jp] = 0.0;

    for (int i = threadIdx.x; i < ndc; i += blockDim.x) {
        double pv = psi[gpos_flat[goff + i] + (col_start + n) * Nd];
        const double* chi_base = Chi_flat + coff + i;
        #pragma unroll 1
        for (int jp = 0; jp < np; jp++)
            dots[jp] = fma(chi_base[jp * ndc], pv, dots[jp]);
    }

    extern __shared__ double smem[];
    #pragma unroll 1
    for (int jp = 0; jp < np; jp++) {
        double val = blockReduceSum_d(dots[jp], smem);
        if (threadIdx.x == 0)
            atomicAdd(&alpha[(abase + jp) * ncol_stride + (col_start + n)], val * dV);
    }
}

// ============================================================
// V4: 128 threads/block + __launch_bounds__(128, 10)
// 128 threads = 4 warps; with ≤40 regs → up to 12 blocks/SM
// For small systems (4 atoms × 20 cols = 80 blocks), this means
// near-perfect occupancy even on 84 SMs
// ============================================================
__global__ __launch_bounds__(128, 10)
void fused_gather_chitpsi_kernel_v4(
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
    int n = blockIdx.y;
    if (iat >= n_atoms || n >= ncol_this) return;

    int ndc = ndc_arr[iat];
    int np = nproj_arr[iat];
    if (ndc == 0 || np == 0) return;
    int goff = gpos_offsets[iat];
    int coff = chi_offsets[iat];
    int abase = IP_displ[iat];

    double dots[NL_MAX_NP];
    #pragma unroll 1
    for (int jp = 0; jp < np; jp++) dots[jp] = 0.0;

    for (int i = threadIdx.x; i < ndc; i += blockDim.x) {
        double pv = psi[gpos_flat[goff + i] + (col_start + n) * Nd];
        const double* chi_base = Chi_flat + coff + i;
        #pragma unroll 1
        for (int jp = 0; jp < np; jp++)
            dots[jp] = fma(chi_base[jp * ndc], pv, dots[jp]);
    }

    extern __shared__ double smem[];
    #pragma unroll 1
    for (int jp = 0; jp < np; jp++) {
        double val = blockReduceSum_d(dots[jp], smem);
        if (threadIdx.x == 0)
            atomicAdd(&alpha[(abase + jp) * ncol_stride + (col_start + n)], val * dV);
    }
}

// ============================================================
// Scatter kernels: same treatment
// ============================================================

// V1 baseline: 2D grid, alpha in shared memory
__global__ void fused_chialpha_scatter_kernel_v1(
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
    int n = blockIdx.y;
    if (iat >= n_atoms || n >= ncol_this) return;
    int ndc = ndc_arr[iat];
    int np = nproj_arr[iat];
    if (ndc == 0 || np == 0) return;
    int goff = gpos_offsets[iat];
    int coff = chi_offsets[iat];
    int abase = IP_displ[iat];

    extern __shared__ double alpha_sh[];
    if (threadIdx.x < np)
        alpha_sh[threadIdx.x] = alpha[(abase + threadIdx.x) * ncol_stride + (col_start + n)];
    __syncthreads();

    for (int ig = threadIdx.x; ig < ndc; ig += blockDim.x) {
        double val = 0.0;
        const double* chi_row_base = Chi_flat + coff + ig;
        #pragma unroll 4
        for (int jp = 0; jp < np; ++jp)
            val = fma(chi_row_base[jp * ndc], alpha_sh[jp], val);
        atomicAdd(&Hpsi[gpos_flat[goff + ig] + (col_start + n) * Nd], val);
    }
}

// V2: V1 + __launch_bounds__(256, 6)
__global__ __launch_bounds__(256, 6)
void fused_chialpha_scatter_kernel_v2(
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
    int n = blockIdx.y;
    if (iat >= n_atoms || n >= ncol_this) return;
    int ndc = ndc_arr[iat];
    int np = nproj_arr[iat];
    if (ndc == 0 || np == 0) return;
    int goff = gpos_offsets[iat];
    int coff = chi_offsets[iat];
    int abase = IP_displ[iat];

    extern __shared__ double alpha_sh[];
    if (threadIdx.x < np)
        alpha_sh[threadIdx.x] = alpha[(abase + threadIdx.x) * ncol_stride + (col_start + n)];
    __syncthreads();

    for (int ig = threadIdx.x; ig < ndc; ig += blockDim.x) {
        double val = 0.0;
        const double* chi_row_base = Chi_flat + coff + ig;
        #pragma unroll 1
        for (int jp = 0; jp < np; ++jp)
            val = fma(chi_row_base[jp * ndc], alpha_sh[jp], val);
        atomicAdd(&Hpsi[gpos_flat[goff + ig] + (col_start + n) * Nd], val);
    }
}

// V3: 128 threads + __launch_bounds__(128, 10)
__global__ __launch_bounds__(128, 10)
void fused_chialpha_scatter_kernel_v3(
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
    int n = blockIdx.y;
    if (iat >= n_atoms || n >= ncol_this) return;
    int ndc = ndc_arr[iat];
    int np = nproj_arr[iat];
    if (ndc == 0 || np == 0) return;
    int goff = gpos_offsets[iat];
    int coff = chi_offsets[iat];
    int abase = IP_displ[iat];

    extern __shared__ double alpha_sh[];
    if (threadIdx.x < np)
        alpha_sh[threadIdx.x] = alpha[(abase + threadIdx.x) * ncol_stride + (col_start + n)];
    __syncthreads();

    for (int ig = threadIdx.x; ig < ndc; ig += blockDim.x) {
        double val = 0.0;
        const double* chi_row_base = Chi_flat + coff + ig;
        #pragma unroll 1
        for (int jp = 0; jp < np; ++jp)
            val = fma(chi_row_base[jp * ndc], alpha_sh[jp], val);
        atomicAdd(&Hpsi[gpos_flat[goff + ig] + (col_start + n) * Nd], val);
    }
}

int main() {
    const int n_atoms = 4;
    const int nproj_per_atom = 8;
    const int ndc_per_atom = 800;
    const int total_nproj = n_atoms * nproj_per_atom;
    const int Nd = 25 * 26 * 27;
    const int ncol = 20;
    const int NREPS = 200;
    const double dV = 0.01;

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

    int total_ndc = n_atoms * ndc_per_atom;
    int* h_gpos = (int*)malloc(total_ndc * sizeof(int));
    srand(42);
    for (int i = 0; i < total_ndc; i++) h_gpos[i] = rand() % Nd;

    double *d_psi, *d_Chi_flat, *d_alpha, *d_Hpsi;
    int *d_gpos_flat, *d_gpos_offsets, *d_chi_offsets, *d_ndc_arr, *d_nproj_arr, *d_IP_displ;

    CUDA_CHECK(cudaMalloc(&d_psi, (size_t)Nd * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Chi_flat, (size_t)total_ndc * nproj_per_atom * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_alpha, (size_t)total_nproj * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Hpsi, (size_t)Nd * ncol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gpos_flat, total_ndc * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gpos_offsets, (n_atoms + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_chi_offsets, (n_atoms + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ndc_arr, n_atoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nproj_arr, n_atoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_IP_displ, n_atoms * sizeof(int)));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniformDouble(gen, d_psi, Nd * ncol);
    curandGenerateUniformDouble(gen, d_Chi_flat, total_ndc * nproj_per_atom);
    curandGenerateUniformDouble(gen, d_alpha, total_nproj * ncol);
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

    dim3 grid2d(n_atoms, ncol);
    size_t smem256 = (256 / 32) * sizeof(double);  // 8 doubles for block reduce
    size_t smem128 = (128 / 32) * sizeof(double);   // 4 doubles
    size_t smem_scatter = nproj_per_atom * sizeof(double);

    printf("=== Nonlocal High-Occupancy Benchmark ===\n");
    printf("    atoms=%d ndc=%d nproj=%d ncol=%d grid=(%d,%d)=%d blocks\n\n",
           n_atoms, ndc_per_atom, nproj_per_atom, ncol,
           n_atoms, ncol, n_atoms * ncol);

    // --- GATHER benchmarks ---
    auto bench_gather = [&](const char* name, auto launcher) {
        for (int i = 0; i < 5; i++) {
            CUDA_CHECK(cudaMemset(d_alpha, 0, total_nproj * ncol * sizeof(double)));
            launcher();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++) {
            CUDA_CHECK(cudaMemset(d_alpha, 0, total_nproj * ncol * sizeof(double)));
            launcher();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        printf("  gather %-14s: %8.4f ms\n", name, ms / NREPS);
    };

    bench_gather("v2 (no LB)", [&]() {
        fused_gather_chitpsi_kernel_v2<<<grid2d, 256, smem256>>>(
            d_psi, d_Chi_flat, d_gpos_flat, d_gpos_offsets, d_chi_offsets,
            d_ndc_arr, d_nproj_arr, d_IP_displ, d_alpha,
            Nd, ncol, ncol, 0, dV, n_atoms);
    });

    bench_gather("v3 (LB 256,6)", [&]() {
        fused_gather_chitpsi_kernel_v3<<<grid2d, 256, smem256>>>(
            d_psi, d_Chi_flat, d_gpos_flat, d_gpos_offsets, d_chi_offsets,
            d_ndc_arr, d_nproj_arr, d_IP_displ, d_alpha,
            Nd, ncol, ncol, 0, dV, n_atoms);
    });

    bench_gather("v4 (LB 128,10)", [&]() {
        fused_gather_chitpsi_kernel_v4<<<grid2d, 128, smem128>>>(
            d_psi, d_Chi_flat, d_gpos_flat, d_gpos_offsets, d_chi_offsets,
            d_ndc_arr, d_nproj_arr, d_IP_displ, d_alpha,
            Nd, ncol, ncol, 0, dV, n_atoms);
    });

    printf("\n");

    // --- SCATTER benchmarks ---
    auto bench_scatter = [&](const char* name, auto launcher) {
        for (int i = 0; i < 5; i++) {
            CUDA_CHECK(cudaMemset(d_Hpsi, 0, Nd * ncol * sizeof(double)));
            launcher();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++) {
            CUDA_CHECK(cudaMemset(d_Hpsi, 0, Nd * ncol * sizeof(double)));
            launcher();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        printf("  scatter %-13s: %8.4f ms\n", name, ms / NREPS);
    };

    bench_scatter("v1 (no LB)", [&]() {
        fused_chialpha_scatter_kernel_v1<<<grid2d, 256, smem_scatter>>>(
            d_Hpsi, d_Chi_flat, d_gpos_flat, d_gpos_offsets, d_chi_offsets,
            d_ndc_arr, d_nproj_arr, d_IP_displ, d_alpha,
            Nd, ncol, ncol, 0, n_atoms);
    });

    bench_scatter("v2 (LB 256,6)", [&]() {
        fused_chialpha_scatter_kernel_v2<<<grid2d, 256, smem_scatter>>>(
            d_Hpsi, d_Chi_flat, d_gpos_flat, d_gpos_offsets, d_chi_offsets,
            d_ndc_arr, d_nproj_arr, d_IP_displ, d_alpha,
            Nd, ncol, ncol, 0, n_atoms);
    });

    bench_scatter("v3 (LB 128,10)", [&]() {
        fused_chialpha_scatter_kernel_v3<<<grid2d, 128, smem_scatter>>>(
            d_Hpsi, d_Chi_flat, d_gpos_flat, d_gpos_offsets, d_chi_offsets,
            d_ndc_arr, d_nproj_arr, d_IP_displ, d_alpha,
            Nd, ncol, ncol, 0, n_atoms);
    });

    printf("\n");
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_psi); cudaFree(d_Chi_flat); cudaFree(d_alpha); cudaFree(d_Hpsi);
    cudaFree(d_gpos_flat); cudaFree(d_gpos_offsets); cudaFree(d_chi_offsets);
    cudaFree(d_ndc_arr); cudaFree(d_nproj_arr); cudaFree(d_IP_displ);
    free(h_gpos);
    return 0;
}
