// Distributed A^T * A — pure GPU analysis
//
// NCCL requires separate physical GPUs for multi-rank, so on 1 GPU we:
// 1. Measure compute time for different M_local sizes (simulating P-way split)
// 2. Measure GPU-side allreduce simulation (cudaMemcpy D2D as proxy for NCCL)
// 3. Use known NCCL allreduce latency model: ~3-5 us base + BW term
//
// NCCL AllReduce for small payloads (N*N doubles):
//   - Intra-node NVLink: ~3-5 us for < 64KB (latency-bound)
//   - Inter-node IB:     ~5-10 us for < 64KB
//   - N=30: 30*30*8 = 7.2 KB → pure latency, ~3-5 us on NVLink
//   - N=50: 50*50*8 = 20 KB → still latency-bound, ~5 us

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS %s:%d: %d\n", __FILE__, __LINE__, (int)s); exit(1); } } while(0)

// Custom dot kernel
__global__ void ata_dot_kernel(
    const double* __restrict__ A,
    double* __restrict__ C,
    int M, int N)
{
    int col_i = blockIdx.x;
    int col_j = blockIdx.y;
    if (col_j < col_i) return;

    extern __shared__ double sdata[];
    double sum = 0.0;
    for (int row = threadIdx.x; row < M; row += blockDim.x)
        sum += A[row + col_i * M] * A[row + col_j * M];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        C[col_i * N + col_j] = sdata[0];
        if (col_i != col_j)
            C[col_j * N + col_i] = sdata[0];
    }
}

// Simulate allreduce: sum P buffers into one (GPU-side, measures kernel overhead)
__global__ void sum_buffers_kernel(const double* __restrict__ src, double* __restrict__ dst, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) dst[idx] += src[idx];
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // ============================================================
    // Part 1: Measure NCCL AllReduce proxy cost (GPU D2D operations)
    // For N*N doubles, NCCL allreduce on NVLink is ~3-5 us.
    // We measure cudaMemcpyD2D + kernel sum as a lower bound.
    // ============================================================
    printf("=== GPU-side AllReduce proxy (D2D memcpy + sum kernel) ===\n");
    printf("  Simulates NCCL ring-allreduce: each step = send+recv one chunk\n");
    printf("  Real NCCL adds ~2-3 us base latency on top.\n\n");
    printf("%-6s  %8s  %10s  %10s\n", "N", "Bytes", "D2D copy", "Sum kernel");
    printf("------  --------  ----------  ----------\n");

    for (int N : {12, 30, 50, 100}) {
        size_t C_bytes = (size_t)N * N * sizeof(double);
        double *d_src, *d_dst;
        CUDA_CHECK(cudaMalloc(&d_src, C_bytes));
        CUDA_CHECK(cudaMalloc(&d_dst, C_bytes));
        CUDA_CHECK(cudaMemset(d_src, 0, C_bytes));
        CUDA_CHECK(cudaMemset(d_dst, 0, C_bytes));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        int iters = 5000;
        // D2D copy
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i)
            CUDA_CHECK(cudaMemcpyAsync(d_dst, d_src, C_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        double d2d_us = ms / iters * 1000;

        // Sum kernel
        int bs = 256;
        int nn = N * N;
        int grid = (nn + bs - 1) / bs;
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i)
            sum_buffers_kernel<<<grid, bs>>>(d_src, d_dst, nn);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&ms, start, stop);
        double sum_us = ms / iters * 1000;

        printf("N=%-4d  %6.1f KB  %7.2f us  %7.2f us\n", N, C_bytes / 1024.0, d2d_us, sum_us);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_dst));
    }

    // ============================================================
    // Part 2: Compute cost for different (M_local, N, nranks) setups
    // M_local = M_global / nranks — what each GPU would compute
    // ============================================================
    printf("\n=== Compute cost: DOT kernel for local A^T*A ===\n");
    printf("  Each GPU computes C_local = A_local^T * A_local\n");
    printf("  Then NCCL AllReduce(sum) the N*N result.\n\n");

    printf("%-8s  %-4s  %-5s  %-8s  %10s  %10s  %10s  %10s\n",
           "M_global", "N", "nGPU", "M_local", "DOT(ms)", "GEMM(ms)", "NCCL_est", "Total_DOT");
    printf("--------  ----  -----  --------  ----------  ----------  ----------  ----------\n");

    // NCCL allreduce latency estimate for small payloads on NVLink
    // Based on published benchmarks: ~3-5 us for < 64KB intra-node
    auto nccl_latency_us = [](int N, int nGPU) -> double {
        double bytes = N * N * 8.0;
        // Ring allreduce: 2*(P-1)/P * bytes / BW + base_latency
        // For small payloads, latency-dominated
        double base_us = 3.0;  // NVLink base latency
        double per_step_us = 1.0;  // per ring step overhead
        return base_us + (nGPU - 1) * per_step_us;
    };

    struct Config { int M; int N; const char* label; };
    Config configs[] = {
        {26*26*26,    12, "26³"},
        {26*26*26,    30, "26³"},
        {48*48*48,    12, "48³"},
        {48*48*48,    30, "48³"},
        {72*72*72,    12, "72³"},
        {72*72*72,    30, "72³"},
        {100*100*100, 12, "100³"},
        {100*100*100, 30, "100³"},
        {100*100*100, 50, "100³"},
    };

    for (auto& cfg : configs) {
        for (int nGPU : {1, 2, 4, 8}) {
            int M_local = cfg.M / nGPU;
            int N = cfg.N;

            double *d_A, *d_C;
            CUDA_CHECK(cudaMalloc(&d_A, (size_t)M_local * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_C, (size_t)N * N * sizeof(double)));

            std::vector<double> h_A(M_local * N);
            for (size_t i = 0; i < h_A.size(); ++i)
                h_A[i] = sin(0.001 * i) * 0.1;
            CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), (size_t)M_local * N * sizeof(double), cudaMemcpyHostToDevice));

            int iters = std::max(10, std::min(5000, (int)(5e8 / (2.0 * M_local * N * N))));
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            float ms;

            // DOT kernel
            int bs = 256;
            dim3 grid(N, N);
            for (int w = 0; w < 5; ++w)
                ata_dot_kernel<<<grid, bs, bs * sizeof(double)>>>(d_A, d_C, M_local, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < iters; ++i)
                ata_dot_kernel<<<grid, bs, bs * sizeof(double)>>>(d_A, d_C, M_local, N);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            cudaEventElapsedTime(&ms, start, stop);
            double dot_ms = ms / iters;

            // cuBLAS DGEMM
            double alpha = 1.0, beta = 0.0;
            for (int w = 0; w < 5; ++w)
                CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                         N, N, M_local, &alpha, d_A, M_local, d_A, M_local, &beta, d_C, N));
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < iters; ++i)
                CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                         N, N, M_local, &alpha, d_A, M_local, d_A, M_local, &beta, d_C, N));
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            cudaEventElapsedTime(&ms, start, stop);
            double gemm_ms = ms / iters;

            double nccl_ms = nccl_latency_us(N, nGPU) / 1000.0;
            double total_dot_ms = dot_ms + nccl_ms;

            printf("%-8d  %-4d  %-5d  %-8d  %8.3f    %8.3f    %8.3f    %8.3f\n",
                   cfg.M, N, nGPU, M_local, dot_ms, gemm_ms, nccl_ms, total_dot_ms);

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_C));
        }
        printf("\n");
    }

    // ============================================================
    // Part 3: Correctness verification of distributed approach
    // ============================================================
    printf("=== Correctness: distributed sum == single-process ===\n");
    for (auto& cfg : configs) {
        int M = cfg.M;
        int N = cfg.N;
        if (M > 200000) continue;  // skip huge for speed

        for (int nGPU : {2, 4}) {
            int M_local = M / nGPU;

            // Single-process reference
            std::vector<double> h_A_full(M * N);
            for (int col = 0; col < N; ++col)
                for (int row = 0; row < M; ++row)
                    h_A_full[row + col * M] = sin(0.001 * (row + col * M)) * 0.1;

            double *d_A_full, *d_C_ref;
            CUDA_CHECK(cudaMalloc(&d_A_full, (size_t)M * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_C_ref, (size_t)N * N * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_A_full, h_A_full.data(), (size_t)M * N * sizeof(double), cudaMemcpyHostToDevice));

            double alpha = 1.0, beta = 0.0;
            CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     N, N, M, &alpha, d_A_full, M, d_A_full, M, &beta, d_C_ref, N));
            std::vector<double> C_ref(N * N);
            CUDA_CHECK(cudaMemcpy(C_ref.data(), d_C_ref, N * N * sizeof(double), cudaMemcpyDeviceToHost));
            cudaFree(d_A_full); cudaFree(d_C_ref);

            // Distributed: sum partial results
            std::vector<double> C_sum(N * N, 0.0);
            for (int g = 0; g < nGPU; ++g) {
                int offset = g * M_local;
                std::vector<double> h_A_local(M_local * N);
                for (int col = 0; col < N; ++col)
                    for (int row = 0; row < M_local; ++row)
                        h_A_local[row + col * M_local] = h_A_full[(offset + row) + col * M];

                double *d_A, *d_C;
                CUDA_CHECK(cudaMalloc(&d_A, (size_t)M_local * N * sizeof(double)));
                CUDA_CHECK(cudaMalloc(&d_C, (size_t)N * N * sizeof(double)));
                CUDA_CHECK(cudaMemcpy(d_A, h_A_local.data(), (size_t)M_local * N * sizeof(double), cudaMemcpyHostToDevice));

                int bs = 256; dim3 grid(N, N);
                ata_dot_kernel<<<grid, bs, bs * sizeof(double)>>>(d_A, d_C, M_local, N);

                std::vector<double> C_part(N * N);
                CUDA_CHECK(cudaMemcpy(C_part.data(), d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost));
                for (int i = 0; i < N * N; ++i)
                    C_sum[i] += C_part[i];

                cudaFree(d_A); cudaFree(d_C);
            }

            double max_err = 0;
            for (int i = 0; i < N * N; ++i)
                max_err = std::max(max_err, std::abs(C_ref[i] - C_sum[i]));

            printf("  M=%7d N=%2d nGPU=%d: err=%.2e %s\n",
                   M, N, nGPU, max_err, max_err < 1e-8 ? "OK" : "FAIL");
        }
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    printf("\nDone.\n");
    return 0;
}
