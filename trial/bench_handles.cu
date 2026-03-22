// Benchmark: CUDA library handle creation/destruction and first-call overhead
// Measures: cuBLAS, cuSOLVER, cuSPARSE, cuFFT, cuRAND
// Goal: quantify why pre-allocated handles matter

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <curand.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

#define CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================
// Benchmark handle creation/destruction overhead
// ============================================================
void bench_handle_creation() {
    printf("=== Handle Creation/Destruction Overhead ===\n");
    int repeats = 10;

    // cuBLAS
    {
        double total = 0;
        for (int i = 0; i < repeats; i++) {
            auto t0 = Clock::now();
            cublasHandle_t h;
            cublasCreate(&h);
            cudaDeviceSynchronize();
            auto t1 = Clock::now();
            cublasDestroy(h);
            total += elapsed_ms(t0, t1);
        }
        printf("  cuBLAS create:    %8.3f ms (avg of %d)\n", total / repeats, repeats);
    }

    // cuSOLVER
    {
        double total = 0;
        for (int i = 0; i < repeats; i++) {
            auto t0 = Clock::now();
            cusolverDnHandle_t h;
            cusolverDnCreate(&h);
            cudaDeviceSynchronize();
            auto t1 = Clock::now();
            cusolverDnDestroy(h);
            total += elapsed_ms(t0, t1);
        }
        printf("  cuSOLVER create:  %8.3f ms (avg of %d)\n", total / repeats, repeats);
    }

    // cuSPARSE
    {
        double total = 0;
        for (int i = 0; i < repeats; i++) {
            auto t0 = Clock::now();
            cusparseHandle_t h;
            cusparseCreate(&h);
            cudaDeviceSynchronize();
            auto t1 = Clock::now();
            cusparseDestroy(h);
            total += elapsed_ms(t0, t1);
        }
        printf("  cuSPARSE create:  %8.3f ms (avg of %d)\n", total / repeats, repeats);
    }

    // cuRAND
    {
        double total = 0;
        for (int i = 0; i < repeats; i++) {
            auto t0 = Clock::now();
            curandGenerator_t g;
            curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
            cudaDeviceSynchronize();
            auto t1 = Clock::now();
            curandDestroyGenerator(g);
            total += elapsed_ms(t0, t1);
        }
        printf("  cuRAND create:    %8.3f ms (avg of %d)\n", total / repeats, repeats);
    }

    // CUDA stream
    {
        double total = 0;
        for (int i = 0; i < repeats; i++) {
            auto t0 = Clock::now();
            cudaStream_t s;
            cudaStreamCreate(&s);
            cudaDeviceSynchronize();
            auto t1 = Clock::now();
            cudaStreamDestroy(s);
            total += elapsed_ms(t0, t1);
        }
        printf("  cudaStream create:%8.3f ms (avg of %d)\n", total / repeats, repeats);
    }

    // CUDA event
    {
        double total = 0;
        for (int i = 0; i < repeats; i++) {
            auto t0 = Clock::now();
            cudaEvent_t e;
            cudaEventCreate(&e);
            cudaDeviceSynchronize();
            auto t1 = Clock::now();
            cudaEventDestroy(e);
            total += elapsed_ms(t0, t1);
        }
        printf("  cudaEvent create: %8.3f ms (avg of %d)\n", total / repeats, repeats);
    }
    printf("\n");
}

// ============================================================
// Benchmark first-call vs warm-call latency
// ============================================================
void bench_first_call_latency() {
    printf("=== First-Call vs Warm-Call Latency ===\n");
    printf("  (First call triggers JIT compilation + workspace allocation)\n\n");

    int N = 64;
    double *d_A, *d_B, *d_C, *d_W;
    CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, N * N * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_A, 0, N * N * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_B, 0, N * N * sizeof(double)));

    // --- cuBLAS dgemm ---
    {
        cublasHandle_t h;
        cublasCreate(&h);
        cudaDeviceSynchronize();

        double alpha = 1.0, beta = 0.0;

        // First call
        auto t0 = Clock::now();
        cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                     &alpha, d_A, N, d_B, N, &beta, d_C, N);
        cudaDeviceSynchronize();
        auto t1 = Clock::now();
        double first = elapsed_ms(t0, t1);

        // Warm calls
        double warm_total = 0;
        int warm_iters = 100;
        cudaDeviceSynchronize();
        t0 = Clock::now();
        for (int i = 0; i < warm_iters; i++) {
            cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                         &alpha, d_A, N, d_B, N, &beta, d_C, N);
        }
        cudaDeviceSynchronize();
        t1 = Clock::now();
        warm_total = elapsed_ms(t0, t1) / warm_iters;

        printf("  cuBLAS dgemm(%dx%d):   first=%.3f ms  warm=%.4f ms  ratio=%.0fx\n",
               N, N, first, warm_total, first / warm_total);

        // With pre-allocated workspace
        size_t ws_size = 4 * 1024 * 1024;
        void* ws;
        CUDA_CHECK(cudaMalloc(&ws, ws_size));
        cublasSetWorkspace(h, ws, ws_size);

        t0 = Clock::now();
        for (int i = 0; i < warm_iters; i++) {
            cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                         &alpha, d_A, N, d_B, N, &beta, d_C, N);
        }
        cudaDeviceSynchronize();
        t1 = Clock::now();
        double ws_warm = elapsed_ms(t0, t1) / warm_iters;
        printf("  cuBLAS dgemm (w/ workspace): warm=%.4f ms\n", ws_warm);

        cudaFree(ws);
        cublasDestroy(h);
    }

    // --- cuSOLVER dsyevd ---
    {
        cusolverDnHandle_t h;
        cusolverDnCreate(&h);
        cudaDeviceSynchronize();

        double* d_eig;
        int* d_info;
        CUDA_CHECK(cudaMalloc(&d_eig, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

        // Query workspace
        int lwork = 0;
        cusolverDnDsyevd_bufferSize(h, CUSOLVER_EIG_MODE_VECTOR,
                                      CUBLAS_FILL_MODE_UPPER,
                                      N, d_A, N, d_eig, &lwork);
        double* d_work;
        CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));

        // First call
        auto t0 = Clock::now();
        cusolverDnDsyevd(h, CUSOLVER_EIG_MODE_VECTOR,
                           CUBLAS_FILL_MODE_UPPER,
                           N, d_A, N, d_eig, d_work, lwork, d_info);
        cudaDeviceSynchronize();
        auto t1 = Clock::now();
        double first = elapsed_ms(t0, t1);

        // Warm calls
        int warm_iters = 50;
        cudaDeviceSynchronize();
        t0 = Clock::now();
        for (int i = 0; i < warm_iters; i++) {
            cusolverDnDsyevd(h, CUSOLVER_EIG_MODE_VECTOR,
                               CUBLAS_FILL_MODE_UPPER,
                               N, d_A, N, d_eig, d_work, lwork, d_info);
        }
        cudaDeviceSynchronize();
        t1 = Clock::now();
        double warm = elapsed_ms(t0, t1) / warm_iters;
        printf("  cuSOLVER dsyevd(%d):   first=%.3f ms  warm=%.4f ms  ratio=%.0fx\n",
               N, first, warm, first / warm);

        cudaFree(d_eig); cudaFree(d_info); cudaFree(d_work);
        cusolverDnDestroy(h);
    }

    // --- cuSOLVER dpotrf ---
    {
        cusolverDnHandle_t h;
        cusolverDnCreate(&h);
        cudaDeviceSynchronize();

        int* d_info;
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

        int lwork = 0;
        cusolverDnDpotrf_bufferSize(h, CUBLAS_FILL_MODE_UPPER,
                                      N, d_A, N, &lwork);
        double* d_work;
        CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));

        auto t0 = Clock::now();
        cusolverDnDpotrf(h, CUBLAS_FILL_MODE_UPPER,
                           N, d_A, N, d_work, lwork, d_info);
        cudaDeviceSynchronize();
        auto t1 = Clock::now();
        double first = elapsed_ms(t0, t1);

        int warm_iters = 100;
        t0 = Clock::now();
        for (int i = 0; i < warm_iters; i++) {
            cusolverDnDpotrf(h, CUBLAS_FILL_MODE_UPPER,
                               N, d_A, N, d_work, lwork, d_info);
        }
        cudaDeviceSynchronize();
        t1 = Clock::now();
        double warm = elapsed_ms(t0, t1) / warm_iters;
        printf("  cuSOLVER dpotrf(%d):   first=%.3f ms  warm=%.4f ms  ratio=%.0fx\n",
               N, first, warm, first / warm);

        cudaFree(d_info); cudaFree(d_work);
        cusolverDnDestroy(h);
    }

    // --- cuBLAS dtrsm ---
    {
        cublasHandle_t h;
        cublasCreate(&h);
        cudaDeviceSynchronize();

        double alpha = 1.0;

        auto t0 = Clock::now();
        cublasDtrsm(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                     CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                     N, N, &alpha, d_A, N, d_B, N);
        cudaDeviceSynchronize();
        auto t1 = Clock::now();
        double first = elapsed_ms(t0, t1);

        int warm_iters = 100;
        t0 = Clock::now();
        for (int i = 0; i < warm_iters; i++) {
            cublasDtrsm(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                         CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                         N, N, &alpha, d_A, N, d_B, N);
        }
        cudaDeviceSynchronize();
        t1 = Clock::now();
        double warm = elapsed_ms(t0, t1) / warm_iters;
        printf("  cuBLAS dtrsm(%dx%d):   first=%.3f ms  warm=%.4f ms  ratio=%.0fx\n",
               N, N, first, warm, first / warm);

        cublasDestroy(h);
    }

    // --- cuSPARSE SpMV ---
    {
        cusparseHandle_t h;
        cusparseCreate(&h);
        cudaDeviceSynchronize();

        // Create a simple diagonal sparse matrix
        int nnz = N;
        int *d_row, *d_col;
        double *d_val, *d_x, *d_y;
        CUDA_CHECK(cudaMalloc(&d_row, (N + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col, nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_val, nnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(double)));

        std::vector<int> h_row(N + 1), h_col(N);
        std::vector<double> h_val(N, 1.0);
        for (int i = 0; i <= N; i++) h_row[i] = i;
        for (int i = 0; i < N; i++) h_col[i] = i;
        CUDA_CHECK(cudaMemcpy(d_row, h_row.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col, h_col.data(), N*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), N*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_x, 0, N * sizeof(double)));

        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;
        cusparseCreateCsr(&matA, N, N, nnz, d_row, d_col, d_val,
                           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_64F);
        cusparseCreateDnVec(&vecY, N, d_y, CUDA_R_64F);

        double alpha = 1.0, beta_v = 0.0;
        size_t bufSize;
        cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta_v, vecY,
                                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize);
        void* d_buf;
        CUDA_CHECK(cudaMalloc(&d_buf, bufSize));

        auto t0 = Clock::now();
        cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      &alpha, matA, vecX, &beta_v, vecY,
                      CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf);
        cudaDeviceSynchronize();
        auto t1 = Clock::now();
        double first = elapsed_ms(t0, t1);

        int warm_iters = 100;
        t0 = Clock::now();
        for (int i = 0; i < warm_iters; i++) {
            cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, matA, vecX, &beta_v, vecY,
                          CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf);
        }
        cudaDeviceSynchronize();
        t1 = Clock::now();
        double warm = elapsed_ms(t0, t1) / warm_iters;
        printf("  cuSPARSE SpMV(%d):     first=%.3f ms  warm=%.4f ms  ratio=%.0fx\n",
               N, first, warm, first / warm);

        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cudaFree(d_buf); cudaFree(d_row); cudaFree(d_col);
        cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
        cusparseDestroy(h);
    }

    // --- cuRAND generate ---
    {
        curandGenerator_t g;
        curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(g, 42);
        cudaDeviceSynchronize();

        double* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, N * N * sizeof(double)));

        auto t0 = Clock::now();
        curandGenerateUniformDouble(g, d_out, N * N);
        cudaDeviceSynchronize();
        auto t1 = Clock::now();
        double first = elapsed_ms(t0, t1);

        int warm_iters = 100;
        t0 = Clock::now();
        for (int i = 0; i < warm_iters; i++) {
            curandGenerateUniformDouble(g, d_out, N * N);
        }
        cudaDeviceSynchronize();
        t1 = Clock::now();
        double warm = elapsed_ms(t0, t1) / warm_iters;
        printf("  cuRAND uniform(%d):    first=%.3f ms  warm=%.4f ms  ratio=%.0fx\n",
               N * N, first, warm, first / warm);

        cudaFree(d_out);
        curandDestroyGenerator(g);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_W);
    printf("\n");
}

// ============================================================
// Benchmark: shared vs per-call handle
// ============================================================
void bench_shared_vs_percall() {
    printf("=== Shared Handle vs Per-Call Handle (10 SCF iterations) ===\n");

    int N = 30; // typical band count
    int Nd = 17576; // 26^3
    int iters = 10;

    double *d_X, *d_S;
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)Nd * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_S, N * N * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_X, 0, (size_t)Nd * N * sizeof(double)));

    double alpha = 1.0, beta = 0.0;

    // Shared handle (pre-created)
    cublasHandle_t shared_h;
    cublasCreate(&shared_h);
    size_t ws_size = 4 * 1024 * 1024;
    void* ws;
    CUDA_CHECK(cudaMalloc(&ws, ws_size));
    cublasSetWorkspace(shared_h, ws, ws_size);
    cudaDeviceSynchronize();

    // Warm up
    cublasDgemm(shared_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 N, N, Nd, &alpha, d_X, Nd, d_X, Nd, &beta, d_S, N);
    cudaDeviceSynchronize();

    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) {
        cublasDgemm(shared_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     N, N, Nd, &alpha, d_X, Nd, d_X, Nd, &beta, d_S, N);
        cudaDeviceSynchronize();
    }
    auto t1 = Clock::now();
    double shared_time = elapsed_ms(t0, t1);
    printf("  Shared handle (%d iters): %.3f ms total, %.3f ms/iter\n",
           iters, shared_time, shared_time / iters);

    // Per-call handle (create + destroy each time)
    t0 = Clock::now();
    for (int i = 0; i < iters; i++) {
        cublasHandle_t h;
        cublasCreate(&h);
        cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                     N, N, Nd, &alpha, d_X, Nd, d_X, Nd, &beta, d_S, N);
        cudaDeviceSynchronize();
        cublasDestroy(h);
    }
    t1 = Clock::now();
    double percall_time = elapsed_ms(t0, t1);
    printf("  Per-call handle (%d iters): %.3f ms total, %.3f ms/iter\n",
           iters, percall_time, percall_time / iters);
    printf("  Overhead ratio: %.1fx\n", percall_time / shared_time);

    cublasDestroy(shared_h);
    cudaFree(ws);
    cudaFree(d_X); cudaFree(d_S);
    printf("\n");
}

// ============================================================
// Benchmark: cuBLAS workspace pre-alloc vs on-demand
// ============================================================
void bench_workspace_prealloc() {
    printf("=== cuBLAS Workspace: Pre-allocated vs On-Demand ===\n");

    int sizes[] = {12, 30, 50, 100};
    int Nd = 17576;

    for (int N : sizes) {
        double *d_X, *d_S;
        CUDA_CHECK(cudaMalloc(&d_X, (size_t)Nd * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_S, N * N * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_X, 0, (size_t)Nd * N * sizeof(double)));

        double alpha = 1.0, beta = 0.0;
        int iters = 200;

        // With pre-allocated workspace
        cublasHandle_t h1;
        cublasCreate(&h1);
        size_t ws_size = 4 * 1024 * 1024;
        void* ws;
        CUDA_CHECK(cudaMalloc(&ws, ws_size));
        cublasSetWorkspace(h1, ws, ws_size);

        // Warm up
        cublasDgemm(h1, CUBLAS_OP_T, CUBLAS_OP_N,
                     N, N, Nd, &alpha, d_X, Nd, d_X, Nd, &beta, d_S, N);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        float ms;

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            cublasDgemm(h1, CUBLAS_OP_T, CUBLAS_OP_N,
                         N, N, Nd, &alpha, d_X, Nd, d_X, Nd, &beta, d_S, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        double with_ws = ms / iters;

        // Without pre-allocated workspace
        cublasHandle_t h2;
        cublasCreate(&h2);
        // No setWorkspace — cuBLAS will malloc internally

        cublasDgemm(h2, CUBLAS_OP_T, CUBLAS_OP_N,
                     N, N, Nd, &alpha, d_X, Nd, d_X, Nd, &beta, d_S, N);
        cudaDeviceSynchronize();

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            cublasDgemm(h2, CUBLAS_OP_T, CUBLAS_OP_N,
                         N, N, Nd, &alpha, d_X, Nd, d_X, Nd, &beta, d_S, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        double without_ws = ms / iters;

        printf("  N=%3d: with_ws=%.4f ms  without_ws=%.4f ms  diff=%.4f ms (%.1f%%)\n",
               N, with_ws, without_ws, without_ws - with_ws,
               (without_ws - with_ws) / without_ws * 100);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        cudaFree(ws);
        cublasDestroy(h1);
        cublasDestroy(h2);
        cudaFree(d_X); cudaFree(d_S);
    }
    printf("\n");
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (CUDA %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Warm up GPU
    {
        double* d;
        cudaMalloc(&d, 1024);
        cudaFree(d);
    }

    bench_handle_creation();
    bench_first_call_latency();
    bench_shared_vs_percall();
    bench_workspace_prealloc();

    printf("Done.\n");
    return 0;
}
