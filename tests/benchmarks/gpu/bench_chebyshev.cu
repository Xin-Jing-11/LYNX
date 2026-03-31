// Standalone benchmark for CheFSI vector kernels:
//   chefsi_init_kernel, chefsi_step_kernel, ata_dot_kernel, atb_dot_kernel
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

// Y[i] = scale * (HX[i] - c * X[i])
__global__ void chefsi_init_kernel(
    const double* __restrict__ HX,
    const double* __restrict__ X,
    double* __restrict__ Y,
    double scale, double c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) Y[idx] = scale * (HX[idx] - c * X[idx]);
}

// Xnew[i] = gamma * (HX[i] - c * Y[i]) - ss * Xold[i]
__global__ void chefsi_step_kernel(
    const double* __restrict__ HX,
    const double* __restrict__ Y,
    const double* __restrict__ Xold,
    double* __restrict__ Xnew,
    double gamma, double c, double ss, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) Xnew[idx] = gamma * (HX[idx] - c * Y[idx]) - ss * Xold[idx];
}

// A^T * A for tall-thin matrices
__global__ void ata_dot_kernel(
    const double* __restrict__ A,
    double* __restrict__ C,
    int M, int N, double scale)
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
        double val = sdata[0] * scale;
        C[col_i * N + col_j] = val;
        if (col_i != col_j) C[col_j * N + col_i] = val;
    }
}

// A^T * B for tall-thin matrices
__global__ void atb_dot_kernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int M, int N, double scale)
{
    int col_i = blockIdx.x;
    int col_j = blockIdx.y;

    extern __shared__ double sdata[];
    double sum = 0.0;
    for (int row = threadIdx.x; row < M; row += blockDim.x)
        sum += A[row + col_i * M] * B[row + col_j * M];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) C[col_i * N + col_j] = sdata[0] * scale;
}

int main() {
    const int Nd = 25 * 26 * 27;  // ~17550
    const int Ns = 20;            // number of bands
    const int total = Nd * Ns;
    const int NREPS = 500;

    double *d_X, *d_Y, *d_HX, *d_Xold, *d_Xnew, *d_S;
    CUDA_CHECK(cudaMalloc(&d_X,    total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Y,    total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_HX,   total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Xold, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Xnew, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_S,    Ns * Ns * sizeof(double)));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniformDouble(gen, d_X, total);
    curandGenerateUniformDouble(gen, d_Y, total);
    curandGenerateUniformDouble(gen, d_HX, total);
    curandGenerateUniformDouble(gen, d_Xold, total);
    curandDestroyGenerator(gen);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int bs = 256;
    int grid_vec = ceildiv(total, bs);

    printf("=== CheFSI Vector Kernels: Nd=%d Ns=%d total=%d (NREPS=%d) ===\n", Nd, Ns, total, NREPS);

    // chefsi_init
    {
        for (int i = 0; i < 5; i++)
            chefsi_init_kernel<<<grid_vec, bs>>>(d_HX, d_X, d_Y, 0.5, 1.0, total);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++)
            chefsi_init_kernel<<<grid_vec, bs>>>(d_HX, d_X, d_Y, 0.5, 1.0, total);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        ms /= NREPS;
        double bytes = total * 3.0 * sizeof(double); // 2 reads + 1 write
        printf("  chefsi_init: %8.4f ms  | %6.1f GB/s\n", ms, bytes / (ms * 1e6));
    }

    // chefsi_step
    {
        for (int i = 0; i < 5; i++)
            chefsi_step_kernel<<<grid_vec, bs>>>(d_HX, d_Y, d_Xold, d_Xnew, 0.5, 1.0, 0.3, total);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++)
            chefsi_step_kernel<<<grid_vec, bs>>>(d_HX, d_Y, d_Xold, d_Xnew, 0.5, 1.0, 0.3, total);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        ms /= NREPS;
        double bytes = total * 4.0 * sizeof(double); // 3 reads + 1 write
        printf("  chefsi_step: %8.4f ms  | %6.1f GB/s\n", ms, bytes / (ms * 1e6));
    }

    // ata_dot (X^T * X)
    {
        dim3 ata_grid(Ns, Ns);
        size_t smem = bs * sizeof(double);
        for (int i = 0; i < 5; i++)
            ata_dot_kernel<<<ata_grid, bs, smem>>>(d_X, d_S, Nd, Ns, 0.01);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++)
            ata_dot_kernel<<<ata_grid, bs, smem>>>(d_X, d_S, Nd, Ns, 0.01);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        ms /= NREPS;
        // Each block reads 2 columns of Nd doubles
        double bytes = (double)Ns * (Ns + 1) / 2.0 * 2.0 * Nd * sizeof(double);
        printf("  ata_dot:     %8.4f ms  | %6.1f GB/s\n", ms, bytes / (ms * 1e6));
    }

    // atb_dot (X^T * HX)
    {
        dim3 atb_grid(Ns, Ns);
        size_t smem = bs * sizeof(double);
        for (int i = 0; i < 5; i++)
            atb_dot_kernel<<<atb_grid, bs, smem>>>(d_X, d_HX, d_S, Nd, Ns, 0.01);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++)
            atb_dot_kernel<<<atb_grid, bs, smem>>>(d_X, d_HX, d_S, Nd, Ns, 0.01);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        ms /= NREPS;
        double bytes = (double)Ns * Ns * 2.0 * Nd * sizeof(double);
        printf("  atb_dot:     %8.4f ms  | %6.1f GB/s\n", ms, bytes / (ms * 1e6));
    }

    printf("\n");
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_HX);
    cudaFree(d_Xold); cudaFree(d_Xnew); cudaFree(d_S);
    return 0;
}
