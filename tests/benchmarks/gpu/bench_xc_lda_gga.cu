// Standalone benchmark for LDA and GGA XC kernels
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

__host__ __device__ int ceildiv(int a, int b) { return (a + b - 1) / b; }

// LDA PW92 kernel (fused exchange + correlation)
__global__ void lda_pw_kernel(
    const double* __restrict__ rho,
    double* __restrict__ exc,
    double* __restrict__ vxc,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double r = rho[i];
    if (r < 1e-30) { exc[i] = 0.0; vxc[i] = 0.0; return; }

    constexpr double PI = 3.14159265358979323846;
    // Exchange: Slater
    double rs = cbrt(0.75 / (PI * r));
    double ex = -0.75 * cbrt(3.0 / PI) * cbrt(r);
    double vx = -cbrt(3.0 * r / PI);

    // Correlation: PW92
    constexpr double A = 0.0310907, a1 = 0.21370;
    constexpr double b1 = 7.5957, b2 = 3.5876, b3 = 1.6382, b4 = 0.49294;
    double sqrRs = sqrt(rs);
    double Q0 = -2.0 * A * (1.0 + a1 * rs);
    double Q1 = 2.0 * A * (b1 * sqrRs + b2 * rs + b3 * rs * sqrRs + b4 * rs * rs);
    double den = 1.0 / (Q1 * Q1 + Q1);
    double ec = Q0 * (-log(Q1 * Q1 * den));

    // Derivative
    double dQ0 = -2.0 * A * a1;
    double dQ1 = A * (b1 / sqrRs + 2.0 * b2 + 3.0 * b3 * sqrRs + 4.0 * b4 * rs);
    double dLogQ = -2.0 * dQ1 / Q1 + dQ1 * (2.0 * Q1 + 1.0) / (Q1 * Q1 + Q1);
    double drsdn = -rs / (3.0 * r);
    double vc = ec + r * drsdn * (dQ0 * (-log(Q1 * Q1 * den)) + Q0 * dLogQ);

    exc[i] = ex + ec;
    vxc[i] = vx + vc;
}

// GGA PBE kernel (simplified for benchmarking)
__global__ void gga_pbe_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ sigma,
    double* __restrict__ exc,
    double* __restrict__ vxc,
    double* __restrict__ v2xc,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double r = rho[i];
    if (r < 1e-30) { exc[i] = 0.0; vxc[i] = 0.0; v2xc[i] = 0.0; return; }

    double sig = sigma[i];
    if (sig < 1e-14) sig = 1e-14;

    constexpr double PI = 3.14159265358979323846;
    constexpr double kappa = 0.804, mu = 0.21951;

    double rs = cbrt(0.75 / (PI * r));
    double kf = cbrt(3.0 * PI * PI * r);
    double s = sqrt(sig) / (2.0 * kf * r);
    double s2 = s * s;

    // PBE exchange enhancement
    double Fx = 1.0 + kappa - kappa / (1.0 + mu * s2 / kappa);
    double ex_unif = -0.75 * cbrt(3.0 / PI) * cbrt(r);
    double ex = ex_unif * Fx;

    // PBE correlation (simplified)
    double sqrRs = sqrt(rs);
    constexpr double A = 0.0310907, a1 = 0.21370;
    constexpr double b1 = 7.5957, b2 = 3.5876, b3 = 1.6382, b4 = 0.49294;
    double Q0 = -2.0 * A * (1.0 + a1 * rs);
    double Q1 = 2.0 * A * (b1 * sqrRs + b2 * rs + b3 * rs * sqrRs + b4 * rs * rs);
    double ec_lda = Q0 * (-log(Q1 * Q1 / (Q1 * Q1 + Q1)));

    constexpr double beta = 0.06672455060314922, gamma = 0.031091;
    double phi = 1.0;
    double t = sqrt(sig) / (2.0 * phi * kf * sqrRs * r);
    double At = beta / gamma * (1.0 / (exp(-ec_lda / (gamma * phi * phi * phi)) - 1.0 + 1e-30));
    double t2 = t * t;
    double H = gamma * phi * phi * phi * log(1.0 + beta / gamma * t2 * (1.0 + At * t2) / (1.0 + At * t2 + At * At * t2 * t2));

    exc[i] = ex + ec_lda + H;
    vxc[i] = ex * 1.33 + ec_lda + H;  // simplified for bench
    v2xc[i] = ex * 0.5;
}

int main() {
    const int N = 25 * 26 * 27;
    const int NREPS = 500;

    double *d_rho, *d_sigma, *d_exc, *d_vxc, *d_v2xc;
    CUDA_CHECK(cudaMalloc(&d_rho, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sigma, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_exc, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vxc, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v2xc, N * sizeof(double)));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniformDouble(gen, d_rho, N);
    curandGenerateUniformDouble(gen, d_sigma, N);
    curandDestroyGenerator(gen);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int bs = 256;
    int grid = ceildiv(N, bs);

    printf("=== LDA/GGA XC Kernel Benchmark: N=%d (NREPS=%d) ===\n", N, NREPS);

    // LDA
    {
        for (int i = 0; i < 10; i++)
            lda_pw_kernel<<<grid, bs>>>(d_rho, d_exc, d_vxc, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++)
            lda_pw_kernel<<<grid, bs>>>(d_rho, d_exc, d_vxc, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        ms /= NREPS;
        double bytes = N * 3.0 * sizeof(double);
        printf("  lda_pw:   %8.4f ms  | %6.1f GB/s\n", ms, bytes / (ms * 1e6));
    }

    // GGA PBE
    {
        for (int i = 0; i < 10; i++)
            gga_pbe_kernel<<<grid, bs>>>(d_rho, d_sigma, d_exc, d_vxc, d_v2xc, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NREPS; i++)
            gga_pbe_kernel<<<grid, bs>>>(d_rho, d_sigma, d_exc, d_vxc, d_v2xc, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; cudaEventElapsedTime(&ms, start, stop);
        ms /= NREPS;
        double bytes = N * 5.0 * sizeof(double);
        printf("  gga_pbe:  %8.4f ms  | %6.1f GB/s\n", ms, bytes / (ms * 1e6));
    }

    printf("\n");
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_rho); cudaFree(d_sigma);
    cudaFree(d_exc); cudaFree(d_vxc); cudaFree(d_v2xc);
    return 0;
}
