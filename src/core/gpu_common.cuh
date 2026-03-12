#pragma once
#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(e)); \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

// Max FD half-order supported (order 12 → FDn=6, 7 coefficients including [0])
#define MAX_FDN 6
#define MAX_FD_COEFF (MAX_FDN + 1)

namespace sparc {
namespace gpu {

// FD stencil coefficients in constant memory
// Diagonal 2nd derivatives (scaled for grid spacing)
extern __constant__ double d_D2x[MAX_FD_COEFF];
extern __constant__ double d_D2y[MAX_FD_COEFF];
extern __constant__ double d_D2z[MAX_FD_COEFF];
// Mixed 2nd derivatives (non-orthogonal cells)
extern __constant__ double d_D2xy[MAX_FD_COEFF];
extern __constant__ double d_D2xz[MAX_FD_COEFF];
extern __constant__ double d_D2yz[MAX_FD_COEFF];
// 1st derivatives
extern __constant__ double d_D1x[MAX_FD_COEFF];
extern __constant__ double d_D1y[MAX_FD_COEFF];
extern __constant__ double d_D1z[MAX_FD_COEFF];

// Upload stencil coefficients from host to constant memory.
// Each array should have (FDn+1) entries.
void upload_stencil_coefficients(
    const double* D2x, const double* D2y, const double* D2z,
    const double* D1x, const double* D1y, const double* D1z,
    const double* D2xy, const double* D2xz, const double* D2yz,
    int FDn);

// Helper: integer ceiling division
inline __host__ __device__ int ceildiv(int a, int b) { return (a + b - 1) / b; }

} // namespace gpu
} // namespace sparc

#endif // USE_CUDA
