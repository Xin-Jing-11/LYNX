#ifdef USE_CUDA
#include "gpu_common.cuh"

namespace lynx {
namespace gpu {

// Define constant memory symbols
__constant__ double d_D2x[MAX_FD_COEFF];
__constant__ double d_D2y[MAX_FD_COEFF];
__constant__ double d_D2z[MAX_FD_COEFF];
__constant__ double d_D2xy[MAX_FD_COEFF];
__constant__ double d_D2xz[MAX_FD_COEFF];
__constant__ double d_D2yz[MAX_FD_COEFF];
__constant__ double d_D1x[MAX_FD_COEFF];
__constant__ double d_D1y[MAX_FD_COEFF];
__constant__ double d_D1z[MAX_FD_COEFF];

void upload_stencil_coefficients(
    const double* D2x, const double* D2y, const double* D2z,
    const double* D1x, const double* D1y, const double* D1z,
    const double* D2xy, const double* D2xz, const double* D2yz,
    int FDn)
{
    int n = FDn + 1;
    size_t bytes = n * sizeof(double);

    CUDA_CHECK(cudaMemcpyToSymbol(d_D2x, D2x, bytes));
    CUDA_CHECK(cudaMemcpyToSymbol(d_D2y, D2y, bytes));
    CUDA_CHECK(cudaMemcpyToSymbol(d_D2z, D2z, bytes));
    CUDA_CHECK(cudaMemcpyToSymbol(d_D1x, D1x, bytes));
    CUDA_CHECK(cudaMemcpyToSymbol(d_D1y, D1y, bytes));
    CUDA_CHECK(cudaMemcpyToSymbol(d_D1z, D1z, bytes));

    if (D2xy) CUDA_CHECK(cudaMemcpyToSymbol(d_D2xy, D2xy, bytes));
    if (D2xz) CUDA_CHECK(cudaMemcpyToSymbol(d_D2xz, D2xz, bytes));
    if (D2yz) CUDA_CHECK(cudaMemcpyToSymbol(d_D2yz, D2yz, bytes));
}

} // namespace gpu
} // namespace lynx
#endif // USE_CUDA
