#pragma once
#ifdef USE_CUDA

#include <cuda_runtime.h>

namespace lynx {
namespace gpu {

// Halo exchange: copy inner domain into extended array with periodic BCs
void halo_exchange_gpu(
    const double* d_x, double* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol,
    bool periodic_x, bool periodic_y, bool periodic_z,
    cudaStream_t stream = 0);

// Batched halo exchange (memsets extended region first)
void halo_exchange_batched_gpu(
    const double* d_x, double* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol,
    bool periodic_x, bool periodic_y, bool periodic_z,
    cudaStream_t stream = 0);

// Batched halo exchange without memset (caller ensures extended region is clean)
void halo_exchange_batched_nomemset_gpu(
    const double* d_x, double* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol,
    bool periodic_x, bool periodic_y, bool periodic_z,
    cudaStream_t stream = 0);

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
