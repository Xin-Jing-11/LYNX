#pragma once
#ifdef USE_CUDA

#include <cuda_runtime.h>

namespace lynx {
namespace gpu {

static constexpr int NL_TILE_FS = 256;
static constexpr int NL_MAX_NP_FS = 64;  // FR pseudopotentials can have up to ~62 projectors/atom

__device__ __forceinline__ double warpReduceSum_fs(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ inline double blockReduceSum_fs(double val, double* smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    val = warpReduceSum_fs(val);
    if (lane == 0) smem[warp] = val;
    __syncthreads();
    if (warp == 0) {
        val = (lane < nwarps) ? smem[lane] : 0.0;
        val = warpReduceSum_fs(val);
    }
    __syncthreads();
    return val;
}

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
