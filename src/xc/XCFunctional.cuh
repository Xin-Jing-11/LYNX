#pragma once
#ifdef USE_CUDA

#include <cuda_runtime.h>

namespace lynx {
namespace gpu {

// GGA pipeline utility kernels (used by evaluate_gpu/evaluate_spin_gpu)
// XC evaluation itself is handled by libxc CUDA device calls.

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
