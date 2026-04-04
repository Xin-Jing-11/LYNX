// DeviceArray.cu — GPU helper functions for DeviceArray<T>.
// These are non-template wrappers around CUDA runtime calls,
// callable from the template methods in DeviceArray.hpp via #ifdef USE_CUDA.

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace lynx::detail {

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

void* gpu_alloc(size_t bytes) {
    void* ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, bytes), "DeviceArray gpu_alloc");
    return ptr;
}

void gpu_free(void* ptr) {
    // cudaFree(nullptr) is a no-op, but guard anyway
    if (ptr) cudaFree(ptr);
}

void gpu_memset(void* ptr, int val, size_t bytes) {
    check_cuda(cudaMemset(ptr, val, bytes), "DeviceArray gpu_memset");
}

void gpu_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    check_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice),
               "DeviceArray gpu_memcpy_h2d");
}

void gpu_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    check_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost),
               "DeviceArray gpu_memcpy_d2h");
}

void gpu_memcpy_d2d(void* dst, const void* src, size_t bytes) {
    check_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice),
               "DeviceArray gpu_memcpy_d2d");
}

}  // namespace lynx::detail
