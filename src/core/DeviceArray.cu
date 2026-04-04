// DeviceArray.cu — GPU helper functions for DeviceArray<T>.
// These are non-template wrappers around CUDA runtime calls,
// callable from the template methods in DeviceArray.hpp via #ifdef USE_CUDA.
//
// All GPU operations use stream-ordered async APIs where possible.
// The stream is obtained from GPUContext (falls back to default stream 0).

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "core/GPUContext.cuh"

namespace lynx::detail {

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// Get the compute stream from GPUContext.
// Falls back to GPUContext's static fallback instance which creates a real stream.
static cudaStream_t get_stream() {
    return gpu::GPUContext::instance().compute_stream;
}

void* gpu_alloc(size_t bytes) {
    void* ptr = nullptr;
    cudaStream_t stream = get_stream();
    check_cuda(cudaMallocAsync(&ptr, bytes, stream), "DeviceArray gpu_alloc");
    return ptr;
}

void gpu_free(void* ptr) {
    if (ptr) {
        cudaStream_t stream = get_stream();
        cudaFreeAsync(ptr, stream);  // no error check needed
    }
}

void gpu_memset(void* ptr, int val, size_t bytes) {
    cudaStream_t stream = get_stream();
    check_cuda(cudaMemsetAsync(ptr, val, bytes, stream), "DeviceArray gpu_memset");
}

void gpu_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    cudaStream_t stream = get_stream();
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream),
               "DeviceArray gpu_memcpy_h2d");
}

void gpu_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    cudaStream_t stream = get_stream();
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream),
               "DeviceArray gpu_memcpy_d2h");
    // D2H needs sync so CPU can read the data immediately
    cudaStreamSynchronize(stream);
}

void gpu_memcpy_d2d(void* dst, const void* src, size_t bytes) {
    cudaStream_t stream = get_stream();
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream),
               "DeviceArray gpu_memcpy_d2d");
}

void* pinned_alloc(size_t bytes) {
    void* ptr = nullptr;
    check_cuda(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault), "DeviceArray pinned_alloc");
    return ptr;
}

void pinned_free(void* ptr) {
    if (ptr) cudaFreeHost(ptr);
}

}  // namespace lynx::detail
