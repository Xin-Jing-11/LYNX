#pragma once
#ifdef USE_CUDA

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include "gpu_common.cuh"

namespace sparc {
namespace gpu {

// RAII singleton managing cuBLAS/cuSOLVER handles and a default stream.
struct GPUContext {
    cublasHandle_t cublas = nullptr;
    cusolverDnHandle_t cusolver = nullptr;
    cudaStream_t stream = nullptr;

    GPUContext() {
        CUDA_CHECK(cudaStreamCreate(&stream));
        cublasCreate(&cublas);
        cublasSetStream(cublas, stream);
        cusolverDnCreate(&cusolver);
        cusolverDnSetStream(cusolver, stream);
    }

    ~GPUContext() {
        if (cusolver) cusolverDnDestroy(cusolver);
        if (cublas) cublasDestroy(cublas);
        if (stream) cudaStreamDestroy(stream);
    }

    GPUContext(const GPUContext&) = delete;
    GPUContext& operator=(const GPUContext&) = delete;

    static GPUContext& instance() {
        static GPUContext ctx;
        return ctx;
    }
};

} // namespace gpu
} // namespace sparc

#endif // USE_CUDA
