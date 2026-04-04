#pragma once

#include "core/DeviceTag.hpp"
#include "core/types.hpp"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef USE_CUDA
// Forward-declare GPU helper functions (implemented in DeviceArray.cu)
namespace lynx::detail {
void* gpu_alloc(size_t bytes);
void  gpu_free(void* ptr);
void  gpu_memset(void* ptr, int val, size_t bytes);
void  gpu_memcpy_h2d(void* dst, const void* src, size_t bytes);
void  gpu_memcpy_d2h(void* dst, const void* src, size_t bytes);
void  gpu_memcpy_d2d(void* dst, const void* src, size_t bytes);
void* pinned_alloc(size_t bytes);
void  pinned_free(void* ptr);
}  // namespace lynx::detail
#endif

namespace lynx {

template<typename T>
class DeviceArray {
public:
    DeviceArray() = default;

    /// Allocate 1D array of n elements on the given device.
    explicit DeviceArray(int n, Device dev = Device::CPU)
        : rows_(n), cols_(1), ld_(n), size_(n), dev_(dev)
    {
        alloc(static_cast<size_t>(size_) * sizeof(T));
        zero();
    }

    /// Allocate 2D array (rows x cols) with padded leading dimension.
    DeviceArray(int rows, int cols, Device dev = Device::CPU)
        : rows_(rows), cols_(cols), ld_(pad_ld(rows)),
          size_(pad_ld(rows) * cols), dev_(dev)
    {
        alloc(static_cast<size_t>(size_) * sizeof(T));
        zero();
    }

    ~DeviceArray() { free_memory(); }

    // Move-only
    DeviceArray(DeviceArray&& other) noexcept { steal(other); }
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            free_memory();
            steal(other);
        }
        return *this;
    }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    /// Raw pointer (host or device depending on device()).
    T* data() { return data_; }
    const T* data() const { return data_; }

    /// Where the data lives.
    Device device() const { return dev_; }
    bool on_gpu() const { return dev_ == Device::GPU; }
    bool on_cpu() const { return dev_ == Device::CPU || dev_ == Device::CPU_PINNED; }
    bool is_pinned() const { return dev_ == Device::CPU_PINNED; }

    /// Element access (CPU only — no bounds checking).
    T& operator()(int i) { return data_[i]; }
    const T& operator()(int i) const { return data_[i]; }
    T& operator()(int i, int j) { return data_[i + j * ld_]; }
    const T& operator()(int i, int j) const { return data_[i + j * ld_]; }

    /// Dimensions.
    int size() const { return size_; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int ld() const { return ld_; }
    bool empty() const { return data_ == nullptr; }

    /// Column pointer (column-major: col j starts at data + j * ld).
    T* col(int j) { return data_ + j * ld_; }
    const T* col(int j) const { return data_ + j * ld_; }

    /// Create a copy on the target device.
    DeviceArray<T> to(Device target) const {
        if (empty()) return DeviceArray<T>();

        DeviceArray<T> dst;
        dst.rows_ = rows_;
        dst.cols_ = cols_;
        dst.ld_   = ld_;
        dst.size_ = size_;
        dst.dev_  = target;
        dst.alloc(byte_size());

        bool src_host = on_cpu();   // CPU or CPU_PINNED
        bool dst_host = (target == Device::CPU || target == Device::CPU_PINNED);

        if (src_host && dst_host) {
            std::memcpy(dst.data_, data_, byte_size());
        }
#ifdef USE_CUDA
        else if (src_host && !dst_host) {
            detail::gpu_memcpy_h2d(dst.data_, data_, byte_size());
        }
        else if (!src_host && dst_host) {
            detail::gpu_memcpy_d2h(dst.data_, data_, byte_size());
        }
        else {
            detail::gpu_memcpy_d2d(dst.data_, data_, byte_size());
        }
#else
        else {
            throw std::runtime_error("DeviceArray: GPU operations require CUDA build");
        }
#endif
        return dst;
    }

    /// Copy data from src (must be same size; can be cross-device).
    void copy_from(const DeviceArray<T>& src) {
        if (src.size_ != size_)
            throw std::invalid_argument("DeviceArray::copy_from: size mismatch");
        if (src.empty() || empty()) return;

        bool src_host = src.on_cpu();   // CPU or CPU_PINNED
        bool dst_host = on_cpu();

        if (src_host && dst_host) {
            std::memcpy(data_, src.data_, byte_size());
        }
#ifdef USE_CUDA
        else if (src_host && !dst_host) {
            detail::gpu_memcpy_h2d(data_, src.data_, byte_size());
        }
        else if (!src_host && dst_host) {
            detail::gpu_memcpy_d2h(data_, src.data_, byte_size());
        }
        else {
            detail::gpu_memcpy_d2d(data_, src.data_, byte_size());
        }
#else
        else {
            throw std::runtime_error("DeviceArray: GPU operations require CUDA build");
        }
#endif
    }

    /// Fill with zeros.
    void zero() {
        if (!data_) return;
        if (on_cpu()) {
            std::memset(data_, 0, byte_size());
        }
#ifdef USE_CUDA
        else {
            detail::gpu_memset(data_, 0, byte_size());
        }
#endif
    }

    /// Fill with value (CPU only).
    void fill(T val) {
        if (!data_) return;
        if (!on_cpu())
            throw std::runtime_error("DeviceArray::fill: CPU/CPU_PINNED only");
        for (int i = 0; i < size_; ++i)
            data_[i] = val;
    }

    /// Deep copy (same device).
    DeviceArray<T> clone() const {
        return to(dev_);
    }

    /// Resize to 1D array of n elements (frees old data, allocates new, zeros).
    /// Preserves the current device (CPU or GPU).
    void resize(int n) {
        Device d = dev_;  // preserve current device
        free_memory();
        rows_ = n; cols_ = 1; ld_ = n; size_ = n;
        dev_ = d;
        alloc(static_cast<size_t>(size_) * sizeof(T));
        zero();
    }

    /// Resize to 2D array (rows x cols) with padded LD (frees old data, allocates new, zeros).
    /// Preserves the current device (CPU or GPU).
    void resize(int rows, int cols) {
        Device d = dev_;  // preserve current device
        free_memory();
        rows_ = rows; cols_ = cols; ld_ = pad_ld(rows); size_ = ld_ * cols;
        dev_ = d;
        alloc(static_cast<size_t>(size_) * sizeof(T));
        zero();
    }

    /// Convenience alias for resize.
    void allocate(int n) { resize(n); }
    void allocate(int rows, int cols) { resize(rows, cols); }

private:
    T* data_ = nullptr;
    int rows_ = 0;
    int cols_ = 0;
    int ld_ = 0;
    int size_ = 0;      // total allocated elements = ld * cols (2D) or n (1D)
    Device dev_ = Device::CPU;

    size_t byte_size() const { return static_cast<size_t>(size_) * sizeof(T); }

    /// Pad leading dimension to 8-element boundary (64-byte aligned for doubles).
    static int pad_ld(int rows) {
        constexpr int align_elems = 8;
        return (rows + align_elems - 1) / align_elems * align_elems;
    }

    void alloc(size_t bytes) {
        if (bytes == 0) return;
        if (dev_ == Device::CPU) {
            void* ptr = nullptr;
#if defined(_WIN32)
            ptr = _aligned_malloc(bytes, MEMORY_ALIGNMENT);
#else
            if (posix_memalign(&ptr, MEMORY_ALIGNMENT, bytes) != 0)
                ptr = nullptr;
#endif
            if (!ptr) throw std::bad_alloc();
            data_ = static_cast<T*>(ptr);
        }
#ifdef USE_CUDA
        else if (dev_ == Device::CPU_PINNED) {
            data_ = static_cast<T*>(detail::pinned_alloc(bytes));
        }
        else if (dev_ == Device::GPU) {
            data_ = static_cast<T*>(detail::gpu_alloc(bytes));
        }
#else
        else {
            throw std::runtime_error("DeviceArray: GPU/pinned allocation requires CUDA build");
        }
#endif
    }

    void free_memory() {
        if (!data_) return;
        if (dev_ == Device::CPU) {
#if defined(_WIN32)
            _aligned_free(data_);
#else
            std::free(data_);
#endif
        }
#ifdef USE_CUDA
        else if (dev_ == Device::CPU_PINNED) {
            detail::pinned_free(data_);
        }
        else if (dev_ == Device::GPU) {
            detail::gpu_free(data_);
        }
#endif
        data_ = nullptr;
        size_ = 0;
    }

    void steal(DeviceArray& other) {
        data_ = other.data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        ld_   = other.ld_;
        size_ = other.size_;
        dev_  = other.dev_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
};

}  // namespace lynx
