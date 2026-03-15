#pragma once

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include "types.hpp"

namespace lynx {

template<typename T>
class NDArray {
public:
    NDArray() = default;

    explicit NDArray(int n) {
        init(n, 1, 1, 1);
    }

    NDArray(int rows, int cols) {
        init(rows, cols, 1, 2);
    }

    NDArray(int d0, int d1, int d2) {
        init(d0, d1, d2, 3);
    }

    ~NDArray() { deallocate(); }

    // Move only
    NDArray(NDArray&& other) noexcept { steal(other); }
    NDArray& operator=(NDArray&& other) noexcept {
        if (this != &other) {
            deallocate();
            steal(other);
        }
        return *this;
    }
    NDArray(const NDArray&) = delete;
    NDArray& operator=(const NDArray&) = delete;

    // Explicit copy
    NDArray clone() const {
        NDArray copy;
        copy.ndim_ = ndim_;
        copy.size_ = size_;
        copy.ld_ = ld_;
        for (int i = 0; i < 3; ++i) copy.dims_[i] = dims_[i];
        if (size_ > 0) {
            copy.allocate(ld_ * (ndim_ >= 2 ? dims_[1] : 1) * (ndim_ >= 3 ? dims_[2] : 1));
            std::memcpy(copy.data_, data_, sizeof(T) * alloc_size_);
            copy.alloc_size_ = alloc_size_;
        }
        return copy;
    }

    // 1D access
    T& operator()(int i) { return data_[i]; }
    const T& operator()(int i) const { return data_[i]; }

    // 2D access (column-major)
    T& operator()(int i, int j) { return data_[i + j * ld_]; }
    const T& operator()(int i, int j) const { return data_[i + j * ld_]; }

    // 3D access
    T& operator()(int i, int j, int k) {
        return data_[i + j * ld_ + k * ld_ * dims_[1]];
    }
    const T& operator()(int i, int j, int k) const {
        return data_[i + j * ld_ + k * ld_ * dims_[1]];
    }

    // Column access
    T* col(int j) { return data_ + j * ld_; }
    const T* col(int j) const { return data_ + j * ld_; }

    T* data() { return data_; }
    const T* data() const { return data_; }

    int size() const { return size_; }
    int rows() const { return dims_[0]; }
    int cols() const { return ndim_ >= 2 ? dims_[1] : 1; }
    int depth() const { return ndim_ >= 3 ? dims_[2] : 1; }
    int ld() const { return ld_; }
    int ndim() const { return ndim_; }

    void fill(T val) {
        for (int i = 0; i < alloc_size_; ++i)
            data_[i] = val;
    }

    void zero() {
        if (data_) std::memset(data_, 0, sizeof(T) * alloc_size_);
    }

    void resize(int n) {
        deallocate();
        init(n, 1, 1, 1);
    }

    void resize(int rows, int cols) {
        deallocate();
        init(rows, cols, 1, 2);
    }

    void resize(int d0, int d1, int d2) {
        deallocate();
        init(d0, d1, d2, 3);
    }

    bool empty() const { return data_ == nullptr; }

    // Check alignment
    bool is_aligned() const {
        return data_ != nullptr &&
               (reinterpret_cast<std::uintptr_t>(data_) % MEMORY_ALIGNMENT) == 0;
    }

private:
    T* data_ = nullptr;
    int dims_[3] = {0, 0, 0};
    int ndim_ = 0;
    int size_ = 0;
    int ld_ = 0;
    int alloc_size_ = 0;

    // Pad leading dimension to multiple of 8 elements (64 bytes for double, 128 for Complex)
    static int pad_ld(int rows) {
        constexpr int align_elems = 8;
        return (rows + align_elems - 1) / align_elems * align_elems;
    }

    void init(int d0, int d1, int d2, int ndim) {
        if (d0 <= 0) throw std::invalid_argument("NDArray dimension must be positive");
        ndim_ = ndim;
        dims_[0] = d0;
        dims_[1] = d1;
        dims_[2] = d2;
        size_ = d0 * d1 * d2;
        ld_ = (ndim >= 2) ? pad_ld(d0) : d0;
        alloc_size_ = ld_ * d1 * d2;
        allocate(alloc_size_);
        zero();
    }

    void allocate(int total) {
        alloc_size_ = total;
        void* ptr = nullptr;
#if defined(_WIN32)
        ptr = _aligned_malloc(sizeof(T) * total, MEMORY_ALIGNMENT);
#else
        if (posix_memalign(&ptr, MEMORY_ALIGNMENT, sizeof(T) * total) != 0)
            ptr = nullptr;
#endif
        if (!ptr) throw std::bad_alloc();
        data_ = static_cast<T*>(ptr);
    }

    void deallocate() {
        if (data_) {
#if defined(_WIN32)
            _aligned_free(data_);
#else
            std::free(data_);
#endif
            data_ = nullptr;
        }
        size_ = 0;
        alloc_size_ = 0;
    }

    void steal(NDArray& other) {
        data_ = other.data_;
        ndim_ = other.ndim_;
        size_ = other.size_;
        ld_ = other.ld_;
        alloc_size_ = other.alloc_size_;
        for (int i = 0; i < 3; ++i) dims_[i] = other.dims_[i];
        other.data_ = nullptr;
        other.size_ = 0;
        other.alloc_size_ = 0;
    }
};

} // namespace lynx
