#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "core/NDArray.hpp"

namespace py = pybind11;

namespace pylynx {

// NDArray<double> -> numpy (zero-copy via buffer protocol when possible)
inline py::array_t<double> to_numpy_1d(const lynx::NDArray<double>& arr) {
    // 1D: contiguous, just wrap
    return py::array_t<double>(
        {arr.size()},
        {sizeof(double)},
        arr.data(),
        py::none()  // no owner — user must keep NDArray alive
    );
}

inline py::array_t<double> to_numpy_2d(const lynx::NDArray<double>& arr) {
    // 2D column-major with padded ld
    // Shape: (rows, cols), strides: (sizeof(double), ld * sizeof(double))
    return py::array_t<double>(
        {arr.rows(), arr.cols()},
        {static_cast<ssize_t>(sizeof(double)),
         static_cast<ssize_t>(arr.ld() * sizeof(double))},
        arr.data(),
        py::none()
    );
}

inline py::array_t<double> to_numpy(const lynx::NDArray<double>& arr) {
    if (arr.ndim() <= 1) return to_numpy_1d(arr);
    return to_numpy_2d(arr);
}

// Copy numpy array into a new NDArray<double>
inline lynx::NDArray<double> from_numpy_1d(py::array_t<double> a) {
    auto buf = a.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Expected 1D array");
    int n = static_cast<int>(buf.shape[0]);
    lynx::NDArray<double> arr(n);
    auto* src = static_cast<double*>(buf.ptr);
    std::memcpy(arr.data(), src, n * sizeof(double));
    return arr;
}

// Copy flat numpy array into caller-provided buffer
inline void numpy_to_buffer(py::array_t<double, py::array::c_style | py::array::forcecast> a,
                            double* dst, int expected_size) {
    auto buf = a.request();
    int total = 1;
    for (int d = 0; d < buf.ndim; ++d) total *= static_cast<int>(buf.shape[d]);
    if (total != expected_size)
        throw std::runtime_error("Array size mismatch: got " + std::to_string(total)
                                 + ", expected " + std::to_string(expected_size));
    std::memcpy(dst, buf.ptr, total * sizeof(double));
}

} // namespace pylynx
