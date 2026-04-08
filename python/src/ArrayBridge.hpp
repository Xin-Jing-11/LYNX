#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include "core/DeviceArray.hpp"

namespace py = pybind11;

namespace pylynx {

// ---------------------------------------------------------------------------
// double overloads
// ---------------------------------------------------------------------------

// DeviceArray<double> -> numpy 1D (zero-copy via buffer protocol)
inline py::array_t<double> to_numpy_1d(const lynx::DeviceArray<double>& arr) {
    if (arr.on_gpu())
        throw std::runtime_error("to_numpy_1d: cannot wrap GPU data in numpy array; copy to CPU first");
    return py::array_t<double>(
        {arr.size()},
        {sizeof(double)},
        arr.data(),
        py::none()  // no owner -- user must keep DeviceArray alive
    );
}

// DeviceArray<double> -> numpy 2D column-major with padded ld
inline py::array_t<double> to_numpy_2d(const lynx::DeviceArray<double>& arr) {
    if (arr.on_gpu())
        throw std::runtime_error("to_numpy_2d: cannot wrap GPU data in numpy array; copy to CPU first");
    // Shape: (rows, cols), strides: (sizeof(double), ld * sizeof(double))
    return py::array_t<double>(
        {arr.rows(), arr.cols()},
        {static_cast<ssize_t>(sizeof(double)),
         static_cast<ssize_t>(arr.ld() * sizeof(double))},
        arr.data(),
        py::none()
    );
}

// Auto-detect 1D vs 2D based on cols
inline py::array_t<double> to_numpy(const lynx::DeviceArray<double>& arr) {
    if (arr.on_gpu())
        throw std::runtime_error("to_numpy: cannot wrap GPU data in numpy array; copy to CPU first");
    if (arr.cols() == 1) return to_numpy_1d(arr);
    return to_numpy_2d(arr);
}

// Copy numpy array into a new DeviceArray<double>
inline lynx::DeviceArray<double> from_numpy_1d(py::array_t<double> a) {
    auto buf = a.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Expected 1D array");
    int n = static_cast<int>(buf.shape[0]);
    lynx::DeviceArray<double> arr(n);
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

// ---------------------------------------------------------------------------
// std::complex<double> overloads (k-point support)
// ---------------------------------------------------------------------------

using cdouble = std::complex<double>;

// DeviceArray<cdouble> -> numpy 1D
inline py::array_t<cdouble> to_numpy_1d(const lynx::DeviceArray<cdouble>& arr) {
    if (arr.on_gpu())
        throw std::runtime_error("to_numpy_1d (complex): cannot wrap GPU data in numpy array; copy to CPU first");
    return py::array_t<cdouble>(
        {arr.size()},
        {sizeof(cdouble)},
        arr.data(),
        py::none()
    );
}

// DeviceArray<cdouble> -> numpy 2D column-major with padded ld
inline py::array_t<cdouble> to_numpy_2d(const lynx::DeviceArray<cdouble>& arr) {
    if (arr.on_gpu())
        throw std::runtime_error("to_numpy_2d (complex): cannot wrap GPU data in numpy array; copy to CPU first");
    return py::array_t<cdouble>(
        {arr.rows(), arr.cols()},
        {static_cast<ssize_t>(sizeof(cdouble)),
         static_cast<ssize_t>(arr.ld() * sizeof(cdouble))},
        arr.data(),
        py::none()
    );
}

// Auto-detect 1D vs 2D based on cols
inline py::array_t<cdouble> to_numpy(const lynx::DeviceArray<cdouble>& arr) {
    if (arr.on_gpu())
        throw std::runtime_error("to_numpy (complex): cannot wrap GPU data in numpy array; copy to CPU first");
    if (arr.cols() == 1) return to_numpy_1d(arr);
    return to_numpy_2d(arr);
}

// Copy numpy array into a new DeviceArray<cdouble>
inline lynx::DeviceArray<cdouble> from_numpy_1d(py::array_t<cdouble> a) {
    auto buf = a.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Expected 1D complex array");
    int n = static_cast<int>(buf.shape[0]);
    lynx::DeviceArray<cdouble> arr(n);
    auto* src = static_cast<cdouble*>(buf.ptr);
    std::memcpy(arr.data(), src, n * sizeof(cdouble));
    return arr;
}

} // namespace pylynx
