#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstring>

#include "operators/Laplacian.hpp"
#include "operators/Gradient.hpp"
#include "operators/Hamiltonian.hpp"
#include "parallel/HaloExchange.hpp"
#include "core/Domain.hpp"

namespace py = pybind11;

namespace pylynx {

// Halo-transparent Laplacian: accepts domain-sized numpy, returns domain-sized numpy
inline py::array_t<double> laplacian_apply_numpy(
    const lynx::Laplacian& lap,
    const lynx::HaloExchange& halo,
    py::array_t<double, py::array::c_style | py::array::forcecast> x_np,
    double a = -0.5, double c = 0.0)
{
    auto buf = x_np.request();
    int Nd_d = static_cast<int>(buf.shape[0]);
    int ncol = (buf.ndim > 1) ? static_cast<int>(buf.shape[1]) : 1;
    const double* x = static_cast<double*>(buf.ptr);

    // Allocate extended array with halos
    int nd_ex = halo.nd_ex();
    std::vector<double> x_ex(nd_ex * ncol, 0.0);

    // Halo exchange
    halo.execute(x, x_ex.data(), ncol);

    // Apply Laplacian
    auto result = py::array_t<double>({Nd_d * ncol});
    double* y = static_cast<double*>(result.request().ptr);
    lap.apply(x_ex.data(), y, a, c, ncol);

    if (ncol > 1) result.resize({Nd_d, ncol});
    return result;
}

// Halo-transparent Laplacian with diagonal potential
inline py::array_t<double> laplacian_apply_with_diag_numpy(
    const lynx::Laplacian& lap,
    const lynx::HaloExchange& halo,
    py::array_t<double, py::array::c_style | py::array::forcecast> x_np,
    py::array_t<double, py::array::c_style | py::array::forcecast> V_np,
    double a = -0.5, double b = 1.0, double c = 0.0)
{
    auto xbuf = x_np.request();
    auto vbuf = V_np.request();
    int Nd_d = static_cast<int>(xbuf.shape[0]);
    int ncol = (xbuf.ndim > 1) ? static_cast<int>(xbuf.shape[1]) : 1;
    const double* x = static_cast<double*>(xbuf.ptr);
    const double* V = static_cast<double*>(vbuf.ptr);

    int nd_ex = halo.nd_ex();
    std::vector<double> x_ex(nd_ex * ncol, 0.0);
    halo.execute(x, x_ex.data(), ncol);

    auto result = py::array_t<double>({Nd_d * ncol});
    double* y = static_cast<double*>(result.request().ptr);
    lap.apply_with_diag(x_ex.data(), V, y, a, b, c, ncol);

    if (ncol > 1) result.resize({Nd_d, ncol});
    return result;
}

// Halo-transparent Gradient
inline py::array_t<double> gradient_apply_numpy(
    const lynx::Gradient& grad,
    const lynx::HaloExchange& halo,
    py::array_t<double, py::array::c_style | py::array::forcecast> x_np,
    int direction)
{
    auto buf = x_np.request();
    int Nd_d = static_cast<int>(buf.shape[0]);
    int ncol = (buf.ndim > 1) ? static_cast<int>(buf.shape[1]) : 1;
    const double* x = static_cast<double*>(buf.ptr);

    int nd_ex = halo.nd_ex();
    std::vector<double> x_ex(nd_ex * ncol, 0.0);
    halo.execute(x, x_ex.data(), ncol);

    auto result = py::array_t<double>({Nd_d * ncol});
    double* y = static_cast<double*>(result.request().ptr);
    grad.apply(x_ex.data(), y, direction, ncol);

    if (ncol > 1) result.resize({Nd_d, ncol});
    return result;
}

// Halo-transparent Hamiltonian (local part only)
inline py::array_t<double> hamiltonian_apply_numpy(
    const lynx::Hamiltonian& H,
    const lynx::HaloExchange& halo,
    py::array_t<double, py::array::c_style | py::array::forcecast> psi_np,
    py::array_t<double, py::array::c_style | py::array::forcecast> Veff_np,
    double c_shift = 0.0)
{
    auto pbuf = psi_np.request();
    int Nd_d = static_cast<int>(pbuf.shape[0]);
    int ncol = (pbuf.ndim > 1) ? static_cast<int>(pbuf.shape[1]) : 1;

    auto result = py::array_t<double>({Nd_d * ncol});
    double* y = static_cast<double*>(result.request().ptr);

    const double* psi = static_cast<double*>(pbuf.ptr);
    const double* Veff = static_cast<double*>(Veff_np.request().ptr);

    H.apply(psi, Veff, y, ncol, c_shift);

    if (ncol > 1) result.resize({Nd_d, ncol});
    return result;
}

} // namespace pylynx
