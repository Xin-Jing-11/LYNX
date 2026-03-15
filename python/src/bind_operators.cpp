#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "operators/Laplacian.hpp"
#include "operators/Gradient.hpp"
#include "operators/Hamiltonian.hpp"
#include "operators/NonlocalProjector.hpp"
#include "OperatorWrappers.hpp"

namespace py = pybind11;

void bind_operators(py::module_& m) {
    using namespace lynx;

    // Laplacian
    py::class_<Laplacian>(m, "Laplacian")
        .def(py::init<>())
        .def(py::init<const FDStencil&, const Domain&>(),
             py::arg("stencil"), py::arg("domain"))
        .def("apply", [](const Laplacian& lap,
                         const HaloExchange& halo,
                         py::array_t<double, py::array::c_style | py::array::forcecast> x,
                         double a, double c) {
            return pylynx::laplacian_apply_numpy(lap, halo, x, a, c);
        }, py::arg("halo"), py::arg("x"),
           py::arg("a") = -0.5, py::arg("c") = 0.0,
           "Apply Laplacian: y = a*Lap(x) + c*x. Handles halo exchange transparently.")
        .def("apply_with_diag", [](const Laplacian& lap,
                                    const HaloExchange& halo,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> x,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> V,
                                    double a, double b, double c) {
            return pylynx::laplacian_apply_with_diag_numpy(lap, halo, x, V, a, b, c);
        }, py::arg("halo"), py::arg("x"), py::arg("V"),
           py::arg("a") = -0.5, py::arg("b") = 1.0, py::arg("c") = 0.0,
           "Apply y = a*Lap(x) + b*V*x + c*x with transparent halo exchange.");

    // Gradient
    py::class_<Gradient>(m, "Gradient")
        .def(py::init<>())
        .def(py::init<const FDStencil&, const Domain&>(),
             py::arg("stencil"), py::arg("domain"))
        .def("apply", [](const Gradient& grad,
                         const HaloExchange& halo,
                         py::array_t<double, py::array::c_style | py::array::forcecast> x,
                         int direction) {
            return pylynx::gradient_apply_numpy(grad, halo, x, direction);
        }, py::arg("halo"), py::arg("x"), py::arg("direction"),
           "Apply gradient in direction (0=x, 1=y, 2=z) with transparent halo exchange.");

    // Hamiltonian
    py::class_<Hamiltonian>(m, "Hamiltonian")
        .def(py::init<>())
        .def("setup", [](Hamiltonian& H,
                         const FDStencil& stencil,
                         const Domain& domain,
                         const FDGrid& grid,
                         const HaloExchange& halo,
                         const NonlocalProjector* vnl) {
            H.setup(stencil, domain, grid, halo, vnl);
        }, py::arg("stencil"), py::arg("domain"), py::arg("grid"),
           py::arg("halo"), py::arg("vnl") = nullptr,
           "Set up Hamiltonian with stencil, domain, grid, halo exchange, and nonlocal projectors.")
        .def("apply", [](const Hamiltonian& H,
                         const HaloExchange& halo,
                         py::array_t<double, py::array::c_style | py::array::forcecast> psi,
                         py::array_t<double, py::array::c_style | py::array::forcecast> Veff,
                         double c) {
            return pylynx::hamiltonian_apply_numpy(H, halo, psi, Veff, c);
        }, py::arg("halo"), py::arg("psi"), py::arg("Veff"),
           py::arg("c") = 0.0,
           "Apply H*psi = -0.5*Lap*psi + Veff*psi + Vnl*psi");

    // NonlocalProjector
    py::class_<NonlocalProjector>(m, "NonlocalProjector")
        .def(py::init<>())
        .def("setup", &NonlocalProjector::setup)
        .def("is_setup", &NonlocalProjector::is_setup)
        .def("total_nproj", &NonlocalProjector::total_nproj);
}
