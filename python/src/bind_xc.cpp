#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "xc/XCFunctional.hpp"

namespace py = pybind11;

void bind_xc(py::module_& m) {
    using namespace lynx;

    py::class_<XCFunctional>(m, "XCFunctional")
        .def(py::init<>())
        .def("setup", &XCFunctional::setup,
             py::arg("type"), py::arg("domain"), py::arg("grid"),
             py::arg("gradient") = nullptr, py::arg("halo") = nullptr)
        .def("type", &XCFunctional::type)
        .def("is_gga", &XCFunctional::is_gga)
        .def("evaluate", [](const XCFunctional& xc,
                            py::array_t<double, py::array::c_style | py::array::forcecast> rho_np,
                            int Nd_d) {
            auto rbuf = rho_np.request();
            const double* rho = static_cast<double*>(rbuf.ptr);

            auto Vxc = py::array_t<double>(Nd_d);
            auto exc = py::array_t<double>(Nd_d);
            double* Vxc_ptr = static_cast<double*>(Vxc.request().ptr);
            double* exc_ptr = static_cast<double*>(exc.request().ptr);

            xc.evaluate(rho, Vxc_ptr, exc_ptr, Nd_d);

            return py::make_tuple(Vxc, exc);
        }, py::arg("rho"), py::arg("Nd_d"),
           py::call_guard<py::gil_scoped_release>(),
           "Evaluate XC potential and energy density. Returns (Vxc, exc).");
}
