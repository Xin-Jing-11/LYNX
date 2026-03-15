#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "solvers/EigenSolver.hpp"
#include "solvers/PoissonSolver.hpp"
#include "solvers/Mixer.hpp"
#include "solvers/LinearSolver.hpp"

namespace py = pybind11;

void bind_solvers(py::module_& m) {
    using namespace lynx;

    // AARParams
    py::class_<AARParams>(m, "AARParams")
        .def(py::init<>())
        .def_readwrite("omega", &AARParams::omega)
        .def_readwrite("beta", &AARParams::beta)
        .def_readwrite("m", &AARParams::m)
        .def_readwrite("p", &AARParams::p)
        .def_readwrite("tol", &AARParams::tol)
        .def_readwrite("max_iter", &AARParams::max_iter);

    // PoissonSolver
    py::class_<PoissonSolver>(m, "PoissonSolver")
        .def(py::init<>())
        .def("setup", &PoissonSolver::setup,
             py::arg("laplacian"), py::arg("stencil"),
             py::arg("domain"), py::arg("grid"), py::arg("halo"))
        .def("solve", [](const PoissonSolver& ps,
                         py::array_t<double, py::array::c_style | py::array::forcecast> rhs_np,
                         double tol) {
            auto buf = rhs_np.request();
            int N = static_cast<int>(buf.shape[0]);
            const double* rhs = static_cast<double*>(buf.ptr);
            auto phi = py::array_t<double>(N);
            double* phi_ptr = static_cast<double*>(phi.request().ptr);
            std::memset(phi_ptr, 0, N * sizeof(double));
            int iters = ps.solve(rhs, phi_ptr, tol);
            return py::make_tuple(phi, iters);
        }, py::arg("rhs"), py::arg("tol") = 1e-8,
           py::call_guard<py::gil_scoped_release>(),
           "Solve Poisson equation. Returns (phi, n_iterations).")
        .def("set_aar_params", &PoissonSolver::set_aar_params);

    // EigenSolver
    py::class_<EigenSolver>(m, "EigenSolver")
        .def(py::init<>())
        .def("setup", &EigenSolver::setup,
             py::arg("H"), py::arg("halo"), py::arg("domain"),
             py::arg("bandcomm"), py::arg("Nband_global") = 0)
        .def("lambda_cutoff", &EigenSolver::lambda_cutoff)
        .def("set_lambda_cutoff", &EigenSolver::set_lambda_cutoff)
        .def("Nband_global", &EigenSolver::Nband_global)
        .def("is_band_parallel", &EigenSolver::is_band_parallel);

    // Mixer
    py::class_<Mixer>(m, "Mixer")
        .def(py::init<>())
        .def("setup", &Mixer::setup,
             py::arg("Nd_d"), py::arg("var"), py::arg("precond_type"),
             py::arg("history_depth"), py::arg("mixing_param"),
             py::arg("laplacian") = nullptr, py::arg("halo") = nullptr,
             py::arg("grid") = nullptr)
        .def("reset", &Mixer::reset);
}
