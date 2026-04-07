#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>
#include "solvers/EigenSolver.hpp"
#include "solvers/PoissonSolver.hpp"
#include "solvers/Mixer.hpp"
#include "solvers/LinearSolver.hpp"
#include "solvers/Preconditioner.hpp"
#include "core/LynxContext.hpp"
#include "parallel/MPIComm.hpp"

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
            int iters;
            {
                py::gil_scoped_release release;
                iters = ps.solve(rhs, phi_ptr, tol);
            }
            return py::make_tuple(phi, iters);
        }, py::arg("rhs"), py::arg("tol") = 1e-8,
           "Solve Poisson equation. Returns (phi, n_iterations).")
        .def("set_aar_params", &PoissonSolver::set_aar_params);

    // EigenSolver — new API: setup(LynxContext&, Hamiltonian&)
    py::class_<EigenSolver>(m, "EigenSolver")
        .def(py::init<>())
        .def("setup", &EigenSolver::setup,
             py::arg("ctx"), py::arg("H"),
             "Set up EigenSolver using LynxContext for all infrastructure.")
        .def("solve", [](EigenSolver& es,
                         py::array_t<double, py::array::f_style | py::array::forcecast> psi_np,
                         py::array_t<double, py::array::c_style | py::array::forcecast> Veff_np,
                         double lambda_cutoff,
                         double eigval_min, double eigval_max,
                         int cheb_degree) {
            auto pbuf = psi_np.request();
            if (pbuf.ndim != 2)
                throw std::runtime_error("psi must be 2D (Nd_d, Nband)");
            int Nd_d = static_cast<int>(pbuf.shape[0]);
            int Nband = static_cast<int>(pbuf.shape[1]);

            // Copy psi in Fortran order (C++ expects column-major)
            std::vector<ssize_t> shape = {Nd_d, Nband};
            std::vector<ssize_t> strides = {
                static_cast<ssize_t>(sizeof(double)),
                static_cast<ssize_t>(Nd_d * sizeof(double))
            };
            auto psi_out = py::array_t<double>(shape, strides);
            double* psi_ptr = static_cast<double*>(psi_out.request().ptr);
            std::memcpy(psi_ptr, pbuf.ptr, Nd_d * Nband * sizeof(double));

            auto eigvals = py::array_t<double>(Nband);
            double* eig_ptr = static_cast<double*>(eigvals.request().ptr);

            const double* Veff = static_cast<double*>(Veff_np.request().ptr);

            {
                py::gil_scoped_release release;
                es.solve(psi_ptr, eig_ptr, Veff, Nd_d, Nband,
                         lambda_cutoff, eigval_min, eigval_max, cheb_degree);
            }

            return py::make_tuple(psi_out, eigvals);
        }, py::arg("psi"), py::arg("Veff"),
           py::arg("lambda_cutoff"), py::arg("eigval_min"), py::arg("eigval_max"),
           py::arg("cheb_degree") = 20,
           "Solve eigenvalue problem via CheFSI. Returns (psi_updated, eigenvalues).")
        .def("lanczos_bounds", [](EigenSolver& es,
                                  py::array_t<double, py::array::c_style | py::array::forcecast> Veff_np,
                                  double tol, int max_iter) {
            auto buf = Veff_np.request();
            int Nd_d = static_cast<int>(buf.shape[0]);
            const double* Veff = static_cast<double*>(buf.ptr);
            double eigmin = 0.0, eigmax = 0.0;
            {
                py::gil_scoped_release release;
                es.lanczos_bounds(Veff, Nd_d, eigmin, eigmax, tol, max_iter);
            }
            return py::make_tuple(eigmin, eigmax);
        }, py::arg("Veff"), py::arg("tol") = 1e-2, py::arg("max_iter") = 1000,
           "Estimate spectral bounds via Lanczos. Returns (eigval_min, eigval_max).")
        .def("lambda_cutoff", &EigenSolver::lambda_cutoff)
        .def("set_lambda_cutoff", &EigenSolver::set_lambda_cutoff)
        .def("Nband_global", &EigenSolver::Nband_global)
        .def("is_band_parallel", &EigenSolver::is_band_parallel);

    // Mixer — new API: setup(Nd_d, var, precond, history, beta, Preconditioner*)
    py::class_<Mixer>(m, "Mixer")
        .def(py::init<>())
        .def("setup", &Mixer::setup,
             py::arg("Nd_d"), py::arg("var"), py::arg("precond_type"),
             py::arg("history_depth"), py::arg("mixing_param"),
             py::arg("preconditioner") = nullptr,
             "Set up density/potential mixer.")
        .def("reset", &Mixer::reset);
}
