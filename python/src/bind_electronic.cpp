#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "electronic/Wavefunction.hpp"
#include "electronic/ElectronDensity.hpp"
#include "electronic/Occupation.hpp"
#include "ArrayBridge.hpp"

namespace py = pybind11;

void bind_electronic(py::module_& m) {
    using namespace lynx;

    // Wavefunction (non-copyable due to move-only NDArray members)
    py::class_<Wavefunction>(m, "Wavefunction")
        .def(py::init<>())
        .def("allocate", py::overload_cast<int, int, int, int, bool, int>(
                 &Wavefunction::allocate),
             py::arg("Nd_d"), py::arg("Nband"), py::arg("Nspin"),
             py::arg("Nkpts"), py::arg("is_complex") = false,
             py::arg("Nspinor") = 1)
        .def("Nd_d", &Wavefunction::Nd_d)
        .def("Nband", &Wavefunction::Nband)
        .def("Nband_global", &Wavefunction::Nband_global)
        .def("Nspin", &Wavefunction::Nspin)
        .def("Nkpts", &Wavefunction::Nkpts)
        .def("is_complex", &Wavefunction::is_complex)
        .def("randomize", &Wavefunction::randomize,
             py::arg("spin"), py::arg("kpt"), py::arg("seed") = 42)
        // Access eigenvalues as numpy
        .def("eigenvalues", [](Wavefunction& wfn, int spin, int kpt) {
            return pylynx::to_numpy(wfn.eigenvalues(spin, kpt));
        }, py::arg("spin") = 0, py::arg("kpt") = 0,
           py::return_value_policy::reference_internal)
        // Access occupations as numpy
        .def("occupations", [](Wavefunction& wfn, int spin, int kpt) {
            return pylynx::to_numpy(wfn.occupations(spin, kpt));
        }, py::arg("spin") = 0, py::arg("kpt") = 0,
           py::return_value_policy::reference_internal)
        // Access psi as numpy (2D: Nd_d x Nband)
        .def("psi", [](Wavefunction& wfn, int spin, int kpt) {
            return pylynx::to_numpy(wfn.psi(spin, kpt));
        }, py::arg("spin") = 0, py::arg("kpt") = 0,
           py::return_value_policy::reference_internal);

    // ElectronDensity
    py::class_<ElectronDensity>(m, "ElectronDensity")
        .def(py::init<>())
        .def("allocate", &ElectronDensity::allocate,
             py::arg("Nd_d"), py::arg("Nspin"))
        .def("Nd_d", &ElectronDensity::Nd_d)
        .def("Nspin", &ElectronDensity::Nspin)
        .def("integrate", &ElectronDensity::integrate, py::arg("dV"))
        .def("rho_total", [](ElectronDensity& ed) {
            return pylynx::to_numpy(ed.rho_total());
        }, py::return_value_policy::reference_internal)
        .def("rho", [](ElectronDensity& ed, int spin) {
            return pylynx::to_numpy(ed.rho(spin));
        }, py::arg("spin"), py::return_value_policy::reference_internal)
        .def("mag", [](ElectronDensity& ed) {
            return pylynx::to_numpy(ed.mag());
        }, py::return_value_policy::reference_internal);

    // Occupation
    py::class_<Occupation>(m, "Occupation")
        .def_static("fermi_dirac", &Occupation::fermi_dirac)
        .def_static("gaussian_smearing", &Occupation::gaussian_smearing);
}
