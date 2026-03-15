#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "atoms/Crystal.hpp"
#include "atoms/AtomType.hpp"
#include "atoms/Pseudopotential.hpp"

namespace py = pybind11;

void bind_atoms(py::module_& m) {
    using namespace lynx;

    // Pseudopotential
    py::class_<Pseudopotential>(m, "Pseudopotential")
        .def(py::init<>())
        .def("load_psp8", &Pseudopotential::load_psp8, py::arg("filename"))
        .def("grid_size", &Pseudopotential::grid_size)
        .def("Zval", &Pseudopotential::Zval)
        .def("lmax", &Pseudopotential::lmax)
        .def("lloc", &Pseudopotential::lloc)
        .def("has_nlcc", &Pseudopotential::has_nlcc)
        .def("nproj_per_atom", &Pseudopotential::nproj_per_atom)
        .def("rc", &Pseudopotential::rc, py::return_value_policy::reference_internal)
        .def("radial_grid", &Pseudopotential::radial_grid,
             py::return_value_policy::reference_internal);

    // AtomType
    py::class_<AtomType>(m, "AtomType")
        .def(py::init<>())
        .def(py::init<const std::string&, double, double, int>(),
             py::arg("element"), py::arg("mass"), py::arg("Zval"), py::arg("n_atoms"))
        .def("element", &AtomType::element)
        .def("mass", &AtomType::mass)
        .def("Zval", &AtomType::Zval)
        .def("n_atoms", &AtomType::n_atoms)
        .def("psd", py::overload_cast<>(&AtomType::psd),
             py::return_value_policy::reference_internal);

    // AtomInfluence
    py::class_<AtomInfluence>(m, "AtomInfluence")
        .def(py::init<>())
        .def_readonly("n_atom", &AtomInfluence::n_atom)
        .def_readonly("coords", &AtomInfluence::coords)
        .def_readonly("atom_index", &AtomInfluence::atom_index);

    // AtomNlocInfluence
    py::class_<AtomNlocInfluence>(m, "AtomNlocInfluence")
        .def(py::init<>())
        .def_readonly("n_atom", &AtomNlocInfluence::n_atom)
        .def_readonly("coords", &AtomNlocInfluence::coords)
        .def_readonly("atom_index", &AtomNlocInfluence::atom_index);

    // Crystal
    py::class_<Crystal>(m, "Crystal")
        .def(py::init<>())
        .def(py::init<std::vector<AtomType>, std::vector<Vec3>,
                      std::vector<int>, const Lattice&>(),
             py::arg("types"), py::arg("positions"),
             py::arg("type_indices"), py::arg("lattice"))
        .def("n_atom_total", &Crystal::n_atom_total)
        .def("n_types", &Crystal::n_types)
        .def("types", py::overload_cast<>(&Crystal::types, py::const_),
             py::return_value_policy::reference_internal)
        .def("positions", &Crystal::positions,
             py::return_value_policy::reference_internal)
        .def("type_indices", &Crystal::type_indices,
             py::return_value_policy::reference_internal)
        .def("total_Zval", &Crystal::total_Zval)
        .def("wrap_positions", &Crystal::wrap_positions);
}
