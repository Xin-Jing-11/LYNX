#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "Calculator.hpp"
#include "ArrayBridge.hpp"

namespace py = pybind11;

void bind_calculator(py::module_& m) {
    using namespace pylynx;

    py::class_<Calculator>(m, "Calculator")
        .def(py::init<>())
        .def(py::init([](const std::string& json_file, bool auto_run) {
            auto calc = std::make_unique<Calculator>();
            calc->load_config(json_file);
            calc->setup(MPI_COMM_SELF);
            if (auto_run) {
                py::gil_scoped_release release;
                calc->run();
            }
            return calc;
        }), py::arg("json_file"), py::arg("auto_run") = true,
           "Create Calculator from JSON config file. If auto_run=True, runs SCF immediately.")

        .def("load_config", &Calculator::load_config, py::arg("json_file"))
        .def("setup", [](Calculator& c) { c.setup(MPI_COMM_SELF); },
             "Set up all internal objects (single-process mode)")
        .def("run", &Calculator::run,
             py::call_guard<py::gil_scoped_release>(),
             "Run SCF calculation. Returns total energy in Hartree.")
        .def("compute_forces", [](Calculator& c) {
            py::gil_scoped_release release;
            auto f = c.compute_forces();
            py::gil_scoped_acquire acquire;
            int N = static_cast<int>(f.size()) / 3;
            auto arr = py::array_t<double>({N, 3});
            std::memcpy(arr.mutable_data(), f.data(), f.size() * sizeof(double));
            return arr;
        }, "Compute atomic forces. Returns numpy array (Natom, 3) in Ha/Bohr.")
        .def("compute_stress", [](Calculator& c) {
            py::gil_scoped_release release;
            auto s = c.compute_stress();
            py::gil_scoped_acquire acquire;
            return py::array_t<double>(6, s.data());
        }, "Compute stress tensor. Returns 6-component Voigt array in Ha/Bohr^3.")

        // Property accessors
        .def_property_readonly("is_setup", &Calculator::is_setup)
        .def_property_readonly("converged", &Calculator::is_converged)
        .def_property_readonly("Nd_d", &Calculator::Nd_d)
        .def_property_readonly("Nelectron", &Calculator::Nelectron)
        .def_property_readonly("Natom", &Calculator::Natom)
        .def_property_readonly("Nspin", &Calculator::Nspin)

        // Energy
        .def_property_readonly("energy", [](const Calculator& c) {
            const auto& e = c.energy();
            py::dict d;
            d["Eband"] = e.Eband;
            d["Exc"] = e.Exc;
            d["Ehart"] = e.Ehart;
            d["Eself"] = e.Eself;
            d["Ec"] = e.Ec;
            d["Entropy"] = e.Entropy;
            d["Etotal"] = e.Etotal;
            d["Eatom"] = e.Eatom;
            return d;
        })
        .def_property_readonly("energy_components", &Calculator::energy,
             py::return_value_policy::reference_internal)
        .def_property_readonly("total_energy", [](const Calculator& c) {
            return c.energy().Etotal;
        })
        .def_property_readonly("fermi_energy", &Calculator::fermi_energy)

        // Density as numpy
        .def_property_readonly("density", [](const Calculator& c) {
            return pylynx::to_numpy(c.scf().density().rho_total());
        }, py::return_value_policy::reference_internal)

        // Internal objects for mid-level usage
        .def_property_readonly("lattice", &Calculator::lattice,
             py::return_value_policy::reference_internal)
        .def_property_readonly("grid", &Calculator::grid,
             py::return_value_policy::reference_internal)
        .def_property_readonly("stencil", &Calculator::stencil,
             py::return_value_policy::reference_internal)
        .def_property_readonly("domain", &Calculator::domain,
             py::return_value_policy::reference_internal)
        .def_property_readonly("kpoints", &Calculator::kpoints,
             py::return_value_policy::reference_internal)
        .def_property_readonly("halo_exchange", &Calculator::halo,
             py::return_value_policy::reference_internal)
        .def_property_readonly("laplacian_op", &Calculator::laplacian,
             py::return_value_policy::reference_internal)
        .def_property_readonly("gradient_op", &Calculator::gradient,
             py::return_value_policy::reference_internal)
        .def_property_readonly("hamiltonian_op", &Calculator::hamiltonian,
             py::return_value_policy::reference_internal)
        .def_property_readonly("nonlocal_projector", &Calculator::nonlocal_projector,
             py::return_value_policy::reference_internal)
        .def_property_readonly("crystal", &Calculator::crystal,
             py::return_value_policy::reference_internal)
        .def_property_readonly("electrostatics", &Calculator::electrostatics,
             py::return_value_policy::reference_internal)
        .def_property_readonly("scf_solver", &Calculator::scf,
             py::return_value_policy::reference_internal)
        .def("get_wavefunction", [](Calculator& c) -> lynx::Wavefunction* {
            return &c.wavefunction();
        }, py::return_value_policy::reference_internal,
           "Get reference to internal Wavefunction object");
}
