#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "physics/SCF.hpp"
#include "physics/Energy.hpp"
#include "physics/Forces.hpp"
#include "physics/Stress.hpp"
#include "physics/Electrostatics.hpp"
#include "ArrayBridge.hpp"

namespace py = pybind11;

void bind_physics(py::module_& m) {
    using namespace lynx;

    // SCFParams
    py::class_<SCFParams>(m, "SCFParams")
        .def(py::init<>())
        .def_readwrite("max_iter", &SCFParams::max_iter)
        .def_readwrite("min_iter", &SCFParams::min_iter)
        .def_readwrite("tol", &SCFParams::tol)
        .def_readwrite("mixing_var", &SCFParams::mixing_var)
        .def_readwrite("mixing_precond", &SCFParams::mixing_precond)
        .def_readwrite("mixing_history", &SCFParams::mixing_history)
        .def_readwrite("mixing_param", &SCFParams::mixing_param)
        .def_readwrite("smearing", &SCFParams::smearing)
        .def_readwrite("elec_temp", &SCFParams::elec_temp)
        .def_readwrite("cheb_degree", &SCFParams::cheb_degree)
        .def_readwrite("rho_trigger", &SCFParams::rho_trigger)
        .def_readwrite("nchefsi", &SCFParams::nchefsi)
        .def_readwrite("poisson_tol", &SCFParams::poisson_tol)
        .def_readwrite("print_eigen", &SCFParams::print_eigen);

    // EnergyComponents
    py::class_<EnergyComponents>(m, "EnergyComponents")
        .def(py::init<>())
        .def_readwrite("Eband", &EnergyComponents::Eband)
        .def_readwrite("Exc", &EnergyComponents::Exc)
        .def_readwrite("Ehart", &EnergyComponents::Ehart)
        .def_readwrite("Eself", &EnergyComponents::Eself)
        .def_readwrite("Ec", &EnergyComponents::Ec)
        .def_readwrite("Entropy", &EnergyComponents::Entropy)
        .def_readwrite("Etotal", &EnergyComponents::Etotal)
        .def_readwrite("Eatom", &EnergyComponents::Eatom)
        .def("to_dict", [](const EnergyComponents& e) {
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
        .def("__repr__", [](const EnergyComponents& e) {
            return "EnergyComponents(Etotal=" + std::to_string(e.Etotal) + " Ha)";
        });

    // SCF
    py::class_<SCF>(m, "SCF")
        .def(py::init<>())
        .def("energy", &SCF::energy, py::return_value_policy::reference_internal)
        .def("density", &SCF::density, py::return_value_policy::reference_internal)
        .def("fermi_energy", &SCF::fermi_energy)
        .def("converged", &SCF::converged)
        // Access converged fields as numpy
        .def("phi", [](const SCF& scf, int Nd_d) {
            return py::array_t<double>({Nd_d}, {sizeof(double)}, scf.phi(), py::none());
        }, py::arg("Nd_d"), py::return_value_policy::reference_internal)
        .def("Vxc", [](const SCF& scf, int Nd_d) {
            return py::array_t<double>({Nd_d}, {sizeof(double)}, scf.Vxc(), py::none());
        }, py::arg("Nd_d"), py::return_value_policy::reference_internal)
        .def("exc", [](const SCF& scf, int Nd_d) {
            return py::array_t<double>({Nd_d}, {sizeof(double)}, scf.exc(), py::none());
        }, py::arg("Nd_d"), py::return_value_policy::reference_internal)
        .def("Veff", [](const SCF& scf, int Nd_d) {
            return py::array_t<double>({Nd_d}, {sizeof(double)}, scf.Veff(), py::none());
        }, py::arg("Nd_d"), py::return_value_policy::reference_internal);

    // Energy (static methods)
    py::class_<Energy>(m, "Energy")
        .def_static("total_energy", &Energy::total_energy);

    // Electrostatics
    py::class_<Electrostatics>(m, "Electrostatics")
        .def(py::init<>())
        .def("Eself", &Electrostatics::Eself)
        .def("Ec", &Electrostatics::Ec)
        .def("Eself_Ec", &Electrostatics::Eself_Ec)
        .def("int_b", &Electrostatics::int_b)
        .def("pseudocharge", [](const Electrostatics& e) {
            return pylynx::to_numpy(e.pseudocharge());
        }, py::return_value_policy::reference_internal)
        .def("pseudocharge_ref", [](const Electrostatics& e) {
            return pylynx::to_numpy(e.pseudocharge_ref());
        }, py::return_value_policy::reference_internal);

    // Forces
    py::class_<Forces>(m, "Forces")
        .def(py::init<>())
        .def("local_forces", [](const Forces& f) {
            const auto& fl = f.local_forces();
            int N = static_cast<int>(fl.size()) / 3;
            return py::array_t<double>({N, 3}, fl.data());
        })
        .def("nonlocal_forces", [](const Forces& f) {
            const auto& fn = f.nonlocal_forces();
            int N = static_cast<int>(fn.size()) / 3;
            return py::array_t<double>({N, 3}, fn.data());
        })
        .def("xc_forces", [](const Forces& f) {
            const auto& fxc = f.xc_forces();
            int N = static_cast<int>(fxc.size()) / 3;
            return py::array_t<double>({N, 3}, fxc.data());
        })
        .def("total_forces", [](const Forces& f) {
            const auto& ft = f.total_forces();
            int N = static_cast<int>(ft.size()) / 3;
            return py::array_t<double>({N, 3}, ft.data());
        });

    // Stress
    py::class_<Stress>(m, "Stress")
        .def(py::init<>())
        .def("pressure", &Stress::pressure)
        .def("total_stress", [](const Stress& s) {
            const auto& st = s.total_stress();
            return py::array_t<double>(6, st.data());
        })
        .def("kinetic_stress", [](const Stress& s) {
            const auto& st = s.kinetic_stress();
            return py::array_t<double>(6, st.data());
        });
}
