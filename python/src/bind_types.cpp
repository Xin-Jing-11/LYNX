#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "core/types.hpp"

namespace py = pybind11;

void bind_types(py::module_& m) {
    using namespace lynx;

    // Enums
    py::enum_<CellType>(m, "CellType")
        .value("Orthogonal", CellType::Orthogonal)
        .value("NonOrthogonal", CellType::NonOrthogonal)
        .value("Helical", CellType::Helical);

    py::enum_<BCType>(m, "BCType")
        .value("Periodic", BCType::Periodic)
        .value("Dirichlet", BCType::Dirichlet);

    py::enum_<SpinType>(m, "SpinType")
        .value("NoSpin", SpinType::None)
        .value("Collinear", SpinType::Collinear)
        .value("NonCollinear", SpinType::NonCollinear);

    py::enum_<MixingVariable>(m, "MixingVariable")
        .value("Density", MixingVariable::Density)
        .value("Potential", MixingVariable::Potential);

    py::enum_<MixingPrecond>(m, "MixingPrecond")
        .value("NoPrecond", MixingPrecond::None)
        .value("Kerker", MixingPrecond::Kerker);

    py::enum_<PoissonSolverType>(m, "PoissonSolverType")
        .value("AAR", PoissonSolverType::AAR)
        .value("CG", PoissonSolverType::CG);

    py::enum_<SmearingType>(m, "SmearingType")
        .value("GaussianSmearing", SmearingType::GaussianSmearing)
        .value("FermiDirac", SmearingType::FermiDirac);

    py::enum_<XCType>(m, "XCType")
        .value("LDA_PZ", XCType::LDA_PZ)
        .value("LDA_PW", XCType::LDA_PW)
        .value("GGA_PBE", XCType::GGA_PBE)
        .value("GGA_PBEsol", XCType::GGA_PBEsol)
        .value("GGA_RPBE", XCType::GGA_RPBE)
        .value("MGGA_SCAN", XCType::MGGA_SCAN)
        .value("MGGA_RSCAN", XCType::MGGA_RSCAN)
        .value("MGGA_R2SCAN", XCType::MGGA_R2SCAN)
        .value("HYB_PBE0", XCType::HYB_PBE0)
        .value("HYB_HSE", XCType::HYB_HSE);

    // Vec3
    py::class_<Vec3>(m, "Vec3")
        .def(py::init<>())
        .def(py::init<double, double, double>())
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("norm", &Vec3::norm)
        .def("__repr__", [](const Vec3& v) {
            return "Vec3(" + std::to_string(v.x) + ", "
                           + std::to_string(v.y) + ", "
                           + std::to_string(v.z) + ")";
        })
        .def("to_list", [](const Vec3& v) {
            return py::make_tuple(v.x, v.y, v.z);
        })
        .def("to_numpy", [](const Vec3& v) {
            auto arr = py::array_t<double>(3);
            auto buf = arr.mutable_unchecked<1>();
            buf(0) = v.x; buf(1) = v.y; buf(2) = v.z;
            return arr;
        });

    // Mat3
    py::class_<Mat3>(m, "Mat3")
        .def(py::init<>())
        .def("__call__", [](const Mat3& m, int i, int j) { return m(i, j); })
        .def("set", [](Mat3& m, int i, int j, double v) { m(i, j) = v; })
        .def("determinant", &Mat3::determinant)
        .def("inverse", &Mat3::inverse)
        .def("transpose", &Mat3::transpose)
        .def("__mul__", &Mat3::operator*)
        .def("to_numpy", [](const Mat3& m) {
            auto arr = py::array_t<double>({3, 3});
            auto buf = arr.mutable_unchecked<2>();
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    buf(i, j) = m(i, j);
            return arr;
        })
        .def_static("from_numpy", [](py::array_t<double> arr) {
            auto buf = arr.unchecked<2>();
            Mat3 m;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    m(i, j) = buf(i, j);
            return m;
        })
        .def("__repr__", [](const Mat3& m) {
            std::string s = "Mat3([[";
            for (int i = 0; i < 3; ++i) {
                if (i > 0) s += "      [";
                for (int j = 0; j < 3; ++j) {
                    s += std::to_string(m(i, j));
                    if (j < 2) s += ", ";
                }
                s += "]";
                if (i < 2) s += ",\n";
            }
            s += "])";
            return s;
        });
}
