#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include "core/Domain.hpp"
#include "core/KPoints.hpp"
#include "operators/FDStencil.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"
#include "parallel/Parallelization.hpp"
#include "ArrayBridge.hpp"

namespace py = pybind11;

void bind_core(py::module_& m) {
    using namespace lynx;

    // Lattice
    py::class_<Lattice>(m, "Lattice")
        .def(py::init<>())
        .def(py::init<const Mat3&, CellType>(),
             py::arg("latvec"), py::arg("cell_type") = CellType::Orthogonal)
        .def("latvec", &Lattice::latvec, py::return_value_policy::reference_internal)
        .def("metric_tensor", &Lattice::metric_tensor, py::return_value_policy::reference_internal)
        .def("jacobian", &Lattice::jacobian)
        .def("cell_type", &Lattice::cell_type)
        .def("is_orthogonal", &Lattice::is_orthogonal)
        .def("lengths", &Lattice::lengths)
        .def("frac_to_cart", &Lattice::frac_to_cart)
        .def("cart_to_frac", &Lattice::cart_to_frac)
        .def("cart_to_nonCart", &Lattice::cart_to_nonCart)
        .def("nonCart_to_cart", &Lattice::nonCart_to_cart)
        .def("reciprocal_latvec", &Lattice::reciprocal_latvec)
        .def("__repr__", [](const Lattice& l) {
            auto lens = l.lengths();
            return "Lattice(a=" + std::to_string(lens.x) +
                   ", b=" + std::to_string(lens.y) +
                   ", c=" + std::to_string(lens.z) + ")";
        });

    // Convenience: create Lattice from numpy 3x3 array
    m.def("make_lattice", [](py::array_t<double> latvec_np, CellType ct) {
        auto buf = latvec_np.unchecked<2>();
        Mat3 lv;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                lv(i, j) = buf(i, j);
        return Lattice(lv, ct);
    }, py::arg("latvec"), py::arg("cell_type") = CellType::Orthogonal,
       "Create Lattice from 3x3 numpy array");

    // FDGrid
    py::class_<FDGrid>(m, "FDGrid")
        .def(py::init<>())
        .def(py::init<int, int, int, const Lattice&, BCType, BCType, BCType>(),
             py::arg("Nx"), py::arg("Ny"), py::arg("Nz"),
             py::arg("lattice"),
             py::arg("bcx") = BCType::Periodic,
             py::arg("bcy") = BCType::Periodic,
             py::arg("bcz") = BCType::Periodic)
        .def("Nx", &FDGrid::Nx)
        .def("Ny", &FDGrid::Ny)
        .def("Nz", &FDGrid::Nz)
        .def("Nd", &FDGrid::Nd)
        .def("dx", &FDGrid::dx)
        .def("dy", &FDGrid::dy)
        .def("dz", &FDGrid::dz)
        .def("dV", &FDGrid::dV)
        .def("bcx", &FDGrid::bcx)
        .def("bcy", &FDGrid::bcy)
        .def("bcz", &FDGrid::bcz)
        .def("__repr__", [](const FDGrid& g) {
            return "FDGrid(" + std::to_string(g.Nx()) + "x"
                             + std::to_string(g.Ny()) + "x"
                             + std::to_string(g.Nz()) + ", Nd="
                             + std::to_string(g.Nd()) + ")";
        });

    // FDStencil
    py::class_<FDStencil>(m, "FDStencil")
        .def(py::init<>())
        .def(py::init<int, const FDGrid&, const Lattice&>(),
             py::arg("order"), py::arg("grid"), py::arg("lattice"))
        .def("order", &FDStencil::order)
        .def("FDn", &FDStencil::FDn)
        .def("max_eigval_half_lap", &FDStencil::max_eigval_half_lap);

    // DomainVertices
    py::class_<DomainVertices>(m, "DomainVertices")
        .def(py::init<>())
        .def_readwrite("xs", &DomainVertices::xs)
        .def_readwrite("xe", &DomainVertices::xe)
        .def_readwrite("ys", &DomainVertices::ys)
        .def_readwrite("ye", &DomainVertices::ye)
        .def_readwrite("zs", &DomainVertices::zs)
        .def_readwrite("ze", &DomainVertices::ze);

    // Domain
    py::class_<Domain>(m, "Domain")
        .def(py::init<>())
        .def(py::init<const FDGrid&, const DomainVertices&>())
        .def("vertices", &Domain::vertices, py::return_value_policy::reference_internal)
        .def("Nx_d", &Domain::Nx_d)
        .def("Ny_d", &Domain::Ny_d)
        .def("Nz_d", &Domain::Nz_d)
        .def("Nd_d", &Domain::Nd_d)
        .def("__repr__", [](const Domain& d) {
            return "Domain(" + std::to_string(d.Nx_d()) + "x"
                             + std::to_string(d.Ny_d()) + "x"
                             + std::to_string(d.Nz_d()) + ", Nd_d="
                             + std::to_string(d.Nd_d()) + ")";
        });

    // Helper to create full domain from grid (single-process mode)
    m.def("full_domain", [](const FDGrid& grid) {
        DomainVertices v;
        v.xs = 0; v.xe = grid.Nx() - 1;
        v.ys = 0; v.ye = grid.Ny() - 1;
        v.zs = 0; v.ze = grid.Nz() - 1;
        return Domain(grid, v);
    }, py::arg("grid"),
       "Create a Domain covering the full grid (for single-process use)");

    // KPoints
    py::class_<KPoints>(m, "KPoints")
        .def(py::init<>())
        .def("generate", &KPoints::generate)
        .def("Nkpts", &KPoints::Nkpts)
        .def("Nkpts_full", &KPoints::Nkpts_full)
        .def("kpts_cart", &KPoints::kpts_cart, py::return_value_policy::reference_internal)
        .def("kpts_red", &KPoints::kpts_red, py::return_value_policy::reference_internal)
        .def("weights", &KPoints::weights, py::return_value_policy::reference_internal)
        .def("normalized_weights", &KPoints::normalized_weights)
        .def("is_gamma_only", &KPoints::is_gamma_only);

    // HaloExchange
    py::class_<HaloExchange>(m, "HaloExchange")
        .def(py::init<>())
        .def(py::init<const Domain&, int>(), py::arg("domain"), py::arg("FDn"))
        .def("nx_ex", &HaloExchange::nx_ex)
        .def("ny_ex", &HaloExchange::ny_ex)
        .def("nz_ex", &HaloExchange::nz_ex)
        .def("nd_ex", &HaloExchange::nd_ex)
        .def("FDn", &HaloExchange::FDn);

    // ParallelParams
    py::class_<ParallelParams>(m, "ParallelParams")
        .def(py::init<>())
        .def_readwrite("npspin", &ParallelParams::npspin)
        .def_readwrite("npkpt", &ParallelParams::npkpt)
        .def_readwrite("npband", &ParallelParams::npband);
}
