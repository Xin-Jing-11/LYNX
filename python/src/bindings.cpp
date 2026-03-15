#include <pybind11/pybind11.h>
#include <mpi.h>

namespace py = pybind11;

// Forward declarations for each bind_* module
void bind_types(py::module_& m);
void bind_core(py::module_& m);
void bind_operators(py::module_& m);
void bind_solvers(py::module_& m);
void bind_electronic(py::module_& m);
void bind_xc(py::module_& m);
void bind_physics(py::module_& m);
void bind_atoms(py::module_& m);
void bind_calculator(py::module_& m);

PYBIND11_MODULE(_core, m) {
    m.doc() = "LYNX DFT simulator — Python bindings";

    // Register all bindings
    bind_types(m);
    bind_core(m);
    bind_operators(m);
    bind_solvers(m);
    bind_electronic(m);
    bind_xc(m);
    bind_physics(m);
    bind_atoms(m);
    bind_calculator(m);

    // Module-level init function
    m.def("_ensure_mpi", []() {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            int argc = 0;
            char** argv = nullptr;
            MPI_Init(&argc, &argv);
        }
    }, "Ensure MPI is initialized");
}
