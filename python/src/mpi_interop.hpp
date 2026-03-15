#pragma once

#include <mpi.h>

namespace pylynx {

// Ensure MPI is initialized
inline void ensure_mpi_initialized() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int argc = 0;
        char** argv = nullptr;
        MPI_Init(&argc, &argv);
    }
}

} // namespace pylynx
