#include <gtest/gtest.h>
#include <mpi.h>
#include "core/LynxContext.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    // Reset LynxContext before MPI_Finalize to avoid freeing MPI comms
    // after MPI has been finalized (the singleton destructor runs at exit).
    lynx::LynxContext::instance().reset();
    MPI_Finalize();
    return result;
}
