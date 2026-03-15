#pragma once

#include <string>
#include "electronic/ElectronDensity.hpp"
#include "core/FDGrid.hpp"
#include "core/Lattice.hpp"

namespace lynx {

// Binary density file I/O for checkpoint/restart.
// Format: 128-byte header + nspin * Nd raw doubles.
class DensityIO {
public:
    // Write density to binary file. Only rank 0 should call this.
    // Returns true on success.
    static bool write(const std::string& filename,
                      const ElectronDensity& density,
                      const FDGrid& grid,
                      const Lattice& lattice);

    // Read density from binary file into pre-allocated ElectronDensity.
    // Validates magic, grid dimensions, and spin count.
    // Returns true on success, false on mismatch or file error.
    static bool read(const std::string& filename,
                     ElectronDensity& density,
                     const FDGrid& grid,
                     const Lattice& lattice);

    static constexpr char MAGIC[8] = {'L','Y','N','X','_','R','H','O'};
    static constexpr uint32_t VERSION = 1;

    struct FileHeader {
        char magic[8];
        uint32_t version;
        uint32_t nspin;
        uint32_t nx, ny, nz;
        uint32_t _pad;          // align to 8-byte boundary
        double latvec[9];       // row-major 3x3
        double dV;
        char _reserved[128 - 8 - 4*5 - 4 - 72 - 8]; // pad to 128 bytes
    };
    static_assert(sizeof(FileHeader) == 128, "FileHeader must be 128 bytes");
};

} // namespace lynx
