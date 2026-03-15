#pragma once

#include <string>
#include "io/InputParser.hpp"
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"

namespace lynx {

class OutputWriter {
public:
    static void print_summary(const SystemConfig& config, const Lattice& lattice,
                              const FDGrid& grid, int world_rank = 0);
};

} // namespace lynx
