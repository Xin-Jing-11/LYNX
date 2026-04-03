#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "core/KPoints.hpp"
#include "core/LynxContext.hpp"
#include "electronic/Wavefunction.hpp"
#include "operators/Gradient.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"

#include <vector>

namespace lynx {

// Kinetic energy density: tau(r) = 0.5 * sum_{n,k,s} f_{nks} |nabla psi_{nks}(r)|^2
// Used by mGGA functionals (SCAN, rSCAN, r2SCAN).
class KineticEnergyDensity {
public:
    KineticEnergyDensity() = default;

    // Allocate tau array.
    // For Nspin==1: tau has Nd_d elements.
    // For Nspin==2: tau has 3*Nd_d elements [up | down | total].
    void allocate(int Nd_d, int Nspin);

    // Compute tau from wavefunctions and occupations (with LynxContext — preferred).
    void compute(const LynxContext& ctx,
                 const Wavefunction& wfn,
                 const std::vector<double>& kpt_weights);

    // Compute tau from wavefunctions and occupations (explicit params).
    void compute(const Wavefunction& wfn,
                 const std::vector<double>& kpt_weights,
                 const FDGrid& grid,
                 const Domain& domain,
                 const HaloExchange& halo,
                 const Gradient& gradient,
                 const KPoints* kpoints,
                 const MPIComm& bandcomm,
                 const MPIComm& kptcomm,
                 const MPIComm* spincomm,
                 int spin_start,
                 int kpt_start,
                 int band_start,
                 int Nspin_global);

    // Access tau data
    double* data() { return tau_.data(); }
    const double* data() const { return tau_.data(); }
    int size() const { return static_cast<int>(tau_.size()); }

    // Validity flag (true after first compute() call)
    bool valid() const { return valid_; }
    void set_valid(bool v) { valid_ = v; }

private:
    NDArray<double> tau_;
    bool valid_ = false;
};

} // namespace lynx
