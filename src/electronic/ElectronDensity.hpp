#pragma once

#include "core/NDArray.hpp"
#include "core/types.hpp"
#include "electronic/Wavefunction.hpp"
#include "parallel/MPIComm.hpp"

namespace sparc {

// Electron density: rho(r) = sum_{n,k,s} f_{nks} |psi_{nks}(r)|^2
class ElectronDensity {
public:
    ElectronDensity() = default;

    void allocate(int Nd_d, int Nspin);

    // Compute density from wavefunctions and occupations
    // kpt_weights: weight of each k-point (usually 1/Nkpts for uniform grid)
    void compute(const Wavefunction& wfn,
                 const std::vector<double>& kpt_weights,
                 double dV,
                 const MPIComm& bandcomm,
                 const MPIComm& kptcomm);

    // Total density (sum over spins)
    NDArray<double>& rho_total() { return rho_total_; }
    const NDArray<double>& rho_total() const { return rho_total_; }

    // Spin-resolved density (index by spin)
    NDArray<double>& rho(int spin) { return rho_[spin]; }
    const NDArray<double>& rho(int spin) const { return rho_[spin]; }

    // Magnetization density (spin-up minus spin-down) — only for collinear
    NDArray<double>& mag() { return mag_; }
    const NDArray<double>& mag() const { return mag_; }

    int Nd_d() const { return Nd_d_; }
    int Nspin() const { return Nspin_; }

    // Integrate rho over domain (returns total electron count)
    double integrate(double dV) const;

private:
    int Nd_d_ = 0;
    int Nspin_ = 0;

    std::vector<NDArray<double>> rho_;  // per-spin density
    NDArray<double> rho_total_;
    NDArray<double> mag_;
};

} // namespace sparc
