#pragma once

#include "core/NDArray.hpp"
#include "core/types.hpp"
#include "electronic/Wavefunction.hpp"
#include "parallel/MPIComm.hpp"

namespace lynx {

// Electron density: rho(r) = sum_{n,k,s} f_{nks} |psi_{nks}(r)|^2
class ElectronDensity {
public:
    ElectronDensity() = default;
    ElectronDensity(const ElectronDensity&) = delete;
    ElectronDensity& operator=(const ElectronDensity&) = delete;
    ElectronDensity(ElectronDensity&&) = default;
    ElectronDensity& operator=(ElectronDensity&&) = default;

    void allocate(int Nd_d, int Nspin);

    // Compute density from wavefunctions and occupations
    // kpt_weights: weight of each k-point (usually 1/Nkpts for uniform grid)
    // Nspin_global: global spin count (for correct spin_fac)
    // spin_start: global index of first local spin channel
    // spincomm: communicator for exchanging spin densities across processes
    // band_start: global index of first local band (for band parallelism)
    void compute(const Wavefunction& wfn,
                 const std::vector<double>& kpt_weights,
                 double dV,
                 const MPIComm& bandcomm,
                 const MPIComm& kptcomm,
                 int Nspin_global = 0,
                 int spin_start = 0,
                 const MPIComm* spincomm = nullptr,
                 int kpt_start = 0,
                 int band_start = 0);

    // Total density (sum over spins)
    NDArray<double>& rho_total() { return rho_total_; }
    const NDArray<double>& rho_total() const { return rho_total_; }

    // Spin-resolved density (index by spin)
    NDArray<double>& rho(int spin) { return rho_[spin]; }
    const NDArray<double>& rho(int spin) const { return rho_[spin]; }

    // Magnetization density (spin-up minus spin-down) — only for collinear
    NDArray<double>& mag() { return mag_; }
    const NDArray<double>& mag() const { return mag_; }

    // Vector magnetization components — for noncollinear/SOC
    NDArray<double>& mag_x() { return mag_x_; }
    const NDArray<double>& mag_x() const { return mag_x_; }
    NDArray<double>& mag_y() { return mag_y_; }
    const NDArray<double>& mag_y() const { return mag_y_; }
    NDArray<double>& mag_z() { return mag_z_; }
    const NDArray<double>& mag_z() const { return mag_z_; }

    int Nd_d() const { return Nd_d_; }
    int Nspin() const { return Nspin_; }

    // Initialize with uniform density rho0 = Nelectron / cell_volume.
    // Allocates and fills rho for scalar (non-SOC) case.
    void initialize_uniform(int Nd_d, int Nspin, int Nelectron, double cell_volume);

    // Initialize with uniform density for noncollinear/SOC case.
    // Allocates noncollinear format with zero magnetization.
    void initialize_uniform_noncollinear(int Nd_d, int Nelectron, double cell_volume);

    // Integrate rho over domain (returns total electron count)
    double integrate(double dV) const;

    // Allocate for noncollinear (spinor) — 1 spin channel + vector magnetization
    void allocate_noncollinear(int Nd_d);

    // Compute density from spinor wavefunctions (SOC/noncollinear)
    // Each band is a 2-component spinor: [psi_up(Nd_d) | psi_dn(Nd_d)]
    void compute_spinor(const Wavefunction& wfn,
                        const std::vector<double>& kpt_weights,
                        double dV,
                        const MPIComm& bandcomm,
                        const MPIComm& kptcomm,
                        int kpt_start = 0,
                        int band_start = 0);

private:
    int Nd_d_ = 0;
    int Nspin_ = 0;

    std::vector<NDArray<double>> rho_;  // per-spin density
    NDArray<double> rho_total_;
    NDArray<double> mag_;
    NDArray<double> mag_x_;  // noncollinear magnetization x
    NDArray<double> mag_y_;  // noncollinear magnetization y
    NDArray<double> mag_z_;  // noncollinear magnetization z
};

} // namespace lynx
