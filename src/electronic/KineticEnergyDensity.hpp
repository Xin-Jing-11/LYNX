#pragma once

#include "core/types.hpp"
#include "core/DeviceArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "core/KPoints.hpp"
#include "core/LynxContext.hpp"
#include "core/DeviceTag.hpp"
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
    ~KineticEnergyDensity();
    KineticEnergyDensity(KineticEnergyDensity&& o) noexcept
        : tau_(std::move(o.tau_)), valid_(o.valid_)
    {
#ifdef USE_CUDA
        gpu_state_raw_ = o.gpu_state_raw_;
        o.gpu_state_raw_ = nullptr;
#endif
    }
    KineticEnergyDensity& operator=(KineticEnergyDensity&& o) noexcept {
        if (this != &o) {
#ifdef USE_CUDA
            cleanup_gpu();
            gpu_state_raw_ = o.gpu_state_raw_;
            o.gpu_state_raw_ = nullptr;
#endif
            tau_ = std::move(o.tau_); valid_ = o.valid_;
        }
        return *this;
    }
    KineticEnergyDensity(const KineticEnergyDensity&) = delete;
    KineticEnergyDensity& operator=(const KineticEnergyDensity&) = delete;

    // Allocate tau array.
    // For Nspin==1: tau has Nd_d elements.
    // For Nspin==2: tau has 3*Nd_d elements [up | down | total].
    void allocate(int Nd_d, int Nspin);

    // Set device for dispatch (CPU or GPU).
    void set_device(Device dev) { dev_ = dev; }
    Device device() const { return dev_; }

    // Compute tau from wavefunctions and occupations.
    // Dispatches to CPU or GPU path based on dev_ member.
    void compute(const LynxContext& ctx,
                 const Wavefunction& wfn,
                 const std::vector<double>& kpt_weights);

    // Compute tau from wavefunctions and occupations (explicit params — CPU only).
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

#ifdef USE_CUDA
    void* gpu_state_raw_ = nullptr;  // Opaque pointer to GPUTauState (defined in .cu)

    void setup_gpu(const LynxContext& ctx, int Nspin);
    void cleanup_gpu();

    // Legacy GPU compute — uploads psi from host (for testing only).
    // Production SCF uses compute_gpu_from_device instead.
    void compute_gpu(const LynxContext& ctx, const Wavefunction& wfn,
                     const std::vector<double>& kpt_weights);

    // GPU compute from per-(spin,kpt) device-resident psi — no psi H2D transfers
    void compute_gpu_from_device(
        const LynxContext& ctx,
        const Wavefunction& wfn,
        const std::vector<double>& kpt_weights,
        const std::vector<const double*>& d_psi_real_ptrs,
        const std::vector<const void*>& d_psi_z_ptrs);

    // Device pointer accessors (for GPU-resident mGGA pipeline)
    double* d_tau();
    double* d_vtau();

    // GPU kernel wrapper: halo + gradient + tau accumulate for one band (defined in .cu)
    void gradient_accumulate_tau_gpu(const double* d_psi_col, double* d_tau_s,
                                     double weight, int Nd);
#endif

    // Access tau data
    double* data() { return tau_.data(); }
    const double* data() const { return tau_.data(); }
    int size() const { return static_cast<int>(tau_.size()); }

    // Validity flag (true after first compute() call)
    bool valid() const { return valid_; }
    void set_valid(bool v) { valid_ = v; }

private:
    Device dev_ = Device::CPU;
    DeviceArray<double> tau_;
    bool valid_ = false;
};

} // namespace lynx
