#pragma once

#include "core/DeviceArray.hpp"
#include "core/types.hpp"
#include "core/LynxContext.hpp"
#include "core/DeviceTag.hpp"
#include "electronic/Wavefunction.hpp"
#include "parallel/MPIComm.hpp"

namespace lynx {

// Electron density: rho(r) = sum_{n,k,s} f_{nks} |psi_{nks}(r)|^2
class ElectronDensity {
public:
    ElectronDensity() = default;
    ~ElectronDensity();
    ElectronDensity(const ElectronDensity&) = delete;
    ElectronDensity& operator=(const ElectronDensity&) = delete;
    ElectronDensity(ElectronDensity&& o) noexcept
        : Nd_d_(o.Nd_d_), Nspin_(o.Nspin_),
          rho_(std::move(o.rho_)), rho_total_(std::move(o.rho_total_)),
          mag_(std::move(o.mag_)), mag_x_(std::move(o.mag_x_)),
          mag_y_(std::move(o.mag_y_)), mag_z_(std::move(o.mag_z_))
    {
#ifdef USE_CUDA
        gpu_state_raw_ = o.gpu_state_raw_;
        o.gpu_state_raw_ = nullptr;
#endif
    }
    ElectronDensity& operator=(ElectronDensity&& o) noexcept {
        if (this != &o) {
#ifdef USE_CUDA
            cleanup_gpu();
            gpu_state_raw_ = o.gpu_state_raw_;
            o.gpu_state_raw_ = nullptr;
#endif
            Nd_d_ = o.Nd_d_; Nspin_ = o.Nspin_;
            rho_ = std::move(o.rho_); rho_total_ = std::move(o.rho_total_);
            mag_ = std::move(o.mag_); mag_x_ = std::move(o.mag_x_);
            mag_y_ = std::move(o.mag_y_); mag_z_ = std::move(o.mag_z_);
        }
        return *this;
    }

    void allocate(int Nd_d, int Nspin);

    // Set device for dispatch (CPU or GPU).
    void set_device(Device dev) { dev_ = dev; }
    Device device() const { return dev_; }

    // Compute density from wavefunctions and occupations.
    // Dispatches to CPU or GPU path based on dev_ member.
    void compute(const LynxContext& ctx,
                 const Wavefunction& wfn,
                 const std::vector<double>& kpt_weights);

    // Compute spinor density. Dispatches based on dev_ member.
    void compute_spinor(const LynxContext& ctx,
                        const Wavefunction& wfn,
                        const std::vector<double>& kpt_weights);

#ifdef USE_CUDA
    void* gpu_state_raw_ = nullptr;  // Opaque pointer to GPUDensityState (defined in .cu)

    void setup_gpu(const LynxContext& ctx, int Nspin);
    void cleanup_gpu();

    // Legacy GPU compute: uploads psi from host — for testing only.
    // Production code must use compute_from_device_ptrs (psi stays GPU-resident).
    void compute_from_device(const LynxContext& ctx,
                             const Wavefunction& wfn,
                             const std::vector<double>& kpt_weights,
                             const double* d_psi_real,      // device psi (gamma, may be null)
                             const void* d_psi_z,           // device psi (kpt, cuDoubleComplex*, may be null)
                             const double* d_occ);          // device occupations

    // Compute density from per-(spin,kpt) device-resident psi pointers.
    // No psi host→device transfers — all psi already on GPU.
    void compute_from_device_ptrs(
        const LynxContext& ctx,
        const Wavefunction& wfn,
        const std::vector<double>& kpt_weights,
        const std::vector<const double*>& d_psi_real_ptrs,  // [s * Nkpts + k] for gamma
        const std::vector<const void*>& d_psi_z_ptrs);      // [s * Nkpts + k] for kpt

    // Legacy GPU compute paths — upload psi from host (for testing only).
    // Production SCF uses compute_from_device_ptrs instead.
    void compute_gpu(const LynxContext& ctx, const Wavefunction& wfn,
                     const std::vector<double>& kpt_weights);
    void compute_spinor_gpu(const LynxContext& ctx, const Wavefunction& wfn,
                            const std::vector<double>& kpt_weights);

    // GPU kernel wrappers for per-band density accumulation (defined in .cu)
    void accumulate_band_gpu(const double* d_psi, const double* d_occ,
                             double* d_rho, int Nd, int Nband, double weight);
    void accumulate_band_kpt_gpu(const void* d_psi_z, const double* d_occ,
                                  double* d_rho, int Nd, int Nband, double weight);
#endif

    // Compute density from wavefunctions and occupations (explicit params — GPU code paths).
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
    DeviceArray<double>& rho_total() { return rho_total_; }
    const DeviceArray<double>& rho_total() const { return rho_total_; }

    // Spin-resolved density (index by spin)
    DeviceArray<double>& rho(int spin) { return rho_[spin]; }
    const DeviceArray<double>& rho(int spin) const { return rho_[spin]; }

    // Magnetization density (spin-up minus spin-down) — only for collinear
    DeviceArray<double>& mag() { return mag_; }
    const DeviceArray<double>& mag() const { return mag_; }

    // Vector magnetization components — for noncollinear/SOC
    DeviceArray<double>& mag_x() { return mag_x_; }
    const DeviceArray<double>& mag_x() const { return mag_x_; }
    DeviceArray<double>& mag_y() { return mag_y_; }
    const DeviceArray<double>& mag_y() const { return mag_y_; }
    DeviceArray<double>& mag_z() { return mag_z_; }
    const DeviceArray<double>& mag_z() const { return mag_z_; }

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

    // Compute density from spinor wavefunctions (SOC/noncollinear, explicit params).
    void compute_spinor(const Wavefunction& wfn,
                        const std::vector<double>& kpt_weights,
                        double dV,
                        const MPIComm& bandcomm,
                        const MPIComm& kptcomm,
                        int kpt_start = 0,
                        int band_start = 0);

private:
    Device dev_ = Device::CPU;
    int Nd_d_ = 0;
    int Nspin_ = 0;

    std::vector<DeviceArray<double>> rho_;  // per-spin density
    DeviceArray<double> rho_total_;
    DeviceArray<double> mag_;
    DeviceArray<double> mag_x_;  // noncollinear magnetization x
    DeviceArray<double> mag_y_;  // noncollinear magnetization y
    DeviceArray<double> mag_z_;  // noncollinear magnetization z
};

} // namespace lynx
