#pragma once

#include "core/types.hpp"
#include "core/DeviceArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "core/DeviceTag.hpp"
#include "core/GPUStatePtr.hpp"
#include "operators/Laplacian.hpp"
#include "operators/Gradient.hpp"
#include "operators/FDStencil.hpp"
#include "operators/Hamiltonian.hpp"
#include "parallel/HaloExchange.hpp"
#include "electronic/ElectronDensity.hpp"
#include "xc/XCFunctional.hpp"
#include "solvers/PoissonSolver.hpp"
#include "core/LynxContext.hpp"

namespace lynx {

class ExactExchange;

// Work arrays produced by EffectivePotential computation.
// These are inputs/outputs that persist across SCF iterations.
struct VeffArrays {
    DeviceArray<double> Veff;       // effective potential (Nd_d * Nspin)
    DeviceArray<double> Vxc;        // XC potential (Nd_d * Nspin)
    DeviceArray<double> exc;        // XC energy density (Nd_d)
    DeviceArray<double> phi;        // electrostatic potential (Nd_d)
    DeviceArray<double> Dxcdgrho;   // GGA: dExc/d(|grad rho|^2) (Nd_d * dxc_ncol)
    DeviceArray<double> vtau;       // d(n*exc)/d(tau) (mGGA) (Nd_d or 2*Nd_d for spin)
    DeviceArray<double> Veff_spinor; // spinor Veff [V_uu|V_dd|Re(V_ud)|Im(V_ud)] (4*Nd_d)

    // Allocate arrays for given system parameters
    void allocate(int Nd_d, int Nspin, XCType xc_type, bool is_soc);
};

// Builds the effective potential from electron density.
// Handles XC evaluation, Poisson solve, NLCC, mGGA fallback, and spinor variants.
class EffectivePotential {
public:
    EffectivePotential() = default;
    ~EffectivePotential();
    EffectivePotential(EffectivePotential&&) noexcept = default;
    EffectivePotential& operator=(EffectivePotential&&) noexcept = default;
    EffectivePotential(const EffectivePotential&) = delete;
    EffectivePotential& operator=(const EffectivePotential&) = delete;

    /// Setup using LynxContext for all infrastructure.
    void setup(const LynxContext& ctx, const Hamiltonian& hamiltonian);

    // Set device for dispatch (CPU or GPU).
    void set_device(Device dev) { dev_ = dev; }
    Device device() const { return dev_; }

    // Compute Veff for standard (scalar) case.
    // Dispatches to CPU or GPU path based on dev_ member.
    void compute(const ElectronDensity& density,
                 const double* rho_b,
                 const double* rho_core,
                 XCType xc_type,
                 double exx_frac_scale,
                 double poisson_tol,
                 VeffArrays& arrays,
                 const double* tau = nullptr,
                 bool tau_valid = false);

    // Compute spinor Veff from noncollinear density.
    // Dispatches to CPU or GPU path based on dev_ member.
    void compute_spinor(const ElectronDensity& density,
                        const double* rho_b,
                        const double* rho_core,
                        XCType xc_type,
                        double poisson_tol,
                        VeffArrays& arrays);

#ifdef USE_CUDA
    GPUStatePtr gpu_state_;  // Opaque pointer to GPUVeffState (defined in .cu)

    void setup_gpu(const LynxContext& ctx, int Nspin,
                        XCType xc_type, const double* rho_b,
                        const double* rho_core);
    void cleanup_gpu();

    // GPU-resident device pointer accessors
    double* gpu_Veff();
    const double* gpu_Veff() const;
    double* gpu_phi();
    double* gpu_exc();
    double* gpu_Vxc();
    double* gpu_rho();
    double* gpu_rho_total();

    // Upload density from host to device buffers.
    void upload_density(const ElectronDensity& density);

    // Download potential arrays from device to host VeffArrays.
    void download_to_host(VeffArrays& arrays);

    // Set device tau/vtau pointers for mGGA GPU pipeline.
    void set_device_tau(double* d_tau, double* d_vtau);

    // GPU compute paths (defined in .cu)
    void compute_gpu(const ElectronDensity& density, const double* rho_b,
                     const double* rho_core, XCType xc_type,
                     double exx_frac_scale, double poisson_tol,
                     VeffArrays& arrays, const double* tau, bool tau_valid);
    void compute_spinor_gpu(const ElectronDensity& density, const double* rho_b,
                            const double* rho_core, XCType xc_type,
                            double poisson_tol, VeffArrays& arrays);

    // GPU kernel wrappers (defined in .cu)
    void poisson_rhs_gpu(const double* d_rho_total, const double* d_pseudocharge,
                          double* d_rhs, int Nd);
    void combine_veff_gpu(const double* d_Vxc, const double* d_phi,
                           double* d_Veff, int Nd, int Nspin);
    void combine_veff_spinor_gpu(const double* d_Vxc_up, const double* d_Vxc_dn,
                                  const double* d_phi,
                                  const double* d_mag_x, const double* d_mag_y,
                                  const double* d_mag_z,
                                  double* d_Veff_spinor, int Nd);
#endif

private:
    Device dev_ = Device::CPU;
    const Domain* domain_ = nullptr;
    const FDGrid* grid_ = nullptr;
    const FDStencil* stencil_ = nullptr;
    const Laplacian* laplacian_ = nullptr;
    const Gradient* gradient_ = nullptr;
    const Hamiltonian* hamiltonian_ = nullptr;
    const HaloExchange* halo_ = nullptr;
    int Nspin_global_ = 1;

    // Solve Poisson equation and shift phi to zero mean
    void solve_poisson(const double* rho, const double* rho_b,
                       int Nd_d, double poisson_tol, double* phi);
};

} // namespace lynx
