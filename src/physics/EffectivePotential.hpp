#pragma once

#include "core/types.hpp"
#include "core/DeviceArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "core/DeviceTag.hpp"
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
    EffectivePotential(EffectivePotential&& o) noexcept
        : domain_(o.domain_), grid_(o.grid_), stencil_(o.stencil_),
          laplacian_(o.laplacian_), gradient_(o.gradient_),
          hamiltonian_(o.hamiltonian_), halo_(o.halo_), Nspin_global_(o.Nspin_global_)
    {
#ifdef USE_CUDA
        gpu_state_raw_ = o.gpu_state_raw_;
        o.gpu_state_raw_ = nullptr;
#endif
    }
    EffectivePotential& operator=(EffectivePotential&& o) noexcept {
        if (this != &o) {
#ifdef USE_CUDA
            cleanup_gpu();
            gpu_state_raw_ = o.gpu_state_raw_;
            o.gpu_state_raw_ = nullptr;
#endif
            domain_ = o.domain_; grid_ = o.grid_; stencil_ = o.stencil_;
            laplacian_ = o.laplacian_; gradient_ = o.gradient_;
            hamiltonian_ = o.hamiltonian_; halo_ = o.halo_;
            Nspin_global_ = o.Nspin_global_;
        }
        return *this;
    }
    EffectivePotential(const EffectivePotential&) = delete;
    EffectivePotential& operator=(const EffectivePotential&) = delete;

    /// Setup using LynxContext for all infrastructure.
    void setup(const LynxContext& ctx, const Hamiltonian& hamiltonian);

    // Compute Veff for standard (scalar) case.
    // density: electron density (with spin-resolved components)
    // rho_b: pseudocharge density (may be null)
    // rho_core: NLCC core density (may be null)
    // xc_type: XC functional type
    // exx_frac_scale: if > 0, scale exchange by (1 - exx_frac_scale)
    // arrays: work arrays (modified in place)
    void compute(const ElectronDensity& density,
                 const double* rho_b,
                 const double* rho_core,
                 XCType xc_type,
                 double exx_frac_scale,
                 double poisson_tol,
                 VeffArrays& arrays,
                 const double* tau = nullptr,
                 bool tau_valid = false);

    // Compute spinor Veff from noncollinear density (rho, mx, my, mz).
    void compute_spinor(const ElectronDensity& density,
                        const double* rho_b,
                        const double* rho_core,
                        XCType xc_type,
                        double poisson_tol,
                        VeffArrays& arrays);

    // --- Device-dispatching interfaces ---

    void compute(const ElectronDensity& density,
                 const double* rho_b,
                 const double* rho_core,
                 XCType xc_type,
                 double exx_frac_scale,
                 double poisson_tol,
                 VeffArrays& arrays,
                 Device dev,
                 const double* tau = nullptr,
                 bool tau_valid = false);

    void compute_spinor(const ElectronDensity& density,
                        const double* rho_b,
                        const double* rho_core,
                        XCType xc_type,
                        double poisson_tol,
                        VeffArrays& arrays,
                        Device dev);

#ifdef USE_CUDA
    void* gpu_state_raw_ = nullptr;  // Opaque pointer to GPUVeffState (defined in .cu)

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

    // Upload density from host to device buffers (for initial Veff computation).
    void upload_density(const ElectronDensity& density);

    // Download potential arrays from device to host VeffArrays (for Energy::compute_all).
    void download_to_host(VeffArrays& arrays);

    // Set device tau/vtau pointers for mGGA GPU pipeline.
    // Called from SCF after KineticEnergyDensity::compute() to wire device tau → XC.
    void set_device_tau(double* d_tau, double* d_vtau);
#endif

private:
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
