#pragma once

#include "core/types.hpp"
#include "core/DeviceArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "core/DeviceTag.hpp"
#include "operators/Gradient.hpp"
#include "parallel/HaloExchange.hpp"

namespace lynx {

// Exchange-correlation functional evaluation using libxc.
// Supports LDA (Slater + PW92/PZ81) and GGA (PBE/PBEsol/RPBE).
class XCFunctional {
public:
    XCFunctional() = default;
    ~XCFunctional();
    XCFunctional(XCFunctional&&) noexcept = default;
    XCFunctional& operator=(XCFunctional&&) noexcept = default;
    XCFunctional(const XCFunctional&) = delete;
    XCFunctional& operator=(const XCFunctional&) = delete;

    void setup(XCType type, const Domain& domain, const FDGrid& grid,
               const Gradient* gradient = nullptr,
               const HaloExchange* halo = nullptr);

    // Evaluate XC energy density and potential (non-spin-polarized).
    // rho: (Nd_d,) electron density
    // Vxc: (Nd_d,) output XC potential
    // exc: (Nd_d,) output XC energy density (per particle)
    // Dxcdgrho: (Nd_d,) output d(rho*exc)/d(|grad rho|^2) for GGA/mGGA (2*vsigma)
    // tau: (Nd_d,) input kinetic energy density (for mGGA)
    // vtau: (Nd_d,) output d(rho*exc)/d(tau) (for mGGA)
    void evaluate(const double* rho, double* Vxc, double* exc, int Nd_d,
                  double* Dxcdgrho = nullptr,
                  const double* tau = nullptr,
                  double* vtau = nullptr) const;

    // Evaluate for spin-polarized (collinear)
    // rho: [Nd_d*3] layout: rho[0..Nd_d-1] = total, rho[Nd_d..2*Nd_d-1] = up,
    //       rho[2*Nd_d..3*Nd_d-1] = down
    // Vxc: [Nd_d*2] layout: Vxc[0..Nd_d-1] = up, Vxc[Nd_d..2*Nd_d-1] = down
    // Dxcdgrho: [Nd_d*3] layout: [v2c | v2x_up | v2x_down]
    // tau: [Nd_d*3] layout: [total | up | down] (for mGGA)
    // vtau: [Nd_d*2] layout: [up | down] (for mGGA)
    void evaluate_spin(const double* rho, double* Vxc, double* exc, int Nd_d,
                       double* Dxcdgrho = nullptr,
                       const double* tau = nullptr,
                       double* vtau = nullptr) const;

    // ── Device-dispatching overloads ─────────────────────────────
    // Forward to CPU methods (Device::CPU) or GPU kernels (Device::GPU).

    void evaluate(const double* rho, double* Vxc, double* exc, int Nd_d,
                  Device dev,
                  double* Dxcdgrho = nullptr,
                  const double* tau = nullptr, double* vtau = nullptr) const;

    void evaluate_spin(const double* rho, double* Vxc, double* exc, int Nd_d,
                       Device dev,
                       double* Dxcdgrho = nullptr,
                       const double* tau = nullptr, double* vtau = nullptr) const;

#ifdef USE_CUDA
    void* gpu_state_raw_ = nullptr;  // Opaque pointer to GPUXCState (defined in .cu)
public:
    void setup_gpu(const class LynxContext& ctx, int Nspin);
    void cleanup_gpu();

    // Upload NLCC core density to GPU state (enables NLCC handling in GPU XC path).
    void set_gpu_nlcc(const double* rho_core, int Nd);

    // Set tau_valid flag on GPU state (for mGGA: true after first tau computation).
    void set_gpu_tau_valid(bool valid);
#endif

    // Set exchange scaling factor (1.0 = full exchange, 0.75 = PBE0 during Fock loop)
    void set_exchange_scale(double s) { exchange_scale_ = s; }

    XCType type() const { return type_; }
    bool is_gga() const {
        return type_ == XCType::GGA_PBE || type_ == XCType::GGA_PBEsol || type_ == XCType::GGA_RPBE
            || type_ == XCType::HYB_PBE0 || type_ == XCType::HYB_HSE;
    }
    bool is_mgga() const {
        return type_ == XCType::MGGA_SCAN || type_ == XCType::MGGA_RSCAN || type_ == XCType::MGGA_R2SCAN;
    }

private:
    XCType type_ = XCType::GGA_PBE;
    const Domain* domain_ = nullptr;
    const FDGrid* grid_ = nullptr;
    const Gradient* gradient_ = nullptr;
    const HaloExchange* halo_ = nullptr;
    double exchange_scale_ = 1.0;  // scaling for exchange (1-exx_frac for hybrids in Fock loop)

    // Get libxc functional IDs for current XC type
    void get_func_ids(int& xc_id, int& cc_id) const;
};

} // namespace lynx
