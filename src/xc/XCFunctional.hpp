#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "operators/Gradient.hpp"
#include "parallel/HaloExchange.hpp"

namespace lynx {

// Exchange-correlation functional evaluation using libxc.
// Supports LDA (Slater + PW92/PZ81) and GGA (PBE/PBEsol/RPBE).
class XCFunctional {
public:
    XCFunctional() = default;

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

    XCType type() const { return type_; }
    bool is_gga() const {
        return type_ == XCType::GGA_PBE || type_ == XCType::GGA_PBEsol || type_ == XCType::GGA_RPBE;
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

    // Get libxc functional IDs for current XC type
    void get_func_ids(int& xc_id, int& cc_id) const;
};

} // namespace lynx
