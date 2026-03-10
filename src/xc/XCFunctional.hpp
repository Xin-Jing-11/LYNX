#pragma once

#include "core/types.hpp"
#include "core/NDArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "operators/Gradient.hpp"
#include "parallel/HaloExchange.hpp"

namespace sparc {

// Exchange-correlation functional evaluation.
// Ported from SPARC reference: exchangeCorrelation.c
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
    void evaluate(const double* rho, double* Vxc, double* exc, int Nd_d) const;

    // Evaluate for spin-polarized (collinear)
    // rho: [DMnd*3] layout: rho[0..DMnd-1] = total, rho[DMnd..2*DMnd-1] = up,
    //       rho[2*DMnd..3*DMnd-1] = down  (matching reference SPARC convention)
    // Vxc: [DMnd*2] layout: Vxc[0..DMnd-1] = up, Vxc[DMnd..2*DMnd-1] = down
    void evaluate_spin(const double* rho, double* Vxc, double* exc, int Nd_d) const;

    XCType type() const { return type_; }
    bool is_gga() const {
        return type_ == XCType::GGA_PBE || type_ == XCType::GGA_PBEsol || type_ == XCType::GGA_RPBE;
    }

private:
    XCType type_ = XCType::GGA_PBE;
    const Domain* domain_ = nullptr;
    const FDGrid* grid_ = nullptr;
    const Gradient* gradient_ = nullptr;
    const HaloExchange* halo_ = nullptr;

    // --- LDA functionals (matching reference SPARC exactly) ---

    // Slater exchange: ex(i) = -C2 * rho^(1/3), vx(i) = -C3 * rho^(1/3)
    static void slater(int DMnd, const double* rho, double* ex, double* vx);

    // PW92 correlation: J.P. Perdew and Y. Wang, PRB 45, 13244 (1992)
    static void pw(int DMnd, const double* rho, double* ec, double* vc);

    // PZ81 correlation: J.P. Perdew and A. Zunger, PRB 23, 5048 (1981)
    static void pz(int DMnd, const double* rho, double* ec, double* vc);

    // Spin-polarized versions
    static void slater_spin(int DMnd, const double* rho, double* ex, double* vx);
    static void pw_spin(int DMnd, const double* rho, double* ec, double* vc);

    // --- GGA functionals (matching reference SPARC exactly) ---

    // PBE exchange: iflag=1(PBE), 2(PBEsol), 3(RPBE), 4(ZY-revPBE)
    // ex: energy density per particle
    // vx: d(rho*ex)/drho part of potential
    // v2x: d(rho*ex)/d(|grad rho|^2) — the gradient-dependent part
    static void pbex(int DMnd, const double* rho, const double* sigma,
                     int iflag, double* ex, double* vx, double* v2x);

    // PBE correlation
    static void pbec(int DMnd, const double* rho, const double* sigma,
                     int iflag, double* ec, double* vc, double* v2c);

    // Spin-polarized GGA
    static void pbex_spin(int DMnd, const double* rho, const double* sigma,
                          int iflag, double* ex, double* vx, double* v2x);
    static void pbec_spin(int DMnd, const double* rho, const double* sigma,
                          int iflag, double* ec, double* vc, double* v2c);

    // Assemble GGA potential including divergence correction
    void apply_gga(const double* rho, double* Vxc, double* exc,
                   const double* Drho_x, const double* Drho_y, const double* Drho_z,
                   const double* v2xc, int Nd_d) const;

    // Get iflag for current XC type
    int get_pbe_iflag() const;
};

} // namespace sparc
