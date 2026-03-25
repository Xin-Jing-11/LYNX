#pragma once

#include "core/types.hpp"
#include "core/Lattice.hpp"
#include <vector>

namespace lynx {

// K-point grid generation and time-reversal symmetry reduction.
// Matches reference LYNX Calculate_kpoints() exactly.
//
// K-points are stored in Cartesian reciprocal coordinates (2π/L units),
// matching reference LYNX convention: k_cart_i = k_red_i * 2π / L_i
class KPoints {
public:
    KPoints() = default;

    // Generate Monkhorst-Pack k-point grid with time-reversal symmetry reduction.
    // Kx, Ky, Kz: grid dimensions
    // shift: Monkhorst-Pack shift in each direction (0 or 0.5 typically)
    // lattice: for cell lengths (range_x, range_y, range_z)
    void generate(int Kx, int Ky, int Kz, Vec3 shift, const Lattice& lattice);

    // Number of k-points after symmetry reduction
    int Nkpts() const { return Nkpts_sym_; }

    // Total k-points before reduction (Kx * Ky * Kz)
    int Nkpts_full() const { return Nkpts_full_; }

    // K-point coordinates in Cartesian reciprocal space (2π/L units)
    // k_cart[i] = {k1[i], k2[i], k3[i]}
    const std::vector<Vec3>& kpts_cart() const { return kpts_cart_; }

    // K-point coordinates in reduced (fractional) reciprocal space [-0.5, 0.5)
    const std::vector<Vec3>& kpts_red() const { return kpts_red_; }

    // K-point weights (1.0 for unique, 2.0 for time-reversal paired)
    // Sum of weights = Nkpts_full
    const std::vector<double>& weights() const { return weights_; }

    // Normalized weights: weights[i] / Nkpts_full (sum = 1.0)
    std::vector<double> normalized_weights() const;

    // Check if this is a Gamma-only calculation (single k-point at origin)
    bool is_gamma_only() const;

    // Half-frequency k-points for exact exchange (EXX)
    // For PBE0/HSE: the full unreduced k-grid is used in the exchange sum.
    // Nkpts_hf = Nkpts_full (all k-points before symmetry reduction)
    int Nkpts_hf() const { return Nkpts_full_; }

    // kpthf_ind[i]: maps full k-point i to symmetry-reduced index
    const std::vector<int>& kpthf_ind() const { return kpthf_ind_; }

    // kpthf_pn[i]: 1 if full k-point i maps directly to sym-reduced k-point,
    //              0 if it maps via time reversal (need conjugate)
    const std::vector<int>& kpthf_pn() const { return kpthf_pn_; }

    // k-point Cartesian coordinates for the full unreduced grid
    const std::vector<Vec3>& kpts_hf_cart() const { return kpts_hf_cart_; }

private:
    int Nkpts_sym_ = 0;
    int Nkpts_full_ = 0;
    std::vector<Vec3> kpts_cart_;
    std::vector<Vec3> kpts_red_;
    std::vector<double> weights_;

    // EXX half-frequency data (populated in generate())
    std::vector<int> kpthf_ind_;       // full -> sym-reduced index map
    std::vector<int> kpthf_pn_;        // 1=direct, 0=time-reversed
    std::vector<Vec3> kpts_hf_cart_;   // full k-grid Cartesian coords
};

} // namespace lynx
