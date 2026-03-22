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

    // --- HF k-point grid for exact exchange ---
    // Setup the full unreduced HF k-point grid and mapping to sym-reduced set.
    // Called after generate(). Nkpts_hf = Nkpts_full (no downsampling).
    void setup_hf_kpoints();

    // Number of full (unreduced) HF k-points
    int Nkpts_hf() const { return Nkpts_hf_; }
    // HF k-points in Cartesian reciprocal space
    const std::vector<Vec3>& kpts_hf_cart() const { return kpts_hf_cart_; }
    // Maps HF k-point index -> sym-reduced k-point index (for psi/occ lookup)
    const std::vector<int>& kpthf_ind() const { return kpthf_ind_; }
    // +1 if k_hf matches k_sym directly, 0 if k_hf = -k_sym (need conj)
    const std::vector<int>& kpthf_pn() const { return kpthf_pn_; }

private:
    int Nkpts_sym_ = 0;
    int Nkpts_full_ = 0;
    std::vector<Vec3> kpts_cart_;
    std::vector<Vec3> kpts_red_;
    std::vector<double> weights_;

    // HF k-point grid
    int Nkpts_hf_ = 0;
    std::vector<Vec3> kpts_hf_cart_;
    std::vector<int> kpthf_ind_;   // [Nkpts_hf] -> index in sym-reduced set
    std::vector<int> kpthf_pn_;    // [Nkpts_hf] -> 1 if direct, 0 if time-reversed
};

} // namespace lynx
