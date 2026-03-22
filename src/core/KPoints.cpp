#include "core/KPoints.hpp"
#include "core/constants.hpp"
#include <cmath>
#include <cstdio>

namespace lynx {

void KPoints::generate(int Kx, int Ky, int Kz, Vec3 shift, const Lattice& lattice) {
    Nkpts_full_ = Kx * Ky * Kz;

    Vec3 L = lattice.lengths();
    double sumx = 2.0 * constants::PI / L.x;
    double sumy = 2.0 * constants::PI / L.y;
    double sumz = 2.0 * constants::PI / L.z;

    // Tolerance for k-point comparison (matches reference TEMP_TOL)
    constexpr double TOL = 1e-12;

    // Pre-allocate for worst case (no reduction)
    kpts_cart_.clear();
    kpts_red_.clear();
    weights_.clear();
    kpts_cart_.reserve(Nkpts_full_);
    kpts_red_.reserve(Nkpts_full_);
    weights_.reserve(Nkpts_full_);

    // M-P grid index ranges (matching reference ABINIT convention)
    int nk1_s = -static_cast<int>(std::floor((Kx - 1) / 2.0));
    int nk1_e = nk1_s + Kx;
    int nk2_s = -static_cast<int>(std::floor((Ky - 1) / 2.0));
    int nk2_e = nk2_s + Ky;
    int nk3_s = -static_cast<int>(std::floor((Kz - 1) / 2.0));
    int nk3_e = nk3_s + Kz;

    int k = 0;  // count of unique k-points

    for (int nk1 = nk1_s; nk1 < nk1_e; ++nk1) {
        for (int nk2 = nk2_s; nk2 < nk2_e; ++nk2) {
            for (int nk3 = nk3_s; nk3 < nk3_e; ++nk3) {
                // Reduced (fractional) k-point coordinates
                double k1_red = nk1 * 1.0 / Kx;
                double k2_red = nk2 * 1.0 / Ky;
                double k3_red = nk3 * 1.0 / Kz;

                // Apply shift and fold to [-0.5, 0.5) (matching reference exactly)
                k1_red = std::fmod(k1_red + shift.x / Kx + 0.5 - TOL, 1.0) - 0.5 + TOL;
                k2_red = std::fmod(k2_red + shift.y / Ky + 0.5 - TOL, 1.0) - 0.5 + TOL;
                k3_red = std::fmod(k3_red + shift.z / Kz + 0.5 - TOL, 1.0) - 0.5 + TOL;

                // Convert to Cartesian reciprocal coordinates
                double k1 = k1_red * 2.0 * constants::PI / L.x;
                double k2 = k2_red * 2.0 * constants::PI / L.y;
                double k3 = k3_red * 2.0 * constants::PI / L.z;

                // Time-reversal symmetry: check if -k is already stored
                bool found = false;
                int found_idx = -1;
                for (int nk = 0; nk < k; ++nk) {
                    bool match_x = (std::fabs(k1 + kpts_cart_[nk].x) < TOL) ||
                                   (std::fabs(k1 + kpts_cart_[nk].x - sumx) < TOL);
                    bool match_y = (std::fabs(k2 + kpts_cart_[nk].y) < TOL) ||
                                   (std::fabs(k2 + kpts_cart_[nk].y - sumy) < TOL);
                    bool match_z = (std::fabs(k3 + kpts_cart_[nk].z) < TOL) ||
                                   (std::fabs(k3 + kpts_cart_[nk].z - sumz) < TOL);
                    if (match_x && match_y && match_z) {
                        found = true;
                        found_idx = nk;
                        break;
                    }
                }

                if (found) {
                    // Time-reversal partner found: double its weight
                    weights_[found_idx] = 2.0;
                } else {
                    // New unique k-point
                    kpts_cart_.push_back({k1, k2, k3});
                    kpts_red_.push_back({k1_red, k2_red, k3_red});
                    weights_.push_back(1.0);
                    k++;
                }
            }
        }
    }

    Nkpts_sym_ = k;
}

std::vector<double> KPoints::normalized_weights() const {
    std::vector<double> nw(Nkpts_sym_);
    double inv_nk = (Nkpts_full_ > 0) ? 1.0 / Nkpts_full_ : 1.0;
    for (int i = 0; i < Nkpts_sym_; ++i) {
        nw[i] = weights_[i] * inv_nk;
    }
    return nw;
}

bool KPoints::is_gamma_only() const {
    if (Nkpts_sym_ != 1) return false;
    constexpr double tol = 1e-12;
    return std::fabs(kpts_cart_[0].x) < tol &&
           std::fabs(kpts_cart_[0].y) < tol &&
           std::fabs(kpts_cart_[0].z) < tol;
}

void KPoints::setup_hf_kpoints() {
    // The HF k-point grid is the FULL unreduced BZ (all Nkpts_full points).
    // These are already generated during generate() but only the reduced set is stored.
    // We need to regenerate the full set and map each to the reduced set.

    Nkpts_hf_ = Nkpts_full_;
    kpts_hf_cart_.clear();
    kpts_hf_cart_.reserve(Nkpts_full_);
    kpthf_ind_.resize(Nkpts_full_);
    kpthf_pn_.resize(Nkpts_full_);

    constexpr double TOL = 1e-12;

    // We need the lattice lengths to compute sumx/sumy/sumz for wrapping
    // These are stored implicitly: kpts_cart_[i].x = k_red * 2π/Lx
    // sumx = 2π/Lx. We can recover from the k-point generation.
    // Actually, the full k-points are exactly the ones we iterated over in generate().
    // Let's just regenerate them.

    // But we don't have Kx/Ky/Kz or shift here. We need them.
    // The simplest fix: store them during generate() and reuse.
    // For now, reconstruct from Nkpts_full and the stored data.

    // Actually, the HF k-points = ALL k-points before time-reversal reduction.
    // They are the same Monkhorst-Pack grid. Each k-point in the full grid
    // either matches a sym-reduced k-point directly (pn=1) or its negative
    // matches one (pn=0).

    // We stored the reduced set. For each full BZ point, find which reduced
    // point it maps to.

    // Problem: we don't store the full BZ points. Let me just store them
    // during generate(). I'll add a member for the full list.

    // For now, a simpler approach: the full BZ points are exactly the
    // reduced points PLUS their time-reversal partners. For each reduced
    // k-point with weight=2, there are TWO full BZ points: k and -k.
    // For weight=1, there's ONE.

    int idx = 0;
    for (int i = 0; i < Nkpts_sym_; i++) {
        // The reduced k-point itself
        kpts_hf_cart_.push_back(kpts_cart_[i]);
        kpthf_ind_[idx] = i;
        kpthf_pn_[idx] = 1;  // direct match
        idx++;

        // If weight=2, add the time-reversal partner -k
        if (std::fabs(weights_[i] - 2.0) < 0.1) {
            Vec3 mk = {-kpts_cart_[i].x, -kpts_cart_[i].y, -kpts_cart_[i].z};
            kpts_hf_cart_.push_back(mk);
            kpthf_ind_[idx] = i;      // maps to same reduced index
            kpthf_pn_[idx] = 0;       // time-reversed, need conj
            idx++;
        }
    }

    // Verify count
    if (idx != Nkpts_full_) {
        std::fprintf(stderr, "WARNING: HF k-point count mismatch: %d vs expected %d\n", idx, Nkpts_full_);
        Nkpts_hf_ = idx;
    }
}

} // namespace lynx
