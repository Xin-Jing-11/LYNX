#pragma once

#include <vector>
#include <string>

namespace lynx {

// Stores pseudopotential data for one element type.
// Radial functions on a radial grid, KB projectors per angular momentum channel.
class Pseudopotential {
public:
    Pseudopotential() = default;

    // Load from psp8 format file
    void load_psp8(const std::string& filename);

    // Radial grid
    const std::vector<double>& radial_grid() const { return r_; }
    int grid_size() const { return static_cast<int>(r_.size()); }
    bool is_r_uniform() const { return is_r_uniform_; }

    // Local potential: rVloc(r) = r * Vloc(r)
    const std::vector<double>& rVloc() const { return rVloc_; }
    const std::vector<double>& rVloc_spline_d() const { return rVloc_d_; }
    double Vloc_0() const { return Vloc_0_; }

    // Nonlocal KB projectors
    int lmax() const { return lmax_; }
    int lloc() const { return lloc_; }  // local angular momentum channel
    const std::vector<int>& ppl() const { return ppl_; }  // projectors per l

    // UdV[l][p] = radial KB projector for channel (l, p), length = grid_size
    const std::vector<std::vector<std::vector<double>>>& UdV() const { return UdV_; }
    const std::vector<std::vector<std::vector<double>>>& UdV_spline_d() const { return UdV_d_; }

    // Gamma[l][p] = KB energy coefficient for channel (l, p)
    const std::vector<std::vector<double>>& Gamma() const { return Gamma_; }

    // Cutoff radii per l
    const std::vector<double>& rc() const { return rc_; }

    // Isolated atom density: rhoIsoAtom(r)
    const std::vector<double>& rho_iso_atom() const { return rho_iso_atom_; }
    const std::vector<double>& rho_iso_atom_spline_d() const { return rho_iso_atom_d_; }

    // Core charge for NLCC
    bool has_nlcc() const { return fchrg_ > 0.0 && !rho_c_.empty(); }
    double fchrg() const { return fchrg_; }
    const std::vector<double>& rho_c() const { return rho_c_; }
    const std::vector<double>& rho_c_table() const { return rho_c_; }
    const std::vector<double>& rho_c_spline_d() const { return rho_c_d_; }

    // Valence charge and XC info from pseudopotential
    double Zval() const { return Zval_; }
    int pspxc() const { return pspxc_; }

    // Total number of projectors per atom (excluding lloc)
    // = sum_{l != lloc} ppl[l] * (2*l + 1)
    int nproj_per_atom() const;

    // Compute spline derivatives for all radial functions
    void compute_splines();

    // Interpolate a radial function at distances r_interp
    static void spline_interp(const std::vector<double>& r_grid,
                              const std::vector<double>& f,
                              const std::vector<double>& f_d,
                              const std::vector<double>& r_interp,
                              std::vector<double>& f_interp);

    // Fast single-point spline interpolation (no heap allocation)
    static double spline_interp_single(const std::vector<double>& r_grid,
                                       const std::vector<double>& f,
                                       const std::vector<double>& f_d,
                                       double r);

private:
    std::vector<double> r_;                 // radial grid
    std::vector<double> rVloc_;             // r * Vloc
    std::vector<double> rVloc_d_;           // spline 1st derivative of rVloc
    double Vloc_0_ = 0.0;                  // Vloc(r=0)

    int lmax_ = -1;
    int lloc_ = 0;                          // local channel
    std::vector<int> ppl_;                  // projectors per l
    std::vector<std::vector<std::vector<double>>> UdV_;    // UdV[l][p][r]
    std::vector<std::vector<std::vector<double>>> UdV_d_;  // spline derivatives
    std::vector<std::vector<double>> Gamma_;               // Gamma[l][p]
    std::vector<double> rc_;                               // cutoff per l

    std::vector<double> rho_iso_atom_;
    std::vector<double> rho_iso_atom_d_;
    std::vector<double> rho_c_;
    std::vector<double> rho_c_d_;
    double fchrg_ = 0.0;
    double Zval_ = 0.0;
    int pspxc_ = 0;
    bool is_r_uniform_ = false;

    // Hermite cubic spline: compute first derivatives (matches reference LYNX getYD_gen)
    static void spline_deriv(const std::vector<double>& x,
                             const std::vector<double>& y,
                             std::vector<double>& yd);
};

} // namespace lynx
