#include "atoms/Pseudopotential.hpp"
#include "core/NumericalMethods.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace lynx {

// Replace Fortran 'D' exponent notation with 'E' for C++ parsing
// e.g. "1.234D+05" -> "1.234E+05"
static void fix_fortran_exp(std::string& s) {
    for (auto& c : s) {
        if (c == 'D' || c == 'd') c = 'E';
    }
}

int Pseudopotential::nproj_per_atom() const {
    int total = 0;
    for (int l = 0; l <= lmax_; ++l) {
        if (l == lloc_) continue;
        total += ppl_[l] * (2 * l + 1);
    }
    return total;
}

void Pseudopotential::load_psp8(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open())
        throw std::runtime_error("Cannot open pseudopotential file: " + filename);

    std::string line;

    // Line 1: comment (element name etc.)
    std::getline(ifs, line);

    // Line 2: zatom, zion, pspd (date)
    double zatom;
    {
        std::getline(ifs, line); fix_fortran_exp(line);
        std::istringstream ss(line);
        ss >> zatom >> Zval_;
    }

    // Line 3: pspcod, pspxc, lmax, lloc, mmax, r2well
    int pspcod, mmax;
    double r2well;
    {
        std::getline(ifs, line); fix_fortran_exp(line);
        std::istringstream ss(line);
        ss >> pspcod >> pspxc_ >> lmax_ >> lloc_ >> mmax >> r2well;
    }

    if (pspcod != 8)
        throw std::runtime_error("Only psp8 format is supported, got pspcod=" + std::to_string(pspcod));

    // Line 4: rchrg, fchrg, qchrg (NLCC parameters)
    double rchrg, qchrg;
    {
        std::getline(ifs, line); fix_fortran_exp(line);
        std::istringstream ss(line);
        ss >> rchrg >> fchrg_ >> qchrg;
    }

    // Line 5: nproj per l channel (lmax+1 values, possibly more)
    {
        std::getline(ifs, line); fix_fortran_exp(line);
        std::istringstream ss(line);
        ppl_.resize(lmax_ + 1, 0);
        for (int l = 0; l <= lmax_; ++l) {
            ss >> ppl_[l];
        }
    }

    // Line 6: extension_switch
    int extension_switch = 0;
    {
        std::getline(ifs, line); fix_fortran_exp(line);
        std::istringstream ss(line);
        ss >> extension_switch;
    }

    // Line 7 (if extension_switch >= 2): nproj_soc per l channel
    // This line exists in FR pseudopotentials and must be consumed before channel data
    // Note: extension_switch=1 just means an extra comment line was present (no SOC data)
    std::vector<int> nprojso;
    if (extension_switch >= 2) {
        std::getline(ifs, line); fix_fortran_exp(line);
        std::istringstream ss(line);
        nprojso.resize(lmax_ + 1, 0);
        // nprojso values are for l=1..lmax (l=0 has no SOC)
        for (int l = 1; l <= lmax_; ++l) {
            if (!(ss >> nprojso[l])) break;
        }
    }

    // Initialize storage
    rc_.resize(lmax_ + 1, 0.0);
    Gamma_.resize(lmax_ + 1);
    UdV_.resize(lmax_ + 1);
    UdV_d_.resize(lmax_ + 1);

    // Read radial grid and potential data per channel
    r_.resize(mmax);
    rVloc_.resize(mmax);
    bool have_rgrid = false;

    // Process each angular momentum channel
    // First come the nonlocal channels (l=0..lmax where l != lloc),
    // then the local channel if lloc > lmax (it comes after lmax channels)
    for (int l = 0; l <= lmax_; ++l) {
        int nprj = ppl_[l];
        Gamma_[l].resize(nprj, 0.0);
        UdV_[l].resize(nprj);
        UdV_d_[l].resize(nprj);

        // Read l header line: l_value [ekb values if nonlocal]
        std::getline(ifs, line); fix_fortran_exp(line);
        std::istringstream hdr(line);
        int l_read;
        hdr >> l_read;

        if (l != lloc_ && nprj > 0) {
            // Read Gamma (ekb) values from the same header line
            for (int p = 0; p < nprj; ++p) {
                hdr >> Gamma_[l][p];
            }

            // Read mmax lines of radial data: index, r, u0(r), u1(r), ...
            for (int i = 0; i < mmax; ++i) {
                std::getline(ifs, line); fix_fortran_exp(line);
                std::istringstream ss(line);
                int idx;
                double ri;
                ss >> idx >> ri;
                if (!have_rgrid) r_[i] = ri;

                for (int p = 0; p < nprj; ++p) {
                    double proj_val;
                    ss >> proj_val;
                    UdV_[l][p].resize(mmax);
                    // psp8 stores r*p_l(r); divide by r to get p_l(r)
                    if (ri > 1e-10) {
                        UdV_[l][p][i] = proj_val / ri;
                    } else {
                        UdV_[l][p][i] = 0.0;  // Will be fixed below
                    }
                }
            }
            // Fix r=0 boundary: copy from r=dr (matching reference LYNX)
            // Critical for l=0 projectors which are non-zero at the origin
            for (int p = 0; p < nprj; ++p) {
                if (mmax > 1) {
                    UdV_[l][p][0] = UdV_[l][p][1];
                }
            }
            have_rgrid = true;
        } else if (l == lloc_) {
            // Local channel: read mmax lines of Vloc data
            for (int i = 0; i < mmax; ++i) {
                std::getline(ifs, line); fix_fortran_exp(line);
                std::istringstream ss(line);
                int idx;
                double ri, vloc_val;
                ss >> idx >> ri >> vloc_val;
                if (!have_rgrid) r_[i] = ri;
                rVloc_[i] = ri * vloc_val;  // Store as r * Vloc
                if (i == 0) Vloc_0_ = vloc_val;  // Store Vloc(r=0) directly from file
            }
            have_rgrid = true;
        } else {
            // l is nonlocal but has 0 projectors
            // Still need to read the radial data (if any)
            // In psp8, channels with 0 projectors don't have data
            // The header line was already read above
        }
    }

    // If lloc > lmax, the local potential data comes after all nonlocal channels
    if (lloc_ > lmax_) {
        // Read local channel header (just the l value)
        std::getline(ifs, line); fix_fortran_exp(line);
        // l_value should be lloc_

        // Read mmax lines of Vloc data
        for (int i = 0; i < mmax; ++i) {
            std::getline(ifs, line); fix_fortran_exp(line);
            std::istringstream ss(line);
            int idx;
            double ri, vloc_val;
            ss >> idx >> ri >> vloc_val;
            if (!have_rgrid) r_[i] = ri;
            rVloc_[i] = ri * vloc_val;  // Store as r * Vloc
            if (i == 0) Vloc_0_ = vloc_val;  // Store Vloc(r=0) directly from file
        }
        have_rgrid = true;
    }

    // Vloc_0 fallback: if not set above (shouldn't happen for valid psp8 files)
    if (Vloc_0_ == 0.0 && mmax > 1) {
        if (r_[0] < 1e-14) {
            Vloc_0_ = rVloc_[1] / r_[1];
        } else {
            Vloc_0_ = rVloc_[0] / r_[0];
        }
    }

    // Read rc values from the r_core comment in line 1
    // If not found there, estimate from the radial grid
    // For now, set rc to the extent where the projectors decay
    for (int l = 0; l <= lmax_; ++l) {
        if (ppl_[l] > 0 && l != lloc_) {
            // Find the last nonzero point of the projector
            rc_[l] = r_.back();
            for (int i = mmax - 1; i >= 0; --i) {
                if (std::abs(UdV_[l][0][i]) > 1e-12) {
                    rc_[l] = r_[i];
                    break;
                }
            }
        } else {
            rc_[l] = 0.0;
        }
    }

    // Parse r_core from line 1 if available
    // Line 1 format: "Si    ONCVPSP-4.0.1  r_core=   1.91059   1.91059   1.91059"
    // We saved it in the first read

    // Read SOC projectors if extension_switch >= 2
    // NOTE: in psp8 format, SOC data comes BEFORE NLCC and isolated atom density
    if (extension_switch >= 2 && lmax_ >= 1) {
        has_soc_ = true;
        ppl_soc_.resize(lmax_ + 1, 0);
        Gamma_soc_.resize(lmax_ + 1);
        UdV_soc_.resize(lmax_ + 1);
        UdV_soc_d_.resize(lmax_ + 1);

        // SOC data: for each l from 1 to lmax, read header + mmax lines
        // l=0 has no SOC (no spin-orbit for s orbitals)
        for (int l = 1; l <= lmax_; ++l) {
            int nprj = nprojso[l];  // SOC projectors per l from the nprojso line
            ppl_soc_[l] = nprj;
            Gamma_soc_[l].resize(nprj, 0.0);
            UdV_soc_[l].resize(nprj);
            UdV_soc_d_[l].resize(nprj);

            if (nprj == 0) continue;

            // Read SOC header line: l_value [ekb_soc values]
            std::getline(ifs, line); fix_fortran_exp(line);
            std::istringstream hdr(line);
            int l_read;
            hdr >> l_read;
            for (int p = 0; p < nprj; ++p) {
                hdr >> Gamma_soc_[l][p];
            }

            // Read mmax lines: index, r, soc_proj_0(r), soc_proj_1(r), ...
            for (int i = 0; i < mmax; ++i) {
                std::getline(ifs, line); fix_fortran_exp(line);
                std::istringstream ss(line);
                int idx;
                double ri;
                ss >> idx >> ri;

                for (int p = 0; p < nprj; ++p) {
                    double proj_val;
                    ss >> proj_val;
                    UdV_soc_[l][p].resize(mmax);
                    // Divide by r (same convention as standard projectors)
                    if (ri > 1e-10) {
                        UdV_soc_[l][p][i] = proj_val / ri;
                    } else {
                        UdV_soc_[l][p][i] = 0.0;
                    }
                }
            }
            // Fix r=0 boundary
            for (int p = 0; p < nprj; ++p) {
                if (mmax > 1) {
                    UdV_soc_[l][p][0] = UdV_soc_[l][p][1];
                }
            }
        }
    }

    // Read NLCC data (model core charge) — comes after SOC in psp8 format
    if (fchrg_ > 1e-10) {
        rho_c_.resize(mmax, 0.0);
        for (int i = 0; i < mmax; ++i) {
            if (!std::getline(ifs, line)) break;
            fix_fortran_exp(line);
            std::istringstream ss(line);
            int idx;
            double ri;
            ss >> idx >> ri;
            double rho_c_val;
            ss >> rho_c_val;
            rho_c_[i] = rho_c_val / (4.0 * M_PI);
        }
    }

    // Read isolated atom density — comes after NLCC in psp8 format
    rho_iso_atom_.resize(mmax, 0.0);
    for (int i = 0; i < mmax; ++i) {
        if (!std::getline(ifs, line)) break;
        if (line.empty()) break;
        fix_fortran_exp(line);
        std::istringstream ss(line);
        int idx;
        double ri;
        ss >> idx >> ri;
        double rho_val;
        ss >> rho_val;
        rho_iso_atom_[i] = rho_val / (4.0 * M_PI);
    }

    // Check if radial grid is uniform
    if (r_.size() > 2) {
        double dr0 = r_[1] - r_[0];
        is_r_uniform_ = true;
        for (size_t i = 2; i < r_.size(); ++i) {
            if (std::abs((r_[i] - r_[i-1]) - dr0) > 1e-10 * std::max(1.0, dr0)) {
                is_r_uniform_ = false;
                break;
            }
        }
    }

    compute_splines();
}

void Pseudopotential::compute_splines() {
    // Helper: compute spline derivatives using NumericalMethods
    auto setup_deriv = [](const std::vector<double>& x, const std::vector<double>& y,
                          std::vector<double>& yd) {
        SplineData s = spline_setup(x, y);
        yd = std::move(s.dydx);
    };

    // Spline rVloc
    setup_deriv(r_, rVloc_, rVloc_d_);

    // Spline each projector
    for (int l = 0; l <= lmax_; ++l) {
        if (l == lloc_) continue;
        for (int p = 0; p < ppl_[l]; ++p) {
            setup_deriv(r_, UdV_[l][p], UdV_d_[l][p]);
        }
    }

    // Spline SOC projectors
    if (has_soc_) {
        for (int l = 1; l <= lmax_; ++l) {
            for (int p = 0; p < ppl_soc_[l]; ++p) {
                setup_deriv(r_, UdV_soc_[l][p], UdV_soc_d_[l][p]);
            }
        }
    }

    // Spline isolated atom density if available
    if (!rho_iso_atom_.empty())
        setup_deriv(r_, rho_iso_atom_, rho_iso_atom_d_);

    // Spline core charge if available
    if (!rho_c_.empty())
        setup_deriv(r_, rho_c_, rho_c_d_);
}

// Compute first derivatives at node points — delegates to NumericalMethods
void Pseudopotential::spline_deriv(const std::vector<double>& x,
                                   const std::vector<double>& y,
                                   std::vector<double>& yd) {
    SplineData s = spline_setup(x, y);
    yd = std::move(s.dydx);
}

// Hermite cubic spline interpolation — delegates to NumericalMethods
void Pseudopotential::spline_interp(const std::vector<double>& r_grid,
                                    const std::vector<double>& f,
                                    const std::vector<double>& yd,
                                    const std::vector<double>& r_interp,
                                    std::vector<double>& f_interp) {
    // Build SplineData from pre-computed components (no re-computation of derivatives)
    SplineData spline;
    spline.x = r_grid;
    spline.y = f;
    spline.dydx = yd;

    int m = static_cast<int>(r_interp.size());
    f_interp.resize(m);
    spline_eval_array(spline, r_interp.data(), f_interp.data(), m);
}

double Pseudopotential::spline_interp_single(const std::vector<double>& r_grid,
                                              const std::vector<double>& f,
                                              const std::vector<double>& yd,
                                              double r) {
    // Build SplineData from pre-computed components (no re-computation of derivatives)
    SplineData spline;
    spline.x = r_grid;
    spline.y = f;
    spline.dydx = yd;

    return spline_eval(spline, r);
}

} // namespace lynx
