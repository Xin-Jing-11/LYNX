#include "atoms/Pseudopotential.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace sparc {

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
        std::getline(ifs, line);
        std::istringstream ss(line);
        ss >> zatom >> Zval_;
    }

    // Line 3: pspcod, pspxc, lmax, lloc, mmax, r2well
    int pspcod, mmax;
    double r2well;
    {
        std::getline(ifs, line);
        std::istringstream ss(line);
        ss >> pspcod >> pspxc_ >> lmax_ >> lloc_ >> mmax >> r2well;
    }

    if (pspcod != 8)
        throw std::runtime_error("Only psp8 format is supported, got pspcod=" + std::to_string(pspcod));

    // Line 4: rchrg, fchrg, qchrg (NLCC parameters)
    double rchrg, qchrg;
    {
        std::getline(ifs, line);
        std::istringstream ss(line);
        ss >> rchrg >> fchrg_ >> qchrg;
    }

    // Line 5: nproj per l channel (lmax+1 values, possibly more)
    {
        std::getline(ifs, line);
        std::istringstream ss(line);
        ppl_.resize(lmax_ + 1, 0);
        for (int l = 0; l <= lmax_; ++l) {
            ss >> ppl_[l];
        }
    }

    // Line 6: extension_switch
    std::getline(ifs, line);

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
        std::getline(ifs, line);
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
                std::getline(ifs, line);
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
                        UdV_[l][p][i] = 0.0;  // p_l(0) = 0 for l>0; finite for l=0 but r→0 term negligible
                    }
                }
            }
            have_rgrid = true;
        } else if (l == lloc_) {
            // Local channel: read mmax lines of Vloc data
            for (int i = 0; i < mmax; ++i) {
                std::getline(ifs, line);
                std::istringstream ss(line);
                int idx;
                double ri, vloc_val;
                ss >> idx >> ri >> vloc_val;
                if (!have_rgrid) r_[i] = ri;
                rVloc_[i] = ri * vloc_val;  // Store as r * Vloc
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
        std::getline(ifs, line);
        // l_value should be lloc_

        // Read mmax lines of Vloc data
        for (int i = 0; i < mmax; ++i) {
            std::getline(ifs, line);
            std::istringstream ss(line);
            int idx;
            double ri, vloc_val;
            ss >> idx >> ri >> vloc_val;
            if (!have_rgrid) r_[i] = ri;
            rVloc_[i] = ri * vloc_val;  // Store as r * Vloc
        }
        have_rgrid = true;
    }

    // Vloc(r=0): extrapolate from first/second point
    if (r_[0] < 1e-14 && mmax > 1) {
        Vloc_0_ = rVloc_[1] / r_[1];
    } else if (mmax > 0) {
        Vloc_0_ = rVloc_[0] / r_[0];
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

    // Try to read NLCC data (model core charge)
    if (fchrg_ > 1e-10) {
        rho_c_.resize(mmax, 0.0);
        for (int i = 0; i < mmax; ++i) {
            if (!std::getline(ifs, line)) break;
            std::istringstream ss(line);
            int idx;
            double ri;
            ss >> idx >> ri;
            double rho_c_val;
            ss >> rho_c_val;
            rho_c_[i] = rho_c_val / (4.0 * M_PI);
        }
    }

    // Try to read isolated atom density
    rho_iso_atom_.resize(mmax, 0.0);
    for (int i = 0; i < mmax; ++i) {
        if (!std::getline(ifs, line)) break;
        if (line.empty()) break;
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
    // Spline rVloc
    spline_deriv(r_, rVloc_, rVloc_d_);

    // Spline each projector
    for (int l = 0; l <= lmax_; ++l) {
        if (l == lloc_) continue;
        for (int p = 0; p < ppl_[l]; ++p) {
            spline_deriv(r_, UdV_[l][p], UdV_d_[l][p]);
        }
    }

    // Spline isolated atom density if available
    if (!rho_iso_atom_.empty())
        spline_deriv(r_, rho_iso_atom_, rho_iso_atom_d_);

    // Spline core charge if available
    if (!rho_c_.empty())
        spline_deriv(r_, rho_c_, rho_c_d_);
}

void Pseudopotential::spline_deriv(const std::vector<double>& x,
                                   const std::vector<double>& y,
                                   std::vector<double>& y2) {
    int n = static_cast<int>(x.size());
    y2.resize(n, 0.0);
    if (n < 3) return;

    std::vector<double> u(n, 0.0);

    // Natural spline: y2[0] = y2[n-1] = 0
    y2[0] = 0.0;
    u[0] = 0.0;

    for (int i = 1; i < n - 1; ++i) {
        double sig = (x[i] - x[i-1]) / (x[i+1] - x[i-1]);
        double p = sig * y2[i-1] + 2.0;
        y2[i] = (sig - 1.0) / p;
        u[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
             - (y[i] - y[i-1]) / (x[i] - x[i-1]);
        u[i] = (6.0 * u[i] / (x[i+1] - x[i-1]) - sig * u[i-1]) / p;
    }

    y2[n-1] = 0.0;
    for (int k = n - 2; k >= 0; --k)
        y2[k] = y2[k] * y2[k+1] + u[k];
}

void Pseudopotential::spline_interp(const std::vector<double>& r_grid,
                                    const std::vector<double>& f,
                                    const std::vector<double>& f_d,
                                    const std::vector<double>& r_interp,
                                    std::vector<double>& f_interp) {
    int n = static_cast<int>(r_grid.size());
    int m = static_cast<int>(r_interp.size());
    f_interp.resize(m);

    for (int j = 0; j < m; ++j) {
        double r = r_interp[j];

        // Clamp to grid range
        if (r <= r_grid[0]) {
            f_interp[j] = f[0];
            continue;
        }
        if (r >= r_grid[n - 1]) {
            f_interp[j] = f[n - 1];
            continue;
        }

        // Binary search for interval
        int lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (r_grid[mid] > r) hi = mid;
            else lo = mid;
        }

        double h = r_grid[hi] - r_grid[lo];
        if (h < 1e-30) {
            f_interp[j] = f[lo];
            continue;
        }
        double a = (r_grid[hi] - r) / h;
        double b = (r - r_grid[lo]) / h;
        f_interp[j] = a * f[lo] + b * f[hi]
                     + ((a * a * a - a) * f_d[lo] + (b * b * b - b) * f_d[hi]) * h * h / 6.0;
    }
}

} // namespace sparc
