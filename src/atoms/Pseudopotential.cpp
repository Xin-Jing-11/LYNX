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

    // Lines 4..4+lmax: for each l, read rchrg, e99.9, nprj
    ppl_.resize(lmax_ + 1, 0);
    rc_.resize(lmax_ + 1, 0.0);
    Gamma_.resize(lmax_ + 1);
    UdV_.resize(lmax_ + 1);
    UdV_d_.resize(lmax_ + 1);

    for (int l = 0; l <= lmax_; ++l) {
        std::getline(ifs, line);
        std::istringstream ss(line);
        int nprj;
        double rcl;
        ss >> rcl;
        // skip e99.9 and e99.0
        double e1, e2;
        ss >> e1 >> e2 >> nprj;
        rc_[l] = rcl;
        ppl_[l] = nprj;
        Gamma_[l].resize(nprj, 0.0);
        UdV_[l].resize(nprj);
        UdV_d_[l].resize(nprj);

        // Read ekb values (projector energies = Gamma)
        if (nprj > 0) {
            std::getline(ifs, line);
            std::istringstream ss2(line);
            for (int p = 0; p < nprj; ++p)
                ss2 >> Gamma_[l][p];
        }
    }

    // Read optional header lines until we hit the radial data
    // Line: npts (number of radial grid points)
    // Actually in psp8, next is the radial grid data
    // Format: each l channel has mmax lines of: index r Vloc/projector data

    // Read local potential
    r_.resize(mmax);
    rVloc_.resize(mmax);

    for (int i = 0; i < mmax; ++i) {
        std::getline(ifs, line);
        std::istringstream ss(line);
        int idx;
        double ri, vloc_val;
        ss >> idx >> ri >> vloc_val;
        r_[i] = ri;
        rVloc_[i] = vloc_val;  // psp8 stores r*Vloc already
    }

    // Vloc(r=0): extrapolate or use first point
    if (r_[0] < 1e-14) {
        Vloc_0_ = rVloc_[1] / r_[1];  // Use second point
    } else {
        Vloc_0_ = rVloc_[0] / r_[0];
    }

    // Read projectors for each l
    for (int l = 0; l <= lmax_; ++l) {
        if (l == lloc_) continue;
        for (int p = 0; p < ppl_[l]; ++p) {
            UdV_[l][p].resize(mmax);
            for (int i = 0; i < mmax; ++i) {
                std::getline(ifs, line);
                std::istringstream ss(line);
                int idx;
                double ri, proj_val;
                ss >> idx >> ri >> proj_val;
                UdV_[l][p][i] = proj_val;
            }
        }
    }

    // Check if radial grid is uniform
    if (r_.size() > 2) {
        double dr0 = r_[1] - r_[0];
        is_r_uniform_ = true;
        for (size_t i = 2; i < r_.size(); ++i) {
            if (std::abs((r_[i] - r_[i-1]) - dr0) > 1e-10 * dr0) {
                is_r_uniform_ = false;
                break;
            }
        }
    }

    // Try to read isolated atom density (may not be present)
    rho_iso_atom_.resize(mmax, 0.0);
    // psp8 may have additional data blocks — skip for now

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
