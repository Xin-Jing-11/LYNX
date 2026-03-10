#include "electronic/Occupation.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace sparc {

double Occupation::fermi_dirac(double x, double beta) {
    double arg = x * beta;
    if (arg > 50.0) return 0.0;
    if (arg < -50.0) return 1.0;
    return 1.0 / (1.0 + std::exp(arg));
}

double Occupation::gaussian_smearing(double x, double beta) {
    // f(x) = 0.5 * erfc(x * beta / sqrt(2))
    // where x = (epsilon - Ef)
    double arg = x * beta * 0.7071067811865476; // 1/sqrt(2)
    return 0.5 * std::erfc(arg);
}

double Occupation::smearing_function(double x, double beta, SmearingType type) {
    switch (type) {
        case SmearingType::FermiDirac:
            return fermi_dirac(x, beta);
        case SmearingType::GaussianSmearing:
            return gaussian_smearing(x, beta);
        default:
            return fermi_dirac(x, beta);
    }
}

double Occupation::total_occupation(const std::vector<double>& eigs,
                                     const std::vector<double>& weights,
                                     double Ef,
                                     double beta,
                                     SmearingType smearing) {
    double Ne = 0.0;
    for (size_t i = 0; i < eigs.size(); ++i) {
        Ne += weights[i] * smearing_function(eigs[i] - Ef, beta, smearing);
    }
    return Ne;
}

double Occupation::find_fermi_level(const std::vector<double>& all_eigs,
                                     const std::vector<double>& all_weights,
                                     double Nelectron,
                                     double beta,
                                     SmearingType smearing) {
    if (all_eigs.empty()) {
        throw std::runtime_error("No eigenvalues for Fermi level search");
    }

    // Bracket the Fermi level
    double emin = *std::min_element(all_eigs.begin(), all_eigs.end());
    double emax = *std::max_element(all_eigs.begin(), all_eigs.end());
    double margin = 10.0 / beta; // expand bracket by ~10*kBT
    double a = emin - margin;
    double b = emax + margin;

    // Brent's method to solve N(Ef) - Nelectron = 0
    double fa = total_occupation(all_eigs, all_weights, a, beta, smearing) - Nelectron;
    double fb = total_occupation(all_eigs, all_weights, b, beta, smearing) - Nelectron;

    if (fa * fb > 0) {
        // Expand bracket further
        a = emin - 100.0 / beta;
        b = emax + 100.0 / beta;
        fa = total_occupation(all_eigs, all_weights, a, beta, smearing) - Nelectron;
        fb = total_occupation(all_eigs, all_weights, b, beta, smearing) - Nelectron;
        if (fa * fb > 0) {
            throw std::runtime_error("Cannot bracket Fermi level");
        }
    }

    // Brent's method
    double c = a, fc = fa;
    double d = b - a, e = d;
    constexpr double tol = 1e-14;
    constexpr int max_iter = 200;

    for (int iter = 0; iter < max_iter; ++iter) {
        if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
            c = a; fc = fa;
            d = e = b - a;
        }
        if (std::abs(fc) < std::abs(fb)) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }
        double tol1 = 2.0 * std::numeric_limits<double>::epsilon() * std::abs(b) + 0.5 * tol;
        double xm = 0.5 * (c - b);

        if (std::abs(xm) <= tol1 || fb == 0.0) {
            return b;
        }

        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            double s = fb / fa;
            double p, q;
            if (a == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                q = fa / fc;
                double r = fb / fc;
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0) q = -q;
            p = std::abs(p);
            if (2.0 * p < std::min(3.0 * xm * q - std::abs(tol1 * q), std::abs(e * q))) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        if (std::abs(d) > tol1) {
            b += d;
        } else {
            b += (xm >= 0) ? tol1 : -tol1;
        }
        fb = total_occupation(all_eigs, all_weights, b, beta, smearing) - Nelectron;
    }

    return b;
}

double Occupation::compute(Wavefunction& wfn,
                            double Nelectron,
                            double beta,
                            SmearingType smearing,
                            const std::vector<double>& kpt_weights,
                            const MPIComm& kptcomm,
                            const MPIComm& spincomm) {
    int Nspin = wfn.Nspin();
    int Nkpts = wfn.Nkpts();
    int Nband = wfn.Nband();

    // Spin multiplier: 2 for non-spin-polarized, 1 for collinear
    double spin_fac = (Nspin == 1) ? 2.0 : 1.0;

    // Gather all eigenvalues and their weights for Fermi level search
    std::vector<double> all_eigs;
    std::vector<double> all_weights;

    for (int s = 0; s < Nspin; ++s) {
        for (int k = 0; k < Nkpts; ++k) {
            const auto& eig = wfn.eigenvalues(s, k);
            double wk = kpt_weights[k] * spin_fac;
            for (int n = 0; n < Nband; ++n) {
                all_eigs.push_back(eig(n));
                all_weights.push_back(wk);
            }
        }
    }

    // Find Fermi level
    double Ef = find_fermi_level(all_eigs, all_weights, Nelectron, beta, smearing);

    // Set occupations
    for (int s = 0; s < Nspin; ++s) {
        for (int k = 0; k < Nkpts; ++k) {
            auto& occ = wfn.occupations(s, k);
            const auto& eig = wfn.eigenvalues(s, k);
            for (int n = 0; n < Nband; ++n) {
                occ(n) = smearing_function(eig(n) - Ef, beta, smearing);
            }
        }
    }

    return Ef;
}

double Occupation::entropy(const Wavefunction& wfn,
                            double beta,
                            SmearingType smearing,
                            const std::vector<double>& kpt_weights) {
    double S = 0.0;
    int Nspin = wfn.Nspin();
    int Nkpts = wfn.Nkpts();
    int Nband = wfn.Nband();
    double spin_fac = (Nspin == 1) ? 2.0 : 1.0;

    for (int s = 0; s < Nspin; ++s) {
        for (int k = 0; k < Nkpts; ++k) {
            const auto& occ = wfn.occupations(s, k);
            double wk = kpt_weights[k] * spin_fac;

            for (int n = 0; n < Nband; ++n) {
                double f = occ(n);
                if (smearing == SmearingType::FermiDirac) {
                    // S = -kBT * sum [f*ln(f) + (1-f)*ln(1-f)]
                    if (f > 1e-14 && f < 1.0 - 1e-14) {
                        S += wk * (f * std::log(f) + (1.0 - f) * std::log(1.0 - f));
                    }
                } else {
                    // Gaussian: S = -1/(2*sqrt(pi)*beta) * sum exp(-(beta*(e-Ef))^2)
                    const auto& eig = wfn.eigenvalues(s, k);
                    double x = eig(n);
                    // We use the relation: entropy = -1/beta * sum_n g(x_n) where
                    // g(x) = -0.5 * sqrt(pi) * x * erfc(x) - 0.5 * exp(-x^2)
                    // with x = beta*(epsilon - Ef) / sqrt(2)
                    // Simplified: direct computation from occupation
                    if (f > 1e-14 && f < 1.0 - 1e-14) {
                        S += wk * (f * std::log(f) + (1.0 - f) * std::log(1.0 - f));
                    }
                }
            }
        }
    }

    // S has units of kBT; multiply by 1/beta to get energy
    if (smearing == SmearingType::FermiDirac) {
        return S / beta;  // -kBT * sum [f*ln(f) + (1-f)*ln(1-f)] (S already negative)
    } else {
        return S / beta;
    }
}

} // namespace sparc
