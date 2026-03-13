#include "electronic/Occupation.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <mpi.h>

namespace sparc {

double Occupation::fermi_dirac(double x, double beta) {
    double arg = x * beta;
    if (arg > 50.0) return 0.0;
    if (arg < -50.0) return 1.0;
    return 1.0 / (1.0 + std::exp(arg));
}

double Occupation::gaussian_smearing(double x, double beta) {
    // f(x) = 0.5 * erfc(beta * x)
    // where x = (epsilon - Ef)
    // Matches reference SPARC: 0.5 * (1 - erf(beta * (lambda - lambda_f)))
    return 0.5 * std::erfc(beta * x);
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
                            const MPIComm& spincomm,
                            int kpt_start) {
    int Nspin_local = wfn.Nspin();
    int Nkpts_local = wfn.Nkpts();
    // Use Nband_global for eigenvalue/occupation arrays (they always hold all bands)
    int Nband = wfn.Nband_global();

    // Determine global Nspin from spincomm
    int Nspin_global = Nspin_local;
    if (!spincomm.is_null() && spincomm.size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &Nspin_global, 1, MPI_INT, MPI_SUM, spincomm.comm());
    }
    double spin_fac = (Nspin_global == 1) ? 2.0 : 1.0;

    // Gather LOCAL eigenvalues and weights
    // eigenvalues array is always Nband_global in size (all eigenvalues on every proc)
    int local_count = Nspin_local * Nkpts_local * Nband;
    std::vector<double> local_eigs(local_count);
    std::vector<double> local_weights(local_count);
    int idx = 0;
    for (int s = 0; s < Nspin_local; ++s) {
        for (int k = 0; k < Nkpts_local; ++k) {
            const auto& eig = wfn.eigenvalues(s, k);
            int k_glob = kpt_start + k;
            double wk = kpt_weights[k_glob] * spin_fac;
            for (int n = 0; n < Nband; ++n) {
                local_eigs[idx] = eig(n);
                local_weights[idx] = wk;
                idx++;
            }
        }
    }

    // Allgather eigenvalues across kptcomm and spincomm for global Fermi level
    std::vector<double> all_eigs = local_eigs;
    std::vector<double> all_weights = local_weights;

    // Gather across kptcomm (different k-points)
    if (!kptcomm.is_null() && kptcomm.size() > 1) {
        int kpt_np = kptcomm.size();
        std::vector<int> recv_counts(kpt_np), displs(kpt_np);
        MPI_Allgather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, kptcomm.comm());
        displs[0] = 0;
        for (int i = 1; i < kpt_np; ++i) displs[i] = displs[i-1] + recv_counts[i-1];
        int total = displs[kpt_np-1] + recv_counts[kpt_np-1];
        all_eigs.resize(total);
        all_weights.resize(total);
        MPI_Allgatherv(local_eigs.data(), local_count, MPI_DOUBLE,
                       all_eigs.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, kptcomm.comm());
        MPI_Allgatherv(local_weights.data(), local_count, MPI_DOUBLE,
                       all_weights.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, kptcomm.comm());
    }

    // Gather across spincomm (different spin channels)
    if (!spincomm.is_null() && spincomm.size() > 1) {
        int spin_np = spincomm.size();
        int cur_count = static_cast<int>(all_eigs.size());
        std::vector<int> recv_counts(spin_np), displs(spin_np);
        MPI_Allgather(&cur_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, spincomm.comm());
        displs[0] = 0;
        for (int i = 1; i < spin_np; ++i) displs[i] = displs[i-1] + recv_counts[i-1];
        int total = displs[spin_np-1] + recv_counts[spin_np-1];
        std::vector<double> tmp_eigs(total), tmp_weights(total);
        MPI_Allgatherv(all_eigs.data(), cur_count, MPI_DOUBLE,
                       tmp_eigs.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, spincomm.comm());
        MPI_Allgatherv(all_weights.data(), cur_count, MPI_DOUBLE,
                       tmp_weights.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, spincomm.comm());
        all_eigs = std::move(tmp_eigs);
        all_weights = std::move(tmp_weights);
    }

    // Find global Fermi level
    double Ef = find_fermi_level(all_eigs, all_weights, Nelectron, beta, smearing);

    // Set occupations for ALL bands (Nband_global)
    for (int s = 0; s < Nspin_local; ++s) {
        for (int k = 0; k < Nkpts_local; ++k) {
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
                            const std::vector<double>& kpt_weights,
                            double Ef,
                            int kpt_start,
                            int Nspin_global) {
    double S = 0.0;
    int Nspin_local = wfn.Nspin();
    int Nkpts = wfn.Nkpts();
    int Nband = wfn.Nband_global();  // use global band count for eigenvalue/occupation iteration
    if (Nspin_global <= 0) Nspin_global = Nspin_local;
    double spin_fac = (Nspin_global == 1) ? 2.0 : 1.0;

    for (int s = 0; s < Nspin_local; ++s) {
        for (int k = 0; k < Nkpts; ++k) {
            const auto& occ = wfn.occupations(s, k);
            double wk = kpt_weights[kpt_start + k] * spin_fac;

            for (int n = 0; n < Nband; ++n) {
                double f = occ(n);
                if (smearing == SmearingType::FermiDirac) {
                    // S_n = -[f*ln(f) + (1-f)*ln(1-f)]
                    if (f > 1e-14 && f < 1.0 - 1e-14) {
                        S += wk * (-(f * std::log(f) + (1.0 - f) * std::log(1.0 - f)));
                    }
                } else {
                    // Gaussian: S_n = (0.5/sqrt(pi)) * exp(-(beta*(eig_n - Ef))^2)
                    // Reference: Calculate_entropy_term, case 1
                    const auto& eig = wfn.eigenvalues(s, k);
                    double x = beta * (eig(n) - Ef);
                    S += wk * (0.5 / std::sqrt(M_PI)) * std::exp(-x * x);
                }
            }
        }
    }

    // Reference: Entropy *= -occfac / (Nkpts * Beta)
    // occfac = 2 for non-spin, kpt_weights already include 1/Nkpts factor,
    // spin_fac and wk already folded into S above.
    // So: Entropy = -S / beta
    return -S / beta;
}

} // namespace sparc
