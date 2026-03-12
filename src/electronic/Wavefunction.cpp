#include "electronic/Wavefunction.hpp"
#include <cmath>
#include <cstdlib>

namespace sparc {

void Wavefunction::allocate(int Nd_d, int Nband, int Nspin, int Nkpts, bool is_complex) {
    Nd_d_ = Nd_d;
    Nband_ = Nband;
    Nspin_ = Nspin;
    Nkpts_ = Nkpts;
    is_complex_ = is_complex;

    int total = Nspin * Nkpts;
    psi_.clear();
    psi_kpt_.clear();
    eig_.clear();
    occ_.clear();

    eig_.reserve(total);
    occ_.reserve(total);

    if (is_complex) {
        psi_kpt_.reserve(total);
        for (int i = 0; i < total; ++i) {
            psi_kpt_.emplace_back(Nd_d, Nband);
            eig_.emplace_back(Nband);
            occ_.emplace_back(Nband);
        }
    } else {
        psi_.reserve(total);
        for (int i = 0; i < total; ++i) {
            psi_.emplace_back(Nd_d, Nband);
            eig_.emplace_back(Nband);
            occ_.emplace_back(Nband);
        }
    }
}

void Wavefunction::randomize(int spin, int kpt, unsigned seed) {
    auto& X = psi(spin, kpt);
    std::srand(seed);
    for (int j = 0; j < Nband_; ++j) {
        double* col = X.col(j);
        for (int i = 0; i < Nd_d_; ++i) {
            col[i] = -0.5 + 1.0 * ((double)std::rand() / RAND_MAX);
        }
    }
}

void Wavefunction::randomize_kpt(int spin, int kpt, unsigned seed) {
    auto& X = psi_kpt(spin, kpt);
    std::srand(seed);
    for (int j = 0; j < Nband_; ++j) {
        Complex* col = X.col(j);
        for (int i = 0; i < Nd_d_; ++i) {
            double re = -0.5 + 1.0 * ((double)std::rand() / RAND_MAX);
            double im = -0.5 + 1.0 * ((double)std::rand() / RAND_MAX);
            col[i] = Complex(re, im);
        }
    }
}

} // namespace sparc
