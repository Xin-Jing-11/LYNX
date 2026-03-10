#include "electronic/Wavefunction.hpp"
#include <cmath>
#include <random>

namespace sparc {

void Wavefunction::allocate(int Nd_d, int Nband, int Nspin, int Nkpts) {
    Nd_d_ = Nd_d;
    Nband_ = Nband;
    Nspin_ = Nspin;
    Nkpts_ = Nkpts;

    int total = Nspin * Nkpts;
    psi_.clear();
    eig_.clear();
    occ_.clear();
    psi_.reserve(total);
    eig_.reserve(total);
    occ_.reserve(total);

    for (int i = 0; i < total; ++i) {
        psi_.emplace_back(Nd_d, Nband);
        eig_.emplace_back(Nband);
        occ_.emplace_back(Nband);
    }
}

void Wavefunction::randomize(int spin, int kpt, unsigned seed) {
    auto& X = psi(spin, kpt);
    std::mt19937 rng(seed + spin * 1000 + kpt);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < Nband_; ++j) {
        double* col = X.col(j);
        for (int i = 0; i < Nd_d_; ++i) {
            col[i] = dist(rng);
        }
    }
}

} // namespace sparc
