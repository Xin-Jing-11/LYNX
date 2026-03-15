#include "electronic/Wavefunction.hpp"
#include <cmath>
#include <cstdlib>

namespace lynx {

void Wavefunction::allocate(int Nd_d, int Nband, int Nspin, int Nkpts,
                             bool is_complex, int Nspinor) {
    // Non-band-parallel: Nband_local == Nband_global
    allocate(Nd_d, Nband, Nband, Nspin, Nkpts, is_complex, Nspinor);
}

void Wavefunction::allocate(int Nd_d, int Nband_local, int Nband_global,
                             int Nspin, int Nkpts, bool is_complex, int Nspinor) {
    Nd_d_ = Nd_d;
    Nband_ = Nband_local;
    Nband_global_ = Nband_global;
    Nspin_ = Nspin;
    Nkpts_ = Nkpts;
    is_complex_ = is_complex;
    Nspinor_ = Nspinor;

    // For spinor: each band column has Nspinor*Nd_d rows (always complex)
    int row_dim = Nd_d * Nspinor;
    if (Nspinor > 1) is_complex_ = true;  // spinor always complex

    int total = Nspin * Nkpts;
    psi_.clear();
    psi_kpt_.clear();
    eig_.clear();
    occ_.clear();

    eig_.reserve(total);
    occ_.reserve(total);

    if (is_complex_) {
        psi_kpt_.reserve(total);
        for (int i = 0; i < total; ++i) {
            psi_kpt_.emplace_back(row_dim, Nband_local);
            eig_.emplace_back(Nband_global);
            occ_.emplace_back(Nband_global);
        }
    } else {
        psi_.reserve(total);
        for (int i = 0; i < total; ++i) {
            psi_.emplace_back(Nd_d, Nband_local);
            eig_.emplace_back(Nband_global);
            occ_.emplace_back(Nband_global);
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
    int row_dim = Nd_d_ * Nspinor_;  // 2*Nd_d for spinor
    std::srand(seed);
    for (int j = 0; j < Nband_; ++j) {
        Complex* col = X.col(j);
        for (int i = 0; i < row_dim; ++i) {
            double re = -0.5 + 1.0 * ((double)std::rand() / RAND_MAX);
            double im = -0.5 + 1.0 * ((double)std::rand() / RAND_MAX);
            col[i] = Complex(re, im);
        }
    }
}

} // namespace lynx
