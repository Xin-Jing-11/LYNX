#include "electronic/Wavefunction.hpp"
#include <cmath>
#include <cstdlib>

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
    // Match reference SPARC exactly: SetRandMat using C's srand/rand
    // seed = rank_in_spincomm * 100 + 1 (for serial: seed=1)
    // range [-0.5, 0.5], fills flat array Mat[0..m*n-1] contiguously
    // where m = DMndsp (= Nd_d for gamma-point), n = Nband
    auto& X = psi(spin, kpt);
    std::srand(seed);
    // Reference fills m*n values contiguously (column-major, ld=m).
    // Our NDArray may have ld > Nd_d_ due to padding, so fill col-by-col.
    for (int j = 0; j < Nband_; ++j) {
        double* col = X.col(j);
        for (int i = 0; i < Nd_d_; ++i) {
            col[i] = -0.5 + 1.0 * ((double)std::rand() / RAND_MAX);
        }
    }
}

} // namespace sparc
