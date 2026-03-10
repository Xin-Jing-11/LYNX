#pragma once

#include "core/NDArray.hpp"
#include "core/types.hpp"
#include "core/Domain.hpp"

#include <vector>

namespace sparc {

// Stores orbitals, eigenvalues, and occupations for one spin/kpt.
// Orbitals are stored as (Nd_d, Nband) column-major in NDArray.
class Wavefunction {
public:
    Wavefunction() = default;

    // Allocate storage for given domain size and number of bands
    void allocate(int Nd_d, int Nband, int Nspin, int Nkpts);

    // Access orbitals for given spin and k-point
    NDArray<double>& psi(int spin, int kpt) { return psi_[spin * Nkpts_ + kpt]; }
    const NDArray<double>& psi(int spin, int kpt) const { return psi_[spin * Nkpts_ + kpt]; }

    // Eigenvalues: (Nband,) per spin/kpt
    NDArray<double>& eigenvalues(int spin, int kpt) { return eig_[spin * Nkpts_ + kpt]; }
    const NDArray<double>& eigenvalues(int spin, int kpt) const { return eig_[spin * Nkpts_ + kpt]; }

    // Occupations: (Nband,) per spin/kpt
    NDArray<double>& occupations(int spin, int kpt) { return occ_[spin * Nkpts_ + kpt]; }
    const NDArray<double>& occupations(int spin, int kpt) const { return occ_[spin * Nkpts_ + kpt]; }

    int Nd_d() const { return Nd_d_; }
    int Nband() const { return Nband_; }
    int Nspin() const { return Nspin_; }
    int Nkpts() const { return Nkpts_; }

    // Randomize orbitals (initial guess)
    void randomize(int spin, int kpt, unsigned seed = 42);

private:
    int Nd_d_ = 0;
    int Nband_ = 0;
    int Nspin_ = 0;
    int Nkpts_ = 0;

    // Indexed by [spin * Nkpts + kpt]
    std::vector<NDArray<double>> psi_;
    std::vector<NDArray<double>> eig_;
    std::vector<NDArray<double>> occ_;
};

} // namespace sparc
