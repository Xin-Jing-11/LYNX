#pragma once

#include "core/NDArray.hpp"
#include "core/types.hpp"
#include "core/Domain.hpp"

#include <vector>
#include <complex>

namespace sparc {

using Complex = std::complex<double>;

// Stores orbitals, eigenvalues, and occupations for one spin/kpt.
// Orbitals are stored as (Nd_d, Nband) column-major in NDArray.
// Supports both real (Gamma-point) and complex (k-point) wavefunctions.
class Wavefunction {
public:
    Wavefunction() = default;

    // Allocate storage for given domain size and number of bands
    // If is_complex=true, allocates complex psi for k-point calculations
    void allocate(int Nd_d, int Nband, int Nspin, int Nkpts, bool is_complex = false);

    // Real orbital access (Gamma-point)
    NDArray<double>& psi(int spin, int kpt) { return psi_[spin * Nkpts_ + kpt]; }
    const NDArray<double>& psi(int spin, int kpt) const { return psi_[spin * Nkpts_ + kpt]; }

    // Complex orbital access (k-point)
    NDArray<Complex>& psi_kpt(int spin, int kpt) { return psi_kpt_[spin * Nkpts_ + kpt]; }
    const NDArray<Complex>& psi_kpt(int spin, int kpt) const { return psi_kpt_[spin * Nkpts_ + kpt]; }

    // Eigenvalues: (Nband,) per spin/kpt — always real
    NDArray<double>& eigenvalues(int spin, int kpt) { return eig_[spin * Nkpts_ + kpt]; }
    const NDArray<double>& eigenvalues(int spin, int kpt) const { return eig_[spin * Nkpts_ + kpt]; }

    // Occupations: (Nband,) per spin/kpt — always real
    NDArray<double>& occupations(int spin, int kpt) { return occ_[spin * Nkpts_ + kpt]; }
    const NDArray<double>& occupations(int spin, int kpt) const { return occ_[spin * Nkpts_ + kpt]; }

    int Nd_d() const { return Nd_d_; }
    int Nband() const { return Nband_; }
    int Nspin() const { return Nspin_; }
    int Nkpts() const { return Nkpts_; }
    bool is_complex() const { return is_complex_; }

    // Randomize orbitals (initial guess) — real version
    void randomize(int spin, int kpt, unsigned seed = 42);

    // Randomize orbitals — complex version (k-point)
    void randomize_kpt(int spin, int kpt, unsigned seed = 42);

private:
    int Nd_d_ = 0;
    int Nband_ = 0;
    int Nspin_ = 0;
    int Nkpts_ = 0;
    bool is_complex_ = false;

    // Indexed by [spin * Nkpts + kpt]
    std::vector<NDArray<double>> psi_;         // real orbitals (Gamma)
    std::vector<NDArray<Complex>> psi_kpt_;    // complex orbitals (k-point)
    std::vector<NDArray<double>> eig_;
    std::vector<NDArray<double>> occ_;
};

} // namespace sparc
