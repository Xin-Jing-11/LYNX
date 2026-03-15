#pragma once

#include "core/NDArray.hpp"
#include "core/types.hpp"
#include "core/Domain.hpp"

#include <vector>
#include <complex>

namespace lynx {

using Complex = std::complex<double>;

// Stores orbitals, eigenvalues, and occupations for one spin/kpt.
// Orbitals are stored as (Nd_d, Nband) column-major in NDArray.
// Supports both real (Gamma-point) and complex (k-point) wavefunctions.
//
// Band parallelism: psi has Nband_local columns (local bands on this process),
// while eigenvalues and occupations have Nband_global entries (all bands needed
// for Fermi level computation). Nband() returns Nband_local.
class Wavefunction {
public:
    Wavefunction() = default;
    Wavefunction(const Wavefunction&) = delete;
    Wavefunction& operator=(const Wavefunction&) = delete;
    Wavefunction(Wavefunction&&) = default;
    Wavefunction& operator=(Wavefunction&&) = default;

    // Allocate storage for given domain size and number of bands.
    // Nband: local band count (psi columns). Eigenvalues/occupations also sized Nband.
    // If is_complex=true, allocates complex psi for k-point calculations.
    // Nspinor: 1 (default) or 2 (SOC spinor — each band has 2*Nd_d complex entries).
    void allocate(int Nd_d, int Nband, int Nspin, int Nkpts,
                  bool is_complex = false, int Nspinor = 1);

    // Allocate with separate local/global band counts for band parallelism.
    // Nband_local: columns of psi on this process.
    // Nband_global: total bands across all band-parallel procs (for eigenvalues/occupations).
    void allocate(int Nd_d, int Nband_local, int Nband_global,
                  int Nspin, int Nkpts, bool is_complex = false, int Nspinor = 1);

    // Real orbital access (Gamma-point)
    NDArray<double>& psi(int spin, int kpt) { return psi_[spin * Nkpts_ + kpt]; }
    const NDArray<double>& psi(int spin, int kpt) const { return psi_[spin * Nkpts_ + kpt]; }

    // Complex orbital access (k-point)
    NDArray<Complex>& psi_kpt(int spin, int kpt) { return psi_kpt_[spin * Nkpts_ + kpt]; }
    const NDArray<Complex>& psi_kpt(int spin, int kpt) const { return psi_kpt_[spin * Nkpts_ + kpt]; }

    // Eigenvalues: (Nband_global,) per spin/kpt — always real
    NDArray<double>& eigenvalues(int spin, int kpt) { return eig_[spin * Nkpts_ + kpt]; }
    const NDArray<double>& eigenvalues(int spin, int kpt) const { return eig_[spin * Nkpts_ + kpt]; }

    // Occupations: (Nband_global,) per spin/kpt — always real
    NDArray<double>& occupations(int spin, int kpt) { return occ_[spin * Nkpts_ + kpt]; }
    const NDArray<double>& occupations(int spin, int kpt) const { return occ_[spin * Nkpts_ + kpt]; }

    int Nd_d() const { return Nd_d_; }
    int Nband() const { return Nband_; }           // local band count (psi columns)
    int Nband_global() const { return Nband_global_; }  // total bands (eigenvalue count)
    int Nspin() const { return Nspin_; }
    int Nkpts() const { return Nkpts_; }
    bool is_complex() const { return is_complex_; }
    int Nspinor() const { return Nspinor_; }
    // Effective row dimension: Nd_d * Nspinor for spinor wavefunctions
    int Nd_d_spinor() const { return Nd_d_ * Nspinor_; }

    // Randomize orbitals (initial guess) — real version
    void randomize(int spin, int kpt, unsigned seed = 42);

    // Randomize orbitals — complex version (k-point)
    void randomize_kpt(int spin, int kpt, unsigned seed = 42);

private:
    int Nd_d_ = 0;
    int Nband_ = 0;         // local band count (psi columns)
    int Nband_global_ = 0;  // global band count (eigenvalue/occupation size)
    int Nspin_ = 0;
    int Nkpts_ = 0;
    bool is_complex_ = false;
    int Nspinor_ = 1;       // 1 = normal, 2 = SOC spinor

    // Indexed by [spin * Nkpts + kpt]
    std::vector<NDArray<double>> psi_;         // real orbitals (Gamma)
    std::vector<NDArray<Complex>> psi_kpt_;    // complex orbitals (k-point)
    std::vector<NDArray<double>> eig_;
    std::vector<NDArray<double>> occ_;
};

} // namespace lynx
