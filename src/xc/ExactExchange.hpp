#pragma once

#include "core/types.hpp"
#include "core/FDGrid.hpp"
#include "core/Lattice.hpp"
#include "core/KPoints.hpp"
#include "core/NDArray.hpp"
#include "core/LynxContext.hpp"
#include "electronic/Wavefunction.hpp"
#include "operators/Gradient.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"
#include "xc/ExchangePoissonSolver.hpp"

#include <vector>
#include <complex>
#include <array>

namespace lynx {

// Exact Exchange (EXX) operator using the ACE (Adaptively Compressed Exchange) method.
//
// Supports both gamma-point (real) and k-point (complex) calculations.
// LYNX has NO domain decomposition: every process holds the full grid.
// Band parallelism is handled via cyclic orbital rotation across bandcomm.
// K-point parallelism is handled via cyclic rotation across kpt_bridge.
class ExactExchange {
public:
    ExactExchange() = default;

    // Setup with LynxContext (preferred).
    void setup(const LynxContext& ctx,
               const EXXParams& params,
               int Kx_hf = 1, int Ky_hf = 1, int Kz_hf = 1);

    // Setup with explicit params (backward compat).
    void setup(const FDGrid& grid, const Lattice& lattice,
               const KPoints* kpoints,
               const MPIComm& bandcomm,
               const MPIComm& kpt_bridge,
               const MPIComm& spin_bridge,
               const EXXParams& params,
               int Nspin, int Nstates, int Nband_local, int band_start,
               int npband, int npkpt, int kpt_start, int spin_start,
               int Kx_hf = 1, int Ky_hf = 1, int Kz_hf = 1);

    // Build ACE operator from current orbitals (called once per outer Fock iteration)
    void build_ACE(const Wavefunction& wfn);

    // Apply Vx to psi: Hx -= exx_frac * Xi * (Xi^T * X)
    // gamma-point (real)
    void apply_Vx(const double* X, int ldx, int ncol, int DMnd,
                  double* Hx, int ldhx, int spin) const;

    // Apply Vx to psi: Hx -= exx_frac * Xi * (Xi^H * X)
    // k-point (complex)
    void apply_Vx_kpt(const Complex* X, int ldx, int ncol, int DMnd,
                      Complex* Hx, int ldhx, int spin, int kpt) const;

    // Compute exact exchange energy
    double compute_energy(const Wavefunction& wfn);

    // Compute EXX stress tensor using stored ctx_ (preferred, call after setup with ctx).
    std::array<double, 6> compute_stress(const Wavefunction& wfn);

    // Compute EXX stress tensor (explicit params — backward compat).
    std::array<double, 6> compute_stress(const Wavefunction& wfn,
                                          const Gradient& gradient,
                                          const HaloExchange& halo,
                                          const Domain& domain);

    double Eexx() const { return Eexx_; }
    void set_Eexx(double e) { Eexx_ = e; }
    void set_Nstates_occ(int n) { Nstates_occ_ = n; }

    // Debug accessors for unit testing
    const Complex* Xi_kpt_data(int spin, int kpt) const { return Xi_kpt_[spin * Nkpts_local_ + kpt].data(); }
    const double* Xi_data(int spin) const { return Xi_[spin].data(); }

    // Check if EXX is set up
    bool is_setup() const { return setup_done_; }

    // Access Poisson solver (for GPU setup)
    const ExchangePoissonSolver& poisson() const { return poisson_; }

    // Access parameters for GPU
    double dV() const { return dV_; }
    int Nstates() const { return Nstates_; }
    int Nstates_occ() const { return Nstates_occ_; }
    int Nd_d() const { return Nd_d_; }

private:
    const LynxContext* ctx_ = nullptr;
    bool setup_done_ = false;
    bool is_gamma_ = true;

    // Grid info
    int Nd_d_ = 0;   // full grid size
    double dV_ = 0;
    const FDGrid* grid_ = nullptr;
    const Lattice* lattice_ = nullptr;

    // Parameters
    EXXParams params_;
    double exx_frac_ = 0.25;

    // Parallelization
    const MPIComm* bandcomm_ = nullptr;
    const MPIComm* kpt_bridge_ = nullptr;
    const MPIComm* spin_bridge_ = nullptr;
    int Nspin_ = 1;
    int Nstates_ = 0;
    int Nband_local_ = 0;
    int band_start_ = 0;
    int npband_ = 1;
    int npkpt_ = 1;
    int kpt_start_ = 0;
    int spin_start_ = 0;

    // K-point info
    const KPoints* kpoints_ = nullptr;
    int Nkpts_local_ = 0;   // k-points on this process

    // Poisson solver
    ExchangePoissonSolver poisson_;

    // ACE operator storage
    // Gamma: Xi_[spin] has shape [Nd_d, Nstates_occ]
    std::vector<NDArray<double>> Xi_;
    // K-point: Xi_kpt_[spin * Nkpts + kpt] has shape [Nd_d, Nstates_occ]
    std::vector<NDArray<Complex>> Xi_kpt_;

    int Nstates_occ_ = 0;  // number of occupied states (occ > threshold)
    double Eexx_ = 0.0;

    // Internal methods
    void allocate_ACE(const Wavefunction& wfn);
    void solve_for_Xi(const Wavefunction& wfn, int spin);
    void calculate_ACE_operator(const Wavefunction& wfn, int spin);

    void allocate_ACE_kpt(const Wavefunction& wfn);
    void solve_for_Xi_kpt(const Wavefunction& wfn, int spin);
    void calculate_ACE_operator_kpt(const Wavefunction& wfn, int spin, int kpt);

    // Gather all bands from band-parallel processes
    void gather_bandcomm(double* vec, int Nd, int Ns) const;
    void gather_bandcomm_kpt(Complex* vec, int Nd, int Ns) const;

    // Cyclic orbital rotation
    void transfer_orbitals_bandcomm(const double* sendbuf, int send_count,
                                     double* recvbuf, int recv_count, int shift) const;
    void transfer_orbitals_bandcomm_kpt(const Complex* sendbuf, int send_count,
                                         Complex* recvbuf, int recv_count, int shift) const;

    static constexpr double OCC_THRESHOLD = 1e-6;
};

} // namespace lynx
