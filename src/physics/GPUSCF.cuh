#pragma once
#ifdef USE_CUDA

#include <cuComplex.h>
#include "core/types.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "core/KPoints.hpp"
#include "operators/FDStencil.hpp"
#include "operators/Hamiltonian.hpp"
#include "operators/NonlocalProjector.hpp"
#include "electronic/Wavefunction.hpp"
#include "electronic/Occupation.hpp"
#include "xc/XCFunctional.hpp"
#include "physics/Energy.hpp"
#include "parallel/MPIComm.hpp"
#include "parallel/HaloExchange.hpp"
#include "atoms/Crystal.hpp"
#include "xc/ExactExchange.hpp"
#include "xc/GPUExchangePoissonSolver.cuh"
#include <vector>

namespace lynx {

struct SCFParams;  // forward declaration (defined in SCF.hpp)

// GPU-resident SCF runner.
// Encapsulates all GPU state (callbacks, buffers, nonlocal data) and
// runs the SCF loop with data staying on device.
// CPU fallbacks: occupation (tiny), energy evaluation (diagnostic D2H).
class GPUSCFRunner {
public:
    GPUSCFRunner() = default;
    ~GPUSCFRunner();

    // Run full GPU SCF. Returns total energy.
    // Prerequisites: Crystal/Influence/Vnl already set up on CPU.
    double run(Wavefunction& wfn,
               const SCFParams& params,
               const FDGrid& grid,
               const Domain& domain,
               const FDStencil& stencil,
               const Hamiltonian& hamiltonian,
               const HaloExchange& halo,
               const NonlocalProjector* vnl,
               const Crystal& crystal,
               const std::vector<AtomNlocInfluence>& nloc_influence,
               const MPIComm& bandcomm,
               int Nelectron,
               int Natom,
               const double* rho_init,       // initial total density (Nd)
               const double* rho_b,          // pseudocharge (Nd)
               double Eself,
               double Ec,
               XCType xc_type,
               const double* rho_core,       // NLCC core density (may be null)
               bool is_gga,
               // Spin/k-point parameters (optional, default = gamma Nspin=1)
               int Nspin = 1,
               bool is_kpt = false,
               const KPoints* kpoints = nullptr,
               const std::vector<double>& kpt_weights = std::vector<double>{1.0},
               int Nspin_local = 1,
               int spin_start = 0,
               int kpt_start = 0,
               const double* rho_up_init = nullptr,  // spin-up density (Nd, for Nspin=2)
               const double* rho_dn_init = nullptr,   // spin-down density (Nd, for Nspin=2)
               bool is_soc = false,                    // SOC mode (spinor wavefunctions)
               // Hybrid EXX parameters (optional)
               ExactExchange* exx = nullptr,           // CPU EXX operator (non-owning)
               XCType xc_type_hybrid = XCType::LDA_PZ);  // for is_hybrid() check

    // Download final GPU state back to CPU arrays for forces/stress
    void download_results(double* phi, double* Vxc, double* exc,
                          double* Veff, double* Dxcdgrho,
                          double* rho, Wavefunction& wfn);

    // Compute nonlocal forces, kinetic + nonlocal stress on GPU.
    // Psi stays on device. Only tiny results (3*Natom + 6 + 6 + 1) downloaded.
    void compute_force_stress(
        const Wavefunction& wfn,      // for occupations
        const Crystal& crystal,
        const std::vector<AtomNlocInfluence>& nloc_influence,
        const Domain& domain,
        const FDGrid& grid,
        double* f_nloc,               // [3 * n_atom] nonlocal forces
        double* stress_k,             // [6] kinetic stress (Voigt)
        double* stress_nl,            // [6] nonlocal stress (Voigt)
        double* energy_nl);           // scalar nonlocal energy

    // Compute SOC nonlocal forces on GPU.
    // Psi is uploaded from wfn (spinor complex), SOC data already on device.
    void compute_soc_force(
        const Wavefunction& wfn,
        const Crystal& crystal,
        const std::vector<AtomNlocInfluence>& nloc_influence,
        const Domain& domain,
        const FDGrid& grid,
        const std::vector<double>& kpt_weights,
        const KPoints* kpoints,
        int kpt_start,
        double* f_soc);  // [3 * n_atoms]

    // Compute SOC nonlocal stress tensor on GPU.
    // GPU computes gradients; position-weighted beta and reduction done on CPU.
    void compute_soc_stress(
        const Wavefunction& wfn,
        const Crystal& crystal,
        const std::vector<AtomNlocInfluence>& nloc_influence,
        const Domain& domain,
        const FDGrid& grid,
        const std::vector<double>& kpt_weights,
        const KPoints* kpoints,
        int kpt_start,
        double* stress_soc,   // [6] Voigt stress
        double* energy_soc);  // scalar SOC energy

    // Download mGGA tau and vtau from GPU to CPU arrays
    void download_tau_vtau(double* tau, double* vtau, int tau_size, int vtau_size);

    // Download spin-resolved densities from GPU
    void download_spin_densities(double* rho_up, double* rho_dn, int size);

    const EnergyComponents& energy() const { return energy_; }
    bool converged() const { return converged_; }
    double fermi_energy() const { return Ef_; }

private:
    // Grid parameters
    int nx_ = 0, ny_ = 0, nz_ = 0, FDn_ = 0, Nd_ = 0;
    double dV_ = 0.0;

    // Stencil diagonal coefficients
    double diag_coeff_ham_ = 0.0;
    double poisson_diag_ = 0.0, jacobi_m_inv_ = 0.0;
    double kerker_diag_ = 0.0, kerker_m_inv_ = 0.0, kerker_rhs_diag_ = 0.0;
    double precond_tol_ = 0.0;

    // GPU nonlocal projector data (flattened, uploaded once)
    struct GPUNonlocalData {
        double* d_Chi_flat = nullptr;
        int*    d_gpos_flat = nullptr;
        int*    d_gpos_offsets = nullptr;
        int*    d_chi_offsets = nullptr;
        int*    d_ndc_arr = nullptr;
        int*    d_nproj_arr = nullptr;
        int*    d_IP_displ = nullptr;
        double* d_Gamma = nullptr;
        double* d_alpha = nullptr;
        int n_influence = 0;
        int total_phys_nproj = 0;
        int max_ndc = 0;
        int max_nproj = 0;

        void setup(const NonlocalProjector& vnl,
                   const Crystal& crystal,
                   const std::vector<AtomNlocInfluence>& nloc_influence,
                   int Nband);
        void free();
    };

    GPUNonlocalData gpu_vnl_;

    // Device-side persistent arrays (not in SCFBuffers)
    double* d_rho_core_ = nullptr;
    double* d_pseudocharge_ = nullptr;
    double* d_mix_fkm1_ = nullptr;
    double* d_Y_ = nullptr;  // separate from Hpsi for eigensolver

    // Mixer state
    int mix_iter_ = 0;

    // Results
    EnergyComponents energy_;
    bool converged_ = false;
    double Ef_ = 0.0;
    double Ef_prev_ = 0.0;

    // XC type and NLCC flag
    bool has_nlcc_ = false;
    bool is_gga_ = false;
    bool is_mgga_ = false;
    bool is_orth_ = true;
    bool has_mixed_deriv_ = false;  // non-orth mixed derivative terms
    XCType xc_type_ = XCType::GGA_PBE;  // base XC type (for mGGA dispatch: SCAN vs rSCAN/r2SCAN)

    // mGGA (SCAN) device buffers and state
    double* d_tau_ = nullptr;       // [Nd] or [2*Nd] for spin
    double* d_vtau_ = nullptr;      // [Nd] or [2*Nd] for spin
    double* d_vtau_active_ = nullptr;  // points to d_vtau_ or d_vtau_ + Nd_ for per-spin Hamiltonian
    // Persistent mGGA Hamiltonian work buffers (avoid scratch pool pressure for k-point)
    double* d_mgga_dpsi_ = nullptr;    // [Nd] gradient of psi (real)
    double* d_mgga_vtdpsi_ = nullptr;  // [Nd] vtau * gradient
    double* d_mgga_div_ = nullptr;     // [Nd] divergence accumulator
    double* d_mgga_vt_ex_ = nullptr;   // [nd_ex] halo of vtau product (real)
    void* d_mgga_dpsi_z_ = nullptr;    // [Nd] complex gradient
    void* d_mgga_vtdpsi_z_ = nullptr;  // [Nd] complex vtau product
    void* d_mgga_div_z_ = nullptr;     // [Nd] complex divergence
    void* d_mgga_vt_ex_z_ = nullptr;   // [nd_ex] complex halo
    bool tau_valid_ = false;        // set true after first tau computation

    // Spin/k-point parameters
    int Nspin_ = 1;
    int Nspin_local_ = 1;
    int spin_start_ = 0;
    bool is_kpt_ = false;
    int kpt_start_ = 0;
    const KPoints* kpoints_ = nullptr;
    double* d_Y_s1_ = nullptr;  // separate d_Y for spin-down (when Nspin=2)

    // Per-k-point complex psi on GPU (stays on device; no CPU round-trip)
    // Layout: d_psi_z_kpt_[s * Nkpts + k], each is (Nd, Nband) cuDoubleComplex
    std::vector<void*> d_psi_z_kpt_;

    // K-point Bloch phase state (set per k-point in inner loop)
    double kxLx_ = 0, kyLy_ = 0, kzLz_ = 0;
    double* d_bloch_fac_ = nullptr;  // [n_influence * 2] cos/sin per influence atom
    void*   d_alpha_z_ = nullptr;    // cuDoubleComplex alpha for complex nonlocal

    // SOC data (uploaded once during setup)
    bool has_soc_ = false;
    const NonlocalProjector* vnl_ptr_ = nullptr;  // for CPU SOC comparison
    struct GPUSOCData {
        void* d_Chi_soc_flat = nullptr;  // cast to cuDoubleComplex* (complex Chi_soc)
        int*    d_gpos_offsets_soc = nullptr;
        int*    d_chi_soc_offsets = nullptr;
        int*    d_ndc_arr_soc = nullptr;
        int*    d_nproj_soc_arr = nullptr;
        int*    d_IP_displ_soc = nullptr;
        double* d_Gamma_soc = nullptr;
        int*    d_proj_l = nullptr;
        int*    d_proj_m = nullptr;
        void*   d_alpha_soc_up = nullptr;  // cuDoubleComplex
        void*   d_alpha_soc_dn = nullptr;  // cuDoubleComplex
        int n_influence_soc = 0;
        int total_soc_nproj = 0;
        int max_ndc_soc = 0;
        int max_nproj_soc = 0;

        void setup_soc(const NonlocalProjector& vnl,
                       const Crystal& crystal,
                       const std::vector<AtomNlocInfluence>& nloc_influence,
                       int Nband);
        void free_soc();
    };
    GPUSOCData gpu_soc_;

    // --- Static callback trampoline ---
    static GPUSCFRunner* s_instance_;

    static void hamiltonian_apply_cb(const double* d_psi, const double* d_Veff,
                                     double* d_Hpsi, double* d_x_ex, int ncol);
    // Spinor Hamiltonian callback for SOC (operates on 2*Nd_d-length spinor vectors)
    static void hamiltonian_apply_spinor_z_cb(const cuDoubleComplex* d_psi, const double* d_Veff,
                                               cuDoubleComplex* d_Hpsi, cuDoubleComplex* d_x_ex, int ncol);

    static void hamiltonian_apply_z_cb(const cuDoubleComplex* d_psi, const double* d_Veff,
                                       cuDoubleComplex* d_Hpsi, cuDoubleComplex* d_x_ex, int ncol);
    static void poisson_op_cb(const double* d_x, double* d_Ax);
    static void poisson_precond_cb(const double* d_r, double* d_f);
    static void kerker_op_cb(const double* d_x, double* d_Ax);
    static void kerker_precond_cb(const double* d_r, double* d_f);

    // GPU helper functions
    double gpu_sum(const double* d_x, int N);
    void gpu_xc_evaluate(double* d_rho, double* d_exc, double* d_Vxc, int Nd);
    void gpu_xc_evaluate_spin(double* d_rho, double* d_exc, double* d_Vxc, int Nd);
    int gpu_poisson_solve(double* d_rho, double* d_phi, double* d_rhs, int Nd, double tol);
    void gpu_pulay_mix(double* d_x, const double* d_g, int Nd, int m_depth, double beta_mix,
                       int Nd_kerker = 0, double beta_mag = 0.0);

    void setup_bloch_factors(const std::vector<AtomNlocInfluence>& nloc_influence,
                             const Crystal& crystal, const Vec3& kpt_cart);

    // --- EXX (exact exchange) GPU state ---
    ExactExchange* exx_cpu_ = nullptr;   // CPU EXX operator (non-owning, for energy)
    double* d_Xi_ = nullptr;             // [Nd x Nocc] ACE operator on device (gamma, current spin)
    double* d_Y_exx_ = nullptr;          // [Nocc x Nband] scratch for apply_Vx (gamma)
    double* d_psi_full_ = nullptr;       // [Nd x Ns] gathered psi for build_ACE (device)
    int exx_Nocc_ = 0;                   // number of occupied states in Xi
    double exx_frac_ = 0.0;             // exchange fraction (0.25 for PBE0)
    int exx_spin_ = 0;                  // current spin channel for apply_Vx
    int exx_kpt_ = 0;                   // current k-point index for apply_Vx (k-point mode)
    bool exx_active_ = false;           // true during Fock inner SCF (apply_Vx enabled)
    XCType xc_type_full_ = XCType::LDA_PZ;  // full XC type (for hybrid detection)

    // K-point EXX GPU state
    // d_Xi_kpt_[spin * Nkpts + kpt] = device pointer to [Nd x Nocc] complex ACE operator
    std::vector<cuDoubleComplex*> d_Xi_kpt_;
    cuDoubleComplex* d_Y_exx_z_ = nullptr;  // [Nocc x Nband] complex scratch for apply_Vx_kpt

    // GPU Poisson solver for exchange (cuFFT-based)
    gpu::GPUExchangePoissonSolver gpu_poisson_;

    void exx_cleanup();

    void cleanup();
};

} // namespace lynx

#endif // USE_CUDA
