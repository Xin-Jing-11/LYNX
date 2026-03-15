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
               const double* rho_dn_init = nullptr);  // spin-down density (Nd, for Nspin=2)

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

    // Spin/k-point parameters
    int Nspin_ = 1;
    int Nspin_local_ = 1;
    int spin_start_ = 0;
    bool is_kpt_ = false;
    int kpt_start_ = 0;
    const KPoints* kpoints_ = nullptr;
    double* d_Y_s1_ = nullptr;  // separate d_Y for spin-down (when Nspin=2)

    // K-point Bloch phase state (set per k-point in inner loop)
    double kxLx_ = 0, kyLy_ = 0, kzLz_ = 0;
    double* d_bloch_fac_ = nullptr;  // [n_influence * 2] cos/sin per influence atom
    void*   d_alpha_z_ = nullptr;    // cuDoubleComplex alpha for complex nonlocal

    // --- Static callback trampoline ---
    static GPUSCFRunner* s_instance_;

    static void hamiltonian_apply_cb(const double* d_psi, const double* d_Veff,
                                     double* d_Hpsi, double* d_x_ex, int ncol);
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
    void gpu_pulay_mix(double* d_x, const double* d_g, int Nd, int m_depth, double beta_mix);

    void setup_bloch_factors(const std::vector<AtomNlocInfluence>& nloc_influence,
                             const Crystal& crystal, const Vec3& kpt_cart);

    void cleanup();
};

} // namespace lynx

#endif // USE_CUDA
