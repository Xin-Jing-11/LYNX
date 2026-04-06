#pragma once

#include "core/types.hpp"
#include "core/DeviceArray.hpp"
#include "core/Domain.hpp"
#include "core/FDGrid.hpp"
#include "core/DeviceTag.hpp"
#include "core/GPUStatePtr.hpp"
#include "operators/FDStencil.hpp"
#include "operators/Gradient.hpp"
#include "operators/NonlocalProjector.hpp"
#include "parallel/HaloExchange.hpp"
#include <complex>

namespace lynx {

class ExactExchange;  // forward declaration
class LynxContext;    // forward declaration

using Complex = std::complex<double>;

// Hamiltonian operator: H*psi = -0.5*Lap*psi + Veff*psi + Vnl*psi
//
// Takes raw pointers for hot-path compatibility (CUDA-ready).
// All data must be on the local domain.
class Hamiltonian {
public:
    Hamiltonian() = default;
    ~Hamiltonian();
    Hamiltonian(Hamiltonian&& other) noexcept
        : stencil_(other.stencil_), domain_(other.domain_), grid_(other.grid_),
          halo_(other.halo_), vnl_(other.vnl_), vnl_kpt_(other.vnl_kpt_),
          vtau_(other.vtau_), exx_(other.exx_), exx_spin_(other.exx_spin_), exx_kpt_(other.exx_kpt_),
          gpu_state_(std::move(other.gpu_state_))
    {}
    Hamiltonian& operator=(Hamiltonian&& other) noexcept {
        if (this != &other) {
            stencil_ = other.stencil_; domain_ = other.domain_;
            grid_ = other.grid_; halo_ = other.halo_;
            vnl_ = other.vnl_; vnl_kpt_ = other.vnl_kpt_;
            vtau_ = other.vtau_; exx_ = other.exx_;
            exx_spin_ = other.exx_spin_; exx_kpt_ = other.exx_kpt_;
            gpu_state_ = std::move(other.gpu_state_);
        }
        return *this;
    }
    Hamiltonian(const Hamiltonian&) = delete;
    Hamiltonian& operator=(const Hamiltonian&) = delete;

    // Setup with all required components
    void setup(const FDStencil& stencil,
               const Domain& domain,
               const FDGrid& grid,
               const HaloExchange& halo,
               const NonlocalProjector* vnl);  // may be null (Gamma-point)

    // Set device for dispatch (CPU or GPU).
    void set_device(Device dev) { dev_ = dev; }
    Device device() const { return dev_; }

    // --- Real (Gamma-point) interface ---

    // Primary dispatcher: checks dev_ and calls _cpu() or _gpu() path.
    void apply(const double* psi, const double* Veff, double* y,
               int ncol, double c = 0.0) const;

    void apply_local(const double* psi, const double* Veff, double* y,
                     int ncol, double c = 0.0) const;

    // --- Complex (k-point) interface ---

    // Set the nonlocal projector for k-point (complex Chi with Bloch phases)
    void set_vnl_kpt(const NonlocalProjector* vnl_kpt) const { vnl_kpt_ = vnl_kpt; }

    // Apply H*psi for a specific k-point (dispatcher: checks dev_).
    // kpt_cart: k-point in Cartesian reciprocal coords
    // cell_lengths: (Lx, Ly, Lz)
    void apply_kpt(const Complex* psi, const double* Veff, Complex* y,
                   int ncol, const Vec3& kpt_cart, const Vec3& cell_lengths,
                   double c = 0.0) const;

    void apply_local_kpt(const Complex* psi, const double* Veff, Complex* y,
                         int ncol, const Vec3& kpt_cart, const Vec3& cell_lengths,
                         double c = 0.0) const;

    // --- Spinor (SOC) interface ---
    // Apply H*psi for 2-component spinor wavefunctions.
    // psi layout: [spin-up(Nd_d) | spin-down(Nd_d)] per band, ncol bands.
    // Veff_spinor layout: [V_uu(Nd_d) | V_dd(Nd_d) | Re(V_ud)(Nd_d) | Im(V_ud)(Nd_d)]
    void apply_spinor_kpt(const Complex* psi, const double* Veff_spinor, Complex* y,
                          int ncol, int Nd_d, const Vec3& kpt_cart, const Vec3& cell_lengths,
                          double c = 0.0) const;

    // mGGA: set vtau potential (d(n*exc)/d(tau))
    void set_vtau(const double* vtau) { vtau_ = vtau; }
    const double* vtau() const { return vtau_; }

    // EXX: set exact exchange operator for hybrid functionals
    void set_exx(ExactExchange* exx) const { exx_ = exx; }
    void set_exx_context(int spin, int kpt) const { exx_spin_ = spin; exx_kpt_ = kpt; }
    const ExactExchange* exx() const { return exx_; }

    // mGGA term: H_mGGA ψ = -0.5 ∇·(vtau · ∇ψ)
    void apply_mgga(const double* psi, double* y, int ncol) const;
    void apply_mgga_kpt(const Complex* psi, Complex* y, int ncol,
                         const Vec3& kpt_cart, const Vec3& cell_lengths) const;

    // CPU sub-step names (aliases for CPU path)
    void apply_local_cpu(const double* psi, const double* Veff, double* y,
                         int ncol, double c) const { apply_cpu(psi, Veff, y, ncol, c); }
    void apply_mgga_cpu(const double* psi, double* y, int ncol) const { apply_mgga(psi, y, ncol); }

    const FDStencil& stencil() const { return *stencil_; }
    const Domain& domain() const { return *domain_; }

#ifdef USE_CUDA
    // ── GPU state management ─────────────────────────────────────
public:
    void setup_gpu(const LynxContext& ctx,
                   const NonlocalProjector* vnl,
                   const class Crystal& crystal,
                   const std::vector<struct AtomNlocInfluence>& nloc_influence,
                   int Nband);
    void cleanup_gpu();
    void* gpu_state_ptr() { return gpu_state_.as<void>(); }
    const void* gpu_state_ptr() const { return gpu_state_.as<void>(); }

    // Update k-point Bloch phase factors on GPU (kxLx, kyLy, kzLz + d_bloch_fac).
    void set_kpoint_gpu(const Vec3& kpt_cart, const Vec3& cell_lengths);

    // GPU sub-step methods (defined in Hamiltonian.cu)
    void apply_local_gpu(const double* psi, const double* Veff, double* y,
                         int ncol, double c) const;
    void apply_nonlocal_gpu(const double* psi, double* y, int ncol) const;
    void apply_mgga_gpu(const double* psi, double* y, int ncol) const;

    // GPU full-apply methods (defined in Hamiltonian.cu)
    void apply_gpu(const double* psi, const double* Veff, double* y,
                   int ncol, double c) const;
    void apply_kpt_gpu(const Complex* psi, const double* Veff, Complex* y,
                       int ncol, double c) const;
    void apply_spinor_kpt_gpu(const Complex* psi, const double* Veff_spinor, Complex* y,
                              int ncol, int Nd_d, double c) const;

    // GPU force/stress: split into separate force-only and stress-only methods.
    // psi stays on GPU — only scalar results are downloaded.
    // Host-occ overloads: upload h_occ internally, avoiding CUDA calls in .cpp files.

    // Real gamma-point: nonlocal force + energy
    void compute_nonlocal_force_gpu(
        const double* d_psi, const double* h_occ, int Nband,
        double occfac,
        double* h_f_nloc, double* h_energy_nl) const;

    // Real gamma-point: kinetic + nonlocal stress
    void compute_kinetic_nonlocal_stress_gpu(
        const double* d_psi, const double* h_occ, int Nband,
        double occfac,
        double* h_stress_k, double* h_stress_nl) const;

    // Complex k-point: nonlocal force + energy
    void compute_nonlocal_force_kpt_gpu(
        const void* d_psi_z, const double* h_occ, int Nband,
        double spn_fac_wk,
        double* h_f_nloc, double* h_energy_nl) const;

    // Complex k-point: kinetic + nonlocal stress
    void compute_kinetic_nonlocal_stress_kpt_gpu(
        const void* d_psi_z, const double* h_occ, int Nband,
        double spn_fac_wk,
        double* h_stress_k, double* h_stress_nl) const;
#endif

private:
    Device dev_ = Device::CPU;
    GPUStatePtr gpu_state_;                             // RAII wrapper for GPUHamiltonianState (defined in .cu)
    const FDStencil* stencil_ = nullptr;
    const Domain* domain_ = nullptr;
    const FDGrid* grid_ = nullptr;
    const HaloExchange* halo_ = nullptr;
    const NonlocalProjector* vnl_ = nullptr;      // Gamma-point nonlocal
    mutable const NonlocalProjector* vnl_kpt_ = nullptr;   // k-point nonlocal (complex Chi)
    const double* vtau_ = nullptr;                  // mGGA vtau potential
    mutable ExactExchange* exx_ = nullptr;           // EXX operator (hybrid)
    mutable int exx_spin_ = 0;                       // current spin for EXX apply
    mutable int exx_kpt_ = 0;                        // current k-point for EXX apply

    // CPU implementation methods (called by dispatchers when dev_ == CPU)
    void apply_cpu(const double* psi, const double* Veff, double* y,
                   int ncol, double c) const;
    void apply_kpt_cpu(const Complex* psi, const double* Veff, Complex* y,
                       int ncol, const Vec3& kpt_cart, const Vec3& cell_lengths,
                       double c) const;
    void apply_spinor_kpt_cpu(const Complex* psi, const double* Veff_spinor, Complex* y,
                              int ncol, int Nd_d, const Vec3& kpt_cart, const Vec3& cell_lengths,
                              double c) const;

    // Templated stencil application (shared between real and complex)
    template<typename T>
    void lap_plus_diag_orth_impl(const T* x_ex, const double* Veff,
                                  T* y, int ncol, double c) const;
    template<typename T>
    void lap_plus_diag_nonorth_impl(const T* x_ex, const double* Veff,
                                     T* y, int ncol, double c) const;

    // Templated mGGA: H_mGGA ψ = -0.5 ∇·(vtau · ∇ψ)
    template<typename T>
    void apply_mgga_impl(const T* psi, T* y, int ncol,
                          const Vec3& kpt_cart = {0,0,0},
                          const Vec3& cell_lengths = {0,0,0}) const;

    // Legacy real wrappers
    void lap_plus_diag_orth(const double* x_ex, const double* Veff,
                            double* y, int ncol, double c) const;
    void lap_plus_diag_nonorth(const double* x_ex, const double* Veff,
                               double* y, int ncol, double c) const;
};

} // namespace lynx
