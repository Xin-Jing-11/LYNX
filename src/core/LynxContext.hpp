#pragma once

#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include "core/Domain.hpp"
#include "core/KPoints.hpp"
#include "operators/FDStencil.hpp"
#include "operators/Laplacian.hpp"
#include "operators/Gradient.hpp"
#include "parallel/Parallelization.hpp"
#include "parallel/HaloExchange.hpp"
#include "parallel/MPIComm.hpp"
#include "io/InputParser.hpp"

#ifdef USE_CUDA
#include "core/GPUContext.cuh"
#endif

#include <memory>
#include <mpi.h>

#if defined(USE_MKL) && !defined(USE_CUDA)
#include <mkl_dfti.h>
#endif

namespace lynx {

/// Singleton holding all infrastructure objects initialized once at startup.
/// Provides read-only access to grid, domain, stencil, operators, and
/// parallelization state.  Eliminates the need to pass 10+ parameters
/// through every function signature.
class LynxContext {
public:
    /// Singleton access.
    static LynxContext& instance();

    /// Build all infrastructure from a (fully-parsed) SystemConfig.
    /// Must be called exactly once before any accessor is used.
    /// Modifies `config` in place (resolves auto-defaults and parallel params).
    void initialize(SystemConfig& config, MPI_Comm world_comm);

    /// True after initialize() has been called.
    bool is_initialized() const { return initialized_; }

    // ── Grid & spatial ─────────────────────────────────────────────
    const Lattice&      lattice()  const { return *lattice_; }
    const FDGrid&       grid()     const { return *grid_; }
    const FDStencil&    stencil()  const { return *stencil_; }
    const Domain&       domain()   const { return parallel_->domain(); }
    const HaloExchange& halo()     const { return *halo_; }

    // ── Pre-built operators ────────────────────────────────────────
    const Laplacian& laplacian() const { return *laplacian_; }
    const Gradient&  gradient()  const { return *gradient_; }

    // ── Parallelization ────────────────────────────────────────────
    const Parallelization& parallel() const { return *parallel_; }
    const MPIComm& bandcomm()    const { return parallel_->bandcomm(); }
    const MPIComm& kptcomm()     const { return parallel_->kptcomm(); }
    const MPIComm& kpt_bridge()  const { return parallel_->kpt_bridge(); }
    const MPIComm& spincomm()    const { return parallel_->spincomm(); }
    const MPIComm& spin_bridge() const { return parallel_->spin_bridge(); }

    /// The "effective bandcomm" used by SCF: when npband > 1 this is kptcomm,
    /// otherwise the real bandcomm.
    const MPIComm& scf_bandcomm() const { return *scf_bandcomm_; }

    // ── Parallelization indices ────────────────────────────────────
    int Nspin_local()  const { return parallel_->Nspin_local(); }
    int Nkpts_local()  const { return parallel_->Nkpts_local(); }
    int Nband_local()  const { return parallel_->Nband_local(); }
    int spin_start()   const { return parallel_->spin_start(); }
    int kpt_start()    const { return parallel_->kpt_start(); }
    int band_start()   const { return parallel_->band_start(); }

    // ── K-points ───────────────────────────────────────────────────
    const KPoints& kpoints() const { return *kpoints_; }
    bool is_kpt()     const { return is_kpt_; }
    bool is_soc()     const { return is_soc_; }
    int  Nspin()      const { return Nspin_; }
    int  Nspinor()    const { return Nspinor_; }

    // ── Physical params ────────────────────────────────────────────
    int    Nelectron() const { return Nelectron_; }
    int    Natom()     const { return Natom_; }
    int    Nstates()   const { return Nstates_; }
    double dV()        const { return grid_->dV(); }

    // ── MPI rank info ──────────────────────────────────────────────
    int rank()  const { return rank_; }
    int nproc() const { return nproc_; }
    bool is_active() const { return parallel_ && parallel_->is_active(); }

#ifdef USE_CUDA
    // ── GPU context ───────────────────────────────────────────
    gpu::GPUContext& gpu_ctx() const { return *gpu_ctx_; }
    bool has_gpu_ctx() const { return gpu_ctx_ != nullptr; }
#endif

    // ── Setters for deferred initialization (atoms, electrons) ────
    void set_atom_info(int Natom, int Nelectron);

    // ── MKL FFT descriptors (shared by EXX Poisson solver) ────────
#if defined(USE_MKL) && !defined(USE_CUDA)
    DFTI_DESCRIPTOR_HANDLE dfti_r2c() const { return desc_r2c_; }
    DFTI_DESCRIPTOR_HANDLE dfti_c2r() const { return desc_c2r_; }
    DFTI_DESCRIPTOR_HANDLE dfti_fwd() const { return desc_fwd_; }
    DFTI_DESCRIPTOR_HANDLE dfti_inv() const { return desc_inv_; }
    void init_fft_descriptors(int Nx, int Ny, int Nz, int ncol_r2c, int ncol_c2c);
#endif

    // ── Reset (for testing — allows re-initialization) ─────────────
    void reset();

private:
    LynxContext() = default;
    ~LynxContext();  // releases MPI resources if still initialized
    LynxContext(const LynxContext&) = delete;
    LynxContext& operator=(const LynxContext&) = delete;

    bool initialized_ = false;

    // Owned objects
    std::unique_ptr<Lattice>          lattice_;
    std::unique_ptr<FDGrid>           grid_;
    std::unique_ptr<FDStencil>        stencil_;
    std::unique_ptr<KPoints>          kpoints_;
    std::unique_ptr<Parallelization>  parallel_;
    std::unique_ptr<HaloExchange>     halo_;
    std::unique_ptr<Laplacian>        laplacian_;
    std::unique_ptr<Gradient>         gradient_;

#ifdef USE_CUDA
    std::unique_ptr<gpu::GPUContext> gpu_ctx_;
#endif

    // Non-owning pointer to the effective bandcomm
    const MPIComm* scf_bandcomm_ = nullptr;

    // Cached scalars
    int Nspin_    = 1;
    int Nspinor_  = 1;
    int Nelectron_= 0;
    int Natom_    = 0;
    int Nstates_  = 0;
    bool is_kpt_  = false;
    bool is_soc_  = false;
    int rank_     = 0;
    int nproc_    = 1;

#if defined(USE_MKL) && !defined(USE_CUDA)
    DFTI_DESCRIPTOR_HANDLE desc_r2c_ = nullptr;
    DFTI_DESCRIPTOR_HANDLE desc_c2r_ = nullptr;
    DFTI_DESCRIPTOR_HANDLE desc_fwd_ = nullptr;
    DFTI_DESCRIPTOR_HANDLE desc_inv_ = nullptr;
    int fft_ncol_r2c_ = 0;
    int fft_ncol_c2c_ = 0;
    int fft_Nx_ = 0, fft_Ny_ = 0, fft_Nz_ = 0;

    void free_fft_descriptors();
#endif
};

}  // namespace lynx
