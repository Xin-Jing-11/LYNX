#include "core/LynxContext.hpp"
#include "core/ParameterDefaults.hpp"
#include "io/OutputWriter.hpp"

#include <omp.h>
#include <thread>
#include <algorithm>
#include <cstdio>

namespace lynx {

LynxContext& LynxContext::instance() {
    static LynxContext ctx;
    return ctx;
}

LynxContext::~LynxContext() {
    // Safety net: if reset() was not called before MPI_Finalize,
    // check MPI state and only release resources if MPI is still active.
    if (initialized_) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            reset();
        } else {
            // MPI already finalized — release non-MPI resources only.
            // Parallelization destructor would call MPI_Comm_free, so
            // we must release it without destroying it properly.
            // Leak the Parallelization object to avoid the crash.
            parallel_.release();
            halo_.reset();
            laplacian_.reset();
            gradient_.reset();
            lattice_.reset();
            grid_.reset();
            stencil_.reset();
            kpoints_.reset();
        }
    }
}

void LynxContext::reset() {
    lattice_.reset();
    grid_.reset();
    stencil_.reset();
    kpoints_.reset();
    parallel_.reset();
    halo_.reset();
    laplacian_.reset();
    gradient_.reset();
    scf_bandcomm_ = nullptr;
    Nspin_ = 1; Nspinor_ = 1;
    Nelectron_ = 0; Natom_ = 0; Nstates_ = 0;
    is_kpt_ = false; is_soc_ = false;
    rank_ = 0; nproc_ = 1;
    initialized_ = false;
}

void LynxContext::set_atom_info(int Natom, int Nelectron) {
    Natom_ = Natom;
    Nelectron_ = Nelectron;
}

void LynxContext::initialize(SystemConfig& config, MPI_Comm world_comm) {
    MPI_Comm_rank(world_comm, &rank_);
    MPI_Comm_size(world_comm, &nproc_);

    // ── Configure OpenMP threads ───────────────────────────────────
    {
        int num_threads = config.parallel.num_threads;
        if (num_threads <= 0) {
            MPI_Comm node_comm;
            MPI_Comm_split_type(world_comm, MPI_COMM_TYPE_SHARED, 0,
                                MPI_INFO_NULL, &node_comm);
            int nranks_node;
            MPI_Comm_size(node_comm, &nranks_node);
            num_threads = static_cast<int>(std::thread::hardware_concurrency())
                          / nranks_node;
            if (num_threads < 1) num_threads = 1;
            MPI_Comm_free(&node_comm);
        }
        omp_set_num_threads(num_threads);
        if (rank_ == 0)
            std::printf("Using %d OpenMP threads per MPI rank\n", num_threads);
    }

    // ── Lattice, grid, stencil ────────────────────────────────────
    lattice_  = std::make_unique<Lattice>(config.latvec, config.cell_type);
    grid_     = std::make_unique<FDGrid>(config.Nx, config.Ny, config.Nz,
                                         *lattice_,
                                         config.bcx, config.bcy, config.bcz);
    stencil_  = std::make_unique<FDStencil>(config.fd_order, *grid_, *lattice_);

    OutputWriter::print_summary(config, *lattice_, *grid_, rank_);

    // ── K-points ──────────────────────────────────────────────────
    kpoints_ = std::make_unique<KPoints>();
    kpoints_->generate(config.Kx, config.Ky, config.Kz,
                       config.kpt_shift, *lattice_);

    int Nkpts = kpoints_->Nkpts();
    is_kpt_ = !kpoints_->is_gamma_only();
    is_soc_ = (config.spin_type == SpinType::NonCollinear);
    Nspin_   = is_soc_ ? 1 : ((config.spin_type == SpinType::None) ? 1 : 2);
    Nspinor_ = is_soc_ ? 2 : 1;
    if (is_soc_) is_kpt_ = true;  // SOC always complex

    if (rank_ == 0 && is_soc_) {
        std::printf("Spin-orbit coupling (SOC) enabled: Nspin=%d, Nspinor=%d\n",
                    Nspin_, Nspinor_);
    }
    if (rank_ == 0) {
        std::printf("K-points: %dx%dx%d grid, %d full -> %d symmetry-reduced%s\n",
                    config.Kx, config.Ky, config.Kz,
                    kpoints_->Nkpts_full(), Nkpts,
                    is_kpt_ ? " (complex)" : " (gamma-only)");
        for (int i = 0; i < Nkpts && i < 10; ++i) {
            auto& kr = kpoints_->kpts_red()[i];
            std::printf("  k[%2d]: %8.4f %8.4f %8.4f  wt=%.3f\n",
                        i, kr.x, kr.y, kr.z, kpoints_->weights()[i]);
        }
        if (Nkpts > 10)
            std::printf("  ... (%d more)\n", Nkpts - 10);
    }

    // ── Auto-determine parallelization ────────────────────────────
    if (is_soc_) {
        config.parallel.npspin = 1;
    } else if (config.parallel.npspin <= 1 && Nspin_ == 2 && nproc_ >= 2) {
        config.parallel.npspin = 2;
    }

    int nproc_after_spin = nproc_ / config.parallel.npspin;
    if (config.parallel.npkpt <= 1 && Nkpts > 1 && nproc_after_spin > 1) {
        config.parallel.npkpt = std::min(nproc_after_spin, Nkpts);
    }

    int nproc_after_kpt = nproc_after_spin / std::max(1, config.parallel.npkpt);
    if (config.parallel.npband <= 1 && nproc_after_kpt > 1) {
        config.parallel.npband = nproc_after_kpt;
    }

    // ── Create Parallelization ────────────────────────────────────
    parallel_ = std::make_unique<Parallelization>(
        world_comm, config.parallel, *grid_, Nspin_, Nkpts, config.Nstates);

    if (rank_ == 0) {
        auto& v = parallel_->domain().vertices();
        std::printf("\nParallelization: %d procs, domain [%d:%d]x[%d:%d]x[%d:%d] = %d pts\n",
                    nproc_, v.xs, v.xe, v.ys, v.ye, v.zs, v.ze,
                    parallel_->domain().Nd_d());
        std::printf("  npspin=%d, npkpt=%d, npband=%d\n",
                    config.parallel.npspin, config.parallel.npkpt, config.parallel.npband);
        std::printf("  This rank: spin_start=%d Nspin_local=%d kpt_start=%d Nkpts_local=%d "
                    "band_start=%d Nband_local=%d\n",
                    parallel_->spin_start(), parallel_->Nspin_local(),
                    parallel_->kpt_start(), parallel_->Nkpts_local(),
                    parallel_->band_start(), parallel_->Nband_local());
    }

    // ── Operators ─────────────────────────────────────────────────
    halo_      = std::make_unique<HaloExchange>(parallel_->domain(), stencil_->FDn());
    laplacian_ = std::make_unique<Laplacian>(*stencil_, parallel_->domain());
    gradient_  = std::make_unique<Gradient>(*stencil_, parallel_->domain());

    // ── Effective bandcomm for SCF ────────────────────────────────
    scf_bandcomm_ = (config.parallel.npband > 1)
                        ? &parallel_->kptcomm()
                        : &parallel_->bandcomm();

    // ── Nstates is stored but Nelectron/Natom are set later ──────
    Nstates_ = config.Nstates;

    initialized_ = true;
}

}  // namespace lynx
