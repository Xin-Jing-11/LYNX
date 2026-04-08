#include "Calculator.hpp"
#include "io/DensityIO.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace pylynx {

void Calculator::load_config(const std::string& json_file) {
    config_ = lynx::InputParser::parse(json_file);
    lynx::InputParser::validate(config_);
}

void Calculator::set_config(const lynx::SystemConfig& config) {
    config_ = config;
    // Skip validation of Nstates here -- setup() will auto-compute if <= 0
}

void Calculator::setup(MPI_Comm comm) {
    // 1. Reset LynxContext (allows re-use across multiple setup() calls)
    auto& ctx = lynx::LynxContext::instance();
    ctx.reset();

    // 2. Initialize infrastructure (lattice, grid, stencil, parallelization, etc.)
    ctx.initialize(config_, comm);

    // 3. Build crystal + electrostatics via Crystal::setup factory
    atoms_ = lynx::Crystal::setup(config_, ctx);

    // 4. Register atom/electron counts with context.
    //    Crystal::setup updates config_.Nstates via ParameterDefaults.
    //    We must re-initialize the context so Parallelization picks up
    //    the resolved Nstates, Nelectron, etc.
    ctx.reset();
    ctx.initialize(config_, comm);
    ctx.set_atom_info(atoms_.Natom, atoms_.Nelectron);

    // 5. Create nonlocal projector
    vnl_ = lynx::NonlocalProjector::create(ctx, atoms_.crystal, atoms_.nloc_influence);

    // 6. Set up Hamiltonian
    hamiltonian_.setup(ctx.stencil(), ctx.domain(), ctx.grid(), ctx.halo(), &vnl_);

    // 7. Set up SCF solver
    lynx::SCFParams params = lynx::SCFParams::from_config(config_);
    scf_.setup(ctx, hamiltonian_, &vnl_, params);

    // 8. GPU acceleration
#ifdef USE_CUDA
    if (use_gpu_) {
        scf_.set_gpu_data(atoms_.crystal, atoms_.nloc_influence,
                          atoms_.influence, atoms_.elec);
    }
#else
    if (use_gpu_) {
        throw std::runtime_error(
            "GPU acceleration requested but LYNX was built without CUDA support. "
            "Rebuild with -DUSE_CUDA=ON to enable GPU.");
    }
#endif

    // 9. Allocate wavefunctions
    int Nd_d_val = ctx.domain().Nd_d();
    int Nband_local = ctx.Nband_local();
    int Nstates = ctx.Nstates();
    int Nspin_local = ctx.Nspin_local();
    int Nkpts_local = ctx.Nkpts_local();
    bool is_kpt = ctx.is_kpt();
    int Nspinor = ctx.Nspinor();

    wfn_.allocate(Nd_d_val, Nband_local, Nstates, Nspin_local, Nkpts_local,
                  is_kpt, Nspinor);

    // 10. Compute atomic density for initial guess
    std::vector<double> rho_at(Nd_d_val, 0.0);
    atoms_.elec.compute_atomic_density(atoms_.crystal, atoms_.influence,
                                       ctx.domain(), ctx.grid(),
                                       rho_at.data(), atoms_.Nelectron);
    rho_atomic_ = rho_at;

    // 11. Build initial magnetization for spin-polarized calculations
    int Nspin = ctx.Nspin();
    std::vector<double> mag_init;
    if (Nspin == 2) {
        mag_init.resize(Nd_d_val, 0.0);
        double total_spin = 0.0;
        for (size_t it = 0; it < config_.atom_types.size(); ++it) {
            const auto& at_in = config_.atom_types[it];
            for (size_t ia = 0; ia < at_in.coords.size(); ++ia) {
                if (ia < at_in.spin.size()) total_spin += at_in.spin[ia];
            }
        }
        if (std::abs(total_spin) > 1e-12) {
            double scale = total_spin / static_cast<double>(atoms_.Nelectron);
            for (int i = 0; i < Nd_d_val; ++i)
                mag_init[i] = scale * rho_at[i];
        }
    }

    // 12. Set initial density on SCF
    scf_.set_initial_density(rho_at.data(), Nd_d_val,
                             Nspin == 2 ? mag_init.data() : nullptr);

    setup_done_ = true;
}

double Calculator::run() {
    if (!setup_done_)
        throw std::runtime_error("Calculator::run() called before setup()");

    auto& ctx = lynx::LynxContext::instance();

    double Etot = scf_.run(wfn_, ctx.Nelectron(), ctx.Natom(),
                           atoms_.elec.pseudocharge().data(), atoms_.Vloc.data(),
                           atoms_.elec.Eself(), atoms_.elec.Ec(), config_.xc,
                           atoms_.has_nlcc ? atoms_.rho_core.data() : nullptr);
    scf_converged_ = scf_.converged();
    return Etot;
}

std::vector<double> Calculator::compute_forces() {
    if (!setup_done_)
        throw std::runtime_error("compute_forces() called before run()");

    auto& ctx = lynx::LynxContext::instance();
    lynx::Forces forces;
    forces.compute(ctx, config_, wfn_, scf_, atoms_, vnl_);
    return forces.total_forces();
}

std::array<double, 6> Calculator::compute_stress() {
    if (!setup_done_)
        throw std::runtime_error("compute_stress() called before run()");

    auto& ctx = lynx::LynxContext::instance();
    lynx::Stress stress;
    stress.compute(ctx, config_, wfn_, scf_, atoms_, vnl_);
    return stress.total_stress();
}

double Calculator::compute_pressure() {
    if (!setup_done_)
        throw std::runtime_error("compute_pressure() called before run()");

    auto& ctx = lynx::LynxContext::instance();
    lynx::Stress stress;
    stress.compute(ctx, config_, wfn_, scf_, atoms_, vnl_);
    return stress.pressure();
}

// --- Accessors delegating to LynxContext ---

const lynx::Lattice& Calculator::lattice() const {
    return lynx::LynxContext::instance().lattice();
}

const lynx::FDGrid& Calculator::grid() const {
    return lynx::LynxContext::instance().grid();
}

const lynx::FDStencil& Calculator::stencil() const {
    return lynx::LynxContext::instance().stencil();
}

const lynx::Domain& Calculator::domain() const {
    return lynx::LynxContext::instance().domain();
}

const lynx::KPoints& Calculator::kpoints() const {
    return lynx::LynxContext::instance().kpoints();
}

const lynx::HaloExchange& Calculator::halo() const {
    return lynx::LynxContext::instance().halo();
}

const lynx::Laplacian& Calculator::laplacian() const {
    return lynx::LynxContext::instance().laplacian();
}

const lynx::Gradient& Calculator::gradient() const {
    return lynx::LynxContext::instance().gradient();
}

int Calculator::Nd_d() const {
    return lynx::LynxContext::instance().domain().Nd_d();
}

int Calculator::Nelectron() const {
    return lynx::LynxContext::instance().Nelectron();
}

int Calculator::Natom() const {
    return lynx::LynxContext::instance().Natom();
}

int Calculator::Nspin() const {
    return lynx::LynxContext::instance().Nspin();
}

} // namespace pylynx
