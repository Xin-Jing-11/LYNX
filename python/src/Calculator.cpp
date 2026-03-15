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
    // Skip validation of Nstates here — setup() will auto-compute if <= 0
}

void Calculator::setup(MPI_Comm comm) {
    // Lattice & grid
    lattice_ = lynx::Lattice(config_.latvec, config_.cell_type);
    grid_ = lynx::FDGrid(config_.Nx, config_.Ny, config_.Nz, lattice_,
                          config_.bcx, config_.bcy, config_.bcz);
    stencil_ = lynx::FDStencil(config_.fd_order, grid_, lattice_);

    // K-points
    kpoints_.generate(config_.Kx, config_.Ky, config_.Kz,
                      config_.kpt_shift, lattice_);
    int Nkpts = kpoints_.Nkpts();
    bool is_kpt = !kpoints_.is_gamma_only();
    Nspin_ = (config_.spin_type == lynx::SpinType::None) ? 1 : 2;

    // Parallelization
    int nproc;
    MPI_Comm_size(comm, &nproc);

    if (config_.parallel.npspin <= 1 && Nspin_ == 2 && nproc >= 2)
        config_.parallel.npspin = 2;
    int nproc_after_spin = nproc / config_.parallel.npspin;
    if (config_.parallel.npkpt <= 1 && Nkpts > 1 && nproc_after_spin > 1)
        config_.parallel.npkpt = std::min(nproc_after_spin, Nkpts);
    int nproc_after_kpt = nproc_after_spin / std::max(1, config_.parallel.npkpt);
    if (config_.parallel.npband <= 1 && nproc_after_kpt > 1)
        config_.parallel.npband = nproc_after_kpt;

    Nstates_ = config_.Nstates;

    parallel_ = std::make_unique<lynx::Parallelization>(
        comm, config_.parallel, grid_, Nspin_, Nkpts, Nstates_);

    const auto& domain = parallel_->domain();
    const auto& bandcomm = parallel_->bandcomm();
    const auto& kptcomm = parallel_->kptcomm();
    const auto& kpt_bridge = parallel_->kpt_bridge();
    const auto& spincomm = parallel_->spincomm();
    const auto& spin_bridge = parallel_->spin_bridge();
    int Nkpts_local = parallel_->Nkpts_local();
    int Nspin_local = parallel_->Nspin_local();
    int kpt_start = parallel_->kpt_start();
    int spin_start = parallel_->spin_start();
    int Nband_local = parallel_->Nband_local();
    int band_start = parallel_->band_start();

    // Load pseudopotentials & create crystal
    std::vector<lynx::AtomType> atom_types;
    std::vector<lynx::Vec3> all_positions;
    std::vector<int> type_indices;
    int total_Nelectron = 0;

    for (size_t it = 0; it < config_.atom_types.size(); ++it) {
        const auto& at_in = config_.atom_types[it];
        int n_atoms = static_cast<int>(at_in.coords.size());

        lynx::Pseudopotential psd_tmp;
        psd_tmp.load_psp8(at_in.pseudo_file);
        double Zval = psd_tmp.Zval();
        double mass = 1.0;

        lynx::AtomType atype(at_in.element, mass, Zval, n_atoms);
        atype.psd().load_psp8(at_in.pseudo_file);

        for (int ia = 0; ia < n_atoms; ++ia) {
            lynx::Vec3 pos = at_in.coords[ia];
            if (at_in.fractional) {
                if (lattice_.is_orthogonal()) {
                    pos = lattice_.frac_to_cart(pos);
                } else {
                    lynx::Vec3 L = lattice_.lengths();
                    pos = {pos.x * L.x, pos.y * L.y, pos.z * L.z};
                }
            } else {
                if (!lattice_.is_orthogonal()) {
                    pos = lattice_.cart_to_nonCart(pos);
                }
            }
            all_positions.push_back(pos);
            type_indices.push_back(static_cast<int>(it));
        }

        total_Nelectron += static_cast<int>(Zval) * n_atoms;
        atom_types.push_back(std::move(atype));
    }

    Nelectron_ = (config_.Nelectron > 0) ? config_.Nelectron : total_Nelectron;
    Natom_ = static_cast<int>(all_positions.size());
    if (Nstates_ <= 0) Nstates_ = Nelectron_ / 2 + 10;

    crystal_ = lynx::Crystal(std::move(atom_types), all_positions, type_indices, lattice_);

    // Atom influence
    double rc_max = 0.0;
    for (int it = 0; it < crystal_.n_types(); ++it) {
        const auto& psd = crystal_.types()[it].psd();
        for (auto rc : psd.rc()) rc_max = std::max(rc_max, rc);
        if (!psd.radial_grid().empty())
            rc_max = std::max(rc_max, psd.radial_grid().back());
    }
    double h_max = std::max({grid_.dx(), grid_.dy(), grid_.dz()});
    rc_max += 8.0 * h_max;

    crystal_.compute_atom_influence(domain, rc_max, influence_);
    crystal_.compute_nloc_influence(domain, nloc_influence_);

    // Electrostatics
    elec_.compute_pseudocharge(crystal_, influence_, domain, grid_, stencil_);

    int Nd_d_val = domain.Nd_d();
    Vloc_.assign(Nd_d_val, 0.0);
    elec_.compute_Vloc(crystal_, influence_, domain, grid_, Vloc_.data());
    elec_.compute_Ec(Vloc_.data(), Nd_d_val, grid_.dV());

    // Operators
    halo_ = lynx::HaloExchange(domain, stencil_.FDn());
    laplacian_ = lynx::Laplacian(stencil_, domain);
    gradient_ = lynx::Gradient(stencil_, domain);

    vnl_.setup(crystal_, nloc_influence_, domain, grid_);
    hamiltonian_.setup(stencil_, domain, grid_, halo_, &vnl_);

    // SCF setup
    lynx::SCFParams scf_params;
    scf_params.max_iter = config_.max_scf_iter;
    scf_params.min_iter = config_.min_scf_iter;
    scf_params.tol = config_.scf_tol;
    scf_params.mixing_var = config_.mixing_var;
    scf_params.mixing_precond = config_.mixing_precond;
    scf_params.mixing_history = config_.mixing_history;
    scf_params.mixing_param = config_.mixing_param;
    scf_params.smearing = config_.smearing;
    scf_params.elec_temp = config_.elec_temp;
    scf_params.cheb_degree = config_.cheb_degree;
    scf_params.rho_trigger = config_.rho_trigger;

    const auto& scf_bandcomm = (config_.parallel.npband > 1) ? kptcomm : bandcomm;
    scf_.setup(grid_, domain, stencil_, laplacian_, gradient_, hamiltonian_,
               halo_, &vnl_, scf_bandcomm, kpt_bridge, spin_bridge, scf_params,
               Nspin_, Nspin_local, spin_start, &kpoints_, kpt_start,
               Nstates_, band_start);

    // Wavefunctions
    wfn_.allocate(Nd_d_val, Nband_local, Nstates_, Nspin_local, Nkpts_local, is_kpt);

    // Initial density (atomic superposition)
    std::vector<double> rho_at(Nd_d_val, 0.0);
    elec_.compute_atomic_density(crystal_, influence_, domain, grid_,
                                  rho_at.data(), Nelectron_);

    std::vector<double> mag_init;
    if (Nspin_ == 2) {
        mag_init.resize(Nd_d_val, 0.0);
        double total_spin = 0.0;
        for (size_t it = 0; it < config_.atom_types.size(); ++it) {
            const auto& at_in = config_.atom_types[it];
            for (size_t ia = 0; ia < at_in.coords.size(); ++ia) {
                if (ia < at_in.spin.size()) total_spin += at_in.spin[ia];
            }
        }
        if (std::abs(total_spin) > 1e-12) {
            double scale = total_spin / static_cast<double>(Nelectron_);
            for (int i = 0; i < Nd_d_val; ++i)
                mag_init[i] = scale * rho_at[i];
        }
    }

    scf_.set_initial_density(rho_at.data(), Nd_d_val,
                              Nspin_ == 2 ? mag_init.data() : nullptr);

    // NLCC
    rho_core_.assign(Nd_d_val, 0.0);
    has_nlcc_ = elec_.compute_core_density(crystal_, influence_, domain, grid_,
                                            rho_core_.data());

    setup_done_ = true;
}

double Calculator::run() {
    if (!setup_done_)
        throw std::runtime_error("Calculator::run() called before setup()");

    double Etot = scf_.run(wfn_, Nelectron_, Natom_,
                            elec_.pseudocharge().data(), Vloc_.data(),
                            elec_.Eself(), elec_.Ec(), config_.xc,
                            has_nlcc_ ? rho_core_.data() : nullptr);
    scf_converged_ = scf_.converged();
    return Etot;
}

std::vector<double> Calculator::compute_forces() {
    if (!setup_done_)
        throw std::runtime_error("compute_forces() called before run()");

    const auto& domain = parallel_->domain();
    const auto& bandcomm = parallel_->bandcomm();
    const auto& kptcomm = parallel_->kptcomm();
    const auto& kpt_bridge = parallel_->kpt_bridge();
    const auto& spin_bridge = parallel_->spin_bridge();
    int kpt_start = parallel_->kpt_start();
    int band_start = parallel_->band_start();
    const auto& scf_bandcomm = (config_.parallel.npband > 1) ? kptcomm : bandcomm;

    std::vector<double> kpt_weights = kpoints_.normalized_weights();
    lynx::Forces forces;
    return forces.compute(wfn_, crystal_, influence_, nloc_influence_, vnl_,
                          stencil_, gradient_, halo_, domain, grid_,
                          scf_.phi(), scf_.density().rho_total().data(),
                          Vloc_.data(),
                          elec_.pseudocharge().data(),
                          elec_.pseudocharge_ref().data(),
                          scf_.Vxc(),
                          has_nlcc_ ? rho_core_.data() : nullptr,
                          kpt_weights, scf_bandcomm, kpt_bridge, spin_bridge,
                          &kpoints_, kpt_start, band_start);
}

std::array<double, 6> Calculator::compute_stress() {
    if (!setup_done_)
        throw std::runtime_error("compute_stress() called before run()");

    const auto& domain = parallel_->domain();
    const auto& bandcomm = parallel_->bandcomm();
    const auto& kptcomm = parallel_->kptcomm();
    const auto& kpt_bridge = parallel_->kpt_bridge();
    const auto& spin_bridge = parallel_->spin_bridge();
    int kpt_start = parallel_->kpt_start();
    int band_start = parallel_->band_start();
    const auto& scf_bandcomm = (config_.parallel.npband > 1) ? kptcomm : bandcomm;

    std::vector<double> kpt_weights = kpoints_.normalized_weights();
    lynx::Stress stress;
    int Nspin_calc = (config_.spin_type == lynx::SpinType::Collinear) ? 2 : 1;
    const double* rho_up = (Nspin_calc == 2) ? scf_.density().rho(0).data() : nullptr;
    const double* rho_dn = (Nspin_calc == 2) ? scf_.density().rho(1).data() : nullptr;

    return stress.compute(wfn_, crystal_, influence_, nloc_influence_, vnl_,
                          stencil_, gradient_, halo_, domain, grid_,
                          scf_.phi(), scf_.density().rho_total().data(),
                          rho_up, rho_dn,
                          Vloc_.data(),
                          elec_.pseudocharge().data(),
                          elec_.pseudocharge_ref().data(),
                          scf_.exc(), scf_.Vxc(),
                          scf_.Dxcdgrho(),
                          scf_.energy().Exc,
                          elec_.Eself() + elec_.Ec(),
                          config_.xc,
                          Nspin_calc,
                          has_nlcc_ ? rho_core_.data() : nullptr,
                          kpt_weights, scf_bandcomm, kpt_bridge, spin_bridge,
                          &kpoints_, kpt_start, band_start);
}

const lynx::Domain& Calculator::domain() const {
    if (!parallel_)
        throw std::runtime_error("Calculator not set up");
    return parallel_->domain();
}

int Calculator::Nd_d() const {
    if (!parallel_)
        throw std::runtime_error("Calculator not set up");
    return parallel_->domain().Nd_d();
}

} // namespace pylynx
