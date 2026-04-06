#include "physics/Forces.hpp"
#include "operators/Hamiltonian.hpp"
#include "solvers/EigenSolver.hpp"
#include "core/LynxContext.hpp"
#include <mpi.h>
#include <cstdio>

namespace lynx {

void Forces::compute_nonlocal_gpu(
    const Wavefunction& wfn,
    const Crystal& crystal,
    const std::vector<AtomNlocInfluence>& nloc_influence,
    const NonlocalProjector& vnl,
    const std::vector<double>& kpt_weights) {

    int n_atom = crystal.n_atom_total();
    int Nspin_local = wfn.Nspin();
    int Nkpts = wfn.Nkpts();
    int Nband = wfn.Nband();
    bool is_kpt = wfn.is_complex();

    // Determine global Nspin from spincomm
    const auto& spincomm = ctx_->spin_bridge();
    int Nspin_g = Nspin_local;
    if (!spincomm.is_null() && spincomm.size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &Nspin_g, 1, MPI_INT, MPI_SUM, spincomm.comm());
    }
    double occfac = (Nspin_g == 1) ? 2.0 : 1.0;

    f_nloc_.assign(3 * n_atom, 0.0);
    double energy_nl = 0.0;

    int kpt_start = ctx_->kpt_start();
    const KPoints* kpoints = &ctx_->kpoints();
    Vec3 cell_lengths = ctx_->grid().lattice().lengths();

    for (int s = 0; s < Nspin_local; ++s) {
        for (int k = 0; k < Nkpts; ++k) {
            double wk = kpt_weights[kpt_start + k];
            const double* h_occ = wfn.occupations(s, k).data();

            std::vector<double> h_f_tmp(3 * n_atom, 0.0);
            double h_enl_tmp = 0.0;

            if (is_kpt) {
                int k_glob = kpt_start + k;
                Vec3 kpt = kpoints->kpts_cart()[k_glob];
                const_cast<Hamiltonian*>(hamiltonian_)->set_kpoint_gpu(kpt, cell_lengths);

                double spn_fac_wk = occfac * 2.0 * wk;
                hamiltonian_->compute_nonlocal_force_kpt_gpu(
                    eigsolver_->device_psi_z(s, k), h_occ, Nband,
                    spn_fac_wk,
                    h_f_tmp.data(), &h_enl_tmp);
            } else {
                hamiltonian_->compute_nonlocal_force_gpu(
                    eigsolver_->device_psi_real(s, k), h_occ, Nband,
                    occfac,
                    h_f_tmp.data(), &h_enl_tmp);
            }

            for (int i = 0; i < 3 * n_atom; ++i)
                f_nloc_[i] += h_f_tmp[i];
            energy_nl += h_enl_tmp;
        }
    }

    // Allreduce across band, kpt, spin comms
    const auto& bandcomm = ctx_->scf_bandcomm();
    const auto& kptcomm = ctx_->kpt_bridge();
    if (!bandcomm.is_null() && bandcomm.size() > 1)
        MPI_Allreduce(MPI_IN_PLACE, f_nloc_.data(), 3 * n_atom, MPI_DOUBLE, MPI_SUM, bandcomm.comm());
    if (!kptcomm.is_null() && kptcomm.size() > 1)
        MPI_Allreduce(MPI_IN_PLACE, f_nloc_.data(), 3 * n_atom, MPI_DOUBLE, MPI_SUM, kptcomm.comm());
    if (!spincomm.is_null() && spincomm.size() > 1)
        MPI_Allreduce(MPI_IN_PLACE, f_nloc_.data(), 3 * n_atom, MPI_DOUBLE, MPI_SUM, spincomm.comm());

    int rank_world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
    if (rank_world == 0)
        std::printf("GPU nonlocal forces computed: psi stayed on device\n");
}

} // namespace lynx
