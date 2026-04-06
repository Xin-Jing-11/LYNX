#include "physics/Stress.hpp"
#include "operators/Hamiltonian.hpp"
#include "solvers/EigenSolver.hpp"
#include "core/LynxContext.hpp"
#include <mpi.h>
#include <cstdio>

namespace lynx {

void Stress::compute_nonlocal_kinetic_gpu(
    const Wavefunction& wfn,
    const Crystal& crystal,
    const std::vector<AtomNlocInfluence>& nloc_influence,
    const NonlocalProjector& vnl,
    const std::vector<double>& kpt_weights) {

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

    stress_k_.fill(0.0);
    stress_nl_.fill(0.0);

    int kpt_start = ctx_->kpt_start();
    const KPoints* kpoints = &ctx_->kpoints();
    Vec3 cell_lengths = ctx_->grid().lattice().lengths();

    for (int s = 0; s < Nspin_local; ++s) {
        for (int k = 0; k < Nkpts; ++k) {
            double wk = kpt_weights[kpt_start + k];
            const double* h_occ = wfn.occupations(s, k).data();

            std::array<double, 6> h_sk_tmp = {}, h_snl_tmp = {};

            if (is_kpt) {
                int k_glob = kpt_start + k;
                Vec3 kpt = kpoints->kpts_cart()[k_glob];
                const_cast<Hamiltonian*>(hamiltonian_)->set_kpoint_gpu(kpt, cell_lengths);

                double spn_fac_wk = occfac * 2.0 * wk;
                hamiltonian_->compute_kinetic_nonlocal_stress_kpt_gpu(
                    eigsolver_->device_psi_z(s, k), h_occ, Nband,
                    spn_fac_wk,
                    h_sk_tmp.data(), h_snl_tmp.data());
            } else {
                hamiltonian_->compute_kinetic_nonlocal_stress_gpu(
                    eigsolver_->device_psi_real(s, k), h_occ, Nband,
                    occfac,
                    h_sk_tmp.data(), h_snl_tmp.data());
            }

            for (int i = 0; i < 6; ++i) {
                stress_k_[i] += h_sk_tmp[i];
                stress_nl_[i] += h_snl_tmp[i];
            }
        }
    }

    // Allreduce across band, kpt, spin comms
    const auto& bandcomm = ctx_->scf_bandcomm();
    const auto& kptcomm = ctx_->kpt_bridge();
    if (!bandcomm.is_null() && bandcomm.size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, stress_k_.data(), 6, MPI_DOUBLE, MPI_SUM, bandcomm.comm());
        MPI_Allreduce(MPI_IN_PLACE, stress_nl_.data(), 6, MPI_DOUBLE, MPI_SUM, bandcomm.comm());
    }
    if (!kptcomm.is_null() && kptcomm.size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, stress_k_.data(), 6, MPI_DOUBLE, MPI_SUM, kptcomm.comm());
        MPI_Allreduce(MPI_IN_PLACE, stress_nl_.data(), 6, MPI_DOUBLE, MPI_SUM, kptcomm.comm());
    }
    if (!spincomm.is_null() && spincomm.size() > 1) {
        MPI_Allreduce(MPI_IN_PLACE, stress_k_.data(), 6, MPI_DOUBLE, MPI_SUM, spincomm.comm());
        MPI_Allreduce(MPI_IN_PLACE, stress_nl_.data(), 6, MPI_DOUBLE, MPI_SUM, spincomm.comm());
    }

    int rank_world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
    if (rank_world == 0)
        std::printf("GPU kinetic+nonlocal stress computed: psi stayed on device\n");
}

} // namespace lynx
