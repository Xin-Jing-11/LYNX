#include <mpi.h>
#include <cstdio>
#include <string>
#include <array>

#include "io/InputParser.hpp"
#include "core/LynxContext.hpp"
#include "atoms/Crystal.hpp"
#include "atoms/AtomSetup.hpp"
#include "operators/NonlocalProjector.hpp"
#include "operators/Hamiltonian.hpp"
#include "physics/SCF.hpp"
#include "physics/Forces.hpp"
#include "physics/Stress.hpp"
#include "xc/ExactExchange.hpp"

#ifdef USE_CUDA
#include "physics/GPUSCF.cuh"
#endif

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0)
            std::fprintf(stderr, "Usage: lynx <input.json>\n");
        MPI_Finalize();
        return 1;
    }

    try {
        // 1. Parse input
        auto config = lynx::InputParser::parse(argv[1]);
        lynx::InputParser::resolve_pseudopotentials(config);
        lynx::InputParser::validate(config);

        // 2. Initialize infrastructure (grid, parallel, operators)
        auto& ctx = lynx::LynxContext::instance();
        ctx.initialize(config, MPI_COMM_WORLD);

        // 3. Setup atoms and electrostatics
        auto atoms = lynx::Crystal::setup(config, ctx);
        ctx.set_atom_info(atoms.Natom, atoms.Nelectron);

        // 4. Setup operators (NonlocalProjector, Hamiltonian)
        auto vnl = lynx::NonlocalProjector::create(ctx, atoms.crystal, atoms.nloc_influence);
        lynx::Hamiltonian hamiltonian;
        hamiltonian.setup(ctx.stencil(), ctx.domain(), ctx.grid(), ctx.halo(), &vnl);

        // 5. Run SCF
        auto [wfn, scf] = lynx::SCF::run_calculation(config, ctx, atoms.crystal, atoms,
                                                       hamiltonian, vnl);

        // 6. Post-SCF: forces
        if (config.print_forces) {
            lynx::Forces forces;
            forces.compute(ctx, wfn, atoms.crystal,
                           atoms.influence, atoms.nloc_influence, vnl,
                           scf.phi(), scf.density().rho_total().data(),
                           atoms.Vloc.data(),
                           atoms.elec.pseudocharge().data(),
                           atoms.elec.pseudocharge_ref().data(),
                           scf.Vxc(),
                           atoms.has_nlcc ? atoms.rho_core.data() : nullptr);
            forces.print(rank, ctx.is_soc(), atoms.has_nlcc, atoms.Natom);
        }

        // 7. Post-SCF: stress
        if (config.calc_stress || config.calc_pressure) {
            int Nspin_calc = (config.spin_type == lynx::SpinType::Collinear) ? 2 :
                             (config.spin_type == lynx::SpinType::NonCollinear) ? 1 : 1;
            const double* rho_up_ptr = (Nspin_calc == 2) ? scf.density().rho(0).data() : nullptr;
            const double* rho_dn_ptr = (Nspin_calc == 2) ? scf.density().rho(1).data() : nullptr;

            // GPU mGGA stress
            const double* gpu_mgga_ptr = nullptr;
            const double* gpu_dot_ptr = nullptr;
            std::array<double, 6> gpu_mgga_stress = {};
            double gpu_tau_vtau_dot = 0.0;
#ifdef USE_CUDA
            {
                bool is_mgga = (config.xc == lynx::XCType::MGGA_SCAN ||
                                config.xc == lynx::XCType::MGGA_RSCAN ||
                                config.xc == lynx::XCType::MGGA_R2SCAN);
                if (is_mgga && scf.gpu_runner() && !ctx.is_kpt()) {
                    scf.gpu_runner()->compute_mgga_stress(
                        wfn, ctx.domain(), ctx.grid(), Nspin_calc,
                        gpu_mgga_stress.data(), &gpu_tau_vtau_dot);
                    gpu_mgga_ptr = gpu_mgga_stress.data();
                    gpu_dot_ptr = &gpu_tau_vtau_dot;
                }
            }
#endif

            lynx::Stress stress;
            auto sigma = stress.compute(ctx, wfn, atoms.crystal,
                                        atoms.influence, atoms.nloc_influence, vnl,
                                        scf.phi(), scf.density().rho_total().data(),
                                        rho_up_ptr, rho_dn_ptr,
                                        atoms.Vloc.data(),
                                        atoms.elec.pseudocharge().data(),
                                        atoms.elec.pseudocharge_ref().data(),
                                        scf.exc(), scf.Vxc(),
                                        scf.Dxcdgrho(),
                                        scf.energy().Exc,
                                        atoms.elec.Eself() + atoms.elec.Ec(),
                                        config.xc,
                                        Nspin_calc,
                                        atoms.has_nlcc ? atoms.rho_core.data() : nullptr,
                                        scf.vtau(), scf.tau(),
                                        gpu_mgga_ptr, gpu_dot_ptr);

            // Add EXX stress for hybrid functionals
            if (lynx::is_hybrid(config.xc) && scf.exx().is_setup()) {
                auto stress_exx = scf.exx().compute_stress(wfn, ctx.gradient(), ctx.halo(), ctx.domain());
                stress.add_to_total(stress_exx);
            }

            stress.print(rank);
        }

        if (rank == 0) std::printf("\nLYNX calculation complete.\n");

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error on rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Reset LynxContext before MPI_Finalize to free MPI communicators
    // while MPI is still active (the singleton destructor runs too late).
    lynx::LynxContext::instance().reset();
    MPI_Finalize();
    return 0;
}
