/**
 * Compare H*psi between our C++ code and reference SPARC C code.
 *
 * Usage:
 *   1. Run reference SPARC with SPARC_DUMP_HPSI=1 to generate /tmp/ref_*.bin
 *   2. Run this test: mpirun -np 1 ./build/src/sparc_hpsi_compare examples/BaTiO3_quick.json
 *
 * This test:
 *   - Sets up the system from JSON input
 *   - Reads reference Veff, psi, stencil, and H*psi from binary dumps
 *   - Compares our FD stencil coefficients with reference
 *   - Sets psi to same deterministic pattern sin(0.01*(i+1))
 *   - Uses reference Veff (to isolate Hamiltonian from upstream differences)
 *   - Applies H*psi (local only, then full with nonlocal)
 *   - Compares with reference results
 */
#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>

#include "io/InputParser.hpp"
#include "core/Lattice.hpp"
#include "core/FDGrid.hpp"
#include "core/Domain.hpp"
#include "operators/FDStencil.hpp"
#include "operators/Laplacian.hpp"
#include "operators/Gradient.hpp"
#include "operators/Hamiltonian.hpp"
#include "operators/NonlocalProjector.hpp"
#include "atoms/Crystal.hpp"
#include "atoms/AtomType.hpp"
#include "parallel/Parallelization.hpp"
#include "parallel/HaloExchange.hpp"

// Read binary file helper
static std::vector<double> read_bin(const char* path, int skip_ints = 0) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); return {}; }
    // Skip integer headers
    for (int i = 0; i < skip_ints; i++) {
        int dummy; fread(&dummy, sizeof(int), 1, f);
    }
    // Read rest as doubles
    fseek(f, 0, SEEK_END);
    long pos = ftell(f);
    long start = skip_ints * sizeof(int);
    int ndoubles = (pos - start) / sizeof(double);
    fseek(f, start, SEEK_SET);
    std::vector<double> data(ndoubles);
    fread(data.data(), sizeof(double), ndoubles, f);
    fclose(f);
    return data;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "Usage: sparc_hpsi_compare <input.json>\n");
        MPI_Finalize(); return 1;
    }

    try {
        std::string input_file = argv[1];
        auto config = sparc::InputParser::parse(input_file);
        sparc::InputParser::validate(config);

        sparc::Lattice lattice(config.latvec, config.cell_type);
        sparc::FDGrid grid(config.Nx, config.Ny, config.Nz, lattice,
                           config.bcx, config.bcy, config.bcz);
        sparc::FDStencil stencil(config.fd_order, grid, lattice);

        // Setup parallelization (1 proc)
        int Nkpts = 1, Nspin = 1;
        sparc::Parallelization parallel(MPI_COMM_WORLD, config.parallel,
                                        grid, Nspin, Nkpts, config.Nstates);
        const auto& domain = parallel.domain();
        int Nd_d = domain.Nd_d();

        printf("=== H*psi Comparison Test ===\n");
        printf("Grid: %d x %d x %d = %d\n", config.Nx, config.Ny, config.Nz, Nd_d);
        printf("FD order: %d, FDn: %d\n", config.fd_order, stencil.FDn());

        // === Step 1: Compare FD stencil coefficients ===
        printf("\n--- Step 1: FD Stencil Coefficients ---\n");
        {
            FILE* fs = fopen("/tmp/ref_stencil.bin", "rb");
            if (fs) {
                int ref_FDn; fread(&ref_FDn, sizeof(int), 1, fs);
                std::vector<double> ref_cx(ref_FDn+1), ref_cy(ref_FDn+1), ref_cz(ref_FDn+1);
                fread(ref_cx.data(), sizeof(double), ref_FDn+1, fs);
                fread(ref_cy.data(), sizeof(double), ref_FDn+1, fs);
                fread(ref_cz.data(), sizeof(double), ref_FDn+1, fs);
                fclose(fs);

                printf("FDn: ref=%d, ours=%d\n", ref_FDn, stencil.FDn());
                double max_err = 0;
                const double* our_cx = stencil.D2_coeff_x();
                const double* our_cy = stencil.D2_coeff_y();
                const double* our_cz = stencil.D2_coeff_z();
                for (int p = 0; p <= stencil.FDn(); p++) {
                    double ex = std::abs(our_cx[p] - ref_cx[p]);
                    double ey = std::abs(our_cy[p] - ref_cy[p]);
                    double ez = std::abs(our_cz[p] - ref_cz[p]);
                    max_err = std::max({max_err, ex, ey, ez});
                    printf("  p=%d: cx ref=%.15e ours=%.15e err=%.2e\n", p, ref_cx[p], our_cx[p], ex);
                }
                printf("  Max stencil error: %.2e\n", max_err);
            } else {
                printf("  No ref stencil dump found\n");
            }
        }

        // === Step 2: Read reference Veff and compare ===
        printf("\n--- Step 2: Veff ---\n");
        std::vector<double> ref_Veff;
        {
            FILE* fv = fopen("/tmp/ref_Veff.bin", "rb");
            if (fv) {
                int ref_nd; fread(&ref_nd, sizeof(int), 1, fv);
                ref_Veff.resize(ref_nd);
                fread(ref_Veff.data(), sizeof(double), ref_nd, fv);
                fclose(fv);
                printf("  Ref Veff: Nd=%d, range=[%.6e, %.6e]\n", ref_nd,
                       *std::min_element(ref_Veff.begin(), ref_Veff.end()),
                       *std::max_element(ref_Veff.begin(), ref_Veff.end()));
                printf("  Veff[0..4]: %.15e %.15e %.15e %.15e %.15e\n",
                       ref_Veff[0], ref_Veff[1], ref_Veff[2], ref_Veff[3], ref_Veff[4]);
            }
        }

        // === Step 3: Create deterministic psi ===
        printf("\n--- Step 3: Deterministic psi ---\n");
        std::vector<double> psi(Nd_d);
        for (int i = 0; i < Nd_d; i++) {
            psi[i] = std::sin(0.01 * (i + 1));
        }

        // Read reference psi to verify match
        {
            FILE* fp = fopen("/tmp/ref_psi.bin", "rb");
            if (fp) {
                int ref_ncol; fread(&ref_ncol, sizeof(int), 1, fp);
                std::vector<double> ref_psi(Nd_d);
                fread(ref_psi.data(), sizeof(double), Nd_d, fp);
                fclose(fp);
                double max_psi_err = 0;
                for (int i = 0; i < Nd_d; i++)
                    max_psi_err = std::max(max_psi_err, std::abs(psi[i] - ref_psi[i]));
                printf("  Max psi mismatch: %.2e (should be 0)\n", max_psi_err);
            }
        }

        // === Step 4: Setup Hamiltonian and apply local part ===
        printf("\n--- Step 4: Local H*psi (Lap + Veff) ---\n");
        sparc::HaloExchange halo(domain, stencil.FDn());

        // Setup Hamiltonian WITHOUT nonlocal (to test local part separately)
        sparc::Hamiltonian hamiltonian;
        hamiltonian.setup(stencil, domain, grid, halo, nullptr);

        std::vector<double> Hpsi_local(Nd_d, 0.0);
        // Use reference Veff to isolate Hamiltonian from upstream differences
        hamiltonian.apply(psi.data(), ref_Veff.data(), Hpsi_local.data(), 1, 0.0);

        // Read reference local Hpsi
        {
            std::vector<double> ref_Hpsi_local(Nd_d);
            FILE* fl = fopen("/tmp/ref_Hpsi_local.bin", "rb");
            if (fl) {
                fread(ref_Hpsi_local.data(), sizeof(double), Nd_d, fl);
                fclose(fl);

                double max_err = 0, max_rel_err = 0, l2_err = 0, l2_ref = 0;
                int worst_idx = 0;
                for (int i = 0; i < Nd_d; i++) {
                    double err = std::abs(Hpsi_local[i] - ref_Hpsi_local[i]);
                    double rel = (std::abs(ref_Hpsi_local[i]) > 1e-15) ? err / std::abs(ref_Hpsi_local[i]) : 0;
                    if (err > max_err) { max_err = err; worst_idx = i; }
                    max_rel_err = std::max(max_rel_err, rel);
                    l2_err += err * err;
                    l2_ref += ref_Hpsi_local[i] * ref_Hpsi_local[i];
                }
                printf("  Max abs error: %.6e at idx %d\n", max_err, worst_idx);
                printf("  Max rel error: %.6e\n", max_rel_err);
                printf("  L2 relative:   %.6e\n", std::sqrt(l2_err / l2_ref));
                printf("  Our  Hpsi_local[0..4]: %.15e %.15e %.15e %.15e %.15e\n",
                       Hpsi_local[0], Hpsi_local[1], Hpsi_local[2], Hpsi_local[3], Hpsi_local[4]);
                printf("  Ref  Hpsi_local[0..4]: %.15e %.15e %.15e %.15e %.15e\n",
                       ref_Hpsi_local[0], ref_Hpsi_local[1], ref_Hpsi_local[2], ref_Hpsi_local[3], ref_Hpsi_local[4]);
                printf("  Worst: ours=%.15e ref=%.15e at idx=%d\n",
                       Hpsi_local[worst_idx], ref_Hpsi_local[worst_idx], worst_idx);
            }
        }

        // === Step 5: Setup nonlocal projectors and full H*psi ===
        printf("\n--- Step 5: Full H*psi (with nonlocal) ---\n");

        // Load pseudopotentials and create Crystal (same as main.cpp)
        std::vector<sparc::AtomType> atom_types;
        std::vector<sparc::Vec3> all_positions;
        std::vector<int> type_indices;

        for (size_t it = 0; it < config.atom_types.size(); ++it) {
            const auto& at_in = config.atom_types[it];
            int n_atoms = static_cast<int>(at_in.coords.size());
            sparc::Pseudopotential psd_tmp;
            psd_tmp.load_psp8(at_in.pseudo_file);
            double Zval = psd_tmp.Zval();
            sparc::AtomType atype(at_in.element, 1.0, Zval, n_atoms);
            atype.psd().load_psp8(at_in.pseudo_file);

            for (int ia = 0; ia < n_atoms; ++ia) {
                sparc::Vec3 pos = at_in.coords[ia];
                if (at_in.fractional) pos = lattice.frac_to_cart(pos);
                all_positions.push_back(pos);
                type_indices.push_back(static_cast<int>(it));
            }
            atom_types.push_back(std::move(atype));
        }

        sparc::Crystal crystal(std::move(atom_types), all_positions, type_indices, lattice);

        std::vector<sparc::AtomNlocInfluence> nloc_influence;
        crystal.compute_nloc_influence(domain, nloc_influence);

        sparc::NonlocalProjector vnl;
        vnl.setup(crystal, nloc_influence, domain, grid);
        printf("  Nonlocal projectors: %d total\n", vnl.total_nproj());

        // Dump per-atom Chi info for comparison with reference
        printf("\n--- Per-atom Chi info ---\n");
        for (size_t it = 0; it < nloc_influence.size(); it++) {
            const auto& inf = nloc_influence[it];
            const auto& psd = crystal.types()[it].psd();
            int nproj = psd.nproj_per_atom();
            for (int iat = 0; iat < inf.n_atom; iat++) {
                int ndc = inf.ndc[iat];
                double chi_norm = 0;
                const auto& chi = vnl.Chi()[it][iat];
                for (int ig = 0; ig < ndc; ig++)
                    for (int jp = 0; jp < nproj; jp++)
                        chi_norm += chi(ig, jp) * chi(ig, jp);
                chi_norm = std::sqrt(chi_norm);
                printf("OUR_CHI type=%d iat=%d ndc=%d nproj=%d coord=(%.10f,%.10f,%.10f) Chi_norm=%.15e Chi[0]=%.15e\n",
                       (int)it, iat, ndc, nproj,
                       inf.coords[iat].x, inf.coords[iat].y, inf.coords[iat].z,
                       chi_norm, ndc > 0 ? chi(0, 0) : 0.0);
            }
        }

        // Setup full Hamiltonian
        sparc::Hamiltonian hamiltonian_full;
        hamiltonian_full.setup(stencil, domain, grid, halo, &vnl);

        std::vector<double> Hpsi_full(Nd_d, 0.0);
        hamiltonian_full.apply(psi.data(), ref_Veff.data(), Hpsi_full.data(), 1, 0.0);

        // Nonlocal contribution = full - local
        std::vector<double> Hpsi_nloc(Nd_d);
        for (int i = 0; i < Nd_d; i++)
            Hpsi_nloc[i] = Hpsi_full[i] - Hpsi_local[i];

        printf("  Nonlocal contribution: min=%.6e max=%.6e\n",
               *std::min_element(Hpsi_nloc.begin(), Hpsi_nloc.end()),
               *std::max_element(Hpsi_nloc.begin(), Hpsi_nloc.end()));

        // Compare with reference full Hpsi
        {
            std::vector<double> ref_Hpsi_full(Nd_d);
            FILE* ff = fopen("/tmp/ref_Hpsi_full.bin", "rb");
            if (ff) {
                fread(ref_Hpsi_full.data(), sizeof(double), Nd_d, ff);
                fclose(ff);

                // Reference nonlocal contribution
                std::vector<double> ref_Hpsi_local(Nd_d);
                FILE* fl = fopen("/tmp/ref_Hpsi_local.bin", "rb");
                fread(ref_Hpsi_local.data(), sizeof(double), Nd_d, fl);
                fclose(fl);

                std::vector<double> ref_nloc(Nd_d);
                for (int i = 0; i < Nd_d; i++)
                    ref_nloc[i] = ref_Hpsi_full[i] - ref_Hpsi_local[i];
                printf("  Ref nonlocal: min=%.6e max=%.6e\n",
                       *std::min_element(ref_nloc.begin(), ref_nloc.end()),
                       *std::max_element(ref_nloc.begin(), ref_nloc.end()));

                // Compare nonlocal contribution
                double max_nloc_err = 0, l2_nloc = 0, l2_ref_nloc = 0;
                int worst_nloc_idx = 0;
                for (int i = 0; i < Nd_d; i++) {
                    double err = std::abs(Hpsi_nloc[i] - ref_nloc[i]);
                    if (err > max_nloc_err) { max_nloc_err = err; worst_nloc_idx = i; }
                    l2_nloc += err * err;
                    l2_ref_nloc += ref_nloc[i] * ref_nloc[i];
                }
                printf("  Nonlocal max abs error: %.6e at idx %d\n", max_nloc_err, worst_nloc_idx);
                printf("  Nonlocal L2 relative:   %.6e\n",
                       l2_ref_nloc > 0 ? std::sqrt(l2_nloc / l2_ref_nloc) : 0);

                // Compare full Hpsi
                double max_full_err = 0, l2_full = 0, l2_ref_full = 0;
                int worst_full_idx = 0;
                for (int i = 0; i < Nd_d; i++) {
                    double err = std::abs(Hpsi_full[i] - ref_Hpsi_full[i]);
                    if (err > max_full_err) { max_full_err = err; worst_full_idx = i; }
                    l2_full += err * err;
                    l2_ref_full += ref_Hpsi_full[i] * ref_Hpsi_full[i];
                }
                printf("\n  Full H*psi max abs error: %.6e at idx %d\n", max_full_err, worst_full_idx);
                printf("  Full H*psi L2 relative:   %.6e\n", std::sqrt(l2_full / l2_ref_full));
                printf("  Our  Hpsi_full[0..4]: %.15e %.15e %.15e %.15e %.15e\n",
                       Hpsi_full[0], Hpsi_full[1], Hpsi_full[2], Hpsi_full[3], Hpsi_full[4]);
                printf("  Ref  Hpsi_full[0..4]: %.15e %.15e %.15e %.15e %.15e\n",
                       ref_Hpsi_full[0], ref_Hpsi_full[1], ref_Hpsi_full[2], ref_Hpsi_full[3], ref_Hpsi_full[4]);
            }
        }

        printf("\n=== Comparison Complete ===\n");

    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
