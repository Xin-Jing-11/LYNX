#include "io/DensityIO.hpp"
#include <cstdio>
#include <cstring>

namespace lynx {

bool DensityIO::write(const std::string& filename,
                      const ElectronDensity& density,
                      const FDGrid& grid,
                      const Lattice& lattice) {
    FILE* fp = std::fopen(filename.c_str(), "wb");
    if (!fp) return false;

    // Build header
    FileHeader hdr{};
    std::memcpy(hdr.magic, MAGIC, 8);
    hdr.version = VERSION;
    hdr.nspin = static_cast<uint32_t>(density.Nspin());
    hdr.nx = static_cast<uint32_t>(grid.Nx());
    hdr.ny = static_cast<uint32_t>(grid.Ny());
    hdr.nz = static_cast<uint32_t>(grid.Nz());
    hdr._pad = 0;

    const auto& lv = lattice.latvec();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            hdr.latvec[i * 3 + j] = lv(i, j);
    hdr.dV = grid.dV();
    std::memset(hdr._reserved, 0, sizeof(hdr._reserved));

    // Write header
    if (std::fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
        std::fclose(fp);
        return false;
    }

    // Write per-spin density data
    int Nd = grid.Nx() * grid.Ny() * grid.Nz();
    for (int s = 0; s < density.Nspin(); ++s) {
        const double* data = density.rho(s).data();
        if (std::fwrite(data, sizeof(double), Nd, fp) != static_cast<size_t>(Nd)) {
            std::fclose(fp);
            return false;
        }
    }

    std::fclose(fp);
    return true;
}

bool DensityIO::read(const std::string& filename,
                     ElectronDensity& density,
                     const FDGrid& grid,
                     const Lattice& lattice) {
    FILE* fp = std::fopen(filename.c_str(), "rb");
    if (!fp) return false;

    // Read header
    FileHeader hdr{};
    if (std::fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        std::fclose(fp);
        return false;
    }

    // Validate magic
    if (std::memcmp(hdr.magic, MAGIC, 8) != 0) {
        std::fclose(fp);
        return false;
    }

    // Validate version
    if (hdr.version != VERSION) {
        std::fclose(fp);
        return false;
    }

    // Validate grid dimensions
    if (hdr.nx != static_cast<uint32_t>(grid.Nx()) ||
        hdr.ny != static_cast<uint32_t>(grid.Ny()) ||
        hdr.nz != static_cast<uint32_t>(grid.Nz())) {
        std::fclose(fp);
        return false;
    }

    // Validate spin count
    int nspin = static_cast<int>(hdr.nspin);
    if (nspin != density.Nspin()) {
        std::fclose(fp);
        return false;
    }

    // Read per-spin density data
    int Nd = grid.Nx() * grid.Ny() * grid.Nz();
    for (int s = 0; s < nspin; ++s) {
        double* data = density.rho(s).data();
        if (std::fread(data, sizeof(double), Nd, fp) != static_cast<size_t>(Nd)) {
            std::fclose(fp);
            return false;
        }
    }

    std::fclose(fp);

    // Recompute rho_total and magnetization from loaded spin densities
    double* total = density.rho_total().data();
    if (nspin == 1) {
        std::memcpy(total, density.rho(0).data(), Nd * sizeof(double));
    } else {
        const double* up = density.rho(0).data();
        const double* dn = density.rho(1).data();
        double* mag = density.mag().data();
        for (int i = 0; i < Nd; ++i) {
            total[i] = up[i] + dn[i];
            mag[i] = up[i] - dn[i];
        }
    }

    return true;
}

} // namespace lynx
