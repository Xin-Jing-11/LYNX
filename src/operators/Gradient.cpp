#include "operators/Gradient.hpp"

namespace sparc {

Gradient::Gradient(const FDStencil& stencil, const Domain& domain)
    : stencil_(&stencil), domain_(&domain) {}

void Gradient::apply(const double* x, double* y, int direction, int ncol) const {
    int nx = domain_->Nx_d();
    int ny = domain_->Ny_d();
    int nz = domain_->Nz_d();
    int FDn = stencil_->FDn();

    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;

    const double* coeff = nullptr;
    int stride = 0;
    switch (direction) {
        case 0: coeff = stencil_->D1_coeff_x(); stride = 1; break;
        case 1: coeff = stencil_->D1_coeff_y(); stride = nx_ex; break;
        case 2: coeff = stencil_->D1_coeff_z(); stride = nx_ex * ny_ex; break;
    }

    for (int col = 0; col < ncol; ++col) {
        int nd = nx * ny * nz;
        const double* xc = x + col * nx_ex * ny_ex * (nz + 2 * FDn);
        double* yc = y + col * nd;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int idx = (i + FDn) + (j + FDn) * nx_ex + (k + FDn) * nx_ex * ny_ex;
                    double val = 0.0;
                    for (int p = 1; p <= FDn; ++p) {
                        val += coeff[p] * (xc[idx + p * stride] - xc[idx - p * stride]);
                    }
                    yc[i + j * nx + k * nx * ny] = val;
                }
            }
        }
    }
}

} // namespace sparc
