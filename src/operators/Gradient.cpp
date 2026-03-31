#include "operators/Gradient.hpp"
#include <omp.h>

namespace lynx {

Gradient::Gradient(const FDStencil& stencil, const Domain& domain)
    : stencil_(&stencil), domain_(&domain) {}

void Gradient::apply(const double* x, double* y, int direction, int ncol) const {
    apply_impl(x, y, direction, ncol);
}

void Gradient::apply(const Complex* x, Complex* y, int direction, int ncol) const {
    apply_impl(x, y, direction, ncol);
}

template<typename T>
void Gradient::apply_impl(const T* x, T* y, int direction, int ncol) const {
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

    int nd = nx * ny * nz;
    int nxny = nx * ny;
    int nxny_ex = nx_ex * ny_ex;
    int nd_ex = nxny_ex * (nz + 2 * FDn);
    int total_ck = ncol * nz;
    #pragma omp parallel for schedule(static)
    for (int ck = 0; ck < total_ck; ++ck) {
        int col = ck / nz;
        int k = ck % nz;
        const T* xc = x + col * nd_ex;
        T* yc = y + col * nd;

        for (int j = 0; j < ny; ++j) {
            int offset_ex = (k + FDn) * nxny_ex + (j + FDn) * nx_ex + FDn;
            int offset = k * nxny + j * nx;
            #pragma omp simd
            for (int i = 0; i < nx; ++i) {
                int idx = offset_ex + i;
                T val = T(0);
                for (int p = 1; p <= FDn; ++p) {
                    val += coeff[p] * (xc[idx + p * stride] - xc[idx - p * stride]);
                }
                yc[offset + i] = val;
            }
        }
    }
}

// Explicit instantiations
template void Gradient::apply_impl<double>(const double*, double*, int, int) const;
template void Gradient::apply_impl<Complex>(const Complex*, Complex*, int, int) const;

} // namespace lynx
