#ifdef USE_CUDA
#include "core/gpu_common.cuh"
#include "operators/Hamiltonian.cuh"
#include "parallel/HaloExchange.cuh"
#include "operators/Laplacian.cuh"

namespace lynx {
namespace gpu {

// Hamiltonian local part: H_local*psi = -0.5*Lap*psi + Veff*psi + c*psi
// Uses batched halo exchange (8 launches total instead of 8*ncol)
// and V2 Laplacian (single launch with multi-column batching)
void hamiltonian_apply_local_gpu(
    const double* d_psi, const double* d_Veff, double* d_Hpsi,
    double* d_x_ex,
    int nx, int ny, int nz, int FDn, int ncol, double c,
    bool is_orthogonal,
    bool periodic_x, bool periodic_y, bool periodic_z,
    double diag_coeff,
    bool has_xy, bool has_xz, bool has_yz,
    cudaStream_t stream)
{
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;

    // Step 1: Batched halo exchange (all columns in ~8 launches)
    halo_exchange_batched_gpu(d_psi, d_x_ex, nx, ny, nz, FDn, ncol,
                               periodic_x, periodic_y, periodic_z, stream);

    // Step 2: Apply (-0.5*Lap + Veff + c*I) using V2 kernel
    if (is_orthogonal) {
        laplacian_orth_v7_gpu(d_x_ex, d_Veff, d_Hpsi,
                               nx, ny, nz, FDn, nx_ex, ny_ex,
                               -0.5, 1.0, c, diag_coeff, ncol, stream);
    } else {
        laplacian_nonorth_gpu(d_x_ex, d_Veff, d_Hpsi,
                              nx, ny, nz, FDn, nx_ex, ny_ex,
                              -0.5, 1.0, c, diag_coeff,
                              has_xy, has_xz, has_yz, ncol, stream);
    }
}

} // namespace gpu
} // namespace lynx
#endif // USE_CUDA
