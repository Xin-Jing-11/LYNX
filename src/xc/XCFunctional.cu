#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <xc.h>
#include <xc_funcs.h>
#include <cublas_v2.h>

// Helper: init libxc functional for device-side evaluation (GPU kernels).
static inline int xc_func_init_device(xc_func_type* p, int functional, int nspin) {
    return xc_func_init_flags(p, functional, nspin, XC_FLAGS_ON_DEVICE);
}
#include "core/gpu_common.cuh"
#include "core/GPUContext.cuh"
#include "core/LynxContext.hpp"
#include "xc/XCFunctional.hpp"
#include "parallel/HaloExchange.cuh"
#include "operators/Gradient.cuh"

namespace lynx {
namespace gpu {

// Hand-written LDA/GGA/mGGA XC kernels removed — replaced by libxc CUDA device calls.

// ============================================================

// sigma = |grad(rho)|^2 for orthogonal cells
__global__ void sigma_kernel(
    const double* __restrict__ Drho_x,
    const double* __restrict__ Drho_y,
    const double* __restrict__ Drho_z,
    double* __restrict__ sigma,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double dx = Drho_x[i], dy = Drho_y[i], dz = Drho_z[i];
        sigma[i] = dx*dx + dy*dy + dz*dz;
    }
}

// Multiply gradient by v2xc in place: Drho_dir[i] *= v2xc[i]
__global__ void v2xc_scale_kernel(
    double* __restrict__ f,
    const double* __restrict__ v2xc,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) f[i] *= v2xc[i];
}

// Add NLCC core density: rho_xc = max(rho + rho_core, 1e-14)
__global__ void nlcc_add_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ rho_core,
    double* __restrict__ rho_xc,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double val = rho[i] + rho_core[i];
        rho_xc[i] = (val > 1e-14) ? val : 1e-14;
    }
}

// Subtract divergence: Vxc[i] -= DDrho[i]
__global__ void divergence_sub_kernel(
    double* __restrict__ Vxc,
    const double* __restrict__ DDrho,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) Vxc[i] -= DDrho[i];
}

// Jacobi preconditioner: f[i] = scale * r[i]
__global__ void jacobi_scale_kernel(
    const double* __restrict__ r,
    double* __restrict__ f,
    double scale, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) f[i] = scale * r[i];
}

// ============================================================
// GGA PBE Exchange + Correlation — spin-polarized fused kernel

} // namespace gpu

// ============================================================
// Kernels duplicated from GPUSCF.cu for XC Device-dispatch
// (file-static to avoid ODR violations)
// ============================================================

namespace {

__global__ void xc_sigma_nonorth_kernel(
    const double* __restrict__ Drho_x,
    const double* __restrict__ Drho_y,
    const double* __restrict__ Drho_z,
    double* __restrict__ sigma, int N,
    double L00, double L11, double L22, double L01, double L02, double L12)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double dx = Drho_x[i], dy = Drho_y[i], dz = Drho_z[i];
        sigma[i] = L00*dx*dx + L11*dy*dy + L22*dz*dz
                 + 2.0*L01*dx*dy + 2.0*L02*dx*dz + 2.0*L12*dy*dz;
    }
}

__global__ void xc_sigma_3col_kernel(
    const double* __restrict__ Drho_x,
    const double* __restrict__ Drho_y,
    const double* __restrict__ Drho_z,
    double* __restrict__ sigma, int N, int ncol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_N = N * ncol;
    if (idx < total_N) {
        double dx = Drho_x[idx], dy = Drho_y[idx], dz = Drho_z[idx];
        sigma[idx] = dx*dx + dy*dy + dz*dz;
    }
}

__global__ void xc_sigma_3col_nonorth_kernel(
    const double* __restrict__ Drho_x,
    const double* __restrict__ Drho_y,
    const double* __restrict__ Drho_z,
    double* __restrict__ sigma, int N, int ncol,
    double L00, double L11, double L22, double L01, double L02, double L12)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_N = N * ncol;
    if (idx < total_N) {
        double dx = Drho_x[idx], dy = Drho_y[idx], dz = Drho_z[idx];
        sigma[idx] = L00*dx*dx + L11*dy*dy + L22*dz*dz
                   + 2.0*L01*dx*dy + 2.0*L02*dx*dz + 2.0*L12*dy*dz;
    }
}

__global__ void xc_v2xc_scale_3col_kernel(
    double* __restrict__ f,
    const double* __restrict__ v2xc, int N, int ncol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_N = N * ncol;
    if (idx < total_N) f[idx] *= v2xc[idx];
}

__global__ void xc_nlcc_add_spin_kernel(
    const double* __restrict__ rho_up,
    const double* __restrict__ rho_dn,
    const double* __restrict__ rho_core,
    double* __restrict__ rho_xc, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double core_half = 0.5 * rho_core[i];
        double rup = rho_up[i] + core_half;
        double rdn = rho_dn[i] + core_half;
        double rtot = rup + rdn;
        rho_xc[i]       = (rtot > 1e-14) ? rtot : 1e-14;
        rho_xc[N + i]   = (rup > 1e-14) ? rup : 1e-14;
        rho_xc[2*N + i] = (rdn > 1e-14) ? rdn : 1e-14;
    }
}

__global__ void xc_rho_xc_spin_kernel(
    const double* __restrict__ rho_up,
    const double* __restrict__ rho_dn,
    double* __restrict__ rho_xc, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double rup = rho_up[i], rdn = rho_dn[i];
        double rtot = rup + rdn;
        rho_xc[i]       = (rtot > 1e-14) ? rtot : 1e-14;
        rho_xc[N + i]   = (rup > 1e-14) ? rup : 1e-14;
        rho_xc[2*N + i] = (rdn > 1e-14) ? rdn : 1e-14;
    }
}

__global__ void xc_spin_divergence_add_kernel(
    double* __restrict__ Vxc_up,
    double* __restrict__ Vxc_dn,
    const double* __restrict__ DDrho,
    int Nd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Nd) {
        Vxc_up[i] -= DDrho[i] + DDrho[Nd + i];
        Vxc_dn[i] -= DDrho[i] + DDrho[2*Nd + i];
    }
}

__global__ void xc_lapcT_flux_kernel(
    double* __restrict__ Drho_x,
    double* __restrict__ Drho_y,
    double* __restrict__ Drho_z,
    const double* __restrict__ v2xc, int N,
    double L00, double L01, double L02,
    double L10, double L11, double L12,
    double L20, double L21, double L22)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double dx = Drho_x[i], dy = Drho_y[i], dz = Drho_z[i];
        double v = v2xc[i];
        Drho_x[i] = v * (L00*dx + L01*dy + L02*dz);
        Drho_y[i] = v * (L10*dx + L11*dy + L12*dz);
        Drho_z[i] = v * (L20*dx + L21*dy + L22*dz);
    }
}

__global__ void xc_lapcT_flux_3col_kernel(
    double* __restrict__ Drho_x,
    double* __restrict__ Drho_y,
    double* __restrict__ Drho_z,
    const double* __restrict__ v2xc, int N, int ncol,
    double L00, double L01, double L02,
    double L10, double L11, double L12,
    double L20, double L21, double L22)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_N = N * ncol;
    if (idx < total_N) {
        double dx = Drho_x[idx], dy = Drho_y[idx], dz = Drho_z[idx];
        double v = v2xc[idx];
        Drho_x[idx] = v * (L00*dx + L01*dy + L02*dz);
        Drho_y[idx] = v * (L10*dx + L11*dy + L12*dz);
        Drho_z[idx] = v * (L20*dx + L21*dy + L22*dz);
    }
}

// Interleave two separate spin arrays into libxc's interleaved format:
// out[2*i] = a[i], out[2*i+1] = b[i]
__global__ void xc_interleave2_kernel(
    const double* __restrict__ a, const double* __restrict__ b,
    double* __restrict__ out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[2*i]     = a[i];
        out[2*i + 1] = b[i];
    }
}

// Interleave three separate arrays into libxc's [uu,ud,dd] format:
// out[3*i] = a[i], out[3*i+1] = b[i], out[3*i+2] = c[i]
__global__ void xc_interleave3_kernel(
    const double* __restrict__ a, const double* __restrict__ b,
    const double* __restrict__ c, double* __restrict__ out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[3*i]     = a[i];
        out[3*i + 1] = b[i];
        out[3*i + 2] = c[i];
    }
}

// Deinterleave libxc's interleaved vrho[2*N] into separate up/dn arrays,
// adding to existing values: out_a[i] += in[2*i], out_b[i] += in[2*i+1]
__global__ void xc_deinterleave2_add_kernel(
    const double* __restrict__ in,
    double* __restrict__ out_a, double* __restrict__ out_b, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out_a[i] += in[2*i];
        out_b[i] += in[2*i + 1];
    }
}

// Deinterleave libxc's interleaved vrho[2*N] into separate up/dn arrays (overwrite)
__global__ void xc_deinterleave2_kernel(
    const double* __restrict__ in,
    double* __restrict__ out_a, double* __restrict__ out_b, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out_a[i] = in[2*i];
        out_b[i] = in[2*i + 1];
    }
}

// Deinterleave libxc's vsigma[3*N] into separate uu/ud/dd arrays (overwrite)
__global__ void xc_deinterleave3_kernel(
    const double* __restrict__ in,
    double* __restrict__ a, double* __restrict__ b, double* __restrict__ c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        a[i] = in[3*i];
        b[i] = in[3*i + 1];
        c[i] = in[3*i + 2];
    }
}

// Deinterleave and add libxc's vsigma[3*N] into separate arrays
__global__ void xc_deinterleave3_add_kernel(
    const double* __restrict__ in,
    double* __restrict__ a, double* __restrict__ b, double* __restrict__ c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        a[i] += in[3*i];
        b[i] += in[3*i + 1];
        c[i] += in[3*i + 2];
    }
}

// Compute sigma_ud = (sigma_tot - sigma_uu - sigma_dd) / 2
__global__ void xc_sigma_ud_kernel(
    const double* __restrict__ sigma_tot,
    const double* __restrict__ sigma_uu,
    const double* __restrict__ sigma_dd,
    double* __restrict__ sigma_ud, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        sigma_ud[i] = 0.5 * (sigma_tot[i] - sigma_uu[i] - sigma_dd[i]);
    }
}

// Scale array by 2: out[i] = 2 * in[i]
__global__ void xc_scale2_kernel(
    const double* __restrict__ in, double* __restrict__ out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = 2.0 * in[i];
}

} // anonymous namespace

// ============================================================

// ============================================================

struct GPUXCState {
    // Grid parameters
    int nx = 0, ny = 0, nz = 0, FDn = 0, Nd = 0;
    bool is_orth = true;
    bool has_mixed_deriv = false;

    // Metric tensor (row-major 3x3)
    double lapcT[9] = {};

    // XC type flags
    XCType xc_type = XCType::GGA_PBE;
    bool is_gga = false;
    bool is_mgga = false;

    // NLCC
    bool has_nlcc = false;
    double* d_rho_core = nullptr;

    // tau pointer (for mGGA, set externally)
    double* d_tau = nullptr;
    double* d_vtau = nullptr;
    bool tau_valid = false;

    // NOTE: GGA/mGGA workspace buffers (grad_rho, sigma, v2xc, x_ex, rho_xc)
    // are NOT owned here — they come from GPUContext::buf or scratch_pool,
    // allocated on use in evaluate()/evaluate_spin().
};

// ── setup_gpu / cleanup_gpu ──────────────────────────

void XCFunctional::setup_gpu(const LynxContext& ctx, int Nspin) {
    if (!gpu_state_)
        gpu_state_.reset(new GPUXCState());
    auto* gs = gpu_state_.as<GPUXCState>();

    const auto& grid    = ctx.grid();
    const auto& domain  = ctx.domain();
    const auto& stencil = ctx.stencil();

    gs->nx  = grid.Nx();
    gs->ny  = grid.Ny();
    gs->nz  = grid.Nz();
    gs->FDn = stencil.FDn();
    gs->Nd  = domain.Nd_d();
    gs->is_orth         = grid.lattice().is_orthogonal();
    gs->has_mixed_deriv = !gs->is_orth;

    gs->xc_type = type_;
    gs->is_gga  = is_gga() || is_mgga();
    gs->is_mgga = is_mgga();

    // Metric tensor
    {
        const auto& L = grid.lattice().lapc_T();
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                gs->lapcT[i * 3 + j] = L(i, j);
    }

    // GGA/mGGA workspace is NOT allocated here — it comes from
    // GPUContext::buf (non-spin) or scratch_pool (spin) at call time.
}

void XCFunctional::cleanup_gpu() {
    if (!gpu_state_) return;
    auto* gs = gpu_state_.as<GPUXCState>();

    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    auto safe_free = [stream](auto*& p) { if (p) { cudaFreeAsync(p, stream); p = nullptr; } };

    safe_free(gs->d_rho_core);

    gpu_state_.reset();
}

XCFunctional::~XCFunctional() {
    cleanup_gpu();
}

void XCFunctional::set_gpu_nlcc(const double* rho_core, int Nd) {
    auto* gs = gpu_state_.as<GPUXCState>();
    if (!gs) return;
    cudaStream_t stream = gpu::GPUContext::instance().compute_stream;
    auto safe_free = [stream](auto*& p) { if (p) { cudaFreeAsync(p, stream); p = nullptr; } };
    safe_free(gs->d_rho_core);
    if (rho_core) {
        gs->has_nlcc = true;
        CUDA_CHECK(cudaMallocAsync(&gs->d_rho_core, Nd * sizeof(double), stream));
        CUDA_CHECK(cudaMemcpyAsync(gs->d_rho_core, rho_core, Nd * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    } else {
        gs->has_nlcc = false;
    }
}

void XCFunctional::set_gpu_tau_valid(bool valid) {
    auto* gs = gpu_state_.as<GPUXCState>();
    if (gs) gs->tau_valid = valid;
}

// ============================================================
// GPU evaluate — full GGA pipeline on device
// ============================================================
void XCFunctional::evaluate_gpu(const double* rho, double* Vxc, double* exc, int Nd_d,
                                 double* Dxcdgrho,
                                 const double* tau, double* vtau) const {
    auto* gs = gpu_state_.as<GPUXCState>();
    auto& ctx = gpu::GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;
    int bs = 256;
    int grid_sz = gpu::ceildiv(Nd_d, bs);
    int nx_ex = gs->nx + 2 * gs->FDn, ny_ex = gs->ny + 2 * gs->FDn;

    bool use_gga = gs->is_gga || gs->is_mgga;
    if (use_gga) {
        // GGA / mGGA path
        double* d_Drho_x = ctx.buf.grad_rho;
        double* d_Drho_y = ctx.buf.grad_rho + Nd_d;
        double* d_Drho_z = ctx.buf.grad_rho + 2 * Nd_d;
        double* d_sigma  = ctx.buf.aar_r;      // reuse
        double* d_v2xc   = ctx.buf.Dxcdgrho;
        double* d_x_ex   = ctx.buf.aar_x_ex;

        // NLCC: rho + rho_core
        double* d_rho_xc;
        if (gs->has_nlcc && gs->d_rho_core) {
            d_rho_xc = ctx.buf.b;
            gpu::nlcc_add_kernel<<<grid_sz, bs, 0, stream>>>(
                rho, gs->d_rho_core, d_rho_xc, Nd_d);
        } else {
            d_rho_xc = const_cast<double*>(rho);
        }

        // Gradient
        gpu::halo_exchange_gpu(d_rho_xc, d_x_ex, gs->nx, gs->ny, gs->nz,
                               gs->FDn, 1, true, true, true, stream);
        gpu::gradient_gpu(d_x_ex, d_Drho_x, gs->nx, gs->ny, gs->nz,
                          gs->FDn, nx_ex, ny_ex, 0, 1, stream);
        gpu::gradient_gpu(d_x_ex, d_Drho_y, gs->nx, gs->ny, gs->nz,
                          gs->FDn, nx_ex, ny_ex, 1, 1, stream);
        gpu::gradient_gpu(d_x_ex, d_Drho_z, gs->nx, gs->ny, gs->nz,
                          gs->FDn, nx_ex, ny_ex, 2, 1, stream);

        // sigma = |nabla rho|^2
        if (gs->is_orth) {
            gpu::sigma_kernel<<<grid_sz, bs, 0, stream>>>(
                d_Drho_x, d_Drho_y, d_Drho_z, d_sigma, Nd_d);
        } else {
            xc_sigma_nonorth_kernel<<<grid_sz, bs, 0, stream>>>(
                d_Drho_x, d_Drho_y, d_Drho_z, d_sigma, Nd_d,
                gs->lapcT[0], gs->lapcT[4], gs->lapcT[8],
                gs->lapcT[1], gs->lapcT[2], gs->lapcT[5]);
        }

        // XC evaluation via libxc (device pointers)
        {
            int xc_id, cc_id;
            get_func_ids(xc_id, cc_id);
            xc_func_type func_x, func_c;
            xc_func_init_device(&func_x, xc_id, XC_UNPOLARIZED);
            xc_func_init_device(&func_c, cc_id, XC_UNPOLARIZED);

            auto& sp = ctx.scratch_pool;
            size_t sp_cp = sp.checkpoint();

            if (gs->is_mgga && gs->tau_valid && tau) {
                // mGGA: exc, vrho, vsigma, vlapl, vtau
                double* d_lapl = sp.alloc<double>(Nd_d);  // zero laplacian
                CUDA_CHECK(cudaMemsetAsync(d_lapl, 0, Nd_d * sizeof(double), stream));
                double* d_exc_c  = sp.alloc<double>(Nd_d);
                double* d_vrho_c = sp.alloc<double>(Nd_d);
                double* d_vsig_x = sp.alloc<double>(Nd_d);
                double* d_vsig_c = sp.alloc<double>(Nd_d);
                double* d_vlapl  = sp.alloc<double>(Nd_d);
                double* d_vtau_x = sp.alloc<double>(Nd_d);
                double* d_vtau_c = sp.alloc<double>(Nd_d);

                xc_mgga_exc_vxc(&func_x, Nd_d, d_rho_xc, d_sigma, d_lapl, tau,
                                exc, Vxc, d_vsig_x, d_vlapl, d_vtau_x);
                xc_mgga_exc_vxc(&func_c, Nd_d, d_rho_xc, d_sigma, d_lapl, tau,
                                d_exc_c, d_vrho_c, d_vsig_c, d_vlapl, d_vtau_c);
                // Accumulate: exc += exc_c, Vxc += vrho_c, vsig = vsig_x + vsig_c, vtau = vtau_x + vtau_c
                double one = 1.0;
                cublasDaxpy(ctx.cublas, Nd_d, &one, d_exc_c, 1, exc, 1);
                cublasDaxpy(ctx.cublas, Nd_d, &one, d_vrho_c, 1, Vxc, 1);
                cublasDaxpy(ctx.cublas, Nd_d, &one, d_vsig_c, 1, d_vsig_x, 1);
                // d_v2xc = 2 * (vsig_x + vsig_c)
                double two = 2.0;
                CUDA_CHECK(cudaMemcpyAsync(d_v2xc, d_vsig_x, Nd_d * sizeof(double), cudaMemcpyDeviceToDevice, stream));
                cublasDscal(ctx.cublas, Nd_d, &two, d_v2xc, 1);
                // vtau = vtau_x + vtau_c
                cublasDaxpy(ctx.cublas, Nd_d, &one, d_vtau_c, 1, d_vtau_x, 1);
                CUDA_CHECK(cudaMemcpyAsync(const_cast<double*>(vtau), d_vtau_x, Nd_d * sizeof(double), cudaMemcpyDeviceToDevice, stream));
            } else {
                // GGA: exc, vrho, vsigma
                double* d_exc_c  = sp.alloc<double>(Nd_d);
                double* d_vrho_c = sp.alloc<double>(Nd_d);
                double* d_vsig_x = sp.alloc<double>(Nd_d);
                double* d_vsig_c = sp.alloc<double>(Nd_d);

                xc_gga_exc_vxc(&func_x, Nd_d, d_rho_xc, d_sigma, exc, Vxc, d_vsig_x);
                xc_gga_exc_vxc(&func_c, Nd_d, d_rho_xc, d_sigma, d_exc_c, d_vrho_c, d_vsig_c);
                // Accumulate
                double one = 1.0;
                cublasDaxpy(ctx.cublas, Nd_d, &one, d_exc_c, 1, exc, 1);
                cublasDaxpy(ctx.cublas, Nd_d, &one, d_vrho_c, 1, Vxc, 1);
                cublasDaxpy(ctx.cublas, Nd_d, &one, d_vsig_c, 1, d_vsig_x, 1);
                // d_v2xc = 2 * (vsig_x + vsig_c)
                double two = 2.0;
                CUDA_CHECK(cudaMemcpyAsync(d_v2xc, d_vsig_x, Nd_d * sizeof(double), cudaMemcpyDeviceToDevice, stream));
                cublasDscal(ctx.cublas, Nd_d, &two, d_v2xc, 1);
            }
            sp.restore(sp_cp);
            xc_func_end(&func_x);
            xc_func_end(&func_c);
        }

        // Divergence correction
        if (gs->is_orth) {
            gpu::v2xc_scale_kernel<<<grid_sz, bs, 0, stream>>>(d_Drho_x, d_v2xc, Nd_d);
            gpu::v2xc_scale_kernel<<<grid_sz, bs, 0, stream>>>(d_Drho_y, d_v2xc, Nd_d);
            gpu::v2xc_scale_kernel<<<grid_sz, bs, 0, stream>>>(d_Drho_z, d_v2xc, Nd_d);
        } else {
            xc_lapcT_flux_kernel<<<grid_sz, bs, 0, stream>>>(
                d_Drho_x, d_Drho_y, d_Drho_z, d_v2xc, Nd_d,
                gs->lapcT[0], gs->lapcT[1], gs->lapcT[2],
                gs->lapcT[3], gs->lapcT[4], gs->lapcT[5],
                gs->lapcT[6], gs->lapcT[7], gs->lapcT[8]);
        }

        double* d_DDrho = d_sigma;  // reuse
        // x-direction
        gpu::halo_exchange_gpu(d_Drho_x, d_x_ex, gs->nx, gs->ny, gs->nz,
                               gs->FDn, 1, true, true, true, stream);
        gpu::gradient_gpu(d_x_ex, d_DDrho, gs->nx, gs->ny, gs->nz,
                          gs->FDn, nx_ex, ny_ex, 0, 1, stream);
        gpu::divergence_sub_kernel<<<grid_sz, bs, 0, stream>>>(Vxc, d_DDrho, Nd_d);
        // y-direction
        gpu::halo_exchange_gpu(d_Drho_y, d_x_ex, gs->nx, gs->ny, gs->nz,
                               gs->FDn, 1, true, true, true, stream);
        gpu::gradient_gpu(d_x_ex, d_DDrho, gs->nx, gs->ny, gs->nz,
                          gs->FDn, nx_ex, ny_ex, 1, 1, stream);
        gpu::divergence_sub_kernel<<<grid_sz, bs, 0, stream>>>(Vxc, d_DDrho, Nd_d);
        // z-direction
        gpu::halo_exchange_gpu(d_Drho_z, d_x_ex, gs->nx, gs->ny, gs->nz,
                               gs->FDn, 1, true, true, true, stream);
        gpu::gradient_gpu(d_x_ex, d_DDrho, gs->nx, gs->ny, gs->nz,
                          gs->FDn, nx_ex, ny_ex, 2, 1, stream);
        gpu::divergence_sub_kernel<<<grid_sz, bs, 0, stream>>>(Vxc, d_DDrho, Nd_d);

    } else {
        // LDA path — libxc with device pointers
        double* d_rho_xc;
        if (gs->has_nlcc && gs->d_rho_core) {
            d_rho_xc = ctx.buf.b;
            gpu::nlcc_add_kernel<<<grid_sz, bs, 0, stream>>>(
                rho, gs->d_rho_core, d_rho_xc, Nd_d);
        } else {
            d_rho_xc = const_cast<double*>(rho);
        }
        int xc_id, cc_id;
        get_func_ids(xc_id, cc_id);
        xc_func_type func_x, func_c;
        xc_func_init_device(&func_x, xc_id, XC_UNPOLARIZED);
        xc_func_init_device(&func_c, cc_id, XC_UNPOLARIZED);

        // Exchange into exc, Vxc
        xc_lda_exc_vxc(&func_x, Nd_d, d_rho_xc, exc, Vxc);
        // Correlation into temps, then accumulate
        auto& sp = ctx.scratch_pool;
        size_t sp_cp = sp.checkpoint();
        double* d_exc_c  = sp.alloc<double>(Nd_d);
        double* d_vrho_c = sp.alloc<double>(Nd_d);
        xc_lda_exc_vxc(&func_c, Nd_d, d_rho_xc, d_exc_c, d_vrho_c);
        double one = 1.0;
        cublasDaxpy(ctx.cublas, Nd_d, &one, d_exc_c, 1, exc, 1);
        cublasDaxpy(ctx.cublas, Nd_d, &one, d_vrho_c, 1, Vxc, 1);
        sp.restore(sp_cp);
        xc_func_end(&func_x);
        xc_func_end(&func_c);
    }
}

// ============================================================
// GPU evaluate_spin — full GGA pipeline on device (spin-polarized)
// ============================================================
void XCFunctional::evaluate_spin_gpu(const double* rho, double* Vxc, double* exc, int Nd_d,
                                      double* Dxcdgrho,
                                      const double* tau, double* vtau) const {
    // GPU path — mirrors GPUSCF::gpu_xc_evaluate_spin()
    auto* gs = gpu_state_.as<GPUXCState>();
    auto& ctx = gpu::GPUContext::instance();
    cudaStream_t stream = ctx.compute_stream;
    int bs = 256;
    int grid_sz = gpu::ceildiv(Nd_d, bs);
    int nx_ex = gs->nx + 2 * gs->FDn, ny_ex = gs->ny + 2 * gs->FDn;

    // rho layout: [up(Nd)|dn(Nd)], Vxc layout: [up(Nd)|dn(Nd)]
    const double* d_rho_up = rho;
    const double* d_rho_dn = rho + Nd_d;

    bool use_gga = gs->is_gga || gs->is_mgga;
    if (use_gga) {
        // Spin GGA / mGGA path
        auto& sp = ctx.scratch_pool;
        size_t sp_cp = sp.checkpoint();

        double* d_rho_xc  = sp.alloc<double>(3 * Nd_d);
        double* d_sigma    = sp.alloc<double>(3 * Nd_d);
        double* d_Drho_x   = sp.alloc<double>(3 * Nd_d);
        double* d_Drho_y   = sp.alloc<double>(3 * Nd_d);
        double* d_Drho_z   = sp.alloc<double>(3 * Nd_d);
        double* d_v2xc     = ctx.buf.Dxcdgrho;  // [3*Nd]
        double* d_x_ex_tmp = ctx.buf.aar_x_ex;

        // Build rho_xc = [total|up|dn]
        if (gs->has_nlcc && gs->d_rho_core) {
            xc_nlcc_add_spin_kernel<<<grid_sz, bs, 0, stream>>>(
                d_rho_up, d_rho_dn, gs->d_rho_core, d_rho_xc, Nd_d);
        } else {
            xc_rho_xc_spin_kernel<<<grid_sz, bs, 0, stream>>>(
                d_rho_up, d_rho_dn, d_rho_xc, Nd_d);
        }

        // Gradient of all 3 columns
        for (int col = 0; col < 3; col++) {
            gpu::halo_exchange_gpu(d_rho_xc + col * Nd_d, d_x_ex_tmp,
                                   gs->nx, gs->ny, gs->nz, gs->FDn,
                                   1, true, true, true, stream);
            gpu::gradient_gpu(d_x_ex_tmp, d_Drho_x + col * Nd_d,
                              gs->nx, gs->ny, gs->nz, gs->FDn,
                              nx_ex, ny_ex, 0, 1, stream);
            gpu::gradient_gpu(d_x_ex_tmp, d_Drho_y + col * Nd_d,
                              gs->nx, gs->ny, gs->nz, gs->FDn,
                              nx_ex, ny_ex, 1, 1, stream);
            gpu::gradient_gpu(d_x_ex_tmp, d_Drho_z + col * Nd_d,
                              gs->nx, gs->ny, gs->nz, gs->FDn,
                              nx_ex, ny_ex, 2, 1, stream);
        }

        // sigma for 3 columns
        int grid3 = gpu::ceildiv(3 * Nd_d, bs);
        if (gs->is_orth) {
            xc_sigma_3col_kernel<<<grid3, bs, 0, stream>>>(
                d_Drho_x, d_Drho_y, d_Drho_z, d_sigma, Nd_d, 3);
        } else {
            xc_sigma_3col_nonorth_kernel<<<grid3, bs, 0, stream>>>(
                d_Drho_x, d_Drho_y, d_Drho_z, d_sigma, Nd_d, 3,
                gs->lapcT[0], gs->lapcT[4], gs->lapcT[8],
                gs->lapcT[1], gs->lapcT[2], gs->lapcT[5]);
        }

        // XC evaluation via libxc (device pointers, spin-polarized)
        {
            int xc_id, cc_id;
            get_func_ids(xc_id, cc_id);
            xc_func_type func_x, func_c;
            xc_func_init_device(&func_x, xc_id, XC_POLARIZED);
            xc_func_init_device(&func_c, cc_id, XC_POLARIZED);

            // d_rho_xc layout: [total(Nd)|up(Nd)|dn(Nd)]
            // d_sigma   layout: [total(Nd)|uu(Nd)|dd(Nd)]
            double* d_rho_up_xc = d_rho_xc + Nd_d;
            double* d_rho_dn_xc = d_rho_xc + 2 * Nd_d;
            double* d_sigma_uu  = d_sigma + Nd_d;
            double* d_sigma_dd  = d_sigma + 2 * Nd_d;
            double* d_sigma_tot = d_sigma;  // total density sigma

            // Interleave into libxc format
            double* d_rho_2   = sp.alloc<double>(2 * Nd_d);  // [up0,dn0,up1,dn1,...]
            double* d_sigma_3 = sp.alloc<double>(3 * Nd_d);  // [uu0,ud0,dd0,...]
            double* d_sigma_ud = sp.alloc<double>(Nd_d);

            xc_interleave2_kernel<<<grid_sz, bs, 0, stream>>>(
                d_rho_up_xc, d_rho_dn_xc, d_rho_2, Nd_d);
            xc_sigma_ud_kernel<<<grid_sz, bs, 0, stream>>>(
                d_sigma_tot, d_sigma_uu, d_sigma_dd, d_sigma_ud, Nd_d);
            xc_interleave3_kernel<<<grid_sz, bs, 0, stream>>>(
                d_sigma_uu, d_sigma_ud, d_sigma_dd, d_sigma_3, Nd_d);

            // libxc output buffers (interleaved)
            double* d_vrho_2   = sp.alloc<double>(2 * Nd_d);
            double* d_vsig_3   = sp.alloc<double>(3 * Nd_d);
            double* d_vrho_2c  = sp.alloc<double>(2 * Nd_d);
            double* d_vsig_3c  = sp.alloc<double>(3 * Nd_d);
            double* d_exc_c    = sp.alloc<double>(Nd_d);

            if (gs->is_mgga && gs->tau_valid && tau) {
                // mGGA spin path
                double* d_tau_2  = sp.alloc<double>(2 * Nd_d);
                double* d_lapl_2 = sp.alloc<double>(2 * Nd_d);
                CUDA_CHECK(cudaMemsetAsync(d_lapl_2, 0, 2 * Nd_d * sizeof(double), stream));
                xc_interleave2_kernel<<<grid_sz, bs, 0, stream>>>(
                    tau, tau + Nd_d, d_tau_2, Nd_d);

                double* d_vlapl_2  = sp.alloc<double>(2 * Nd_d);
                double* d_vtau_2   = sp.alloc<double>(2 * Nd_d);
                double* d_vtau_2c  = sp.alloc<double>(2 * Nd_d);

                xc_mgga_exc_vxc(&func_x, Nd_d, d_rho_2, d_sigma_3, d_lapl_2, d_tau_2,
                                exc, d_vrho_2, d_vsig_3, d_vlapl_2, d_vtau_2);
                xc_mgga_exc_vxc(&func_c, Nd_d, d_rho_2, d_sigma_3, d_lapl_2, d_tau_2,
                                d_exc_c, d_vrho_2c, d_vsig_3c, d_vlapl_2, d_vtau_2c);

                // Accumulate exc
                double one = 1.0;
                cublasDaxpy(ctx.cublas, Nd_d, &one, d_exc_c, 1, exc, 1);
                // Accumulate vrho and vsigma
                cublasDaxpy(ctx.cublas, 2 * Nd_d, &one, d_vrho_2c, 1, d_vrho_2, 1);
                cublasDaxpy(ctx.cublas, 3 * Nd_d, &one, d_vsig_3c, 1, d_vsig_3, 1);
                cublasDaxpy(ctx.cublas, 2 * Nd_d, &one, d_vtau_2c, 1, d_vtau_2, 1);

                // Deinterleave vrho -> Vxc[up], Vxc[dn]
                xc_deinterleave2_kernel<<<grid_sz, bs, 0, stream>>>(
                    d_vrho_2, Vxc, Vxc + Nd_d, Nd_d);

                // Deinterleave vtau -> vtau[up], vtau[dn]
                xc_deinterleave2_kernel<<<grid_sz, bs, 0, stream>>>(
                    d_vtau_2, const_cast<double*>(vtau), const_cast<double*>(vtau) + Nd_d, Nd_d);
            } else {
                // GGA spin path
                xc_gga_exc_vxc(&func_x, Nd_d, d_rho_2, d_sigma_3, exc, d_vrho_2, d_vsig_3);
                xc_gga_exc_vxc(&func_c, Nd_d, d_rho_2, d_sigma_3, d_exc_c, d_vrho_2c, d_vsig_3c);

                double one = 1.0;
                cublasDaxpy(ctx.cublas, Nd_d, &one, d_exc_c, 1, exc, 1);
                cublasDaxpy(ctx.cublas, 2 * Nd_d, &one, d_vrho_2c, 1, d_vrho_2, 1);
                cublasDaxpy(ctx.cublas, 3 * Nd_d, &one, d_vsig_3c, 1, d_vsig_3, 1);

                // Deinterleave vrho -> Vxc[up], Vxc[dn]
                xc_deinterleave2_kernel<<<grid_sz, bs, 0, stream>>>(
                    d_vrho_2, Vxc, Vxc + Nd_d, Nd_d);
            }

            // Deinterleave vsigma[3*N] -> v2xc layout: [ud(Nd)|uu(Nd)|dd(Nd)]
            // with factor of 2: d_v2xc[col] = 2 * vsigma[col]
            // vsig_3 = [uu0,ud0,dd0,...] combined from exchange + correlation
            // d_v2xc layout: [v2xc_c(Nd) | v2xc_x_up(Nd) | v2xc_x_dn(Nd)]
            //   = [2*vsigma_ud | 2*vsigma_uu | 2*vsigma_dd]
            double* d_vsig_uu = sp.alloc<double>(Nd_d);
            double* d_vsig_ud = sp.alloc<double>(Nd_d);
            double* d_vsig_dd = sp.alloc<double>(Nd_d);
            xc_deinterleave3_kernel<<<grid_sz, bs, 0, stream>>>(
                d_vsig_3, d_vsig_uu, d_vsig_ud, d_vsig_dd, Nd_d);
            // d_v2xc = [2*ud | 2*uu | 2*dd]
            xc_scale2_kernel<<<grid_sz, bs, 0, stream>>>(d_vsig_ud, d_v2xc, Nd_d);
            xc_scale2_kernel<<<grid_sz, bs, 0, stream>>>(d_vsig_uu, d_v2xc + Nd_d, Nd_d);
            xc_scale2_kernel<<<grid_sz, bs, 0, stream>>>(d_vsig_dd, d_v2xc + 2 * Nd_d, Nd_d);

            xc_func_end(&func_x);
            xc_func_end(&func_c);
        }

        // Divergence correction for 3 columns
        if (gs->is_orth) {
            xc_v2xc_scale_3col_kernel<<<grid3, bs, 0, stream>>>(d_Drho_x, d_v2xc, Nd_d, 3);
            xc_v2xc_scale_3col_kernel<<<grid3, bs, 0, stream>>>(d_Drho_y, d_v2xc, Nd_d, 3);
            xc_v2xc_scale_3col_kernel<<<grid3, bs, 0, stream>>>(d_Drho_z, d_v2xc, Nd_d, 3);
        } else {
            xc_lapcT_flux_3col_kernel<<<grid3, bs, 0, stream>>>(
                d_Drho_x, d_Drho_y, d_Drho_z, d_v2xc, Nd_d, 3,
                gs->lapcT[0], gs->lapcT[1], gs->lapcT[2],
                gs->lapcT[3], gs->lapcT[4], gs->lapcT[5],
                gs->lapcT[6], gs->lapcT[7], gs->lapcT[8]);
        }

        // Accumulate divergence
        double* d_DDrho = d_sigma;  // reuse
        CUDA_CHECK(cudaMemsetAsync(d_DDrho, 0, 3 * Nd_d * sizeof(double), stream));

        for (int dir = 0; dir < 3; dir++) {
            double* d_Drho_dir = (dir == 0) ? d_Drho_x : (dir == 1) ? d_Drho_y : d_Drho_z;
            for (int col = 0; col < 3; col++) {
                double* d_DDcol = sp.alloc<double>(Nd_d);
                gpu::halo_exchange_gpu(d_Drho_dir + col * Nd_d, d_x_ex_tmp,
                                       gs->nx, gs->ny, gs->nz, gs->FDn,
                                       1, true, true, true, stream);
                gpu::gradient_gpu(d_x_ex_tmp, d_DDcol,
                                  gs->nx, gs->ny, gs->nz, gs->FDn,
                                  nx_ex, ny_ex, dir, 1, stream);
                double one = 1.0;
                cublasDaxpy(ctx.cublas, Nd_d, &one, d_DDcol, 1,
                            d_DDrho + col * Nd_d, 1);
                sp.restore(sp.checkpoint() - Nd_d * sizeof(double));
            }
        }

        // Apply spin divergence
        xc_spin_divergence_add_kernel<<<grid_sz, bs, 0, stream>>>(
            Vxc, Vxc + Nd_d, d_DDrho, Nd_d);

        sp.restore(sp_cp);
    } else {
        // LDA spin path — libxc with device pointers
        auto& sp_lda = ctx.scratch_pool;
        size_t sp_lda_cp = sp_lda.checkpoint();

        const double* d_rup;
        const double* d_rdn;
        if (gs->has_nlcc && gs->d_rho_core) {
            double* d_rho_xc_3 = sp_lda.alloc<double>(3 * Nd_d);
            xc_nlcc_add_spin_kernel<<<grid_sz, bs, 0, stream>>>(
                d_rho_up, d_rho_dn, gs->d_rho_core, d_rho_xc_3, Nd_d);
            d_rup = d_rho_xc_3 + Nd_d;
            d_rdn = d_rho_xc_3 + 2 * Nd_d;
        } else {
            d_rup = d_rho_up;
            d_rdn = d_rho_dn;
        }

        // Interleave for libxc: rho_2 = [up0,dn0,up1,dn1,...]
        double* d_rho_2 = sp_lda.alloc<double>(2 * Nd_d);
        xc_interleave2_kernel<<<grid_sz, bs, 0, stream>>>(d_rup, d_rdn, d_rho_2, Nd_d);

        int xc_id, cc_id;
        get_func_ids(xc_id, cc_id);
        xc_func_type func_x, func_c;
        xc_func_init_device(&func_x, xc_id, XC_POLARIZED);
        xc_func_init_device(&func_c, cc_id, XC_POLARIZED);

        // Exchange: vrho_2 = [vup0,vdn0,...], exc
        double* d_vrho_2  = sp_lda.alloc<double>(2 * Nd_d);
        xc_lda_exc_vxc(&func_x, Nd_d, d_rho_2, exc, d_vrho_2);
        // Correlation
        double* d_exc_c   = sp_lda.alloc<double>(Nd_d);
        double* d_vrho_2c = sp_lda.alloc<double>(2 * Nd_d);
        xc_lda_exc_vxc(&func_c, Nd_d, d_rho_2, d_exc_c, d_vrho_2c);

        // Accumulate exc += exc_c
        double one = 1.0;
        cublasDaxpy(ctx.cublas, Nd_d, &one, d_exc_c, 1, exc, 1);
        // Accumulate vrho
        cublasDaxpy(ctx.cublas, 2 * Nd_d, &one, d_vrho_2c, 1, d_vrho_2, 1);
        // Deinterleave vrho -> Vxc[up], Vxc[dn]
        xc_deinterleave2_kernel<<<grid_sz, bs, 0, stream>>>(
            d_vrho_2, Vxc, Vxc + Nd_d, Nd_d);

        sp_lda.restore(sp_lda_cp);
        xc_func_end(&func_x);
        xc_func_end(&func_c);
    }
}

} // namespace lynx

#endif // USE_CUDA
