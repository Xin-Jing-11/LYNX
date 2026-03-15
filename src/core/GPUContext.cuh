#pragma once
#ifdef USE_CUDA

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <string>
#include <vector>
#include <cassert>
#include "gpu_common.cuh"

namespace lynx {
namespace gpu {

// ============================================================
// GPUMemoryPool — Stream-ordered async memory pool
// ============================================================
// Uses cudaMallocAsync/cudaFreeAsync with the device's built-in memory pool.
// The pool grows automatically — no fixed-size limit, no OOM crash.
// Freed blocks are reused by subsequent allocations on the same stream.
//
// API preserves bump-allocator semantics (alloc/checkpoint/restore)
// for backward compatibility, but internally each alloc is a separate
// cudaMallocAsync call. checkpoint/restore free blocks allocated after
// the checkpoint.
class GPUMemoryPool {
public:
    GPUMemoryPool() = default;

    // Initialize pool. hint_bytes is ignored (pool grows dynamically).
    void init(size_t /*hint_bytes*/) {
        if (initialized_) return;
        initialized_ = true;
        // Configure the device pool to not release memory back to OS
        cudaMemPool_t pool;
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetDefaultMemPool(&pool, device);
        uint64_t threshold = UINT64_MAX;  // never release
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    }

    ~GPUMemoryPool() {
        free_all();
    }

    GPUMemoryPool(const GPUMemoryPool&) = delete;
    GPUMemoryPool& operator=(const GPUMemoryPool&) = delete;

    // Allocate from pool via cudaMallocAsync
    void* alloc(size_t bytes) {
        if (bytes == 0) bytes = 256;  // avoid zero-size alloc
        void* ptr = nullptr;
        cudaError_t err = cudaMallocAsync(&ptr, bytes, stream_);
        if (err != cudaSuccess || !ptr) {
            std::fprintf(stderr, "GPUMemoryPool: cudaMallocAsync failed for %zu bytes: %s\n",
                         bytes, cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
        allocs_.push_back(ptr);
        used_ += bytes;
        if (used_ > high_water_) high_water_ = used_;
        return ptr;
    }

    // Typed alloc
    template<typename T>
    T* alloc(size_t count) {
        return static_cast<T*>(alloc(count * sizeof(T)));
    }

    // Reset: free all allocations (blocks return to pool for reuse)
    void reset() {
        free_all();
        used_ = 0;
    }

    // Checkpoint: save current allocation count
    size_t checkpoint() const { return allocs_.size(); }

    // Restore: free allocations made after checkpoint
    void restore(size_t cp) {
        while (allocs_.size() > cp) {
            cudaFreeAsync(allocs_.back(), stream_);
            allocs_.pop_back();
        }
    }

    size_t used() const { return used_; }
    size_t capacity() const { return used_; }  // grows dynamically
    size_t high_water() const { return high_water_; }

private:
    cudaStream_t stream_ = nullptr;
    bool initialized_ = false;
    std::vector<void*> allocs_;
    size_t used_ = 0;
    size_t high_water_ = 0;

    void free_all() {
        for (auto* p : allocs_)
            cudaFreeAsync(p, stream_);
        // Sync to ensure frees complete before destructor returns
        if (!allocs_.empty())
            cudaStreamSynchronize(stream_);
        allocs_.clear();
    }
};

// ============================================================
// PinnedMemoryPool — Pinned host memory for async H2D/D2H
// ============================================================
class PinnedMemoryPool {
public:
    PinnedMemoryPool() = default;

    void init(size_t total_bytes) {
        if (base_) return;
        total_bytes_ = total_bytes;
        CUDA_CHECK(cudaMallocHost(&base_, total_bytes_));
        offset_ = 0;
    }

    ~PinnedMemoryPool() {
        if (base_) cudaFreeHost(base_);
    }

    PinnedMemoryPool(const PinnedMemoryPool&) = delete;
    PinnedMemoryPool& operator=(const PinnedMemoryPool&) = delete;

    void* alloc(size_t bytes) {
        constexpr size_t ALIGN = 64;
        size_t aligned_offset = (offset_ + ALIGN - 1) & ~(ALIGN - 1);
        if (aligned_offset + bytes > total_bytes_) {
            std::fprintf(stderr, "PinnedMemoryPool: OOM — requested %zu bytes\n", bytes);
            std::exit(EXIT_FAILURE);
        }
        void* ptr = static_cast<char*>(base_) + aligned_offset;
        offset_ = aligned_offset + bytes;
        return ptr;
    }

    template<typename T>
    T* alloc(size_t count) {
        return static_cast<T*>(alloc(count * sizeof(T)));
    }

    void reset() { offset_ = 0; }
    size_t checkpoint() const { return offset_; }
    void restore(size_t cp) { offset_ = cp; }

private:
    void*  base_ = nullptr;
    size_t total_bytes_ = 0;
    size_t offset_ = 0;
};

// ============================================================
// SCF Buffer Descriptors — Named GPU buffers for the SCF loop
// ============================================================
struct SCFBuffers {
    // Problem dimensions (set by init())
    int Nd = 0;           // Grid points
    int Nd_ex = 0;        // Extended grid (with halos)
    int Ns = 0;           // Number of bands (local)
    int Ns_global = 0;    // Number of bands (global, for band parallelism)
    int Nspin = 0;        // Spin channels
    int FDn = 0;          // FD half-order
    int ld = 0;           // Leading dimension (padded Nd)
    int ld_ex = 0;        // Leading dimension of extended arrays

    // Persistent GPU arrays (live for entire SCF)
    double* psi = nullptr;         // (ld, Ns) — wavefunctions
    double* Hpsi = nullptr;        // (ld, Ns) — H*psi result
    double* Veff = nullptr;        // (Nd, Nspin) — effective potential
    double* Vxc = nullptr;         // (Nd, Nspin) — XC potential
    double* phi = nullptr;         // (Nd) — electrostatic potential
    double* rho = nullptr;         // (Nd, Nspin) — electron density
    double* rho_total = nullptr;   // (Nd) — total density
    double* b = nullptr;           // (Nd) — RHS for Poisson
    double* exc = nullptr;         // (Nd) — XC energy density
    double* Vc = nullptr;          // (Nd) — pseudocharge potential
    double* eigenvalues = nullptr; // (Ns_global) — eigenvalues
    double* occupations = nullptr; // (Ns_global) — occupation numbers

    // CheFSI workspace
    double* Xold = nullptr;       // (ld, Ns) — Chebyshev filter temp
    double* Xnew = nullptr;       // (ld, Ns) — Chebyshev filter temp
    double* x_ex = nullptr;       // (Nd_ex, Ns) — halo-extended workspace

    // Subspace matrices (small, N×N)
    double* Hs = nullptr;         // (Ns_global, Ns_global) — projected Hamiltonian
    double* Ms = nullptr;         // (Ns_global, Ns_global) — overlap / temp
    double* Q = nullptr;          // (Ns_global, Ns_global) — eigenvectors
    double* eig_work = nullptr;   // cuSOLVER workspace

    // AAR solver workspace (Poisson + preconditioner)
    double* aar_r = nullptr;      // (Nd) — residual
    double* aar_f = nullptr;      // (Nd) — preconditioned residual
    double* aar_Ax = nullptr;     // (Nd) — operator output
    double* aar_X = nullptr;      // (Nd, m) — Anderson history
    double* aar_F = nullptr;      // (Nd, m) — Anderson history
    double* aar_x_ex = nullptr;   // (Nd_ex) — single-column halo

    // Mixer workspace (Pulay)
    double* mix_R = nullptr;      // (Nd*ncol, m) — residual history
    double* mix_F = nullptr;      // (Nd*ncol, m) — iterate history
    double* mix_fk = nullptr;     // (Nd*ncol) — current residual
    double* mix_xkm1 = nullptr;   // (Nd*ncol) — previous iterate

    // Nonlocal projector data (persistent, uploaded once)
    double* Chi_flat = nullptr;   // Concatenated Chi matrices
    int*    gpos_flat = nullptr;  // Concatenated grid positions
    int*    atom_offsets = nullptr;// CSR offsets
    double* Gamma_flat = nullptr; // KB energy coefficients
    double* alpha = nullptr;      // (total_nproj, Ns) — inner products

    // GGA workspace
    double* Dxcdgrho = nullptr;   // (Nd, dxc_ncol) — GGA functional derivative
    double* grad_rho = nullptr;   // (Nd, 3) — density gradient

    // Band-parallel Allgather buffer
    double* X_full = nullptr;     // (ld, Ns_global) — gathered wavefunctions

    // Per-spin psi: psi_spin[s] = psi + s * ld * Ns (for Nspin=2)
    // Or separate allocations. For simplicity, allocate psi big enough for Nspin.
    double* psi_s1 = nullptr;     // (ld, Ns) spin-down psi (when Nspin=2)
    double* Hpsi_s1 = nullptr;    // (ld, Ns) spin-down H*psi
    double* eigenvalues_s1 = nullptr; // (Ns_global) spin-down eigenvalues
    double* occupations_s1 = nullptr; // (Ns_global) spin-down occupations

    // Complex buffers for k-point support (allocated when is_kpt)
    void*   psi_z = nullptr;      // cuDoubleComplex* (ld, Ns)
    void*   Hpsi_z = nullptr;     // cuDoubleComplex* (ld, Ns)
    void*   Xold_z = nullptr;     // cuDoubleComplex* (ld, Ns)
    void*   Xnew_z = nullptr;     // cuDoubleComplex* (ld, Ns)
    void*   x_ex_z = nullptr;     // cuDoubleComplex* (Nd_ex, Ns)
    void*   Hs_z = nullptr;       // cuDoubleComplex* (Ns, Ns)
    void*   Ms_z = nullptr;       // cuDoubleComplex* (Ns, Ns)
    void*   alpha_z = nullptr;    // cuDoubleComplex* (total_nproj, Ns)

    // cuSOLVER workspace
    int     eig_work_size = 0;
    double* cusolver_work = nullptr;
    int*    cusolver_devinfo = nullptr;

    // Flags
    bool is_kpt = false;
    int  Nspin_alloc = 1;  // Nspin used during allocation
};

// ============================================================
// GPUHandles — All CUDA library handles in one place
// ============================================================
// Pre-created at init, shared across all GPU operations.
// Each handle triggers ~0.2-0.5 ms of setup; first use of each
// kernel variant triggers 1-56 ms of JIT compilation.
// The warmup() method pre-triggers all JIT paths we'll need.
struct GPUHandles {
    // Dense linear algebra
    cublasHandle_t   cublas = nullptr;
    cusolverDnHandle_t cusolver = nullptr;

    // Sparse (for NonlocalProjector CSR, Laplacian if needed)
    cusparseHandle_t cusparse = nullptr;

    // RNG (for Lanczos initial vector, wavefunction randomization)
    curandGenerator_t curand = nullptr;

    // cuBLAS workspace (pre-allocated to avoid hidden mallocs)
    void*  cublas_workspace = nullptr;
    size_t cublas_workspace_size = 0;

    // cuSOLVER persistent workspace + devinfo
    int*   cusolver_devinfo = nullptr;

    void init(cudaStream_t stream) {
        // cuBLAS
        cublasStatus_t cb = cublasCreate(&cublas);
        if (cb != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "cuBLAS create failed: %d\n", (int)cb);
            std::exit(EXIT_FAILURE);
        }
        cublasSetStream(cublas, stream);
        cublasSetMathMode(cublas, CUBLAS_DEFAULT_MATH);

        // Pre-allocate cuBLAS workspace (4 MB)
        cublas_workspace_size = 4 * 1024 * 1024;
        CUDA_CHECK(cudaMalloc(&cublas_workspace, cublas_workspace_size));
        cublasSetWorkspace(cublas, cublas_workspace, cublas_workspace_size);

        // cuSOLVER
        cusolverStatus_t cs = cusolverDnCreate(&cusolver);
        if (cs != CUSOLVER_STATUS_SUCCESS) {
            std::fprintf(stderr, "cuSOLVER create failed: %d\n", (int)cs);
            std::exit(EXIT_FAILURE);
        }
        cusolverDnSetStream(cusolver, stream);
        CUDA_CHECK(cudaMalloc(&cusolver_devinfo, sizeof(int)));

        // cuSPARSE
        cusparseStatus_t csp = cusparseCreate(&cusparse);
        if (csp != CUSPARSE_STATUS_SUCCESS) {
            std::fprintf(stderr, "cuSPARSE create failed: %d\n", (int)csp);
            std::exit(EXIT_FAILURE);
        }
        cusparseSetStream(cusparse, stream);

        // cuRAND
        curandStatus_t cr = curandCreateGenerator(&curand, CURAND_RNG_PSEUDO_DEFAULT);
        if (cr != CURAND_STATUS_SUCCESS) {
            std::fprintf(stderr, "cuRAND create failed: %d\n", (int)cr);
            std::exit(EXIT_FAILURE);
        }
        curandSetStream(curand, stream);
        curandSetPseudoRandomGeneratorSeed(curand, 42);
    }

    // Pre-trigger JIT compilation for all kernel variants we use.
    // Without this, first SCF iteration pays 50-100 ms penalty.
    void warmup(cudaStream_t stream) {
        int N = 16;
        double *d_A, *d_B, *d_C, *d_eig;
        int* d_info;
        CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_C, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_eig, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_A, 0, N * N * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_B, 0, N * N * sizeof(double)));

        double alpha = 1.0, beta = 0.0;

        // cuBLAS: dgemm (used for A^T*A fallback, orbital rotation)
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
        // cuBLAS: dgemm with transpose (used for subspace projection)
        cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                     N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
        // cuBLAS: dtrsm (used for Cholesky QR)
        cublasDtrsm(cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                     CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                     N, N, &alpha, d_A, N, d_B, N);
        // cuBLAS: ddot, dnrm2 (used in AAR solver)
        double dot_result;
        cublasDdot(cublas, N, d_A, 1, d_B, 1, &dot_result);
        cublasDnrm2(cublas, N, d_A, 1, &dot_result);

        // cuSOLVER: dpotrf (Cholesky)
        int lwork = 0;
        cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER,
                                      N, d_A, N, &lwork);
        double* d_work;
        CUDA_CHECK(cudaMalloc(&d_work, std::max(lwork, 1) * sizeof(double)));
        cusolverDnDpotrf(cusolver, CUBLAS_FILL_MODE_UPPER,
                           N, d_A, N, d_work, lwork, d_info);

        // cuSOLVER: dsyevd (eigenvalue decomposition)
        cusolverDnDsyevd_bufferSize(cusolver, CUSOLVER_EIG_MODE_VECTOR,
                                      CUBLAS_FILL_MODE_UPPER,
                                      N, d_A, N, d_eig, &lwork);
        cudaFree(d_work);
        CUDA_CHECK(cudaMalloc(&d_work, std::max(lwork, 1) * sizeof(double)));
        cusolverDnDsyevd(cusolver, CUSOLVER_EIG_MODE_VECTOR,
                           CUBLAS_FILL_MODE_UPPER,
                           N, d_A, N, d_eig, d_work, lwork, d_info);

        // cuRAND: generate random doubles (for Lanczos, wavefunction init)
        curandGenerateUniformDouble(curand, d_A, N * N);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaFree(d_eig); cudaFree(d_info); cudaFree(d_work);
    }

    void destroy() {
        if (curand) { curandDestroyGenerator(curand); curand = nullptr; }
        if (cusparse) { cusparseDestroy(cusparse); cusparse = nullptr; }
        if (cusolver_devinfo) { cudaFree(cusolver_devinfo); cusolver_devinfo = nullptr; }
        if (cusolver) { cusolverDnDestroy(cusolver); cusolver = nullptr; }
        if (cublas_workspace) { cudaFree(cublas_workspace); cublas_workspace = nullptr; }
        if (cublas) { cublasDestroy(cublas); cublas = nullptr; }
    }

    // Query overhead: total time to create all handles (diagnostic)
    static void print_handle_info() {
        std::printf("GPU Handles:\n");
        std::printf("  cuBLAS + cuSOLVER + cuSPARSE + cuRAND\n");
        std::printf("  Pre-allocated cuBLAS workspace: 4 MB\n");
        std::printf("  Warmup: triggers JIT for dgemm, dtrsm, dpotrf, dsyevd, ddot, dnrm2, curand\n");
    }
};

// ============================================================
// GPUContext — Central GPU resource manager
// ============================================================
// One instance per MPI rank. Owns all GPU memory, handles, streams.
// Zero cudaMalloc/cudaFree during SCF iterations.
struct GPUContext {
    // All library handles (cuBLAS, cuSOLVER, cuSPARSE, cuRAND)
    GPUHandles handles;

    // Convenience aliases for backward compatibility
    cublasHandle_t&      cublas = handles.cublas;
    cusolverDnHandle_t&  cusolver = handles.cusolver;
    cusparseHandle_t&    cusparse = handles.cusparse;
    curandGenerator_t&   curand = handles.curand;

    // Streams
    cudaStream_t compute_stream = nullptr; // Main compute
    cudaStream_t copy_stream = nullptr;    // Async H2D/D2H overlap

    // Memory pools
    GPUMemoryPool    device_pool;    // Main GPU memory arena
    GPUMemoryPool    scratch_pool;   // Reset-per-phase scratch space
    PinnedMemoryPool pinned_pool;    // Pinned host memory

    // Pre-allocated SCF buffers (named pointers into device_pool)
    SCFBuffers buf;

    // ---- Lifecycle ----

    GPUContext() {
        // Create streams first (handles need them)
        CUDA_CHECK(cudaStreamCreate(&compute_stream));
        CUDA_CHECK(cudaStreamCreate(&copy_stream));

        // Create all handles, bind to compute_stream
        handles.init(compute_stream);

        // Pre-trigger JIT compilation — avoids 50-100 ms hit on first SCF iteration
        handles.warmup(compute_stream);
    }

    ~GPUContext() {
        handles.destroy();
        if (copy_stream) cudaStreamDestroy(copy_stream);
        if (compute_stream) cudaStreamDestroy(compute_stream);
    }

    GPUContext(const GPUContext&) = delete;
    GPUContext& operator=(const GPUContext&) = delete;

    // ---- Initialize: pre-allocate all SCF buffers ----

    // Pad leading dimension to 8-element boundary (matches NDArray)
    static int pad_ld(int rows) {
        return (rows + 7) / 8 * 8;
    }

    void init_scf_buffers(int Nd, int nx, int ny, int nz, int FDn,
                          int Ns_local, int Ns_global, int Nspin,
                          int aar_m, int mix_m, int mix_ncol,
                          int total_nproj, size_t chi_size, size_t gpos_size,
                          bool is_gga, bool band_parallel,
                          bool is_kpt_in = false) {
        auto& b = buf;
        b.Nd = Nd;
        b.Ns = Ns_local;
        b.Ns_global = Ns_global;
        b.Nspin = Nspin;
        b.Nspin_alloc = Nspin;
        b.is_kpt = is_kpt_in;
        b.FDn = FDn;
        b.ld = pad_ld(Nd);

        int nx_ex = nx + 2 * FDn;
        int ny_ex = ny + 2 * FDn;
        int nz_ex = nz + 2 * FDn;
        b.Nd_ex = nx_ex * ny_ex * nz_ex;
        b.ld_ex = b.Nd_ex; // extended arrays are 1D

        // ---- Estimate total memory needed ----
        size_t total = 0;
        auto account = [&](size_t bytes) { total += (bytes + 255) & ~255ULL; };

        // Persistent SCF arrays
        account(b.ld * Ns_local * sizeof(double));      // psi
        account(b.ld * Ns_local * sizeof(double));      // Hpsi
        account(Nd * Nspin * sizeof(double));            // Veff
        account(Nd * Nspin * sizeof(double));            // Vxc
        account(Nd * sizeof(double));                    // phi
        account(Nd * Nspin * sizeof(double));            // rho
        account(Nd * sizeof(double));                    // rho_total
        account(Nd * sizeof(double));                    // b (Poisson RHS)
        account(Nd * sizeof(double));                    // exc
        account(Nd * sizeof(double));                    // Vc
        account(Ns_global * sizeof(double));             // eigenvalues
        account(Ns_global * sizeof(double));             // occupations

        // CheFSI workspace
        account(b.ld * Ns_local * sizeof(double));      // Xold
        account(b.ld * Ns_local * sizeof(double));      // Xnew
        account((size_t)b.Nd_ex * Ns_local * sizeof(double)); // x_ex

        // Subspace matrices
        int N2 = Ns_global * Ns_global;
        account(N2 * sizeof(double));                    // Hs
        account(N2 * sizeof(double));                    // Ms
        account(N2 * sizeof(double));                    // Q

        // AAR workspace (Poisson solver)
        account(Nd * sizeof(double));                    // aar_r
        account(Nd * sizeof(double));                    // aar_f
        account(Nd * sizeof(double));                    // aar_Ax
        account(Nd * aar_m * sizeof(double));            // aar_X
        account(Nd * aar_m * sizeof(double));            // aar_F
        account((size_t)b.Nd_ex * sizeof(double));       // aar_x_ex

        // Mixer workspace
        int mix_N = Nd * mix_ncol;
        account(mix_N * mix_m * sizeof(double));         // mix_R
        account(mix_N * mix_m * sizeof(double));         // mix_F
        account(mix_N * sizeof(double));                 // mix_fk
        account(mix_N * sizeof(double));                 // mix_xkm1

        // Nonlocal projector
        if (chi_size > 0) {
            account(chi_size * sizeof(double));          // Chi_flat
            account(gpos_size * sizeof(int));            // gpos_flat
            account((total_nproj + 1) * sizeof(int));    // atom_offsets (approx)
            account(total_nproj * sizeof(double));       // Gamma_flat
            account(total_nproj * Ns_local * sizeof(double)); // alpha
        }

        // GGA workspace
        if (is_gga) {
            int dxc_ncol = (Nspin == 2) ? 3 : 1;
            account(Nd * dxc_ncol * sizeof(double));     // Dxcdgrho
            account(Nd * 3 * sizeof(double));            // grad_rho
        }

        // Band-parallel Allgather buffer
        if (band_parallel) {
            account(b.ld * Ns_global * sizeof(double));  // X_full
        }

        // Spin-polarized: duplicate psi/Hpsi/eig/occ
        if (Nspin >= 2) {
            account(b.ld * Ns_local * sizeof(double));   // psi_s1
            account(b.ld * Ns_local * sizeof(double));   // Hpsi_s1
            account(Ns_global * sizeof(double));          // eigenvalues_s1
            account(Ns_global * sizeof(double));          // occupations_s1
        }

        // Complex buffers for k-point (2x size each for cuDoubleComplex)
        if (is_kpt_in) {
            account(b.ld * Ns_local * 2 * sizeof(double)); // psi_z
            account(b.ld * Ns_local * 2 * sizeof(double)); // Hpsi_z
            account(b.ld * Ns_local * 2 * sizeof(double)); // Xold_z
            account(b.ld * Ns_local * 2 * sizeof(double)); // Xnew_z
            account((size_t)b.Nd_ex * Ns_local * 2 * sizeof(double)); // x_ex_z
            account(N2 * 2 * sizeof(double));              // Hs_z
            account(N2 * 2 * sizeof(double));              // Ms_z
            if (chi_size > 0)
                account(total_nproj * Ns_local * 2 * sizeof(double)); // alpha_z
        }

        // Add 10% headroom
        total = (size_t)(total * 1.1);

        // ---- Allocate the main pool ----
        device_pool.init(total);

        // ---- Assign named pointers ----
        b.psi         = device_pool.alloc<double>(b.ld * Ns_local);
        b.Hpsi        = device_pool.alloc<double>(b.ld * Ns_local);
        b.Veff        = device_pool.alloc<double>(Nd * Nspin);
        b.Vxc         = device_pool.alloc<double>(Nd * Nspin);
        b.phi         = device_pool.alloc<double>(Nd);
        b.rho         = device_pool.alloc<double>(Nd * Nspin);
        b.rho_total   = device_pool.alloc<double>(Nd);
        b.b           = device_pool.alloc<double>(Nd);
        b.exc         = device_pool.alloc<double>(Nd);
        b.Vc          = device_pool.alloc<double>(Nd);
        b.eigenvalues = device_pool.alloc<double>(Ns_global);
        b.occupations = device_pool.alloc<double>(Ns_global);

        // CheFSI
        b.Xold = device_pool.alloc<double>(b.ld * Ns_local);
        b.Xnew = device_pool.alloc<double>(b.ld * Ns_local);
        b.x_ex = device_pool.alloc<double>((size_t)b.Nd_ex * Ns_local);

        // Subspace
        b.Hs = device_pool.alloc<double>(N2);
        b.Ms = device_pool.alloc<double>(N2);
        b.Q  = device_pool.alloc<double>(N2);

        // AAR
        b.aar_r    = device_pool.alloc<double>(Nd);
        b.aar_f    = device_pool.alloc<double>(Nd);
        b.aar_Ax   = device_pool.alloc<double>(Nd);
        b.aar_X    = device_pool.alloc<double>(Nd * aar_m);
        b.aar_F    = device_pool.alloc<double>(Nd * aar_m);
        b.aar_x_ex = device_pool.alloc<double>(b.Nd_ex);

        // Mixer
        b.mix_R    = device_pool.alloc<double>(mix_N * mix_m);
        b.mix_F    = device_pool.alloc<double>(mix_N * mix_m);
        b.mix_fk   = device_pool.alloc<double>(mix_N);
        b.mix_xkm1 = device_pool.alloc<double>(mix_N);

        // Nonlocal projector
        if (chi_size > 0) {
            b.Chi_flat     = device_pool.alloc<double>(chi_size);
            b.gpos_flat    = device_pool.alloc<int>(gpos_size);
            b.atom_offsets = device_pool.alloc<int>(total_nproj + 1);
            b.Gamma_flat   = device_pool.alloc<double>(total_nproj);
            b.alpha        = device_pool.alloc<double>(total_nproj * Ns_local);
        }

        // GGA
        if (is_gga) {
            int dxc_ncol = (Nspin == 2) ? 3 : 1;
            b.Dxcdgrho = device_pool.alloc<double>(Nd * dxc_ncol);
            b.grad_rho = device_pool.alloc<double>(Nd * 3);
        }

        // Band-parallel
        if (band_parallel) {
            b.X_full = device_pool.alloc<double>(b.ld * Ns_global);
        }

        // Spin-polarized: second set of psi/Hpsi/eigenvalues/occupations
        if (Nspin >= 2) {
            b.psi_s1         = device_pool.alloc<double>(b.ld * Ns_local);
            b.Hpsi_s1        = device_pool.alloc<double>(b.ld * Ns_local);
            b.eigenvalues_s1 = device_pool.alloc<double>(Ns_global);
            b.occupations_s1 = device_pool.alloc<double>(Ns_global);
        }

        // Complex buffers for k-point support (each cuDoubleComplex = 16 bytes)
        if (is_kpt_in) {
            size_t zld = b.ld * 2;  // cuDoubleComplex has 2 doubles
            b.psi_z  = device_pool.alloc<double>(zld * Ns_local);
            b.Hpsi_z = device_pool.alloc<double>(zld * Ns_local);
            b.Xold_z = device_pool.alloc<double>(zld * Ns_local);
            b.Xnew_z = device_pool.alloc<double>(zld * Ns_local);
            b.x_ex_z = device_pool.alloc<double>((size_t)b.Nd_ex * 2 * Ns_local);
            b.Hs_z   = device_pool.alloc<double>(N2 * 2);
            b.Ms_z   = device_pool.alloc<double>(N2 * 2);
            if (chi_size > 0) {
                b.alpha_z = device_pool.alloc<double>(total_nproj * Ns_local * 2);
            }
        }

        // cuSOLVER devinfo alias into SCFBuffers for backward compatibility
        b.cusolver_devinfo = handles.cusolver_devinfo;

        // ---- Scratch pool for temporary per-phase allocations ----
        // Uses same async pool — grows on demand, checkpoint/restore for scoping
        scratch_pool.init(0);

        // ---- Pinned host memory (for async transfers of small data) ----
        // Eigenvalues, occupations, subspace matrices, scalars
        size_t pinned_size = (Ns_global * 2 + N2 * 3 + 1024) * sizeof(double);
        pinned_pool.init(pinned_size);

        // Zero all persistent buffers
        CUDA_CHECK(cudaMemset(b.psi, 0, b.ld * Ns_local * sizeof(double)));
        CUDA_CHECK(cudaMemset(b.rho, 0, Nd * Nspin * sizeof(double)));
        CUDA_CHECK(cudaMemset(b.Veff, 0, Nd * Nspin * sizeof(double)));
        CUDA_CHECK(cudaMemset(b.phi, 0, Nd * sizeof(double)));
    }

    // Print memory and handle usage summary
    void print_memory_info() const {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);

        std::printf("GPU Resources:\n");
        std::printf("  Handles: cuBLAS + cuSOLVER + cuSPARSE + cuRAND (all pre-warmed)\n");
        std::printf("  cuBLAS workspace: %.1f MB\n", handles.cublas_workspace_size / 1e6);
        std::printf("  Device pool: %.1f MB used / %.1f MB total (high water: %.1f MB)\n",
                    device_pool.used() / 1e6, device_pool.capacity() / 1e6,
                    device_pool.high_water() / 1e6);
        std::printf("  Scratch pool: %.1f MB capacity\n", scratch_pool.capacity() / 1e6);
        std::printf("  GPU free: %.1f MB / %.1f MB total\n", free_mem / 1e6, total_mem / 1e6);
    }

    // Singleton access
    static GPUContext& instance() {
        static GPUContext ctx;
        return ctx;
    }
};

} // namespace gpu
} // namespace lynx

#endif // USE_CUDA
