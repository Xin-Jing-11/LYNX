# GPU Kernel Profiling Report — LYNX on RTX 5080 (sm_120, Blackwell)

**GPU**: NVIDIA GeForce RTX 5080, 16GB VRAM, CUDA 13.1
**Grid**: 25×26×27 (Nd=17,550), FDn=6, ncol=20 (Si4 benchmark dimensions)

## Phase 1: Baseline Timing (CUDA Events, 200-500 reps)

| Kernel | Time (ms) | Effective BW (GB/s) | Notes |
|--------|-----------|---------------------|-------|
| laplacian_v2 (multi-col loop) | 0.0861 | 1272 | |
| laplacian_v6 (1-col/block) | 0.0739 | 1483 | |
| **laplacian_v7 (precomp+1col)** | **0.0523** | **2093** | **Fastest Laplacian** |
| laplacian_v8 (precomp+multi) | 0.0902 | 1215 | |
| laplacian_v3 (shared mem) | 0.1043 | 1050 | |
| gradient_v1 (per-col launch) | 0.0819 | 446 | |
| gradient_v2 (batched) | 0.0307 | 1189 | |
| **gather_chitpsi (orig)** | **0.2210** | — | **Slowest kernel** |
| scatter_chialpha (orig) | 0.0420 | — | |
| gamma_scale | 0.0021 | — | Negligible |
| chefsi_init | 0.0041 | 2055 | Near peak BW |
| chefsi_step | 0.0061 | 1827 | Near peak BW |
| ata_dot (X^T*X) | 0.0225 | 2616 | |
| atb_dot (X^T*HX) | 0.0307 | 3654 | |
| mgga_scan | 0.0450 | 22 (compute-heavy) | |
| lda_pw | 0.0123 | 34 | |
| gga_pbe | 0.0164 | 43 | |

## Phase 2: ncu Hardware Metrics

| Kernel | Compute % | Mem BW % | Occupancy % | Regs/Thread | L1 Hit % | L2 Hit % | Bottleneck |
|--------|-----------|----------|-------------|-------------|----------|----------|------------|
| laplacian_v2 | 45.9 | 7.6 | 19.9 | **96** | 58.1 | 82.1 | Register pressure |
| laplacian_v6 | **79.0** | 9.0 | **79.5** | 40 | 56.0 | 82.9 | Compute-bound (good) |
| laplacian_v7 | **77.7** | 12.9 | **79.1** | 40 | 55.5 | 82.6 | Compute-bound (good) |
| laplacian_v8 | 42.7 | 7.3 | 20.2 | **64** | 58.1 | 81.8 | Register pressure |
| gradient_v2 | 40.8 | 12.9 | 20.3 | 39 | 84.2 | 15.9 | Low occupancy |
| gather_orig | **1.4** | **0.7** | 16.7 | 48 | 48.9 | 87.7 | **Only 4 blocks!** |
| scatter_orig | **2.6** | **3.6** | 15.8 | 40 | 64.6 | 32.3 | **Only 4 blocks!** |
| chefsi_step | 23.3 | **75.8** | **79.7** | 18 | 0 | 0.7 | Memory-bound (optimal) |
| ata_dot | 35.9 | 9.9 | 46.7 | 17 | 16.9 | 94.8 | Limited blocks (20×20) |
| mgga_scan | **67.2** | 1.0 | 16.5 | **48** | 0 | 33.7 | Compute-bound, low occ |

## Phase 3: Optimizations Applied

### 1. Nonlocal Gather — 2D Grid (atom × column) — **18.6× speedup**

**Root cause**: Original launches only `n_atoms` blocks (4 for Si4). RTX 5080 has 84 SMs, so 95% of GPU is idle.

**Fix**: Split column loop out of kernel, launch `(n_atoms, ncol)` 2D grid = 80 blocks. Also removed shared memory tile (psi_tile), relying on L2 cache for scattered reads, and used FMA intrinsics.

| Version | Time (ms) | Speedup | Blocks |
|---------|-----------|---------|--------|
| orig | 0.229 | 1.0× | 4 |
| v1 (2D grid + smem tile) | 0.018 | 12.5× | 80 |
| **v2 (2D grid + no tile + FMA)** | **0.012** | **18.6×** | 80 |

### 2. Nonlocal Scatter — 2D Grid — **4.6× speedup**

**Root cause**: Same as gather — only 4 blocks.

**Fix**: Launch `(n_atoms, ncol)` grid. Each block handles one (atom, column) pair. Shared memory reduced from `np * ncol` to just `np` doubles.

| Version | Time (ms) | Speedup |
|---------|-----------|---------|
| orig | 0.047 | 1.0× |
| **v1 (2D grid)** | **0.010** | **4.6×** |

### 3. Laplacian — V7 is already optimal

V7 (precomputed `a*D2x` coefficients + 1-column-per-block) is the clear winner at 0.0523ms. It achieves:
- 79% occupancy (40 regs/thread)
- 77.7% compute utilization
- FMA-friendly: `val += d_aD2x[p] * sum` compiles to single FMA

**V2 and V8 suffer from register spill** (96 and 64 regs respectively) due to the multi-column inner loop requiring accumulators for all columns simultaneously. This kills occupancy to ~20%.

**Recommendation**: Use V7 (laplacian_orth_v7_gpu) as the production kernel. V3 shared memory approach is slower because the working set for the xy-plane halo fits in L1 cache already.

### 4. Gradient — V2 batched is 2.7× faster than V1

V2 avoids per-column kernel launch overhead but has low occupancy (20%). Could apply V6 pattern (1-col-per-block) for further gains, but gradient is not the bottleneck.

### 5. CheFSI vector kernels — already memory-bandwidth bound

`chefsi_step` achieves 75.8% DRAM throughput with 79.7% occupancy and only 18 regs/thread. This is near-optimal for a simple element-wise kernel. No optimization needed.

### 6. SCAN XC — compute-bound, block size insensitive

67.2% compute utilization, only 16.5% occupancy. The kernel does ~20 transcendental operations (cbrt, exp, log, pow, sqrt) per grid point. Block sizes 64-256 all give identical performance (0.045ms), while 512 is 2× slower (register pressure forces spill).

**Recommendation**: Keep block size at 128 or 256. The kernel is inherently compute-bound due to algorithmic complexity. Further optimization would require algorithmic changes (e.g., table interpolation for transcendentals), which is not worth the accuracy risk.

## Summary: Before/After Optimization

| Kernel | Before (ms) | After (ms) | Speedup | Technique |
|--------|-------------|------------|---------|-----------|
| **gather_chitpsi** | **0.229** | **0.012** | **18.6×** | 2D grid (atom×col), remove smem tile, FMA |
| **scatter_chialpha** | **0.047** | **0.010** | **4.6×** | 2D grid (atom×col), register alpha |
| laplacian (best) | 0.052 (V7) | 0.052 (V7) | 1.0× | Already optimal — use V7 |
| gradient | 0.031 (V2) | 0.031 (V2) | 1.0× | Batched is sufficient |
| chefsi_step | 0.006 | 0.006 | 1.0× | Memory-bound, near peak |
| mgga_scan | 0.045 | 0.045 | 1.0× | Compute-bound, no easy gains |

## Recommendations for Production Code

1. **Apply nonlocal 2D grid** to `NonlocalProjector.cu` — change gather/scatter from 1D `<<<n_atoms>>>` to 2D `<<<dim3(n_atoms, ncol)>>>`. This is the single highest-impact change (~0.25ms → ~0.02ms per H*psi application).

2. **Ensure V7 is the default Laplacian** — confirm `laplacian_orth_v7_gpu` is called in GPUSCF.cu, not V2 or V8.

3. **Apply same 2D grid pattern to complex (k-point) nonlocal** in `ComplexOperators.cu` — `fused_gather_chitpsi_z_kernel` likely has the same single-atom-per-block bottleneck.

4. **No changes needed** for CheFSI vector kernels, XC kernels, or gradient kernels at current system sizes.
