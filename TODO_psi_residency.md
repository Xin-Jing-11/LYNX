# TODO: GPU Psi Residency — Zero Host Transfers

## Current Problem
psi is downloaded/uploaded between CPU and GPU during SCF loop:
- Multi-spin/kpt: EigenSolver has ONE device buffer, swaps via host
- Forces/stress: psi downloaded to CPU post-SCF, computed on CPU
- mGGA tau: psi downloaded to CPU for gradient computation

## Strategy: Allocate-Solve-Accumulate-Free

Maintain a pointer list (nkpt * nspin) but allocate/free device memory on demand.

### LDA / GGA / SCAN (independent eigenvalue problems)
```
for each (spin, kpt):
    1. Allocate ONE psi buffer (Nd * Nband) on GPU
    2. Solve eigenvalue problem → psi on GPU
    3. Accumulate density from this psi (GPU kernel)
    4. Accumulate tau from this psi if mGGA (GPU kernel)
    5. Accumulate force/stress components from this psi (GPU kernel)
    6. Free this psi buffer
```
Only ONE psi buffer active at a time. Density, tau, force, stress accumulators
persist on GPU across all (spin, kpt). Final results (energy, force array,
stress tensor) downloaded to host at the very end.

### Hybrid functionals (PBE0 / HSE)
The ACE operator depends on ALL k-points within one spin channel.
Must allocate all k-point psi buffers for one spin simultaneously:
```
for each spin:
    Allocate psi for ALL k-points in this spin channel
    Solve all eigenvalue problems (ACE needs cross-kpt access)
    Accumulate density/force/stress for all kpts
    Free all psi for this spin channel
```
This is more complex — implement after LDA/GGA/SCAN path works.

## Files to Change
- `src/solvers/EigenSolver.{hpp,cu}` — per-(spin,kpt) buffer management
- `src/physics/SCF.cpp` — delete all upload_psi/download_psi calls
- `src/physics/Forces.cpp` — dispatch to GPU via _gpu() methods
- `src/physics/Stress.cpp` — dispatch to GPU via _gpu() methods
- `src/electronic/KineticEnergyDensity.cpp` — read device psi directly
- `src/electronic/ElectronDensity.cpp` — read device psi directly

## Key Rule (from CLAUDE.md)
psi NEVER transfers between CPU and GPU. Born on GPU, lives on GPU, dies on GPU.
Only final scalar results (energy, forces, stress) go to host.
