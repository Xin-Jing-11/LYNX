# LYNX — Agent Workflow Instructions

## Project Overview
LYNX is a real-space density functional theory (DFT) code being ported from SPARC (C/CUDA) to modern C++17/CUDA.
Priority: **accuracy first**, performance second.

## Success Criteria
A feature port is considered complete only when it passes BOTH accuracy and performance checks:

### Accuracy (blocking — must pass before merge)
- Energy, forces, AND stress must all match SPARC within tolerances
- A feature that matches energy but not forces/stress is NOT done

### Performance (required for production readiness)
- **CPU**: at least comparable speed to SPARC for the same system; must scale (speedup) with more CPU cores
- **GPU**: at least **10x faster** than CPU for an end-to-end SCF test on a representative system

## Reference Code
- SPARC reference: `/home/xx/Desktop/dev_SPARC_GPU/`
- **NEVER** fetch from online repos or external URLs for reference code
- Always compare numerical results against SPARC reference values
- When porting: read and understand the SPARC algorithm first, then write idiomatic C++ — do NOT line-by-line translate C to C++

## Accuracy Targets (vs SPARC)
- Energy: < 1e-6 Ha
- Forces: < 1e-5 Ha/Bohr
- Stress: < 0.01 GPa
- Verify with actual test runs, not theoretical reasoning

## Comparison Protocol (MANDATORY)
When comparing LYNX results against SPARC, **every** comparison must use identical settings.
Check ALL of these before running any comparison:
1. **Mesh**: same grid dimensions (Nx, Ny, Nz) and FD order
2. **Lattice vectors**: identical cell vectors (watch for Bohr vs Angstrom)
3. **Atom positions**: identical fractional/Cartesian coordinates
4. **Temperature** (smearing): same electronic temperature and smearing type
5. **Pseudopotential**: same psp8 file (same path or byte-identical copy)
6. **SCF tolerance**: same convergence criterion (e.g., 1e-6 Ha/atom)
7. **Mixing parameters**: same beta, Pulay history depth, Kerker cutoff
8. **K-points**: same Monkhorst-Pack grid and shift
9. **XC functional**: same functional type and parameters

If ANY setting differs, the comparison is invalid. Always document the shared settings in test comments.

## Code Standards
- Language: C++17, CUDA for GPU kernels
- Namespace: `lynx`
- Storage: column-major `DeviceArray<T>`, 64-byte aligned (CPU), `cudaMallocAsync` (GPU)
- Use libxc for XC functionals where available
- No unnecessary abstractions or over-engineering

## Operator Dispatch Rules (MANDATORY)
**One algorithm, kernel-level dispatch. Follows PyTorch/TensorFlow pattern.**

1. **Algorithm lives in `.cpp` only.** Each operator has ONE public method (e.g., `solve()`, `evaluate()`, `mix()`). The algorithm (iteration loop, convergence check, sub-steps) is written once in `.cpp`. NEVER duplicate algorithm logic in `.cu` files.

2. **`.cu` files contain only GPU method implementations.** No loops, no algorithm logic. Just kernel launches and cuBLAS/cuSOLVER calls inside `_gpu()` methods.

3. **Dispatch via `_cpu()` / `_gpu()` methods on the same class:**
   ```
   Operator.hpp  — declares solve(), apply_foo(), apply_foo_cpu(), apply_foo_gpu()
   Operator.cpp  — implements solve() (algorithm), apply_foo() (dispatcher), apply_foo_cpu()
   Operator.cu   — implements apply_foo_gpu()
   ```
   The dispatcher is a small method:
   ```cpp
   void Operator::apply_foo(const double* x, double* y) {
       if (dev_ == Device::GPU) apply_foo_gpu(x, y);
       else                     apply_foo_cpu(x, y);
   }
   ```

4. **Device is a member, not a parameter.** Set `dev_` at setup/construction. Public methods do NOT take `Device dev` — they check `dev_` internally.

5. **No global state for dispatch.** No `thread_local`, no file-static pointers, no C-style callback trampolines. If a sub-operation needs context, it's a method on `this`.

6. **No standalone GPU solver functions.** Delete `gpu::aar_gpu()`, `gpu::eigensolver_solve_gpu()`, etc. The iteration loop belongs in the operator's `.cpp` method, not in a standalone `.cu` function.

## GPU Data Residency Rules (MANDATORY)
**Minimize host↔device transfers. Data lives on GPU for the entire SCF loop.**

1. **psi NEVER transfers between CPU and GPU.** On GPU builds, psi is born on GPU (curand), lives on GPU for the entire SCF loop, and stays on GPU for forces/stress. On CPU builds, psi is on CPU. There is NO upload or download of psi, ever. This applies to ALL spins and ALL k-points — allocate per-spin/kpt device buffers, do NOT share one buffer and swap via host.

2. **GPU-resident arrays** (stay on GPU from allocation to deallocation):
   - psi (wavefunctions) — all spins, all k-points, simultaneously on device
   - Veff, Vxc, phi, exc (potentials) — computed on GPU, stay on GPU
   - rho (density) — computed on GPU, stay on GPU
   - tau, vtau (mGGA) — computed on GPU, stay on GPU

3. **Allowed CPU↔GPU transfers per SCF iteration**:
   - eigvals D2H (~KB) — needed for CPU Fermi level / occupation computation
   - occ H2D (~KB) — upload occupations after CPU computation
   - rho, Veff, phi, exc, Vxc D2H — for Energy::compute_all() on CPU (once/iter)
   - That's it. No psi transfer. No Veff upload. No rho upload.

4. **Forces/stress run on GPU in GPU builds.** Do NOT download psi to CPU for force/stress computation. GPU force/stress kernels already exist (GPUForce.cu, GPUStress.cu) — use them. The entire pipeline from SCF → forces → stress runs on GPU without psi ever touching the host.

5. **One-time transfers** (before/after SCF+forces+stress):
   - H2D: initial density, pseudocharge (rho_b), rho_core (NLCC), nonlocal projector data (Chi, Gamma)
   - D2H: final results (energy, forces array, stress tensor) — NOT psi

6. **Never do**:
   - Upload psi from CPU (randomize on GPU instead)
   - Download psi to CPU and re-upload next iteration
   - Share one device psi buffer across spins/kpts (allocate per-spin/kpt)
   - Download psi for forces/stress (compute on GPU)
   - Allocate GPU memory with `cudaMalloc` inside hot loops (use `cudaMallocAsync` with stream pool)
   - Transfer data that the next operator on GPU already has

## Worktree Layout
| Worktree | Path | Branch | Focus |
|----------|------|--------|-------|
| main | `/home/xx/Desktop/LYNX` | master | Stable baseline |

Agents create temporary worktrees at `.worktrees/<name>` on `feature/<name>` branches. These are deleted after merging (see rule 9).

## Build & Test
```bash
# IMPORTANT: initialize submodules first (pseudopotentials + libxc)
git submodule update --init

# CPU build
cd build && cmake .. && make -j$(nproc)

# GPU build
cd build && cmake .. -DUSE_CUDA=ON && make -j$(nproc)

# Run all tests
cd build && ctest --output-on-failure

# Run specific test
cd build && ./test_lynx --gtest_filter=EndToEnd.TestName

# Run with MPI
cd build && mpirun -np 4 ./test_lynx --gtest_filter=EndToEnd.TestName
```

## Workflow Rules
1. **Build and test before claiming success** — run the actual test, paste the output
2. **NEVER modify master directly** — ALL work must be done in a git worktree on a feature branch (`git worktree add .worktrees/<name> feature/<name>`). Stage and commit changes in the worktree. Only merge to master after explicit user approval + rebase + retest. Agents that modify master directly are violating this rule.
3. **One concern per commit** — keep commits focused and descriptive
4. **When stuck on accuracy**: dump intermediate values (rho, Vxc, tau, etc.) and compare grid-point-by-grid-point against SPARC
5. **Do not remove or weaken existing tests** — only add stricter tolerances or new tests
6. **Stress testing**: use deformed cells (non-cubic) to exercise all 6 Voigt components
7. **Never merge to master without explicit user approval** — report results first, wait for go-ahead
8. **Before merging to master**: rebase your feature branch onto the latest master first, rebuild, and rerun ALL tests. Only if tests still pass after rebasing can you request merge approval. This prevents merge conflicts from silently breaking previously-working code.
9. **After merging to master**: delete the local worktree (`rm -rf .worktrees/<name> && git worktree prune`), delete the local branch (`git branch -d feature/<name>`), and delete the remote branch (`git push origin --delete feature/<name>`). Leave no stale worktrees or branches behind.

## Test Requirements (MANDATORY before merge)
Every new XC functional or major feature MUST be tested on the **standard benchmark suite** below.
Results must be recorded in `tests/benchmark_results.md` as a table with ALL columns filled.

### Standard Benchmark Systems
| System | Atoms | Cell | Grid | K-points | Spin | PSP |
|--------|-------|------|------|----------|------|-----|
| Si4 ortho | 4 | Deformed ortho 10.0×10.26×10.5 | 25×26×27 | Gamma | No | Si LDA |
| Si4 kpt | 4 | Same as above | 25×26×27 | 2×2×2 (0.5 shift) | No | Si LDA |
| Fe2 spin gamma | 2 | Non-ortho BCC 2.84³ sheared | 29×29×29 | Gamma | Yes | Fe LDA |
| Fe2 spin kpt | 2 | Same as above | 29×29×29 | 2×2×2 (0.5 shift) | Yes | Fe LDA |

### Required Result Table Columns
For each system, record:

| Column | Description |
|--------|-------------|
| Test name | E2E test name in test_EndToEnd.cpp |
| System | Material + cell description |
| Grid | Nx×Ny×Nz |
| K-points | Grid + shift |
| Spin | None / Collinear |
| SPARC Etotal (Ha) | Reference energy from SPARC |
| LYNX Etotal (Ha) | LYNX computed energy |
| Energy error (Ha) | |Etotal_LYNX - Etotal_SPARC| |
| SPARC max force (Ha/Bohr) | Max component from SPARC |
| Force error (Ha/Bohr) | Max |F_LYNX - F_SPARC| |
| SPARC max stress (GPa) | Max component from SPARC |
| Stress error (GPa) | Max |σ_LYNX - σ_SPARC| |
| SCF iterations | Number of SCF cycles |
| CPU time (s) | Wall clock time on CPU |
| GPU time (s) | Wall clock time on GPU (if applicable) |

### Pass criteria
- Energy error < 1e-6 Ha
- Force error < 1e-5 Ha/Bohr
- Stress error < 0.01 GPa
- All 4 benchmark systems must pass
- Results logged in `tests/benchmark_results.md`
- E2E tests added to `tests/test_EndToEnd.cpp`

### Zero-failure policy (MANDATORY)
- **ALL tests must pass before merging to master.** There is no such thing as a "pre-existing failure" — if `ctest` reports ANY failure, the branch is not ready to merge.
- If you encounter failing tests that existed before your changes, **fix them** as part of your branch before requesting merge approval.
- Do NOT rationalize failures as "pre-existing" or "unrelated" — every red test is your responsibility to investigate and fix.
- `ctest --output-on-failure` must show **0 failures** on both CPU and GPU builds before merge.

## Key Architecture
- SCF loop: `src/physics/SCF.cpp` (CPU), `src/physics/GPUSCF.cu` (GPU)
- XC functionals: `src/xc/XCFunctional.{cpp,cu,hpp}`
- Hamiltonian: `src/operators/Hamiltonian.{cpp,hpp}` — includes mGGA term
- Forces/Stress: `src/physics/Forces.cpp`, `src/physics/Stress.cpp`
- GPU kernels: `src/physics/GPUForceStress.cu`, `src/operators/*.cu`
- Exact exchange: `src/xc/ExactExchange.{cpp,hpp}`, `src/xc/ExchangePoissonSolver.{cpp,hpp}`
- Tests: `tests/test_EndToEnd.cpp`, test data in `tests/data/`

## Common Pitfalls
- Spin factor: density gets `spin_fac` (2 for non-spin), tau does NOT
- NLCC: always add `rho_core` to density before XC evaluation
- Non-orthogonal cells: must apply metric tensor to sigma, laplacian, divergences
- Pseudocharge: uses Fortran D-notation in psp8 files (e.g., `1.234D+02`)
- Unit conversion: stress internal unit is Ha/Bohr³, multiply by 29421.01569650548 for GPa
