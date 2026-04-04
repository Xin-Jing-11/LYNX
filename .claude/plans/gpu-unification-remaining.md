# GPU Unification: Remaining Work

## Completed
- [x] GPUSCF.cu deleted (5584 lines), GPUSCF.cuh deleted (299 lines)
- [x] NDArray replaced by DeviceArray throughout src/
- [x] One SCF loop in SCF.cpp with Device dispatch
- [x] All 8 operators have setup_gpu()/cleanup_gpu()
- [x] EigenSolver GPU path wired (real + k-point CheFSI)
- [x] ElectronDensity GPU path wired (real + k-point)
- [x] Hamiltonian GPU dispatch (local + nonlocal + mGGA + EXX)
- [x] XCFunctional GPU dispatch (LDA/GGA/mGGA)
- [x] PoissonSolver GPU dispatch (AAR solver)
- [x] Mixer GPU dispatch (Pulay + Kerker)
- [x] 117/117 tests pass (CUDA build)

## Remaining (performance + completeness)

### P0: GPU-resident data (performance blocker)
Currently each operator uploads/downloads per call (H2D/D2H every SCF iteration).
To match the old GPUSCF.cu performance, data must stay GPU-resident:
- Allocate psi, Veff, rho, phi on GPU once before SCF loop
- Pass device pointers between operators
- Download only for Energy::compute_all() (once per iter, not hot)
- Estimated: ~200 lines in SCF.cpp + accessor methods on operators

### P1: EffectivePotential GPU path
Currently falls back to CPU. Needs to chain XC(GPU) + Poisson(GPU) + combine kernel.
The individual pieces work — just need to wire them together in EffectivePotential.cu.

### P1: KineticEnergyDensity GPU path
Currently falls back to CPU. Needs GPU gradient + tau accumulation kernels (already exist in .cu file).

### P2: Spinor (SOC) GPU paths
EigenSolver::solve_spinor_kpt and ElectronDensity::compute_spinor fall back to CPU.
Needs spinor callback wiring and spinor density GPU kernel integration.

### P2: k-point Bloch factor upload
Hamiltonian::set_kpoint_gpu() updates kxLx/kyLy/kzLz but d_bloch_fac is not yet
uploaded. Nonlocal projector silently skips when d_bloch_fac is null (safe but incorrect
for k-point GPU). Need to port setup_bloch_factors() from old GPUSCF.cu.

### P2: GPU mGGA stress
Disabled in Stress.cpp (line 71 TODO). Needs standalone GPU stress function.

### P3: Static callback cleanup
EigenSolver.cu, PoissonSolver.cu, Mixer.cu use file-static pointers for C-style
callbacks. Replace with std::function or thread-local storage for thread safety.

### P3: Remove SCFBuffers from GPUContext
The monolithic SCFBuffers struct in GPUContext.cuh is unused now (each operator
owns its own buffers). Can be removed to clean up GPUContext.
