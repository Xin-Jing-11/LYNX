# SPARC C++ Rewrite — TODO & Progress

## Completed

### Phase 1: Core Infrastructure
- [x] NDArray<T> with column-major storage, 64-byte alignment
- [x] Lattice (ortho/non-ortho/helical), FDGrid, Domain
- [x] FDStencil (Taylor expansion coefficients, all FD orders)
- [x] InputParser (JSON), OutputWriter
- [x] MPIComm (RAII), CartTopology, Parallelization
- [x] Example inputs: BaTiO3.json, Si8.json

### Phase 2: Operators & Atoms
- [x] HaloExchange (MPI + serial periodic ghost zones)
- [x] Laplacian (ortho + non-ortho mixed derivatives)
- [x] Gradient (per-axis first derivatives)
- [x] Hamiltonian (H*psi = -0.5*Lap + Veff + Vnl)
- [x] NonlocalProjector (KB projectors: setup + apply, Chi^T*psi, Gamma, Chi*alpha)
- [x] AtomType, Pseudopotential (psp8 reader, spline interpolation, NLCC)
- [x] Crystal (atom positions, periodic images, influence computation)

### Phase 3: Electronic Structure & SCF
- [x] EigenSolver (CheFSI + Lanczos bounds estimation)
- [x] Mixer (Pulay + Kerker preconditioner)
- [x] PoissonSolver (AAR/CG with Jacobi preconditioner)
- [x] LinearSolver (AAR + CG)
- [x] Wavefunction, ElectronDensity, Occupation (Gaussian/Fermi-Dirac)
- [x] XCFunctional: LDA_PZ, LDA_PW, GGA_PBE, GGA_PBEsol, GGA_RPBE
- [x] SCF loop with convergence matching reference SPARC
- [x] Energy (Eband, Exc, Ehart, Eself, Ec, Entropy, Etotal)
- [x] Electrostatics (pseudocharge, Vloc, Eself, Ec, atomic density, NLCC core density)
- [x] BaTiO3: Etot matches to 3e-6 Ha (0.08 meV/atom)

### Phase 4: Forces & Stress
- [x] Local forces (-integral bJ * grad_phi + correction)
- [x] Nonlocal forces (-spn_fac * g_n * Gamma * <Chi|psi> * <Chi|grad_psi>)
- [x] NLCC XC forces (integral Vxc * grad_rho_core_J)
- [x] Force symmetrization (zero-sum constraint)
- [x] Kinetic stress
- [x] XC stress (diagonal + GGA gradient correction)
- [x] NLCC XC stress (integral grad_rho_core * (x-R) * Vxc / cell_measure)
- [x] Electrostatic stress
- [x] Nonlocal stress
- [x] Pressure
- [x] BaTiO3: forces match to 1e-7 Ha/Bohr, stress to ~0.01 GPa

### GPU Operator Kernels (standalone, not integrated)
- [x] Laplacian.cu, Gradient.cu, Hamiltonian.cu, NonlocalProjector.cu
- [x] HaloExchange.cu, gpu_common.cu, GPUContext.cuh
- [x] test_GPUKernels.cu (10 tests, correctness + benchmarks)
- [x] Performance: Laplacian 89x, Gradient 76x, Hamiltonian 35x speedup (RTX 5080)

### Parallelization
- [x] Domain decomposition (npNdx * npNdy * npNdz)
- [x] Band parallelization (npband)
- [x] K-point parallelization (npkpt)
- [x] Spin parallelization (npspin)
- [x] Bridge communicators for cross-group reduction

---

## TODO

### Single-Core Functional Gaps (bugs/incompleteness)
- [ ] **Spin-polarized GGA correlation** — `pbec_spin()` in XCFunctional.cpp is a placeholder; collinear spin + PBE gives wrong Ec
- [ ] **Spherical harmonics l >= 3** — NonlocalProjector returns 0.0 for f-orbitals; needed for rare earths/actinides
- [ ] **Per-type pseudocharge cutoff** — currently using global rc_max; reference uses adaptive `Calculate_PseudochargeCutoff` per atom type
- [x] **Non-orthogonal forces/stress coordinate transform** — `nonCart2Cart_grad` + `nonCart2Cart_coord` in Stress.cpp; MgO GGA_PBE matches ref to 0.01 GPa

### Output & Usability
- [ ] Eigenvalue printing (per k-point, per spin)
- [ ] Electron density output (.cube format)
- [ ] Band structure calculation & output
- [ ] Restart / checkpoint (write density to file, read on restart)

### Additional XC Functionals
- [ ] Meta-GGA: SCAN, rSCAN, r2SCAN (mGGA Hamiltonian term + stress)
- [ ] Hybrid: HSE06, PBE0 (exact exchange via ACE or conventional)
- [ ] vdW-DF (van der Waals density functional)
- [ ] DFT-D3 dispersion correction

### Geometry Optimization
- [ ] LBFGS (Limited-memory BFGS)
- [ ] FIRE (Fast Inertial Relaxation Engine)
- [ ] NLCG (Nonlinear Conjugate Gradient)
- [ ] Cell relaxation (volume/shape optimization)
- [ ] Atom constraints (fix atoms / selective dynamics)

### Molecular Dynamics
- [ ] NVE (microcanonical ensemble)
- [ ] NVT with Nose-Hoover thermostat
- [ ] Velocity Verlet / Leapfrog integrators
- [ ] MD restart files
- [ ] Trajectory output

### Advanced Electronic Structure
- [ ] Spin-orbit coupling (SOC)
- [ ] Non-collinear magnetism
- [ ] Charged systems (net charge handling)

### GPU Integration
- [ ] CPU/GPU dispatch in operator classes (#ifdef USE_CUDA)
- [ ] GPU EigenSolver (CheFSI on GPU)
- [ ] GPU LinearSolver / PoissonSolver
- [ ] GPU SCF integration (full GPU pipeline)
- [ ] GPU-aware MPI for multi-GPU

### Other
- [ ] ScaLAPACK distributed eigensolver integration
- [ ] D2D (domain-to-domain) transfer for redistribution
- [ ] Socket interface (ASE coupling)
- [ ] OFDFT (orbital-free DFT)
