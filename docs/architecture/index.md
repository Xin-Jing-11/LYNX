---
title: Architecture
nav_order: 2
has_children: true
permalink: /architecture/
---

# Architecture

LYNX is built in clear layers. Each layer depends only on those below it — no physics logic in GPU kernels, no kernel calls in physics algorithms.

## Layer Diagram

```
Python (pylynx / ASE)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                     C++ Engine                      │
│                                                     │
│  physics/   SCF loop, Energy, Forces, Stress        │
│  solvers/   CheFSI eigensolver · AAR Poisson        │
│             Pulay+Kerker mixer                      │
│  operators/ Laplacian · Gradient · Hamiltonian      │
│             NonlocalProjector · FDStencil           │
│  electronic/ Wavefunction · ElectronDensity         │
│              Occupation                             │
│  xc/        XCFunctional (libxc) · ExactExchange    │
│  atoms/     Crystal · Pseudopotential (psp8)        │
│  parallel/  MPI · HaloExchange                      │
│  core/      DeviceArray · Lattice · FDGrid          │
│             Domain · KPoints                        │
└─────────────────────────────────────────────────────┘
```

## Design Principles

**One algorithm, two execution paths.** Every operator has a single public method (e.g., `apply()`, `solve()`). That method dispatches to `_cpu()` or `_gpu()` based on a member `dev_` flag set at construction. The algorithm — iteration loops, convergence checks, sub-steps — lives once in `.cpp`. GPU-specific code is isolated to `.cu` files containing only kernel launches.

**GPU-resident data.** Wavefunctions are born on the GPU (random initialization via cuRAND) and never leave the device. The full SCF → forces → stress pipeline runs on-device. Only small scalar arrays (eigenvalues, occupations) cross the PCIe bus per SCF iteration.

**Python without overhead.** The Python API is a thin pybind11 binding over the C++ engine. NumPy arrays share memory with `DeviceArray<T>` buffers — no copies for CPU arrays. Three granularity levels let users trade control for convenience.

## Sections

- [CPU/GPU Dispatch]({{ site.baseurl }}/architecture/dispatch/) — the `_cpu()/_gpu()` pattern in detail
- [Python API]({{ site.baseurl }}/architecture/python-api/) — three levels of access, zero-copy numpy interop
- [External Libraries]({{ site.baseurl }}/architecture/external-libs/) — libxc, MPI, pybind11, ASE, pseudopotentials
