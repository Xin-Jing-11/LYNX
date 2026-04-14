---
title: External Libraries
parent: Architecture
nav_order: 3
---

# External Libraries

LYNX is designed so that each external dependency plugs in at a single, well-defined seam.

## libxc — XC functional zoo

libxc provides 600+ exchange-correlation functionals (LDA, GGA, metaGGA, hybrid). LYNX links against libxc and routes functional selection through a single string at input time:

```python
lynx.calculate(atoms, xc="PBE")     # GGA-PBE
lynx.calculate(atoms, xc="LDA")     # LDA
lynx.calculate(atoms, xc="SCAN")    # metaGGA
```

Adding a new functional requires no code changes — just pass a valid XC name. The `XCFunctional` class in `src/xc/` handles the libxc API calls.

## MPI — Parallelization

LYNX parallelizes over spins, k-points, and bands. The `MPIComm` and `HaloExchange` classes in `src/parallel/` encapsulate all MPI calls. The rest of the code does not call MPI directly.

```bash
mpirun -np 8 ./build/src/lynx Si8.json   # 8 MPI ranks
```

Decomposition is automatic: ranks are assigned to spin/k-point groups, and within a group to band blocks. `HaloExchange` handles ghost-node synchronization for finite-difference stencils transparently.

## pybind11 — Python bindings

pybind11 is fetched automatically by CMake (`FetchContent`) — no manual install required. The binding sources live in `python/src/`. Key design choices:

- **Buffer protocol:** `DeviceArray<T>` exposes a buffer interface so numpy can access CPU arrays without copying.
- **Automatic unit conversion:** `LynxCalculator` converts Å→Bohr and eV→Hartree on input, back on output.
- **No GIL tricks:** each `calc(atoms)` call releases the GIL so Python threads are not blocked during SCF.

```bash
cmake -DBUILD_PYTHON=ON ..
make -j
export PYTHONPATH=/path/to/LYNX/python:$PYTHONPATH
```

## ASE — Atoms and workflows

The `LynxCalculator` class in `python/lynx/ase.py` implements the ASE [Calculator interface](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculator.html). It adapts `lynx.DFT` to accept ASE `Atoms` objects and returns forces/stress in ASE units. No changes to the C++ engine were needed — the adapter is pure Python.

This means LYNX works transparently with all ASE tools: geometry optimizers, molecular dynamics drivers (including i-PI via the ASE socket interface), equation-of-state fitting, and structure generation.

## ONCV Pseudopotentials (PseudoDojo)

Pseudopotential files are included as git submodules from [PseudoDojo](http://www.pseudo-dojo.org/):

```
psps/ONCVPSP-PBE-PDv0.4/   # GGA-PBE
psps/ONCVPSP-LDA-PDv0.4/   # LDA
```

Each element has a subdirectory with a `.psp8` file (e.g., `Si/Si.psp8`). The `Pseudopotential` class in `src/atoms/` reads the psp8 format and precomputes projector data.

Initialize submodules after cloning:

```bash
git submodule update --init
```

## Build flags summary

| Flag | Effect |
|------|--------|
| (default) | CPU-only build, MPI, OpenBLAS/MKL, libxc (submodule) |
| `-DUSE_CUDA=ON` | Enables GPU kernels, cuBLAS, cuSOLVER, cuRAND |
| `-DBUILD_PYTHON=ON` | Builds `_core.so` pybind11 extension |
