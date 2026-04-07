# LYNX

A real-space density functional theory (DFT) simulator with C++/CUDA kernels and a Python interface.

LYNX solves the Kohn-Sham equations on a finite-difference grid using Chebyshev-filtered subspace iteration (CheFSI), supporting both CPU and GPU execution. The Python interface (`pylynx`) exposes the full C++ engine via pybind11, enabling scripted workflows, ASE integration, and direct numpy interop.

## Quick start

### Build

```bash
# Clone with submodules (pseudopotentials + libxc)
git clone --recurse-submodules https://github.com/Xin-Jing-11/LYNX.git
cd LYNX

# Or if already cloned without submodules:
git submodule update --init

# CPU only
mkdir build && cd build
cmake ..
make -j

# With GPU support
cmake -DUSE_CUDA=ON ..
make -j

# With Python bindings
cmake -DBUILD_PYTHON=ON ..
make -j

# Run tests (all 118 must pass)
ctest --output-on-failure
```

Dependencies: MPI, BLAS/LAPACK (OpenBLAS or MKL), C++17 compiler. Optional: CUDA toolkit, ScaLAPACK, pybind11 (fetched automatically).

> **Note:** Submodules must be initialized before building or testing. Without them, pseudopotential files and libxc are missing, causing build or test failures.

### Pseudopotentials

LYNX uses ONCV norm-conserving pseudopotentials from the [PseudoDojo](http://www.pseudo-dojo.org/) project, included as git submodules.

After cloning, fetch the pseudopotential files:

```bash
git clone --recurse-submodules https://github.com/SMSHBSB/LYNX.git

# Or if already cloned:
git submodule update --init
```

This populates:

| Directory | Functional | Source |
|-----------|-----------|--------|
| `psps/ONCVPSP-PBE-PDv0.4/` | GGA-PBE | [PseudoDojo](https://github.com/PseudoDojo/ONCVPSP-PBE-PDv0.4) |
| `psps/ONCVPSP-LDA-PDv0.4/` | LDA | [PseudoDojo](https://github.com/PseudoDojo/ONCVPSP-LDA-PDv0.4) |

Each element has a subdirectory with `.psp8` files, e.g. `psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8`.

### Run (C++)

```bash
mpirun -np 4 ./build/src/lynx examples/Si8.json
```

### Run (Python)

```python
import lynx
from lynx.config import DFTConfig

config = DFTConfig(
    cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
    fractional=[[0, 0, 0], [0.25, 0.25, 0.25]],
    symbols=['Si', 'Si'],
    pseudo_files={'Si': 'psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8'},
    Nstates=10,
)
calc = config.create_calculator(auto_run=True)
print(calc.total_energy)  # Hartree
```

### Run (ASE)

```python
from ase.build import bulk
from lynx.ase_interface import LynxCalculator

atoms = bulk('Si', 'diamond', a=5.43)
atoms.calc = LynxCalculator(kpts=(4, 4, 4))
print(atoms.get_potential_energy())  # eV
print(atoms.get_forces())           # eV/Angstrom
```

---

## Project structure

```
LYNX/
  src/
    main.cpp                  C++ entry point
    core/                     NDArray, Lattice, FDGrid, Domain, KPoints
    operators/                Laplacian, Gradient, Hamiltonian, NonlocalProjector, FDStencil
    solvers/                  EigenSolver (CheFSI), PoissonSolver (AAR), Mixer (Pulay+Kerker)
    electronic/               Wavefunction, ElectronDensity, Occupation
    xc/                       XCFunctional (LDA/GGA)
    physics/                  SCF, Energy, Forces, Stress, Electrostatics
    atoms/                    Crystal, AtomType, Pseudopotential (psp8)
    parallel/                 MPIComm, HaloExchange, Parallelization
    io/                       InputParser (JSON), OutputWriter, DensityIO
  python/
    lynx/                     Python package
      __init__.py             Re-exports all bindings
      config.py               DFTConfig — programmatic setup (no JSON)
      ase_interface.py        ASE Calculator adapter
      units.py                Bohr/Hartree <-> Angstrom/eV constants
    src/                      pybind11 C++ binding sources
    tests/                    pytest test suite (19 tests)
    examples/                 Python usage examples
  examples/
    Si8_comparison/           Same calculation via C++, Python, and ASE
  psps/                       Pseudopotential files (.psp8)
  tests/                      C++ tests (GoogleTest)
```

---

## Python interface

The Python interface provides three granularity levels, all producing identical results with the C++ executable.

### Level 1 — High-level Calculator

Run a full DFT calculation in one call.

**From a JSON file** (same format the C++ executable uses):
```python
import lynx

calc = lynx.Calculator("Si8.json")
print(calc.total_energy)      # Hartree
print(calc.energy)            # dict with Eband, Exc, Ehart, Etotal, ...
forces = calc.compute_forces()  # numpy (Natom, 3) in Ha/Bohr
rho = calc.density              # numpy (Nd_d,)
```

**From a Python script** (no JSON file):
```python
from lynx.config import DFTConfig

config = DFTConfig(
    cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
    fractional=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    symbols=['Si', 'Si'],
    pseudo_files={'Si': 'psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8'},
    Nx=24, Ny=24, Nz=24,
    Nstates=10,
    xc='GGA_PBE',
    kpts=(2, 2, 2),
)
calc = config.create_calculator(auto_run=True)
```

**From an ASE Atoms object** (automatic Angstrom/eV conversion):
```python
from ase.build import bulk
from lynx.config import DFTConfig

atoms = bulk('Si', 'diamond', a=5.43)             # Angstrom
config = DFTConfig.from_ase(atoms, kpts=(4, 4, 4))  # converts to Bohr
calc = config.create_calculator(auto_run=True)
print(calc.total_energy)                           # Hartree
```

### Level 2 — Mid-level SCF control

Set up the calculation without running, then access internal objects.

```python
calc = lynx.Calculator("Si8.json", auto_run=False)
# calc is set up but SCF has not run

# Access internal components
print(calc.grid)           # FDGrid(24x24x24)
print(calc.domain)         # Domain(24x24x24)
print(calc.Nelectron)      # 32
print(calc.crystal)        # Crystal object

# Run SCF manually
calc.run()
print(calc.total_energy)

# Access wavefunctions, density, potentials
wfn = calc.get_wavefunction()
eigs = wfn.eigenvalues(spin=0, kpt=0)  # numpy array
rho = calc.density                      # numpy array
```

### Level 3 — Low-level operators on numpy arrays

Create finite-difference operators and apply them to numpy arrays directly. Halo exchange is handled transparently.

```python
import numpy as np
import lynx

lattice = lynx.make_lattice(np.diag([10.0, 10.0, 10.0]))
grid = lynx.FDGrid(48, 48, 48, lattice)
stencil = lynx.FDStencil(12, grid, lattice)
domain = lynx.full_domain(grid)
halo = lynx.HaloExchange(domain, stencil.FDn())

lap = lynx.Laplacian(stencil, domain)
grad = lynx.Gradient(stencil, domain)

x = np.random.randn(domain.Nd_d())
y = lap.apply(halo, x, a=-0.5, c=0.0)     # y = -0.5 * Lap(x)
dfdx = grad.apply(halo, x, direction=0)    # df/dx
```

### ASE Calculator

The `LynxCalculator` class implements the ASE [Calculator interface](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculator.html). All ASE-facing values use ASE units (Angstrom, eV). Conversion is automatic.

```python
from ase.build import bulk
from lynx.ase_interface import LynxCalculator

atoms = bulk('Si', 'diamond', a=5.43)
atoms.calc = LynxCalculator(
    xc='GGA_PBE',
    kpts=(4, 4, 4),
    mesh_spacing=0.5,     # Bohr (grid spacing)
    max_scf_iter=100,
    scf_tol=1e-6,
    # psp_dir='psps/',    # or set LYNX_PSP_PATH env var
)

energy = atoms.get_potential_energy()   # eV
forces = atoms.get_forces()            # eV/Angstrom
stress = atoms.get_stress()            # eV/Angstrom^3 (Voigt: xx yy zz yz xz xy)
```

### Units

| Interface | Length | Energy | Forces | Stress |
|-----------|--------|--------|--------|--------|
| C++ / JSON | Bohr | Hartree | Ha/Bohr | Ha/Bohr^3 |
| `DFTConfig()` | Bohr | Hartree | Ha/Bohr | Ha/Bohr^3 |
| `DFTConfig.from_ase()` | auto | auto | auto | auto |
| `LynxCalculator` (ASE) | Angstrom | eV | eV/Ang | eV/Ang^3 |

Conversion constants are in `lynx.units`: `BOHR_TO_ANG`, `HA_TO_EV`, `HA_BOHR_TO_EV_ANG`, etc.

### Building the Python bindings

```bash
mkdir build && cd build
cmake -DBUILD_PYTHON=ON ..
make -j

# The _core.so is placed in python/lynx/ automatically.
# Add to your Python path:
export PYTHONPATH=/path/to/LYNX/python:$PYTHONPATH

# Or install as a package:
cd python && pip install -e .
```

### Running the tests

```bash
cd /path/to/LYNX
PYTHONPATH=python python -m pytest python/tests/ -v
```

19 tests covering imports, enums, types, lattice/grid construction, Laplacian/Gradient operators, full SCF calculations, `DFTConfig`, ASE integration, and unit conversions.

---

## Examples

### `examples/Si8_comparison/`

The same 8-atom Si diamond calculation run three ways, producing identical results (`Etotal = -33.3685427384 Ha`):

| File | Interface | Units |
|------|-----------|-------|
| `run_cpp.sh` | C++ executable + `Si8.json` | Bohr / Hartree |
| `run_python.py` | Python `DFTConfig` (no JSON) | Bohr / Hartree |
| `run_ase.py` | ASE `Atoms` + `LynxCalculator` | Angstrom / eV |

### `python/examples/`

| File | Description |
|------|-------------|
| `01_script_si2.py` | Minimal Si2 from Python script |
| `02_ase_si_diamond.py` | ASE Calculator usage |
| `03_ase_atoms_to_lynx.py` | ASE Atoms -> DFTConfig -> access internals |
| `04_custom_structure.py` | Build arbitrary structures with ASE |
| `05_operators_numpy.py` | Level 3: Laplacian/Gradient on numpy |

---

## Input format (JSON)

```json
{
  "lattice": {
    "vectors": [[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
    "cell_type": "orthogonal"
  },
  "grid": {
    "Nx": 40, "Ny": 40, "Nz": 40,
    "fd_order": 12,
    "boundary_conditions": ["periodic", "periodic", "periodic"]
  },
  "atoms": [{
    "element": "Si",
    "pseudo_file": "path/to/Si.psp8",
    "fractional": true,
    "coordinates": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
  }],
  "electronic": {
    "xc": "GGA_PBE",
    "Nstates": 20,
    "temperature": 300,
    "smearing": "gaussian"
  },
  "scf": {
    "max_iter": 100,
    "tolerance": 1e-6,
    "mixing": "density",
    "preconditioner": "kerker",
    "mixing_history": 7,
    "mixing_parameter": 0.3
  }
}
```

All lengths in Bohr, temperature in Kelvin. Supported XC functionals: `LDA_PZ`, `LDA_PW`, `GGA_PBE`, `GGA_PBEsol`, `GGA_RPBE`.

---

## Features

- **Real-space finite-difference** method with configurable FD order (6-24)
- **Chebyshev-filtered subspace iteration** (CheFSI) eigensolver with Lanczos bounds
- **XC functionals**: LDA (PZ, PW) and GGA (PBE, PBEsol, RPBE), spin-polarized
- **K-points**: Monkhorst-Pack grid with time-reversal symmetry reduction
- **Pseudopotentials**: Optimized norm-conserving Vanderbilt (ONCV) in psp8 format
- **Forces and stress**: local, nonlocal, XC (including NLCC) contributions
- **GPU acceleration**: full GPU-resident SCF pipeline (CUDA), 35-89x speedup per operator
- **Parallelization**: MPI with spin, k-point, and band decomposition
- **Python bindings**: three granularity levels, zero-copy numpy interop, ASE integration
