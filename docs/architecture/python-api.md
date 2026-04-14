---
title: Python API
parent: Architecture
nav_order: 2
---

# Python API

The Python interface (`pylynx`) wraps the C++ engine via pybind11 with three granularity levels. All three produce identical results — choose the level that matches how much control you need.

## Level 1 — Full calculation in one call

```python
import lynx

atoms = lynx.Atoms(
    cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
    positions=[[0, 0, 0], [2.565, 2.565, 2.565]],
    symbols=["Si", "Si"],
    units="bohr",
)
result = lynx.calculate(atoms, xc="PBE")
print(result.energy)          # Hartree
print(result.forces)          # numpy (Natom, 3), Ha/Bohr
```

Or with more control via `lynx.DFT`:

```python
import lynx

calc = lynx.DFT(xc="PBE", kpts=(2, 2, 2), n_bands=20)
result = calc(atoms)
print(result.energy)
print(result.eigenvalues)    # numpy array of KS eigenvalues
```

## Level 2 — SCF control and internal access

```python
import lynx

calc = lynx.DFT(xc="PBE", kpts=(2, 2, 2))

# Run and inspect internals
result = calc(atoms)
print(result.energy_decomposition)    # Eband, Exc, Ehart, Etotal, ...
print(result.density)                 # numpy (Nd,) electron density
print(result.eigenvalues)             # numpy array of KS eigenvalues

# Run on GPU
calc_gpu = lynx.DFT(xc="PBE", kpts=(2, 2, 2), device="gpu")
result = calc_gpu(atoms)
```

## Level 3 — Low-level operators on numpy arrays

Direct access to finite-difference operators:

```python
import numpy as np
import lynx
from lynx.ops import Laplacian, Gradient

grid = lynx.Grid(cell=[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
                 shape=[48, 48, 48], fd_order=12)
lap  = Laplacian(grid)
grad = Gradient(grid)

x    = np.random.randn(grid.ndof)
y    = lap(x, a=-0.5)          # y = -0.5 * nabla^2(x)
dfdx = grad(x, direction=0)   # df/dx
```

`Laplacian` and `Gradient` also support the `@` operator: `lap @ x` is equivalent to `lap(x)`.

Level 3 is useful for testing finite-difference operators independently, building custom post-processing, or exploring the grid directly.

## Zero-copy numpy interop

`DeviceArray<double>` (the internal C++ array type) implements the Python buffer protocol. When an array lives on CPU, numpy accesses it directly — no copy. When it lives on GPU, calling `.cpu()` returns a CPU-side numpy array.

```python
rho = result.density    # numpy view of the internal array — no copy
```

## ASE Calculator

`LynxCalculator` is a drop-in ASE calculator. All values use ASE units (Å, eV) — conversion to Bohr/Hartree is automatic.

```python
from ase.build import bulk
from lynx.ase import LynxCalculator

atoms = bulk('Si', 'diamond', a=5.43)
atoms.calc = LynxCalculator(
    xc='PBE',
    kpts=(4, 4, 4),
    mesh_spacing=0.5,   # Bohr — sets grid spacing
    scf_tol=1e-6,
)

energy = atoms.get_potential_energy()   # eV
forces = atoms.get_forces()            # eV/Å, shape (Natom, 3)
stress = atoms.get_stress()            # eV/Å³, Voigt: (xx, yy, zz, yz, xz, xy)
```

## Unit conventions

| Interface | Length | Energy | Forces | Stress |
|-----------|--------|--------|--------|--------|
| C++ / JSON | Bohr | Hartree | Ha/Bohr | Ha/Bohr³ |
| `lynx.Atoms(units="bohr")` | Bohr | Hartree | Ha/Bohr | Ha/Bohr³ |
| `LynxCalculator` (ASE) | Å | eV | eV/Å | eV/Å³ |

Constants are in `lynx.units`: `BOHR_TO_ANG`, `HA_TO_EV`, `HA_BOHR_TO_EV_ANG`.
