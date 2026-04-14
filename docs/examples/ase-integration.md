---
title: ASE Integration
parent: Examples
nav_order: 3
---

# ASE Integration

`LynxCalculator` is a drop-in ASE calculator. All values use ASE units (Angstrom, eV).

## Basic usage

```python
from ase.build import bulk
from lynx.ase import LynxCalculator

atoms = bulk('Si', 'diamond', a=5.43)
atoms.calc = LynxCalculator(
    xc='PBE',
    kpts=(4, 4, 4),
    mesh_spacing=0.5,   # Bohr — grid spacing
    scf_tol=1e-6,
)

energy = atoms.get_potential_energy()   # eV
forces = atoms.get_forces()             # eV/Angstrom, shape (Natom, 3)
stress = atoms.get_stress()             # eV/Angstrom^3, Voigt: (xx, yy, zz, yz, xz, xy)
```

## Geometry optimization

```python
from ase.build import bulk
from ase.optimize import BFGS
from lynx.ase import LynxCalculator

atoms = bulk('Si', 'diamond', a=5.43)
atoms.calc = LynxCalculator(xc='PBE', kpts=(4, 4, 4))

opt = BFGS(atoms, trajectory='si_relax.traj')
opt.run(fmax=0.01)   # eV/Angstrom
print(atoms.get_potential_energy())
```

## Equation of state

```python
from ase.build import bulk
from ase.eos import EquationOfState
from ase import units
from lynx.ase import LynxCalculator

atoms = bulk('Si', 'diamond', a=5.43)

volumes, energies = [], []
for scale in [0.95, 0.97, 1.00, 1.03, 1.05]:
    a = atoms.copy()
    a.set_cell(atoms.cell * scale, scale_atoms=True)
    a.calc = LynxCalculator(xc='PBE', kpts=(4, 4, 4))
    volumes.append(a.get_volume())
    energies.append(a.get_potential_energy())

eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()
print(f"Bulk modulus: {B / units.GPa:.1f} GPa")
```

A complete runnable version of this example is at `examples/ase_ipi/06_equation_of_state.py`.

## Calculator options

| Option | Default | Description |
|--------|---------|-------------|
| `xc` | `'PBE'` | XC functional (`'PBE'`, `'LDA'`, `'SCAN'`, `'HSE06'`) |
| `kpts` | `(1,1,1)` | Monkhorst-Pack grid |
| `kpt_shift` | `(0,0,0)` | K-point shift in fractional coordinates |
| `mesh_spacing` | auto | Grid spacing in Bohr |
| `grid_shape` | auto | Explicit `(Nx, Ny, Nz)` grid |
| `fd_order` | `12` | Finite-difference stencil order |
| `scf_tol` | `1e-6` | SCF convergence threshold (Ha/atom) |
| `mixing_beta` | `0.3` | Pulay mixing parameter |
| `temperature` | `315.77` | Electronic temperature in Kelvin |
| `n_bands` | auto | Number of bands |
| `device` | `'cpu'` | `'cpu'` or `'gpu'` |
| `verbose` | `0` | Verbosity level (0 = silent) |

## GPU acceleration

```python
atoms.calc = LynxCalculator(xc='PBE', kpts=(4, 4, 4), device='gpu')
energy = atoms.get_potential_energy()
```

**What's next:** [MD simulation example]({{ site.baseurl }}/examples/md-simulation/) shows LynxCalculator in an NVE run with automatic density restart.
