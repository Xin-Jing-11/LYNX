---
title: Python Quickstart
parent: Examples
nav_order: 2
---

# Python Quickstart

## Setup

```bash
mkdir build && cd build
cmake -DBUILD_PYTHON=ON ..
make -j
cd ../python && pip install -e .
```

## One-liner calculation

```python
import lynx

atoms = lynx.Atoms(
    cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
    positions=[[0, 0, 0], [2.565, 2.565, 2.565]],
    symbols=["Si", "Si"],
    units="bohr",
)
result = lynx.calculate(atoms, xc="PBE")
print(result.energy)   # Hartree
print(result.forces)   # numpy (Natom, 3), Ha/Bohr
```

## More control with lynx.DFT

```python
import lynx

calc = lynx.DFT(xc="PBE", kpts=(2, 2, 2))
result = calc(atoms)

print(result.energy)       # total energy, Hartree
print(result.eigenvalues)  # Kohn-Sham eigenvalues
print(result.energies)     # energy decomposition dict
```

## Spin-polarized calculation

```python
calc = lynx.DFT(xc="PBE", kpts=(2, 2, 2), spin="collinear")
result = calc(atoms)
print(result.energy)
```

## GPU acceleration

```python
calc_gpu = lynx.DFT(xc="PBE", kpts=(2, 2, 2), device="gpu")
result = calc_gpu(atoms)
print(result.energy)
```

## Common DFT parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `xc` | `"PBE"` | XC functional (`"PBE"`, `"LDA"`, `"SCAN"`, `"HSE06"`) |
| `kpts` | `(1,1,1)` | Monkhorst-Pack k-point grid |
| `kpt_shift` | `(0,0,0)` | K-point shift in fractional coordinates |
| `mesh_spacing` | auto | Target grid spacing in Bohr |
| `grid_shape` | auto | Explicit `(Nx, Ny, Nz)` grid |
| `fd_order` | `12` | Finite-difference stencil order |
| `temperature` | `315.77` | Electronic temperature in Kelvin |
| `smearing` | `"gaussian"` | Smearing method (`"gaussian"` or `"fermi-dirac"`) |
| `n_bands` | auto | Number of Kohn-Sham bands |
| `spin` | `None` | `None`, `"collinear"`, or `"noncollinear"` |
| `scf_tol` | `1e-6` | SCF convergence threshold (Ha/atom) |
| `mixing_beta` | `0.3` | Pulay mixing parameter |
| `mixing_history` | `7` | Pulay history depth |
| `device` | `"cpu"` | `"cpu"` or `"gpu"` |

**What's next:** [ASE integration]({{ site.baseurl }}/examples/ase-integration/) shows how to use LYNX inside the broader ASE ecosystem.
