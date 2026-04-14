# ASE & i-PI Integration Examples

Examples showing how to use LYNX as a force engine for molecular dynamics,
geometry optimization, and advanced atomistic simulations.

## Prerequisites

```bash
# Build LYNX with Python bindings
cd build && cmake .. -DBUILD_PYTHON=ON && make -j$(nproc)
cd ../python && pip install -e .

# Install ASE
pip install ase

# Install i-PI (for socket examples)
pip install ipi
```

## ASE Calculator Examples

LYNX provides a drop-in ASE calculator via `lynx.ase.LynxCalculator`.
Any ASE tool that needs energy/forces/stress works out of the box.

| Example | Description |
|---------|-------------|
| `01_geometry_optimization.py` | Relax atomic positions with BFGS/LBFGS |
| `02_cell_optimization.py` | Relax cell + positions (variable-cell) |
| `03_md_nve.py` | Microcanonical (NVE) molecular dynamics |
| `04_md_nvt.py` | Canonical (NVT) MD with Langevin thermostat |
| `05_md_npt.py` | Isothermal-isobaric (NPT) MD |
| `06_equation_of_state.py` | Energy-volume curve and bulk modulus |
| `07_neb.py` | Nudged elastic band for transition states |
| `08_phonons.py` | Finite-displacement phonon calculation |

## i-PI Socket Examples

LYNX can act as a force engine for i-PI, communicating atomic positions,
forces, energy, and stress over a UNIX or TCP socket.

| Example | Description |
|---------|-------------|
| `09_ipi_client.py` | LYNX as an i-PI socket client |
| `ipi_nvt.xml` | i-PI input: NVT thermostat driving LYNX |
| `ipi_npt.xml` | i-PI input: NPT barostat driving LYNX |
| `run_ipi_nvt.sh` | Launch i-PI server + LYNX client together |

## Typical Workflow

```python
from ase.build import bulk
from lynx.ase import LynxCalculator

# 1. Build structure with ASE
atoms = bulk("Si", "diamond", a=5.43)

# 2. Attach LYNX as force engine
atoms.calc = LynxCalculator(xc="PBE", kpts=[2, 2, 2], mesh_spacing=0.4)

# 3. Use any ASE tool
from ase.optimize import BFGS
opt = BFGS(atoms)
opt.run(fmax=0.01)  # eV/Angstrom
```

## GPU Acceleration

All examples support GPU by adding `device="gpu"`:

```python
atoms.calc = LynxCalculator(xc="PBE", kpts=[2, 2, 2], device="gpu")
```
