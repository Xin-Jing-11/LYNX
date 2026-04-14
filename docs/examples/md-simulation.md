---
title: MD Simulation
parent: Examples
nav_order: 4
---

# MD Simulation

Run NVE molecular dynamics using ASE's VelocityVerlet driver with LynxCalculator.
A complete, runnable version of this example lives at `examples/ase_ipi/03_md_nve.py`.

## NVE run

```python
from ase.build import bulk
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase import units
from lynx.ase import LynxCalculator

# Build structure
atoms = bulk('Si', 'diamond', a=5.43)

# Attach LYNX calculator (relaxed SCF tolerance is fine for MD)
atoms.calc = LynxCalculator(
    xc='LDA_PZ',
    kpts=(2, 2, 2),
    mesh_spacing=0.5,   # Bohr
    max_scf=60,
    scf_tol=1e-4,       # 1e-4 Ha/atom is adequate for MD forces
    mixing_beta=0.3,
    temperature=315.77,
    verbose=0,
)

# Initialize velocities at 300 K and remove center-of-mass drift
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
momenta = atoms.get_momenta()
momenta -= momenta.mean(axis=0)
atoms.set_momenta(momenta)

# Set up NVE dynamics (1 fs timestep)
dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs,
                     trajectory='si_nve.traj', logfile='si_nve.log')
dyn.run(50)   # 50 steps = 50 fs
```

## Density restart

`LynxCalculator` automatically reuses the converged electron density from the
previous MD step to warm-start the SCF solver of the next step. This is handled
internally: the `calculate()` method saves `result.density` after each step and
passes it as `initial_density` to the next `DFT()` call.

No extra code is required — simply reuse the same calculator object across all
MD steps (which ASE does by default when you set `atoms.calc` once):

```python
# One calculator object for the entire trajectory: density restart is automatic.
atoms.calc = LynxCalculator(xc='PBE', kpts=(2, 2, 2), scf_tol=1e-4)

dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
dyn.run(100)
```

Typical SCF iteration savings: 40–60% fewer iterations per step after the first.

## Reading a trajectory

```python
from ase.io import read
import matplotlib.pyplot as plt

traj = read('si_nve.traj', index=':')
energies = [a.get_potential_energy() for a in traj]

plt.plot(energies)
plt.xlabel('MD step')
plt.ylabel('Energy (eV)')
plt.savefig('si_nve_energy.png')
```

## NVT and NPT

Additional thermostat and barostat examples are provided in `examples/ase_ipi/`:

| Script | Ensemble | Driver |
|--------|----------|--------|
| `03_md_nve.py` | NVE | ASE VelocityVerlet |
| `04_md_nvt.py` | NVT | ASE Langevin |
| `05_md_npt.py` | NPT | ASE NPT |
| `09_ipi_client.py` | NVT/NPT | i-PI socket |

## i-PI socket interface

For advanced sampling (PIMD, replica exchange, metadynamics), LYNX can act as
an i-PI force engine over a UNIX or TCP socket:

```bash
# Terminal 1: start i-PI server
i-pi ipi_nvt.xml

# Terminal 2: start LYNX as i-PI client
python examples/ase_ipi/09_ipi_client.py
```

The i-PI input files `ipi_nvt.xml` and `ipi_npt.xml` are in `examples/ase_ipi/`.

**What's next:** [Architecture overview]({{ site.baseurl }}/architecture/) explains how LynxCalculator wraps the C++ engine and how psi stays GPU-resident across MD steps.
