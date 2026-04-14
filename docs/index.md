---
title: Home
layout: home
nav_order: 1
---

<div style="text-align: center; padding: 3rem 0 2rem;">
  <h1 style="font-size: 3rem; font-weight: 700; letter-spacing: -1px; margin-bottom: 0.5rem;">LYNX</h1>
  <p style="font-size: 1.2rem; color: #555; max-width: 560px; margin: 0 auto 2rem; line-height: 1.6;">
    A real-space DFT engine designed for clarity —<br>
    GPU acceleration without algorithmic duplication,<br>
    Python access without overhead.
  </p>
  <a href="https://github.com/Xin-Jing-11/LYNX" class="btn btn-primary fs-5 mb-4 mb-md-0 mr-2">View on GitHub</a>
  &nbsp;
  <a href="/LYNX/architecture/" class="btn btn-outline fs-5 mb-4">Get Started →</a>
</div>

---

## Design Highlights

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.5rem; margin: 1.5rem 0 2.5rem;">
  <div style="background: #f8f9fb; border: 1px solid #e1e4e8; border-radius: 8px; padding: 1.5rem;">
    <h3 style="margin-top: 0;">⚡ PyTorch-style dispatch</h3>
    <p>Algorithm in <code>.cpp</code>, kernels in <code>.cu</code>. One <code>_cpu()/_gpu()</code> switch per operator. No algorithmic duplication across execution paths.</p>
    <a href="/LYNX/architecture/dispatch/">Learn more →</a>
  </div>
  <div style="background: #f8f9fb; border: 1px solid #e1e4e8; border-radius: 8px; padding: 1.5rem;">
    <h3 style="margin-top: 0;">🚀 GPU-resident SCF pipeline</h3>
    <p>Wavefunctions never leave the GPU. Full SCF → forces → stress on device. 35–89× speedup per operator over CPU.</p>
    <a href="/LYNX/architecture/">Learn more →</a>
  </div>
  <div style="background: #f8f9fb; border: 1px solid #e1e4e8; border-radius: 8px; padding: 1.5rem;">
    <h3 style="margin-top: 0;">🐍 Three-level Python API</h3>
    <p>From a one-liner <code>Calculator</code> to raw numpy operators. Zero-copy interop via pybind11. Drop-in ASE support.</p>
    <a href="/LYNX/architecture/python-api/">Learn more →</a>
  </div>
</div>

---

## Quick Example

**Python — one call:**

```python
from lynx.config import DFTConfig

config = DFTConfig(
    cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
    fractional=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    symbols=['Si', 'Si'],
    pseudo_files={'Si': 'psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8'},
    Nstates=10, xc='GGA_PBE',
)
calc = config.create_calculator(auto_run=True)
print(calc.total_energy)   # Hartree
```

**ASE — drop-in calculator:**

```python
from ase.build import bulk
from lynx.ase_interface import LynxCalculator

atoms = bulk('Si', 'diamond', a=5.43)
atoms.calc = LynxCalculator(xc='GGA_PBE', kpts=(4, 4, 4))
print(atoms.get_potential_energy())   # eV
print(atoms.get_forces())             # eV/Å
```

**C++ — JSON input:**

```bash
mpirun -np 4 ./build/src/lynx examples/Si8.json
```

```json
{
  "lattice": { "vectors": [[10.26,0,0],[0,10.26,0],[0,0,10.26]] },
  "grid": { "Nx": 40, "Ny": 40, "Nz": 40, "fd_order": 12 },
  "atoms": [{ "element": "Si", "pseudo_file": "psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8",
              "fractional": true,
              "coordinates": [[0,0,0],[0.25,0.25,0.25]] }],
  "electronic": { "xc": "GGA_PBE", "Nstates": 10 },
  "scf": { "max_iter": 100, "tolerance": 1e-6 }
}
```

---

## What's Inside

LYNX is built in layers. The **[Architecture](/LYNX/architecture/)** section explains the software design — how operators, solvers, and the Python API fit together. The **[Theory](/LYNX/theory/)** section covers the physics, from real-space discretization to forces and stress. The **[Examples](/LYNX/examples/)** section shows runnable code for common workflows.
