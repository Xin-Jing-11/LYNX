---
title: Home
layout: default
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
    <p>Wavefunctions never leave the GPU. Full SCF → forces → stress on device. 35–89× per-operator microbenchmark speedup over CPU.</p>
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
import lynx

atoms = lynx.Atoms(
    cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
    positions=[[0, 0, 0], [2.565, 2.565, 2.565]],
    symbols=["Si", "Si"],
    units="bohr",
)
result = lynx.calculate(atoms, xc="PBE")
print(result.energy)   # Hartree
```

**ASE — drop-in calculator:**

```python
from ase.build import bulk
from lynx.ase import LynxCalculator

atoms = bulk('Si', 'diamond', a=5.43)
atoms.calc = LynxCalculator(xc='PBE', kpts=(4, 4, 4))
print(atoms.get_potential_energy())   # eV
print(atoms.get_forces())             # eV/Å
```

**C++ — JSON input:**

```bash
mpirun -np 4 ./build/src/lynx examples/Si8.json
```

See [C++ Quickstart](/LYNX/examples/cpp-quickstart/) for the full input format.

---

## What's Inside

LYNX is built in layers. The **[Architecture](/LYNX/architecture/)** section explains the software design — how operators, solvers, and the Python API fit together. The **[Theory](/LYNX/theory/)** section covers the physics, from real-space discretization to forces and stress. The **[Examples](/LYNX/examples/)** section shows runnable code for common workflows.
