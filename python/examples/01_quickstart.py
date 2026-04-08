"""LYNX Quickstart — one-liner DFT calculation.

Usage:
    python examples/01_quickstart.py
"""
import lynx

# Define Si2 crystal in Bohr
atoms = lynx.Atoms(
    cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
    fractional=[[0, 0, 0], [0.25, 0.25, 0.25]],
    symbols=["Si", "Si"],
    units="bohr",
    psp_dir="psps",
)

# One-liner DFT
result = lynx.calculate(atoms, xc="LDA_PZ", grid_shape=[25, 25, 25])

# Access results
print(f"Energy:     {result.energy:.8f} Ha ({result.energy_eV:.6f} eV)")
print(f"Converged:  {result.converged}")
print(f"Fermi:      {result.fermi_energy:.8f} Ha")
if result.forces is not None:
    import numpy as np
    print(f"Max force:  {np.max(np.abs(result.forces)):.6e} Ha/Bohr")
