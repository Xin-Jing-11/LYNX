"""Variable-cell optimization with ASE + LYNX.

Relaxes both atomic positions AND cell shape/volume.
Uses ASE's FrechetCellFilter (or ExpCellFilter) to expose cell
degrees of freedom to the optimizer.

Usage:
    python 02_cell_optimization.py
"""

from ase.build import bulk
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from lynx.ase import LynxCalculator

# ---------------------------------------------------------------
# 1. Build structure with slightly wrong lattice constant
# ---------------------------------------------------------------
atoms = bulk("Si", "diamond", a=5.50)  # true a ~ 5.43 Ang
print(f"Initial cell volume: {atoms.get_volume():.2f} Ang^3")
print(f"Initial lattice constant: {atoms.cell.cellpar()[0]:.4f} Ang")

# ---------------------------------------------------------------
# 2. Attach LYNX calculator
# ---------------------------------------------------------------
atoms.calc = LynxCalculator(
    xc="PBE",
    kpts=[2, 2, 2],
    mesh_spacing=0.4,
    max_scf=100,
    scf_tol=1e-6,
    device="cpu",
    verbose=0,
)

# ---------------------------------------------------------------
# 3. Wrap with ExpCellFilter for variable-cell optimization
# ---------------------------------------------------------------
# ExpCellFilter exposes cell DOFs as extra "atom" positions.
# The optimizer then relaxes both positions and cell simultaneously.
ecf = ExpCellFilter(atoms)

opt = BFGS(ecf, trajectory="si_cellopt.traj", logfile="si_cellopt.log")
opt.run(fmax=0.01)  # eV/Ang for forces, eV/Ang^3 for stress

print(f"\nConverged in {opt.nsteps} steps")
print(f"Final energy:  {atoms.get_potential_energy():.6f} eV")
print(f"Final volume:  {atoms.get_volume():.2f} Ang^3")
print(f"Final lattice: {atoms.cell.cellpar()[0]:.4f} Ang")

# ---------------------------------------------------------------
# 4. Alternative: FrechetCellFilter (ASE >= 3.23)
# ---------------------------------------------------------------
# FrechetCellFilter is more robust for anisotropic relaxation.
# Uncomment if your ASE version supports it:
#
# from ase.constraints import FrechetCellFilter
# fcf = FrechetCellFilter(atoms)
# opt = BFGS(fcf)
# opt.run(fmax=0.01)
