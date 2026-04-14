"""Geometry optimization with ASE + LYNX.

Relaxes atomic positions to minimize forces using BFGS or LBFGS.
The cell shape is fixed; only atom positions move.

Usage:
    python 01_geometry_optimization.py
"""

from ase.build import bulk
from ase.optimize import BFGS, LBFGS
from lynx.ase import LynxCalculator

# ---------------------------------------------------------------
# 1. Build a slightly distorted Si diamond structure
# ---------------------------------------------------------------
atoms = bulk("Si", "diamond", a=5.43)

# Displace one atom so there are nonzero forces
atoms.positions[1] += [0.05, 0.03, -0.02]  # Angstrom

print(f"Initial structure: {len(atoms)} atoms")
print(f"Initial positions (Ang):\n{atoms.positions}")

# ---------------------------------------------------------------
# 2. Attach LYNX calculator
# ---------------------------------------------------------------
atoms.calc = LynxCalculator(
    xc="PBE",
    kpts=[2, 2, 2],
    mesh_spacing=0.4,      # Bohr — auto-determines grid
    max_scf=100,
    scf_tol=1e-6,
    device="cpu",           # change to "gpu" for GPU acceleration
    verbose=0,
)

# ---------------------------------------------------------------
# 3. Optimize with BFGS
# ---------------------------------------------------------------
print("\n--- BFGS optimization ---")
opt = BFGS(atoms, trajectory="si_opt_bfgs.traj", logfile="si_opt_bfgs.log")
opt.run(fmax=0.01)  # converge when max force < 0.01 eV/Ang

print(f"\nConverged in {opt.nsteps} steps")
print(f"Final energy: {atoms.get_potential_energy():.6f} eV")
print(f"Max force:    {abs(atoms.get_forces()).max():.6f} eV/Ang")
print(f"Final positions (Ang):\n{atoms.positions}")

# ---------------------------------------------------------------
# 4. Alternative: LBFGS (better for large systems)
# ---------------------------------------------------------------
# Reset distortion
atoms2 = bulk("Si", "diamond", a=5.43)
atoms2.positions[1] += [0.05, 0.03, -0.02]
atoms2.calc = atoms.calc

print("\n--- LBFGS optimization ---")
opt2 = LBFGS(atoms2, trajectory="si_opt_lbfgs.traj", logfile="si_opt_lbfgs.log")
opt2.run(fmax=0.01)

print(f"Converged in {opt2.nsteps} steps")
print(f"Final energy: {atoms2.get_potential_energy():.6f} eV")

# ---------------------------------------------------------------
# 5. Read trajectory (optional analysis)
# ---------------------------------------------------------------
from ase.io import read
traj = read("si_opt_bfgs.traj", index=":")
print(f"\nTrajectory has {len(traj)} frames")
for i, frame in enumerate(traj):
    e = frame.get_potential_energy()
    fmax = abs(frame.get_forces()).max()
    print(f"  Step {i}: E = {e:.6f} eV, Fmax = {fmax:.6f} eV/Ang")
