"""Nudged Elastic Band (NEB) with ASE + LYNX.

Finds the minimum energy path between two configurations.
This example demonstrates vacancy migration in Si.

Usage:
    python 07_neb.py
"""

import numpy as np
from ase.build import bulk
from ase.neb import NEB
from ase.optimize import BFGS
from ase.io import write
from lynx.ase import LynxCalculator

# ---------------------------------------------------------------
# Helper: create a LYNX calculator
# ---------------------------------------------------------------
def make_calc():
    return LynxCalculator(
        xc="LDA_PZ",
        kpts=[1, 1, 1],       # Gamma for supercell
        mesh_spacing=0.5,
        max_scf=100,
        scf_tol=1e-6,
        device="cpu",
        verbose=0,
    )

# ---------------------------------------------------------------
# 1. Create initial state: Si supercell with vacancy
# ---------------------------------------------------------------
si = bulk("Si", "diamond", a=5.43).repeat((2, 2, 2))  # 16 atoms
n_atoms = len(si)

# Remove atom 0 to create vacancy
initial = si.copy()
del initial[0]

# Relax initial state
initial.calc = make_calc()
opt = BFGS(initial, logfile=None)
opt.run(fmax=0.05)
print(f"Initial state relaxed: E = {initial.get_potential_energy():.6f} eV")

# ---------------------------------------------------------------
# 2. Create final state: vacancy migrated to neighbor site
# ---------------------------------------------------------------
final = si.copy()
del final[1]  # remove a neighbor instead

# Relax final state
final.calc = make_calc()
opt = BFGS(final, logfile=None)
opt.run(fmax=0.05)
print(f"Final state relaxed:   E = {final.get_potential_energy():.6f} eV")

# ---------------------------------------------------------------
# 3. Set up NEB with interpolated images
# ---------------------------------------------------------------
n_images = 5  # number of intermediate images

images = [initial.copy()]
for i in range(n_images):
    image = initial.copy()
    image.calc = make_calc()
    images.append(image)
images.append(final.copy())

# Linear interpolation of positions between endpoints
neb = NEB(images, climb=True)  # CI-NEB for accurate barrier
neb.interpolate()

# ---------------------------------------------------------------
# 4. Optimize the NEB path
# ---------------------------------------------------------------
print(f"\nOptimizing NEB with {n_images} images (CI-NEB)...")
opt = BFGS(neb, trajectory="si_neb.traj", logfile="si_neb.log")
opt.run(fmax=0.05)

print(f"NEB converged in {opt.nsteps} steps")

# ---------------------------------------------------------------
# 5. Extract energy profile
# ---------------------------------------------------------------
print(f"\n{'Image':>6} {'Energy (eV)':>14} {'dE (eV)':>10}")
print("-" * 34)

e_initial = images[0].get_potential_energy()
energies = []

for i, image in enumerate(images):
    e = image.get_potential_energy()
    de = e - e_initial
    energies.append(de)
    marker = " <-- barrier" if de == max(energies) and i > 0 else ""
    print(f"{i:6d} {e:14.6f} {de:10.6f}{marker}")

barrier = max(energies)
print(f"\nMigration barrier: {barrier:.4f} eV")

# ---------------------------------------------------------------
# 6. Save path for visualization
# ---------------------------------------------------------------
write("si_neb_path.xyz", images)
print("NEB path saved to si_neb_path.xyz")
