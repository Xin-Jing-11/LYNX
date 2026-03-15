"""
Example 4: Build any structure with ASE and run with LYNX.

ASE provides many structure builders that work with Angstrom units:
  - bulk()       : standard crystal structures (FCC, BCC, diamond, etc.)
  - molecule()   : common molecules from a database
  - Atoms()      : arbitrary structures from scratch
  - surface()    : slab models for surface calculations
  - read()       : read from CIF, POSCAR, xyz, etc.

All of these produce ase.Atoms objects that can be passed to DFTConfig.from_ase().
"""
import numpy as np
from ase import Atoms

import lynx
from lynx.config import DFTConfig
from lynx.units import HA_TO_EV, ANG_TO_BOHR

lynx.init()

# ---- Method 1: Build manually with ASE Atoms (Angstrom) ----
# 8-atom Si conventional cell
a = 5.43  # Angstrom
atoms = Atoms(
    symbols=['Si'] * 8,
    scaled_positions=[
        [0.00, 0.00, 0.00],
        [0.25, 0.25, 0.25],
        [0.50, 0.50, 0.00],
        [0.75, 0.75, 0.25],
        [0.50, 0.00, 0.50],
        [0.75, 0.25, 0.75],
        [0.00, 0.50, 0.50],
        [0.25, 0.75, 0.75],
    ],
    cell=[a, a, a],  # Cubic cell, shorthand for diagonal
    pbc=True,
)

print(f"Cell (Ang): {atoms.cell.lengths()}")
print(f"N atoms: {len(atoms)}")
print(f"Volume: {atoms.cell.volume:.2f} Ang^3")

# ---- Convert and run ----
config = DFTConfig.from_ase(
    atoms,
    xc='GGA_PBE',
    kpts=(1, 1, 1),
    mesh_spacing=0.5,
    Nstates=25,
    max_scf_iter=50,
    scf_tol=1e-4,
)

calc = config.create_calculator(auto_run=True)
print(f"\nSi8 total energy: {calc.total_energy:.6f} Ha = {calc.total_energy * HA_TO_EV:.4f} eV")
print(f"Per atom: {calc.total_energy / 8 * HA_TO_EV:.4f} eV/atom")


# ---- Method 2: From CIF/POSCAR (requires file) ----
# from ase.io import read
# atoms = read('POSCAR')  # or 'structure.cif', 'structure.xyz', etc.
# config = DFTConfig.from_ase(atoms, xc='GGA_PBE', kpts=(4,4,4))
# calc = config.create_calculator(auto_run=True)
