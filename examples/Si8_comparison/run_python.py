#!/usr/bin/env python3
"""
Si8 Diamond — Pure Python script (no JSON, no ASE).

Same system as Si8.json and run_ase.py, configured entirely from Python.
All values in LYNX internal units: Bohr (length), Hartree (energy).

Usage (from this directory):
    python run_python.py
"""
import os
import sys
import numpy as np

# Ensure the lynx package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import lynx
from lynx.config import DFTConfig

lynx.init()

# ---- System definition (Bohr) ----
a = 10.26  # lattice constant in Bohr

PSP_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'psps')
si_psp = os.path.join(PSP_DIR, '14_Si_4_1.9_1.9_pbe_n_v1.0.psp8')

config = DFTConfig(
    # Lattice vectors (Bohr)
    cell=[[a, 0, 0],
          [0, a, 0],
          [0, 0, a]],

    # 8-atom diamond fractional coordinates
    fractional=[
        [0.00, 0.00, 0.00],
        [0.25, 0.25, 0.25],
        [0.50, 0.50, 0.00],
        [0.75, 0.75, 0.25],
        [0.50, 0.00, 0.50],
        [0.75, 0.25, 0.75],
        [0.00, 0.50, 0.50],
        [0.25, 0.75, 0.75],
    ],
    symbols=['Si'] * 8,
    pseudo_files={'Si': si_psp},

    # Grid: 24x24x24, FD order 12
    Nx=24, Ny=24, Nz=24,
    fd_order=12,

    # Electronic
    xc='GGA_PBE',
    Nstates=24,
    elec_temp=300.0,
    smearing='gaussian',

    # SCF
    max_scf_iter=100,
    scf_tol=1e-6,
    mixing_param=0.3,
)

# ---- Run ----
print("========================================")
print(" Si8 Diamond — Python script (DFTConfig)")
print("========================================")
print()

calc = config.create_calculator(auto_run=True)

# ---- Results (Hartree / Bohr) ----
E = calc.energy
print()
print(f"SCF {'converged' if calc.converged else 'NOT CONVERGED'}")
print(f"  Eband   = {E['Eband']:18.10f} Ha")
print(f"  Exc     = {E['Exc']:18.10f} Ha")
print(f"  Ehart   = {E['Ehart']:18.10f} Ha")
print(f"  Eself   = {E['Eself']:18.10f} Ha")
print(f"  Ec      = {E['Ec']:18.10f} Ha")
print(f"  Entropy = {E['Entropy']:18.10f} Ha")
print(f"  Etotal  = {E['Etotal']:18.10f} Ha")
print(f"  Eatom   = {E['Etotal']/calc.Natom:18.10f} Ha/atom")
print(f"  Ef      = {calc.fermi_energy:18.10f} Ha")
print()

# Forces
forces = calc.compute_forces()
print("Forces (Ha/Bohr):")
for i in range(calc.Natom):
    print(f"  Atom {i+1:3d}: {forces[i,0]:14.10f} {forces[i,1]:14.10f} {forces[i,2]:14.10f}")
print()

# Density check
rho = calc.density
print(f"Density integral: {np.sum(rho)*calc.grid.dV():.6f} electrons (expect {calc.Nelectron})")
