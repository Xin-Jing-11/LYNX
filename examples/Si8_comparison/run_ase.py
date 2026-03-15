#!/usr/bin/env python3
"""
Si8 Diamond — ASE Calculator interface.

Same system as Si8.json and run_python.py, but defined using ASE's
structure-building tools with Angstrom/eV units.

The LynxCalculator adapter converts ASE units (Angstrom, eV) to LYNX
internal units (Bohr, Hartree) automatically — no manual conversion needed.

Usage (from this directory):
    python run_ase.py

Requirements:
    pip install ase
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from ase import Atoms
from lynx.ase_interface import LynxCalculator

# ---- Build the structure with ASE (Angstrom) ----
#
# Si diamond conventional cell: a = 10.26 Bohr.
# Convert exactly to Angstrom so all three examples use the same cell.
#
from lynx.units import BOHR_TO_ANG
a = 10.26 * BOHR_TO_ANG  # Angstrom (= 5.4294... Ang)

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
    cell=[a, a, a],
    pbc=True,
)

print("========================================")
print(" Si8 Diamond — ASE Calculator interface")
print("========================================")
print()
print(f"Cell (Angstrom): {atoms.cell.lengths()}")
print(f"Atoms: {len(atoms)} x {atoms.get_chemical_symbols()[0]}")
print(f"Volume: {atoms.cell.volume:.2f} Ang^3")
print()

# ---- Attach LYNX calculator ----
#
# mesh_spacing is in Bohr (LYNX internal parameter).
# All other inputs/outputs are in ASE units (Angstrom, eV).
#
PSP_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'psps')
si_psp = os.path.join(PSP_DIR, 'ONCVPSP-PBE-PDv0.4', 'Si', 'Si.psp8')

calc = LynxCalculator(
    xc='GGA_PBE',
    mesh_spacing=10.26 / 24,  # match 24x24x24 grid on a=10.26 Bohr cell
    Nstates=24,
    elec_temp=300.0,
    smearing='gaussian',
    max_scf_iter=100,
    scf_tol=1e-6,
    mixing_param=0.3,
    pseudo_files={'Si': si_psp},
)
atoms.calc = calc

# ---- Compute properties (ASE units: eV, eV/Ang) ----
energy = atoms.get_potential_energy()

print(f"Total energy: {energy:.6f} eV")
print(f"Energy/atom:  {energy/len(atoms):.6f} eV/atom")
print()

forces = atoms.get_forces()
print("Forces (eV/Angstrom):")
for i, (sym, f) in enumerate(zip(atoms.get_chemical_symbols(), forces)):
    print(f"  {sym} {i+1:3d}: {f[0]:14.8f} {f[1]:14.8f} {f[2]:14.8f}")
print()

# ---- Cross-check: convert to Hartree for comparison with C++ / Python ----
from lynx.units import HA_TO_EV, HA_BOHR_TO_EV_ANG

print("--- Cross-check (converted to Hartree/Bohr) ---")
print(f"  Etotal = {energy / HA_TO_EV:.10f} Ha")
print(f"  Eatom  = {energy / len(atoms) / HA_TO_EV:.10f} Ha/atom")
print()
print("  Forces (Ha/Bohr):")
for i, f in enumerate(forces):
    fx = f[0] / HA_BOHR_TO_EV_ANG
    fy = f[1] / HA_BOHR_TO_EV_ANG
    fz = f[2] / HA_BOHR_TO_EV_ANG
    print(f"    Atom {i+1:3d}: {fx:14.10f} {fy:14.10f} {fz:14.10f}")
