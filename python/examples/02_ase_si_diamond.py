"""
Example 2: Use LYNX as an ASE calculator.

ASE uses Angstrom/eV units. The LynxCalculator adapter converts automatically.
All input/output through ASE is in Angstrom and eV — no need to think about Bohr.

Requirements:
    pip install ase
"""
from ase import Atoms
from ase.build import bulk
from lynx.ase_interface import LynxCalculator

# ---- Create structure with ASE (Angstrom) ----
# bulk() creates standard crystal structures with experimental lattice constants
atoms = bulk('Si', 'diamond', a=5.43)  # 2-atom primitive cell, a in Angstrom
print(f"ASE cell (Angstrom):\n{atoms.cell.array}")
print(f"Positions (Angstrom):\n{atoms.positions}")
print(f"Symbols: {atoms.get_chemical_symbols()}")
print(f"PBC: {atoms.pbc}")

# ---- Attach LYNX calculator ----
calc = LynxCalculator(
    xc='GGA_PBE',
    kpts=(2, 2, 2),
    mesh_spacing=0.5,    # Bohr (grid spacing — this is a LYNX internal parameter)
    Nstates=10,
    max_scf_iter=100,
    scf_tol=1e-5,
    # psp_dir='psps/',   # uncomment if pseudopotentials are in a custom directory
)
atoms.calc = calc

# ---- Compute properties (all returned in ASE units: eV, eV/Ang) ----
energy = atoms.get_potential_energy()
print(f"\nTotal energy: {energy:.6f} eV")
print(f"Energy per atom: {energy / len(atoms):.6f} eV/atom")

forces = atoms.get_forces()
print(f"\nForces (eV/Angstrom):")
for i, (sym, f) in enumerate(zip(atoms.get_chemical_symbols(), forces)):
    print(f"  {sym} {i}: [{f[0]:12.8f}, {f[1]:12.8f}, {f[2]:12.8f}]")
