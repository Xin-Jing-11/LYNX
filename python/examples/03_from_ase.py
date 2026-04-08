"""Build structures with ASE, calculate with LYNX.

Requires: pip install ase
"""
try:
    from ase.build import bulk
except ImportError:
    print("This example requires ASE: pip install ase")
    raise SystemExit(1)

import lynx

# Build Si diamond with ASE (Angstrom)
ase_atoms = bulk("Si", "diamond", a=5.43)
print(f"ASE atoms: {ase_atoms}")

# Convert to LYNX (auto Angstrom -> Bohr)
atoms = lynx.Atoms.from_ase(ase_atoms, psp_dir="psps")
print(f"LYNX atoms: {atoms}")
print(f"Cell (Bohr): diag = [{atoms.cell[0,0]:.4f}, {atoms.cell[1,1]:.4f}, {atoms.cell[2,2]:.4f}]")

# Run DFT
result = lynx.calculate(atoms, xc="LDA_PZ", grid_shape=[25, 25, 25], verbose=0)
print(f"\nEnergy: {result.energy:.8f} Ha ({result.energy_eV:.6f} eV)")
