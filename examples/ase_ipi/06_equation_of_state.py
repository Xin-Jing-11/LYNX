"""Equation of state (E-V curve) with ASE + LYNX.

Computes energy vs. volume for a range of lattice constants,
then fits to the Birch-Murnaghan equation of state to extract
the equilibrium lattice constant, bulk modulus, and minimum energy.

Usage:
    python 06_equation_of_state.py
"""

import numpy as np
from ase.build import bulk
from ase.eos import EquationOfState
from lynx.ase import LynxCalculator

# ---------------------------------------------------------------
# 1. Set up calculator
# ---------------------------------------------------------------
calc = LynxCalculator(
    xc="PBE",
    kpts=[4, 4, 4],
    mesh_spacing=0.35,
    max_scf=100,
    scf_tol=1e-6,
    device="cpu",
    verbose=0,
)

# ---------------------------------------------------------------
# 2. Compute E(V) for a range of lattice constants
# ---------------------------------------------------------------
a0 = 5.43  # approximate equilibrium lattice constant (Ang)
strains = np.linspace(-0.04, 0.04, 9)  # +/- 4% strain

volumes = []
energies = []

print(f"{'a (Ang)':>10} {'V (Ang^3)':>12} {'E (eV)':>14}")
print("-" * 40)

for strain in strains:
    a = a0 * (1 + strain)
    atoms = bulk("Si", "diamond", a=a)
    atoms.calc = calc

    e = atoms.get_potential_energy()
    v = atoms.get_volume()

    volumes.append(v)
    energies.append(e)
    print(f"{a:10.4f} {v:12.4f} {e:14.8f}")

# ---------------------------------------------------------------
# 3. Fit Birch-Murnaghan EOS
# ---------------------------------------------------------------
eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
v0, e0, B = eos.fit()

# B is in eV/Ang^3, convert to GPa
B_GPa = B / 0.006241509  # 1 eV/Ang^3 = 160.217 GPa ... use ASE's units
from ase import units
B_GPa = B * 160.21766208  # eV/Ang^3 -> GPa

# Back-calculate lattice constant from equilibrium volume
# For diamond: V = a^3 / 4 (2 atoms in primitive cell)
a_eq = (4 * v0) ** (1.0 / 3.0)

print(f"\n--- Birch-Murnaghan fit ---")
print(f"Equilibrium lattice constant: {a_eq:.4f} Ang")
print(f"Equilibrium volume:           {v0:.4f} Ang^3")
print(f"Minimum energy:               {e0:.8f} eV")
print(f"Bulk modulus:                  {B_GPa:.2f} GPa")

# ---------------------------------------------------------------
# 4. Save plot (optional, requires matplotlib)
# ---------------------------------------------------------------
try:
    eos.plot("si_eos.png")
    print("\nEOS plot saved to si_eos.png")
except Exception:
    print("\n(matplotlib not available — skipping plot)")
