"""Phonon calculation with ASE + LYNX.

Computes phonon frequencies using the finite displacement method.
Can produce phonon band structures and density of states.

Usage:
    python 08_phonons.py
"""

import numpy as np
from ase.build import bulk
from ase.phonons import Phonons
from lynx.ase import LynxCalculator

# ---------------------------------------------------------------
# 1. Build primitive cell
# ---------------------------------------------------------------
atoms = bulk("Si", "diamond", a=5.43)
print(f"Primitive cell: {len(atoms)} atoms")

# ---------------------------------------------------------------
# 2. Attach LYNX calculator
# ---------------------------------------------------------------
calc = LynxCalculator(
    xc="LDA_PZ",
    kpts=[4, 4, 4],
    mesh_spacing=0.35,
    max_scf=100,
    scf_tol=1e-6,
    device="cpu",
    verbose=0,
)

# ---------------------------------------------------------------
# 3. Set up phonon calculation
# ---------------------------------------------------------------
# supercell: 2x2x2 repetitions for force constants
# delta: finite displacement in Angstrom
ph = Phonons(atoms, calc, supercell=(2, 2, 2), delta=0.01)

# Run all displacement calculations
# This computes forces for +/- displacements of each atom
print("Running displacement calculations...")
ph.run()
print("Displacements complete.")

# ---------------------------------------------------------------
# 4. Extract force constants and compute phonon band structure
# ---------------------------------------------------------------
ph.read(acoustic=True)  # symmetrize acoustic modes

# High-symmetry path for FCC (diamond)
path = atoms.cell.bandpath("GXWKGLUWLK", npoints=100)
bs = ph.get_band_structure(path)

print(f"\nPhonon band structure computed along: {path.path}")
print(f"Number of q-points: {len(path.kpts)}")

# ---------------------------------------------------------------
# 5. Compute phonon density of states
# ---------------------------------------------------------------
dos = ph.get_dos(kpts=(10, 10, 10))
print(f"Phonon DOS computed on 10x10x10 q-grid")

# ---------------------------------------------------------------
# 6. Print zone-center frequencies
# ---------------------------------------------------------------
# Gamma-point frequencies
gamma_freq = bs.energies[0, 0, :]  # first path segment, first point
# Convert from eV to cm^-1 (1 eV = 8065.54 cm^-1)
freq_cm = gamma_freq * 8065.54

print(f"\nGamma-point phonon frequencies:")
for i, f in enumerate(freq_cm):
    mode = "acoustic" if abs(f) < 10 else "optical"
    print(f"  Mode {i+1}: {f:8.2f} cm^-1  ({mode})")

# ---------------------------------------------------------------
# 7. Save results (optional, requires matplotlib)
# ---------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Band structure
    bs.plot(ax=axes[0])
    axes[0].set_ylabel("Frequency (eV)")
    axes[0].set_title("Si phonon band structure")

    # DOS
    energies = dos.get_energies()
    weights = dos.get_weights()
    axes[1].plot(energies * 8065.54, weights)
    axes[1].set_xlabel("Frequency (cm$^{-1}$)")
    axes[1].set_ylabel("DOS")
    axes[1].set_title("Si phonon DOS")

    plt.tight_layout()
    plt.savefig("si_phonons.png", dpi=150)
    print("\nPhonon plot saved to si_phonons.png")
except Exception:
    print("\n(matplotlib not available — skipping plot)")

# Clean up displacement files
ph.clean()
