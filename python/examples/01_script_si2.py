"""
Example 1: Run a Si2 DFT calculation entirely from Python — no JSON file.

Uses LYNX internal units: Bohr (length), Hartree (energy).
"""
import lynx
from lynx.config import DFTConfig
import numpy as np

lynx.init()

# ---- Define the system in Bohr ----
a = 10.26  # lattice constant in Bohr (Si diamond conventional cell)

config = DFTConfig(
    cell=[[a, 0, 0],
          [0, a, 0],
          [0, 0, a]],
    fractional=[[0.00, 0.00, 0.00],
                [0.25, 0.25, 0.25]],
    symbols=['Si', 'Si'],
    pseudo_files={'Si': 'psps/14_Si_4_1.9_1.9_pbe_n_v1.0.psp8'},
    Nx=20, Ny=20, Nz=20,
    Nstates=10,
    xc='GGA_PBE',
    kpts=(1, 1, 1),
    max_scf_iter=100,
    scf_tol=1e-6,
)

# ---- Run ----
calc = config.create_calculator(auto_run=True)

# ---- Results (in Hartree / Bohr) ----
print(f"\nTotal energy:  {calc.total_energy:.10f} Ha")
print(f"Fermi energy:  {calc.fermi_energy:.10f} Ha")
print(f"Converged:     {calc.converged}")

E = calc.energy
print(f"\nEnergy breakdown:")
print(f"  Eband   = {E['Eband']:.10f} Ha")
print(f"  Exc     = {E['Exc']:.10f} Ha")
print(f"  Ehart   = {E['Ehart']:.10f} Ha")
print(f"  Etotal  = {E['Etotal']:.10f} Ha")
print(f"  Eatom   = {E['Etotal']/calc.Natom:.10f} Ha/atom")

# Density
rho = calc.density
print(f"\nDensity: shape={rho.shape}, integral={np.sum(rho)*calc.grid.dV():.6f} electrons")
