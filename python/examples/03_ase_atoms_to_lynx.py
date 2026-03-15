"""
Example 3: Build structure with ASE, run with LYNX directly.

This shows how to use ASE's structure-building tools (bulk, molecule, surface, etc.)
and then run LYNX without going through the ASE Calculator interface.
Useful when you want direct access to LYNX internals (density, wavefunctions, etc.)
but still want to define structures with ASE's convenient API.

ASE units:  Angstrom, eV
LYNX units: Bohr, Hartree
DFTConfig.from_ase() handles the conversion.
"""
import numpy as np
from ase import Atoms
from ase.build import bulk

import lynx
from lynx.config import DFTConfig
from lynx.units import HA_TO_EV, BOHR_TO_ANG

lynx.init()

# ---- Build structure with ASE (all in Angstrom) ----
atoms = bulk('Si', 'diamond', a=5.43)
print(f"Cell (Angstrom):\n{atoms.cell.array}")
print(f"Fractional positions:\n{atoms.get_scaled_positions()}")

# ---- Convert ASE Atoms -> LYNX DFTConfig ----
# from_ase() converts Angstrom -> Bohr automatically
config = DFTConfig.from_ase(
    atoms,
    xc='GGA_PBE',
    kpts=(2, 2, 2),
    mesh_spacing=0.5,
    Nstates=10,
    max_scf_iter=100,
    scf_tol=1e-5,
)

# ---- Run SCF ----
calc = config.create_calculator(auto_run=True)

# ---- Access LYNX results directly (in Hartree/Bohr) ----
print(f"\nTotal energy: {calc.total_energy:.10f} Ha")
print(f"             = {calc.total_energy * HA_TO_EV:.6f} eV")
print(f"Fermi energy: {calc.fermi_energy:.10f} Ha")
print(f"Converged: {calc.converged}")

# Access electron density
rho = calc.density
print(f"\nDensity grid: {rho.shape[0]} points")
print(f"Integral: {np.sum(rho) * calc.grid.dV():.6f} electrons (expect {calc.Nelectron})")

# Access wavefunctions
wfn = calc.get_wavefunction()
print(f"\nWavefunction: {wfn.Nband()} bands, {wfn.Nspin()} spins, {wfn.Nkpts()} kpts")
eigs = wfn.eigenvalues(0, 0)  # spin=0, kpt=0
print(f"Eigenvalues (Ha): {eigs[:5]}")
print(f"Eigenvalues (eV): {eigs[:5] * HA_TO_EV}")

# Access grid info
grid = calc.grid
print(f"\nGrid: {grid.Nx()}x{grid.Ny()}x{grid.Nz()} = {grid.Nd()} points")
print(f"Grid spacing: dx={grid.dx():.4f} dy={grid.dy():.4f} dz={grid.dz():.4f} Bohr")
print(f"            = dx={grid.dx()*BOHR_TO_ANG:.4f} dy={grid.dy()*BOHR_TO_ANG:.4f} dz={grid.dz()*BOHR_TO_ANG:.4f} Ang")
