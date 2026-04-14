"""NPT (isothermal-isobaric) molecular dynamics with ASE + LYNX.

Constant temperature and pressure MD. The cell volume fluctuates
to maintain the target pressure.

Requires stress tensor from the calculator.

Usage:
    python 05_md_npt.py
"""

import numpy as np
from ase.build import bulk
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase import units
from lynx.ase import LynxCalculator

# ---------------------------------------------------------------
# 1. Build system
# ---------------------------------------------------------------
atoms = bulk("Si", "diamond", a=5.43).repeat((2, 2, 2))  # 16 atoms
print(f"System: {len(atoms)} Si atoms")
print(f"Initial volume: {atoms.get_volume():.2f} Ang^3")

# ---------------------------------------------------------------
# 2. Attach LYNX calculator (stress needed for NPT)
# ---------------------------------------------------------------
atoms.calc = LynxCalculator(
    xc="LDA_PZ",
    kpts=[1, 1, 1],
    mesh_spacing=0.5,
    max_scf=100,
    scf_tol=1e-6,
    device="cpu",
    verbose=0,
)

# ---------------------------------------------------------------
# 3. Initialize velocities
# ---------------------------------------------------------------
target_T = 300      # Kelvin
target_P = 0.0      # GPa (zero external pressure)
MaxwellBoltzmannDistribution(atoms, temperature_K=target_T)

# ---------------------------------------------------------------
# 4. Set up NPT dynamics (Parrinello-Rahman style)
# ---------------------------------------------------------------
dt = 1.0 * units.fs

# ASE NPT uses pressure in eV/Ang^3
pressure_eV_A3 = target_P * units.GPa

dyn = NPT(
    atoms,
    timestep=dt,
    temperature_K=target_T,
    externalstress=pressure_eV_A3,
    ttime=25 * units.fs,          # thermostat time constant
    pfactor=((75 * units.fs) ** 2)  # barostat coupling (~ bulk modulus * ttime^2)
        * 160.2 * units.GPa,       # rough Si bulk modulus
    trajectory="si_npt.traj",
    logfile="si_npt.log",
)

# Print header
print(f"\n{'Step':>6} {'Time (fs)':>10} {'Epot (eV)':>12} "
      f"{'T (K)':>8} {'V (Ang^3)':>12} {'P (GPa)':>10}")
print("-" * 64)

def print_status():
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    temp = ekin / (1.5 * units.kB * len(atoms))
    vol = atoms.get_volume()
    # Pressure from stress: P = -1/3 * Tr(stress)
    stress = atoms.get_stress(voigt=False)  # 3x3
    pressure = -np.trace(stress) / 3.0 / units.GPa
    step = dyn.nsteps
    time_fs = step * dt / units.fs
    print(f"{step:6d} {time_fs:10.1f} {epot:12.6f} "
          f"{temp:8.1f} {vol:12.2f} {pressure:10.4f}")

dyn.attach(print_status, interval=1)

# ---------------------------------------------------------------
# 5. Run NPT MD
# ---------------------------------------------------------------
n_steps = 10  # increase for production
dyn.run(n_steps)

# ---------------------------------------------------------------
# 6. Analyze volume fluctuations
# ---------------------------------------------------------------
traj = Trajectory("si_npt.traj")
volumes = [frame.get_volume() for frame in traj]
traj.close()

volumes = np.array(volumes)
print(f"\nVolume statistics:")
print(f"  Initial: {volumes[0]:.2f} Ang^3")
print(f"  Final:   {volumes[-1]:.2f} Ang^3")
print(f"  Mean:    {volumes.mean():.2f} Ang^3")
print(f"  Stdev:   {volumes.std():.2f} Ang^3")
