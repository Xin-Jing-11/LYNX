"""NVT (canonical) molecular dynamics with ASE + LYNX.

Constant temperature MD using the Langevin thermostat.

Usage:
    python 04_md_nvt.py              # CPU
    python 04_md_nvt.py --device gpu  # GPU
"""

import argparse
import numpy as np
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase import units
from lynx.ase import LynxCalculator

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
parser.add_argument("--steps", type=int, default=5)
args = parser.parse_args()

# ---------------------------------------------------------------
# 1. Build system (2-atom Si primitive cell)
# ---------------------------------------------------------------
atoms = bulk("Si", "diamond", a=5.43)
print(f"System: {len(atoms)} Si atoms, device={args.device}")

# ---------------------------------------------------------------
# 2. Attach LYNX calculator (relaxed SCF for MD)
# ---------------------------------------------------------------
atoms.calc = LynxCalculator(
    xc="LDA_PZ",
    kpts=[2, 2, 2],
    mesh_spacing=0.5,
    max_scf=60,
    scf_tol=1e-4,       # relaxed for MD
    mixing_beta=0.3,
    temperature=315.77,
    device=args.device,
    verbose=0,
)

# ---------------------------------------------------------------
# 3. Initialize velocities
# ---------------------------------------------------------------
target_T = 300  # Kelvin
MaxwellBoltzmannDistribution(atoms, temperature_K=target_T)

# ---------------------------------------------------------------
# 4. Set up Langevin thermostat
# ---------------------------------------------------------------
dt = 1.0 * units.fs
friction = 0.01 / units.fs

dyn = Langevin(
    atoms,
    timestep=dt,
    temperature_K=target_T,
    friction=friction,
    trajectory="si_nvt_langevin.traj",
    logfile="si_nvt_langevin.log",
)

print(f"\n{'Step':>6} {'Time (fs)':>10} {'Epot (eV)':>12} "
      f"{'Ekin (eV)':>12} {'T (K)':>8}")
print("-" * 54)

def print_status():
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    temp = ekin / (1.5 * units.kB * len(atoms))
    step = dyn.nsteps
    time_fs = step * dt / units.fs
    print(f"{step:6d} {time_fs:10.1f} {epot:12.6f} {ekin:12.6f} {temp:8.1f}")

dyn.attach(print_status, interval=1)

# ---------------------------------------------------------------
# 5. Run NVT MD
# ---------------------------------------------------------------
dyn.run(args.steps)

# ---------------------------------------------------------------
# 6. Analyze temperature distribution
# ---------------------------------------------------------------
traj = Trajectory("si_nvt_langevin.traj")
temperatures = []
for frame in traj:
    ekin = frame.get_kinetic_energy()
    T = ekin / (1.5 * units.kB * len(frame))
    temperatures.append(T)
traj.close()

temperatures = np.array(temperatures)
print(f"\nTemperature statistics:")
print(f"  Target: {target_T} K")
print(f"  Mean:   {temperatures.mean():.1f} K")
print(f"  Stdev:  {temperatures.std():.1f} K")
