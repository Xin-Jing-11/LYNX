"""NVE (microcanonical) molecular dynamics with ASE + LYNX.

Constant energy, constant volume MD using the Velocity Verlet integrator.
Useful for checking energy conservation of the force engine.

Usage:
    python 03_md_nve.py              # CPU
    python 03_md_nve.py --device gpu  # GPU
"""

import argparse
import numpy as np
from ase.build import bulk
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase import units
from lynx.ase import LynxCalculator

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
parser.add_argument("--steps", type=int, default=5)
args = parser.parse_args()

# ---------------------------------------------------------------
# 1. Build a small Si cell (2 atoms — fast for testing)
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
    scf_tol=1e-4,       # relaxed for MD — 1e-4 is fine
    mixing_beta=0.3,
    temperature=315.77,
    device=args.device,
    verbose=0,
)

# ---------------------------------------------------------------
# 3. Initialize velocities at 300 K
# ---------------------------------------------------------------
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Remove center-of-mass momentum
momenta = atoms.get_momenta()
momenta -= momenta.mean(axis=0)
atoms.set_momenta(momenta)

# ---------------------------------------------------------------
# 4. Set up NVE dynamics
# ---------------------------------------------------------------
dt = 1.0 * units.fs
dyn = VelocityVerlet(atoms, timestep=dt, trajectory="si_nve.traj",
                     logfile="si_nve.log")

print(f"\n{'Step':>6} {'Time (fs)':>10} {'Etot (eV)':>14} {'Ekin (eV)':>12} "
      f"{'Epot (eV)':>12} {'T (K)':>8}")
print("-" * 68)

def print_status():
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    temp = ekin / (1.5 * units.kB * len(atoms))
    step = dyn.nsteps
    time_fs = step * dt / units.fs
    print(f"{step:6d} {time_fs:10.1f} {epot + ekin:14.6f} "
          f"{ekin:12.6f} {epot:12.6f} {temp:8.1f}")

dyn.attach(print_status, interval=1)

# ---------------------------------------------------------------
# 5. Run MD
# ---------------------------------------------------------------
dyn.run(args.steps)

# ---------------------------------------------------------------
# 6. Check energy conservation
# ---------------------------------------------------------------
traj = Trajectory("si_nve.traj")
energies = []
for frame in traj:
    epot = frame.get_potential_energy()
    ekin = frame.get_kinetic_energy()
    energies.append(epot + ekin)
traj.close()

energies = np.array(energies)
drift = energies[-1] - energies[0]
std = np.std(energies)
print(f"\nEnergy conservation:")
print(f"  E_total drift: {drift:.6f} eV over {args.steps} steps")
print(f"  E_total stdev: {std:.6f} eV")
