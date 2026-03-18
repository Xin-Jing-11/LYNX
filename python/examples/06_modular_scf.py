"""
Example 6: Modular SCF — run DFT with pluggable, swappable components.

Demonstrates the algorithm-agnostic Python DFT framework:
  1. SystemSetup builds all C++ infrastructure from a DFTConfig
  2. default_operators() returns ABC-wrapped components
  3. SCFDriver composes them into a self-consistent field loop
  4. Any component can be swapped for a user-provided implementation

Units: Bohr (length), Hartree (energy) — same as LYNX internal.
"""
import numpy as np
import lynx

lynx.init()

# ---- 1. Find pseudopotential ----
import os
psp = lynx.find_psp("Si", xc="PBE")
if psp is None:
    raise FileNotFoundError("Si PBE pseudopotential not found in psps/")

# ---- 2. Build system from DFTConfig ----
config = lynx.DFTConfig(
    cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
    fractional=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    symbols=["Si", "Si"],
    pseudo_files={"Si": psp},
    Nx=20, Ny=20, Nz=20,
    Nstates=14,    # 12 occupied + 2 buffer for smearing
    xc="GGA_PBE",
)

system = lynx.SystemSetup(config)
print(f"System: {system.Nelectron} electrons, {system.Nd_d} grid points, "
      f"Nspin={system.Nspin}")

# ---- 3. Get default operators (all FD-based, wrapping C++) ----
ops = system.default_operators(xc_type=lynx.XCType.GGA_PBE)
params = system.scf_params()

print(f"Components: {', '.join(ops.keys())}")
print(f"  Hamiltonian:  {type(ops['hamiltonian']).__name__} (ndof={ops['hamiltonian'].ndof})")
print(f"  EigenSolver:  {type(ops['eigensolver']).__name__}")
print(f"  Poisson:      {type(ops['poisson']).__name__}")
print(f"  XC:           {type(ops['xc']).__name__} (is_gga={ops['xc'].is_gga})")
print(f"  Mixer:        {type(ops['mixer']).__name__}")
print(f"  Occupation:   {type(ops['occupation']).__name__}")

# ---- 4. Run SCF with default components ----
driver = lynx.SCFDriver(
    hamiltonian=ops["hamiltonian"],
    eigensolver=ops["eigensolver"],
    poisson=ops["poisson"],
    xc=ops["xc"],
    mixer=ops["mixer"],
    occupation=ops["occupation"],
    **params,
    max_iter=50,
    tol=1e-3,
    rho_trigger=4,
    on_iteration=lambda it, err, E: print(f"  iter {it:3d}: error={err:.4e}  E={E:.6f} Ha"),
)

# Initial wavefunctions (random) and density (flat, since atomic density
# may be unavailable depending on the pseudopotential)
psi_init = system.random_wavefunctions()
Nd_d = system.Nd_d
dV = system.dV
rho_init = np.full(Nd_d, params["Nelectron"] / (Nd_d * dV))

print("\n--- Running SCF with default FD components ---")
result = driver.run(psi_init, rho_init)

print(f"\nConverged: {result.converged}")
print(f"Total energy: {result.total_energy:.6f} Ha")
print(f"Fermi energy: {result.fermi_energy:.6f} Ha")
print(f"Iterations:   {result.n_iterations}")
print(f"Eigenvalues:  {result.eigenvalues[0][:6]}")


# ---- 5. Swap the mixer: use a custom damped mixer ----
print("\n\n--- Now swap in a simple linear mixer ---")

class SimpleMixer(lynx.abc.DensityMixer):
    """Textbook linear (damped) mixing: rho_next = (1-beta)*rho_in + beta*rho_out."""
    def __init__(self, beta=0.1):
        self.beta = beta
    def mix(self, rho_in, rho_out):
        return (1.0 - self.beta) * rho_in + self.beta * rho_out
    def reset(self):
        pass

driver2 = lynx.SCFDriver(
    hamiltonian=ops["hamiltonian"],
    eigensolver=ops["eigensolver"],
    poisson=ops["poisson"],
    xc=ops["xc"],
    mixer=SimpleMixer(beta=0.1),        # <-- swapped!
    occupation=ops["occupation"],
    **params,
    max_iter=80,
    tol=1e-3,
    rho_trigger=4,
    on_iteration=lambda it, err, E: print(f"  iter {it:3d}: error={err:.4e}  E={E:.6f} Ha")
                                    if it <= 5 or it % 10 == 0 else None,
)

print("Running with SimpleMixer (slower convergence expected)...")
result2 = driver2.run(psi_init.copy(), rho_init.copy())
print(f"Converged: {result2.converged}, iterations: {result2.n_iterations}")
print(f"Total energy: {result2.total_energy:.6f} Ha")
