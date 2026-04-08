"""Swap DFT components — like custom nn.Module layers.

Replace the default eigensolver, mixer, or XC functional.
"""
import lynx
from lynx import solvers, xc

# Create calculator with custom components
calc = lynx.DFT(
    xc=xc.PBE(),                                # explicit XC object
    eigensolver=solvers.CheFSI(degree=25),       # tuned Chebyshev
    mixer=solvers.PulayMixer(beta=0.2, history=10),  # aggressive mixing
    kpts=[1, 1, 1],
    grid_shape=[25, 25, 25],
    verbose=0,
)

print(f"XC: {calc.xc}")
print(f"Eigensolver: {calc.eigensolver}")
print(f"Mixer: {calc.mixer}")

# Swap components after creation
calc.xc = "LDA_PZ"
print(f"\nAfter swap -> XC: {calc.xc}")

atoms = lynx.Atoms(
    cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
    fractional=[[0, 0, 0], [0.25, 0.25, 0.25]],
    symbols=["Si", "Si"],
    units="bohr",
    psp_dir="psps",
)

result = calc(atoms)
print(f"Energy: {result.energy:.8f} Ha")
