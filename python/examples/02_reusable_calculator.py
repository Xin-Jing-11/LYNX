"""Reusable DFT calculator — like nn.Module.

The DFT object can be configured once and applied to multiple structures.
"""
import lynx

# Configure calculator once
calc = lynx.DFT(
    xc="LDA_PZ",
    kpts=[1, 1, 1],
    grid_shape=[25, 25, 25],
    temperature=315.77,
    max_scf=100,
    scf_tol=1e-6,
    verbose=1,
)

print(f"Calculator: {calc}")

# Apply to Si2
si2 = lynx.Atoms(
    cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
    fractional=[[0, 0, 0], [0.25, 0.25, 0.25]],
    symbols=["Si", "Si"],
    units="bohr",
    psp_dir="psps",
)

result = calc(si2)
print(f"\nSi2 energy: {result.energy:.8f} Ha")
print(result.energies)
