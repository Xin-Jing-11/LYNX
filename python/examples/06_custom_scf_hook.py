"""Custom SCF hooks — like nn.Module hooks.

Subclass DFT to add per-step callbacks.
"""
import lynx

class VerboseDFT(lynx.DFT):
    """DFT with custom per-step output."""

    def on_converged(self, result):
        print(f"\n*** Converged! E = {result.energy:.8f} Ha ***")
        print(result.energies)

atoms = lynx.Atoms(
    cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
    fractional=[[0, 0, 0], [0.25, 0.25, 0.25]],
    symbols=["Si", "Si"],
    units="bohr",
    psp_dir="psps",
)

calc = VerboseDFT(xc="LDA_PZ", grid_shape=[25, 25, 25], verbose=0)
result = calc(atoms)
