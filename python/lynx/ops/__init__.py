"""Low-level operators -- like torch.nn.functional.

Usage:
    grid = lynx.Grid(cell, shape=[25, 25, 25])

    lap = lynx.ops.Laplacian(grid)
    y = lap(f)           # or: y = lap @ f

    grad = lynx.ops.Gradient(grid)
    df_dx = grad(f, direction=0)

    H = lynx.ops.Hamiltonian(grid)
    H.set_potential(Veff)
    Hpsi = H(psi)       # or: Hpsi = H @ psi
"""

from .laplacian import Laplacian
from .gradient import Gradient
from .hamiltonian import Hamiltonian

__all__ = ["Laplacian", "Gradient", "Hamiltonian"]
