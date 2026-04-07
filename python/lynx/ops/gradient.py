"""Gradient operator."""
import numpy as np


class Gradient:
    """Discrete gradient on a finite-difference grid.

    Wraps the C++ Gradient for production performance.
    """

    def __init__(self, grid):
        self._grid = grid
        self._core_grad = None
        self._core_halo = None

    def _ensure_setup(self):
        if self._core_grad is not None:
            return
        try:
            from lynx import _core
            lattice = _core.make_lattice(
                _core.Mat3(*self._grid.cell.ravel()),
                _core.CellType.Orthogonal
            )
            fd_grid = _core.FDGrid(
                self._grid.Nx, self._grid.Ny, self._grid.Nz,
                lattice,
                _core.BCType.Periodic, _core.BCType.Periodic, _core.BCType.Periodic
            )
            stencil = _core.FDStencil(self._grid.fd_order, fd_grid, lattice)
            domain = _core.full_domain(fd_grid)
            self._core_halo = _core.HaloExchange(domain, stencil.FDn())
            self._core_grad = _core.Gradient(stencil, domain)
        except (ImportError, Exception):
            raise RuntimeError("C++ backend not available")

    def __call__(self, f, direction=0):
        """Apply gradient in given direction.

        Args:
            f: input array (ndof,)
            direction: 0=x, 1=y, 2=z

        Returns:
            Gradient array (ndof,)
        """
        self._ensure_setup()
        f = np.asarray(f, dtype=np.float64)
        return self._core_grad.apply(self._core_halo, f, direction)

    @property
    def ndof(self) -> int:
        return self._grid.ndof

    def __repr__(self):
        return f"Gradient(grid={self._grid.shape})"
