"""Laplacian operator."""
import numpy as np


class Laplacian:
    """Discrete Laplacian on a finite-difference grid.

    Wraps the C++ Laplacian for production performance.
    Supports callable interface and @ operator.
    """

    def __init__(self, grid):
        """
        Args:
            grid: lynx.Grid instance
        """
        self._grid = grid
        self._core_lap = None  # set when C++ context is available
        self._core_halo = None

    def _ensure_setup(self):
        """Lazy initialization from C++ context if available."""
        if self._core_lap is not None:
            return
        try:
            from lynx import _core
            lattice = _core.make_lattice(
                _core.Mat3(*self._grid.cell.ravel()),
                _core.CellType.Orthogonal  # simplified
            )
            fd_grid = _core.FDGrid(
                self._grid.Nx, self._grid.Ny, self._grid.Nz,
                lattice,
                _core.BCType.Periodic, _core.BCType.Periodic, _core.BCType.Periodic
            )
            stencil = _core.FDStencil(self._grid.fd_order, fd_grid, lattice)
            domain = _core.full_domain(fd_grid)
            self._core_halo = _core.HaloExchange(domain, stencil.FDn())
            self._core_lap = _core.Laplacian(stencil, domain)
        except (ImportError, Exception):
            raise RuntimeError("C++ backend not available. Build with -DBUILD_PYTHON=ON")

    def __call__(self, f, a=1.0, c=0.0):
        """Apply Laplacian: y = a * nabla^2 f + c * f.

        Args:
            f: input array (ndof,) or (ndof, ncol)
            a: scaling factor for Laplacian (default 1.0)
            c: diagonal shift (default 0.0)

        Returns:
            Result array, same shape as f
        """
        self._ensure_setup()
        f = np.asarray(f, dtype=np.float64)
        return self._core_lap.apply(self._core_halo, f, a=a, c=c)

    def __matmul__(self, f):
        """Support lap @ f syntax."""
        return self(f)

    @property
    def ndof(self) -> int:
        return self._grid.ndof

    def __repr__(self):
        return f"Laplacian(grid={self._grid.shape})"
