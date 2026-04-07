"""Grid configuration for finite-difference DFT.

Usage:
    grid = lynx.Grid(cell=atoms.cell, shape=[25, 25, 25])
    grid = lynx.Grid(cell=atoms.cell, spacing=0.3)

    print(grid.ndof)    # total grid points
    print(grid.shape)   # (Nx, Ny, Nz)
    print(grid.spacing) # (dx, dy, dz) in Bohr
"""

import numpy as np


class Grid:
    """Finite-difference grid for real-space DFT.

    Like torch.Size but for 3D spatial grids.
    """

    def __init__(self, cell, *, shape=None, spacing=None, fd_order=12):
        """
        Args:
            cell: (3,3) lattice vectors in Bohr, or (3,) for orthorhombic
            shape: (Nx, Ny, Nz) grid dimensions (mutually exclusive with spacing)
            spacing: target mesh spacing in Bohr (auto-computes shape)
            fd_order: finite-difference stencil order (default 12)
        """
        # Validate cell
        cell = np.asarray(cell, dtype=float)
        if cell.shape == (3,):
            cell = np.diag(cell)
        if cell.shape != (3, 3):
            raise ValueError(f"cell must be (3,3) or (3,), got {cell.shape}")
        self._cell = cell.copy()
        self._fd_order = fd_order

        # Compute grid dimensions
        if shape is not None and spacing is not None:
            raise ValueError("Specify shape OR spacing, not both")

        if shape is not None:
            self._shape = tuple(int(s) for s in shape)
        elif spacing is not None:
            # Auto-compute from spacing
            lengths = np.linalg.norm(cell, axis=1)
            self._shape = tuple(max(int(np.ceil(L / spacing)), 1) for L in lengths)
        else:
            raise ValueError("Must specify either shape or spacing")

        # Compute derived quantities
        lengths = np.linalg.norm(cell, axis=1)
        self._spacing = tuple(L / n for L, n in zip(lengths, self._shape))
        self._dV = abs(np.linalg.det(cell)) / (self._shape[0] * self._shape[1] * self._shape[2])

    @property
    def cell(self) -> np.ndarray:
        """(3,3) lattice vectors in Bohr."""
        return self._cell.copy()

    @property
    def shape(self) -> tuple:
        """(Nx, Ny, Nz) grid dimensions."""
        return self._shape

    @property
    def Nx(self) -> int: return self._shape[0]
    @property
    def Ny(self) -> int: return self._shape[1]
    @property
    def Nz(self) -> int: return self._shape[2]

    @property
    def ndof(self) -> int:
        """Total number of grid points."""
        return self._shape[0] * self._shape[1] * self._shape[2]

    @property
    def spacing(self) -> tuple:
        """(dx, dy, dz) mesh spacing in Bohr."""
        return self._spacing

    @property
    def dV(self) -> float:
        """Volume element in Bohr^3."""
        return self._dV

    @property
    def fd_order(self) -> int:
        return self._fd_order

    def __repr__(self):
        return (f"Grid(shape={self._shape}, "
                f"spacing=({self._spacing[0]:.3f}, {self._spacing[1]:.3f}, {self._spacing[2]:.3f}) Bohr, "
                f"fd_order={self._fd_order})")
