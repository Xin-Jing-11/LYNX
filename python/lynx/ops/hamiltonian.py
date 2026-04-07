"""Hamiltonian operator."""
import numpy as np


class Hamiltonian:
    """Kohn-Sham Hamiltonian: H = -0.5*nabla^2 + Veff + Vnl.

    Supports callable and @ operator interfaces.
    """

    def __init__(self, grid=None, Veff=None):
        """
        Args:
            grid: lynx.Grid instance (optional, set during DFT setup)
            Veff: effective potential array (optional, set via set_potential)
        """
        self._grid = grid
        self._Veff = Veff
        self._core_H = None  # set during DFT setup

    def set_potential(self, Veff):
        """Update the effective potential."""
        self._Veff = np.asarray(Veff, dtype=np.float64)

    def apply(self, psi, Veff=None, shift=0.0):
        """Apply H to wavefunctions: Hpsi = (-0.5*nabla^2 + Veff + Vnl)*psi.

        Args:
            psi: (ndof,) or (ndof, nband) wavefunctions
            Veff: effective potential (uses stored if None)
            shift: diagonal shift (Hpsi += shift*psi)

        Returns:
            Hpsi, same shape as psi
        """
        if self._core_H is None:
            raise RuntimeError("Hamiltonian not initialized. Use via lynx.DFT().")

        if Veff is None:
            Veff = self._Veff
        if Veff is None:
            raise ValueError("No effective potential set. Call set_potential() first.")

        psi = np.asarray(psi, dtype=np.float64)
        Veff = np.asarray(Veff, dtype=np.float64)
        # Delegate to C++ operator wrapper
        return self._core_H.apply(psi, Veff, shift)

    def __call__(self, psi, Veff=None, shift=0.0):
        return self.apply(psi, Veff, shift)

    def __matmul__(self, psi):
        """Support H @ psi syntax (uses stored Veff)."""
        return self.apply(psi)

    @property
    def ndof(self) -> int:
        if self._grid is not None:
            return self._grid.ndof
        raise RuntimeError("Grid not set")

    def __repr__(self):
        shape = self._grid.shape if self._grid else "?"
        return f"Hamiltonian(grid={shape})"
