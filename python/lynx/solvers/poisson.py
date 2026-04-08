"""Poisson equation solvers."""
from abc import ABC, abstractmethod


class PoissonSolver(ABC):
    """Abstract base class for Poisson solvers."""

    @abstractmethod
    def solve(self, rhs, tol=1e-8):
        """Solve -nabla^2 phi = rhs.

        Returns:
            (phi, n_iterations)
        """
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class AAR(PoissonSolver):
    """Anderson Acceleration with Richardson (default).

    Wraps the C++ PoissonSolver.
    """

    def __init__(self, tol=1e-8, max_iter=1000):
        self.tol = tol
        self.max_iter = max_iter
        self._core_solver = None  # set during DFT setup

    def solve(self, rhs, tol=None):
        if self._core_solver is None:
            raise RuntimeError("AAR not initialized. Use via lynx.DFT().")
        raise NotImplementedError("Standalone solve not yet implemented")

    def __repr__(self):
        return f"AAR(tol={self.tol})"
