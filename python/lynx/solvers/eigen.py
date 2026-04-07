"""Eigensolvers for the Kohn-Sham equations."""
from abc import ABC, abstractmethod
import numpy as np


class EigenSolver(ABC):
    """Abstract base class for eigensolvers.

    Subclass this to implement custom eigensolvers (e.g., LOBPCG, Davidson).
    The solve() method receives a Hamiltonian operator and wavefunctions.
    """

    @abstractmethod
    def solve(self, H, psi, Veff, **kwargs):
        """Solve the eigenvalue problem H*psi = E*psi.

        Args:
            H: Hamiltonian operator with .apply(psi, Veff) or @ operator
            psi: (Nd, Nband) initial wavefunctions
            Veff: (Nd,) effective potential
            **kwargs: solver-specific options (cheb_degree, bounds, etc.)

        Returns:
            (psi_out, eigenvalues): updated wavefunctions and eigenvalues
        """
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class CheFSI(EigenSolver):
    """Chebyshev-filtered subspace iteration (default).

    Wraps the C++ EigenSolver. This is the production eigensolver.
    """

    def __init__(self, degree=20):
        self.degree = degree
        self._core_solver = None  # set during DFT setup

    def solve(self, H, psi, Veff, **kwargs):
        if self._core_solver is None:
            raise RuntimeError("CheFSI not initialized. Use via lynx.DFT().")
        # When used standalone, delegate to C++ solver
        # kwargs may include: lambda_cutoff, eigval_min, eigval_max
        raise NotImplementedError("Standalone solve not yet implemented")

    def __repr__(self):
        return f"CheFSI(degree={self.degree})"


class LOBPCG(EigenSolver):
    """Locally Optimal Block Preconditioned Conjugate Gradient.

    Pure Python solver using scipy. Useful for testing and comparison.
    """

    def __init__(self, tol=1e-5, max_iter=500):
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, H, psi, Veff, **kwargs):
        try:
            from scipy.sparse.linalg import lobpcg, LinearOperator
        except ImportError:
            raise ImportError("LOBPCG requires scipy: pip install scipy")

        ndof = psi.shape[0]

        def matvec(x):
            # x may be (ndof,) or (ndof, k)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            return H.apply(x, Veff)

        A = LinearOperator((ndof, ndof), matvec=matvec, dtype=np.float64)
        eigvals, eigvecs = lobpcg(A, psi, tol=self.tol, maxiter=self.max_iter,
                                   largest=False)
        return eigvecs, eigvals

    def __repr__(self):
        return f"LOBPCG(tol={self.tol})"
