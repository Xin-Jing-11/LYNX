"""Abstract base classes for modular, algorithm-agnostic DFT components.

All data exchanged as numpy arrays. Each backend manages its own spatial
representation internally. ABCs operate on single-(spin,kpt) channels.
"""

from abc import ABC, abstractmethod
from numpy import ndarray


class HamiltonianOperator(ABC):
    """Apply H|psi> = T|psi> + Veff|psi> + Vnl|psi>.

    Fused operator — a single call handles kinetic, local potential,
    and nonlocal pseudopotential in one halo exchange (for FD backends).
    """

    @abstractmethod
    def apply(self, psi: ndarray, Veff: ndarray, shift: float = 0.0) -> ndarray:
        """Apply Hamiltonian to wavefunctions.

        Args:
            psi: Wavefunctions, shape (Nd_d,) or (Nd_d, Nband).
            Veff: Effective potential, shape (Nd_d,).
            shift: Diagonal shift c, returning H*psi + c*psi.

        Returns:
            Result array, same shape as psi.
        """
        ...

    @property
    @abstractmethod
    def ndof(self) -> int:
        """Number of degrees of freedom (grid points in domain)."""
        ...


class KineticOperator(ABC):
    """Apply kinetic energy T|psi> = -0.5 * Laplacian |psi>."""

    @abstractmethod
    def apply(self, psi: ndarray) -> ndarray:
        """Apply kinetic operator.

        Args:
            psi: Wavefunctions, shape (Nd_d,) or (Nd_d, Nband).

        Returns:
            T*psi, same shape as psi.
        """
        ...

    @property
    @abstractmethod
    def ndof(self) -> int:
        """Number of degrees of freedom."""
        ...


class NonlocalOperator(ABC):
    """Apply nonlocal pseudopotential Vnl|psi>."""

    @abstractmethod
    def apply(self, psi: ndarray) -> ndarray:
        """Apply nonlocal projector.

        Args:
            psi: Wavefunctions, shape (Nd_d,) or (Nd_d, Nband).

        Returns:
            Vnl*psi, same shape as psi.
        """
        ...


class EigenSolver(ABC):
    """Solve partial eigenvalue problem for lowest eigenpairs."""

    @abstractmethod
    def solve(self, H: HamiltonianOperator, psi: ndarray, Veff: ndarray,
              **kw) -> tuple[ndarray, ndarray]:
        """Compute lowest eigenpairs of H.

        Args:
            H: Hamiltonian operator (for eigensolvers that call H.apply()).
            psi: Initial wavefunctions (Nd_d, Nband).
            Veff: Effective potential (Nd_d,).
            **kw: Solver-specific parameters.

        Returns:
            (psi_updated, eigenvalues) — shapes (Nd_d, Nband) and (Nband,).
        """
        ...


class PoissonSolver(ABC):
    """Solve the Poisson equation: Laplacian(phi) = rhs."""

    @abstractmethod
    def solve(self, rhs: ndarray, tol: float = 1e-8) -> tuple[ndarray, int]:
        """Solve Poisson equation.

        Args:
            rhs: Right-hand side, shape (Nd_d,).
            tol: Convergence tolerance.

        Returns:
            (phi, n_iterations).
        """
        ...


class XCFunctional(ABC):
    """Exchange-correlation potential and energy density."""

    @abstractmethod
    def evaluate(self, rho: ndarray) -> tuple[ndarray, ndarray]:
        """Evaluate XC potential and energy density.

        Args:
            rho: Electron density for one spin channel, shape (Nd_d,).

        Returns:
            (Vxc, exc) — both shape (Nd_d,).
        """
        ...

    @property
    @abstractmethod
    def is_gga(self) -> bool:
        """Whether this is a GGA functional (needs density gradient)."""
        ...


class DensityMixer(ABC):
    """Mix input/output densities for SCF convergence."""

    @abstractmethod
    def mix(self, rho_in: ndarray, rho_out: ndarray) -> ndarray:
        """Mix densities.

        Args:
            rho_in: Input density (flattened).
            rho_out: Output density from eigensolver (flattened).

        Returns:
            Mixed density for next SCF iteration.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset mixer history for a fresh SCF run."""
        ...


class OccupationFunction(ABC):
    """Compute occupation numbers from eigenvalues."""

    @abstractmethod
    def compute(self, eigenvalues: list, Nelectron: float,
                kpt_weights=None) -> tuple[list, float]:
        """Determine Fermi level and occupation numbers.

        Args:
            eigenvalues: List of (Nband,) arrays, one per (spin, kpt) channel.
                         Ordered as [s0k0, s0k1, ..., s1k0, s1k1, ...].
            Nelectron: Total number of electrons.
            kpt_weights: K-point weights array (len = Nkpts), None for gamma-only.

        Returns:
            (occupations, fermi_energy) where occupations is a list of arrays
            matching the structure of eigenvalues.
        """
        ...
