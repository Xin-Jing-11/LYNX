"""Concrete finite-difference operator wrappers around _core.

These wrap the C++ operators (Hamiltonian, Laplacian, Gradient) and handle
halo exchange transparently. Each stores references to the C++ objects.
"""

from .abc import HamiltonianOperator, KineticOperator, NonlocalOperator
from . import _core


class FDHamiltonian(HamiltonianOperator):
    """Finite-difference Hamiltonian: -0.5*Lap + Veff + Vnl.

    Wraps _core.Hamiltonian with fused halo exchange — a single call
    handles kinetic, local potential, and nonlocal pseudopotential terms.
    """

    def __init__(self, hamiltonian: _core.Hamiltonian,
                 halo: _core.HaloExchange, Nd_d: int):
        self._H = hamiltonian
        self._halo = halo
        self._Nd_d = Nd_d

    def apply(self, psi, Veff, shift=0.0):
        return self._H.apply(self._halo, psi, Veff, c=shift)

    @property
    def ndof(self):
        return self._Nd_d


class FDKinetic(KineticOperator):
    """Finite-difference kinetic energy: T = -0.5 * Laplacian."""

    def __init__(self, laplacian: _core.Laplacian,
                 halo: _core.HaloExchange, Nd_d: int):
        self._lap = laplacian
        self._halo = halo
        self._Nd_d = Nd_d

    def apply(self, psi):
        return self._lap.apply(self._halo, psi, a=-0.5, c=0.0)

    @property
    def ndof(self):
        return self._Nd_d


class FDNonlocal(NonlocalOperator):
    """Finite-difference nonlocal pseudopotential projector.

    Note: In the FD backend, the nonlocal operator is fused into
    FDHamiltonian for efficiency (single halo exchange). This standalone
    wrapper exists for API completeness but the apply is not directly
    exposed in _core; use FDHamiltonian instead.
    """

    def __init__(self, projector: _core.NonlocalProjector):
        self._vnl = projector

    def apply(self, psi):
        raise NotImplementedError(
            "Standalone nonlocal apply is not exposed in _core. "
            "Use FDHamiltonian which fuses T + Veff + Vnl in one halo exchange."
        )


class FDGradient:
    """Finite-difference gradient operator.

    Not an ABC — gradient is only needed internally by GGA XC functionals.
    """

    def __init__(self, gradient: _core.Gradient, halo: _core.HaloExchange):
        self._grad = gradient
        self._halo = halo

    def apply(self, x, direction):
        """Apply gradient in given direction (0=x, 1=y, 2=z)."""
        return self._grad.apply(self._halo, x, direction)
