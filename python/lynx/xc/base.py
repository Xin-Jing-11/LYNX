"""Exchange-correlation functional base class."""
from abc import ABC, abstractmethod
import numpy as np


class Functional(ABC):
    """Base class for XC functionals.

    Subclass this to implement custom XC functionals.
    Built-in functionals wrap libxc via C++.
    """

    @abstractmethod
    def evaluate(self, rho, *, tau=None):
        """Evaluate XC potential and energy density.

        Args:
            rho: electron density array (Nd,) or (Nspin, Nd)
            tau: kinetic energy density (mGGA only)

        Returns:
            (Vxc, exc) — potential and energy density arrays
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def is_gga(self) -> bool:
        return False

    @property
    def is_mgga(self) -> bool:
        return False

    @property
    def is_hybrid(self) -> bool:
        return False

    @property
    def exx_fraction(self) -> float:
        return 0.0

    def __repr__(self):
        return f"{self.name}()"
