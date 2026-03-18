"""Concrete exchange-correlation functional wrapper around _core."""

from .abc import XCFunctional as XCFunctionalABC
from . import _core


class LibxcFunctional(XCFunctionalABC):
    """XC functional backed by libxc (via _core.XCFunctional).

    Wraps the C++ XCFunctional which supports LDA and GGA functionals.
    """

    def __init__(self, xc_func: _core.XCFunctional, Nd_d: int):
        """
        Args:
            xc_func: Initialized _core.XCFunctional (after setup()).
            Nd_d: Number of grid points in the domain.
        """
        self._xc = xc_func
        self._Nd_d = Nd_d

    def evaluate(self, rho):
        return self._xc.evaluate(rho, self._Nd_d)

    @property
    def is_gga(self):
        return self._xc.is_gga()
