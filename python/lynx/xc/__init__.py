"""Exchange-correlation functionals.

Usage:
    import lynx
    calc = lynx.DFT(xc="PBE")           # string shorthand
    calc = lynx.DFT(xc=lynx.xc.PBE())   # explicit object
    calc = lynx.DFT(xc=lynx.xc.HSE06(alpha=0.25))  # with params
"""

from .base import Functional

# Each concrete class stores its XCType enum name and optional params.
# Actual C++ setup happens in DFT._setup_xc() when grid info is available.

class _LibxcFunctional(Functional):
    """Base for libxc-backed functionals."""
    _xc_type = None  # set by subclasses (string matching XCType enum)
    _is_gga = False
    _is_mgga = False
    _is_hybrid = False
    _exx_frac = 0.0

    def __init__(self):
        self._core_xc = None  # set during DFT setup

    def evaluate(self, rho, *, tau=None):
        if self._core_xc is None:
            raise RuntimeError("XC functional not initialized. Use via lynx.DFT().")
        # Delegate to C++ — this is used by low-level access
        raise NotImplementedError("Direct evaluate requires grid context")

    @property
    def is_gga(self): return self._is_gga
    @property
    def is_mgga(self): return self._is_mgga
    @property
    def is_hybrid(self): return self._is_hybrid
    @property
    def exx_fraction(self): return self._exx_frac


# --- LDA ---
class LDA_PZ(_LibxcFunctional):
    _xc_type = "LDA_PZ"

class LDA_PW(_LibxcFunctional):
    _xc_type = "LDA_PW"

# --- GGA ---
class PBE(_LibxcFunctional):
    _xc_type = "GGA_PBE"
    _is_gga = True

class PBEsol(_LibxcFunctional):
    _xc_type = "GGA_PBEsol"
    _is_gga = True

class RPBE(_LibxcFunctional):
    _xc_type = "GGA_RPBE"
    _is_gga = True

# --- mGGA ---
class SCAN(_LibxcFunctional):
    _xc_type = "MGGA_SCAN"
    _is_gga = True
    _is_mgga = True

class RSCAN(_LibxcFunctional):
    _xc_type = "MGGA_RSCAN"
    _is_gga = True
    _is_mgga = True

class R2SCAN(_LibxcFunctional):
    _xc_type = "MGGA_R2SCAN"
    _is_gga = True
    _is_mgga = True

# --- Hybrid ---
class PBE0(_LibxcFunctional):
    _xc_type = "HYB_PBE0"
    _is_gga = True
    _is_hybrid = True
    _exx_frac = 0.25

    def __init__(self, alpha=0.25):
        super().__init__()
        self._exx_frac = alpha

    def __repr__(self):
        return f"PBE0(alpha={self._exx_frac})"

class HSE06(_LibxcFunctional):
    _xc_type = "HYB_HSE"
    _is_gga = True
    _is_hybrid = True
    _exx_frac = 0.25

    def __init__(self, alpha=0.25, omega=0.11):
        super().__init__()
        self._exx_frac = alpha
        self.omega = omega

    def __repr__(self):
        return f"HSE06(alpha={self._exx_frac}, omega={self.omega})"


# --- Registry: string name -> class ---
_REGISTRY = {
    "LDA_PZ": LDA_PZ, "LDA": LDA_PZ,
    "LDA_PW": LDA_PW,
    "PBE": PBE, "GGA_PBE": PBE,
    "PBESOL": PBEsol, "GGA_PBESOL": PBEsol,
    "RPBE": RPBE, "GGA_RPBE": RPBE,
    "SCAN": SCAN, "MGGA_SCAN": SCAN,
    "RSCAN": RSCAN, "MGGA_RSCAN": RSCAN,
    "R2SCAN": R2SCAN, "MGGA_R2SCAN": R2SCAN,
    "PBE0": PBE0, "HYB_PBE0": PBE0,
    "HSE06": HSE06, "HSE": HSE06, "HYB_HSE": HSE06,
}

def get(name: str) -> Functional:
    """Look up a functional by name.

    Args:
        name: e.g. "PBE", "SCAN", "HSE06", "LDA_PZ"

    Returns:
        Functional instance
    """
    key = name.upper().replace("-", "_")
    if key not in _REGISTRY:
        available = sorted(set(cls.__name__ for cls in _REGISTRY.values()))
        raise ValueError(f"Unknown XC functional '{name}'. Available: {available}")
    return _REGISTRY[key]()
