"""LYNX — Real-space DFT with PyTorch-style Python interface.

Quick start:
    import lynx

    atoms = lynx.Atoms(
        cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
        positions=[[0, 0, 0], [2.565, 2.565, 2.565]],
        symbols=["Si", "Si"],
        units="bohr",
    )

    result = lynx.calculate(atoms, xc="PBE")
    print(result.energy)
"""

__version__ = "0.2.0"

# Core classes
from lynx.atoms import Atoms
from lynx.dft import DFT
from lynx.result import DFTResult, EnergyDecomposition
from lynx.grid import Grid
from lynx.device import device, cuda_available

# Sub-packages
from lynx import xc
from lynx import solvers
from lynx import ops
from lynx import io

# Units
from lynx import units

# Serialization (top-level convenience)
from lynx.io import save, load


def calculate(atoms, **kwargs):
    """One-liner DFT calculation.

    Args:
        atoms: lynx.Atoms instance
        **kwargs: passed to lynx.DFT constructor

    Returns:
        DFTResult

    Example:
        result = lynx.calculate(atoms, xc="PBE", kpts=[2,2,2])
    """
    return DFT(**kwargs)(atoms)


def init():
    """Initialize MPI (if needed). Called automatically."""
    try:
        from mpi4py import MPI
        return
    except ImportError:
        pass
    # Fallback: initialize MPI via ctypes if mpi4py is not available
    try:
        import ctypes
        import ctypes.util
        libmpi = ctypes.util.find_library("mpi")
        if libmpi:
            lib = ctypes.CDLL(libmpi)
            initialized = ctypes.c_int(0)
            lib.MPI_Initialized(ctypes.byref(initialized))
            if not initialized.value:
                lib.MPI_Init(None, None)
    except Exception:
        pass


# Auto-init
init()


__all__ = [
    # Core
    "Atoms", "DFT", "DFTResult", "EnergyDecomposition", "Grid",
    # Device
    "device", "cuda_available",
    # I/O
    "save", "load",
    # Convenience
    "calculate",
    # Sub-packages
    "xc", "solvers", "ops", "io", "units",
]
