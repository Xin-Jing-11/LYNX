"""
LYNX — Real-space DFT simulator with Python bindings.

Three granularity levels:
  Level 1 (High-level):  lynx.Calculator("input.json")
                          lynx.DFTConfig(...).create_calculator()
  Level 2 (Mid-level):   calc.setup(); access internal operators
  Level 3 (Low-level):   lynx.Laplacian, lynx.Gradient, etc. on numpy arrays

Units:
  LYNX internal: Bohr (length), Hartree (energy)
  ASE interface: Angstrom (length), eV (energy) — converted automatically
"""

__version__ = "0.1.0"

from ._core import (
    # Enums
    CellType,
    BCType,
    SpinType,
    MixingVariable,
    MixingPrecond,
    PoissonSolverType,
    SmearingType,
    XCType,
    # Core types
    Vec3,
    Mat3,
    # Geometry
    Lattice,
    FDGrid,
    FDStencil,
    Domain,
    DomainVertices,
    KPoints,
    # Parallel
    HaloExchange,
    ParallelParams,
    # Operators
    Laplacian,
    Gradient,
    Hamiltonian,
    NonlocalProjector,
    # Solvers
    AARParams,
    PoissonSolver,
    EigenSolver,
    Mixer,
    # Electronic
    Wavefunction,
    ElectronDensity,
    Occupation,
    # XC
    XCFunctional,
    # Physics
    SCFParams,
    SCF,
    EnergyComponents,
    Energy,
    Electrostatics,
    Forces,
    Stress,
    # Atoms
    Pseudopotential,
    AtomType,
    AtomInfluence,
    AtomNlocInfluence,
    Crystal,
    # Config
    SystemConfig,
    AtomTypeInput,
    # High-level
    Calculator,
    # Helpers
    make_lattice,
    full_domain,
)

from ._core import _ensure_mpi
from .config import DFTConfig, find_psp
from . import units

# Modular DFT framework
from . import abc
from .operators import FDHamiltonian, FDKinetic, FDNonlocal, FDGradient
from .solvers import (
    CheFSIEigenSolver, AARPoissonSolver, AndersonMixer,
    FermiDiracOccupation, GaussianOccupation,
)
from .xc import LibxcFunctional
from .scf import SCFDriver, SCFResult
from .system import SystemSetup
from . import postprocessing


def init():
    """Initialize MPI if not already done. Call before any LYNX operations."""
    _ensure_mpi()
