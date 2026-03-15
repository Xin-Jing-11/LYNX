"""
LYNX — Real-space DFT simulator with Python bindings.

Three granularity levels:
  Level 1 (High-level):  lynx.Calculator("input.json")
  Level 2 (Mid-level):   calc.setup(); access internal operators
  Level 3 (Low-level):   lynx.Laplacian, lynx.Gradient, etc. on numpy arrays
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
    # High-level
    Calculator,
    # Helpers
    make_lattice,
    full_domain,
)

from ._core import _ensure_mpi


def init():
    """Initialize MPI if not already done. Call before any LYNX operations."""
    _ensure_mpi()
