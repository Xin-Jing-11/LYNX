"""Solver components for DFT calculations.

Swappable algorithmic components — like torch.optim.

Usage:
    from lynx import solvers
    calc = lynx.DFT(
        eigensolver=solvers.CheFSI(degree=25),
        mixer=solvers.PulayMixer(beta=0.2, history=10),
    )
"""

from .eigen import EigenSolver, CheFSI, LOBPCG
from .poisson import PoissonSolver, AAR
from .mixer import Mixer, PulayMixer, AndersonMixer, SimpleMixer
from .occupation import Occupation, FermiDirac, Gaussian

__all__ = [
    "EigenSolver", "CheFSI", "LOBPCG",
    "PoissonSolver", "AAR",
    "Mixer", "PulayMixer", "AndersonMixer", "SimpleMixer",
    "Occupation", "FermiDirac", "Gaussian",
]
