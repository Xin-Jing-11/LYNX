"""Occupation functions (smearing)."""
from abc import ABC, abstractmethod
import numpy as np


class Occupation(ABC):
    """Abstract base class for occupation functions."""

    @abstractmethod
    def compute(self, eigenvalues, n_electrons, kpt_weights=None):
        """Compute occupations and Fermi energy.

        Args:
            eigenvalues: list of arrays, one per (spin, kpt) channel
            n_electrons: total number of electrons
            kpt_weights: k-point weights (normalized)

        Returns:
            (occupations_list, fermi_energy)
        """
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class FermiDirac(Occupation):
    """Fermi-Dirac smearing."""

    def __init__(self, temperature=315.77):
        """
        Args:
            temperature: electronic temperature in Kelvin
        """
        self.temperature = temperature

    def compute(self, eigenvalues, n_electrons, kpt_weights=None):
        kBT = self.temperature * 3.1668114e-6  # K -> Ha
        if kBT < 1e-14:
            kBT = 1e-14
        beta = 1.0 / kBT

        # Bisection for Fermi level
        all_eigs = np.concatenate([e.ravel() for e in eigenvalues])
        Ef_lo, Ef_hi = all_eigs.min() - 1.0, all_eigs.max() + 1.0

        nspin = 1  # will be inferred from structure
        spin_fac = 2.0 / nspin if kpt_weights is not None else 2.0

        for _ in range(200):
            Ef = 0.5 * (Ef_lo + Ef_hi)
            ne = 0.0
            for i, eig in enumerate(eigenvalues):
                w = kpt_weights[i % len(kpt_weights)] if kpt_weights is not None else 1.0
                occ = 1.0 / (1.0 + np.exp(beta * (eig - Ef)))
                ne += spin_fac * w * np.sum(occ)
            if ne > n_electrons:
                Ef_hi = Ef
            else:
                Ef_lo = Ef
            if abs(ne - n_electrons) < 1e-12:
                break

        Ef = 0.5 * (Ef_lo + Ef_hi)
        occupations = []
        for eig in eigenvalues:
            occ = 1.0 / (1.0 + np.exp(beta * (eig - Ef)))
            occupations.append(occ)

        return occupations, Ef

    def __repr__(self):
        return f"FermiDirac(T={self.temperature} K)"


class Gaussian(Occupation):
    """Gaussian smearing."""

    def __init__(self, temperature=315.77):
        self.temperature = temperature

    def compute(self, eigenvalues, n_electrons, kpt_weights=None):
        from scipy.special import erfc

        kBT = self.temperature * 3.1668114e-6  # K -> Ha
        if kBT < 1e-14:
            kBT = 1e-14
        sigma = kBT

        all_eigs = np.concatenate([e.ravel() for e in eigenvalues])
        Ef_lo, Ef_hi = all_eigs.min() - 1.0, all_eigs.max() + 1.0

        nspin = 1
        spin_fac = 2.0 / nspin if kpt_weights is not None else 2.0

        for _ in range(200):
            Ef = 0.5 * (Ef_lo + Ef_hi)
            ne = 0.0
            for i, eig in enumerate(eigenvalues):
                w = kpt_weights[i % len(kpt_weights)] if kpt_weights is not None else 1.0
                occ = 0.5 * erfc((eig - Ef) / sigma)
                ne += spin_fac * w * np.sum(occ)
            if ne > n_electrons:
                Ef_hi = Ef
            else:
                Ef_lo = Ef
            if abs(ne - n_electrons) < 1e-12:
                break

        Ef = 0.5 * (Ef_lo + Ef_hi)
        occupations = []
        for eig in eigenvalues:
            occ = 0.5 * erfc((eig - Ef) / sigma)
            occupations.append(occ)

        return occupations, Ef

    def __repr__(self):
        return f"Gaussian(T={self.temperature} K)"
