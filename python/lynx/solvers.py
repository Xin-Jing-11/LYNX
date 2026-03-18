"""Concrete solver wrappers and pure-Python mixing/occupation.

CheFSIEigenSolver and AARPoissonSolver wrap _core C++ solvers.
AndersonMixer, FermiDiracOccupation, and GaussianOccupation are
pure Python — fully transparent and customizable.
"""

import numpy as np
from .abc import (
    EigenSolver as EigenSolverABC,
    PoissonSolver as PoissonSolverABC,
    DensityMixer,
    OccupationFunction,
)
from . import _core


class CheFSIEigenSolver(EigenSolverABC):
    """Chebyshev-filtered subspace iteration eigensolver.

    Wraps _core.EigenSolver. The C++ implementation handles the full
    Chebyshev filter + orthogonalize + subspace diag cycle. Python
    only controls *when* to call it.
    """

    def __init__(self, eigensolver: _core.EigenSolver, cheb_degree: int = 20):
        """
        Args:
            eigensolver: Initialized _core.EigenSolver (after setup()).
            cheb_degree: Chebyshev polynomial degree for the filter.
        """
        self._es = eigensolver
        self._cheb_degree = cheb_degree
        self._eigmin = None
        self._eigmax = None

    def solve(self, H, psi, Veff, **kw):
        """Solve using CheFSI.

        The H parameter is unused — the C++ EigenSolver already has the
        Hamiltonian from setup(). It is accepted for ABC compatibility,
        allowing other eigensolvers (e.g. LOBPCG) to use it.

        Extra keyword arguments:
            cheb_degree: Override Chebyshev degree for this call.
            lambda_cutoff: Upper bound for wanted eigenvalues.
            eigval_min, eigval_max: Spectral bounds (skip Lanczos).
            recompute_bounds: Set False to reuse cached bounds (default: True).
        """
        cheb_degree = kw.get('cheb_degree', self._cheb_degree)

        # Recompute bounds by default (Veff changes each SCF iteration)
        if kw.get('recompute_bounds', True) or self._eigmin is None:
            self._eigmin, self._eigmax = self._es.lanczos_bounds(Veff)

        eigmin = kw.get('eigval_min', self._eigmin)
        eigmax = kw.get('eigval_max', self._eigmax)

        lambda_cutoff = kw.get('lambda_cutoff', self._es.lambda_cutoff())

        psi_out, eigvals = self._es.solve(
            psi, Veff, lambda_cutoff, eigmin, eigmax, cheb_degree)

        self._eigmin = eigmin
        self._eigmax = eigmax

        # Update lambda_cutoff: set above highest eigenvalue with buffer
        # so the filter doesn't suppress the highest wanted band
        if len(eigvals) > 0:
            eig_range = eigmax - eigmin if eigmax > eigmin else 1.0
            buffer = max(0.1 * eig_range, 1.0)
            self._es.set_lambda_cutoff(float(eigvals[-1]) + buffer)

        return psi_out, eigvals

    @property
    def cheb_degree(self):
        return self._cheb_degree

    @cheb_degree.setter
    def cheb_degree(self, value):
        self._cheb_degree = value


class AARPoissonSolver(PoissonSolverABC):
    """Anderson Acceleration with Richardson (AAR) Poisson solver.

    Wraps _core.PoissonSolver.
    """

    def __init__(self, poisson_solver: _core.PoissonSolver):
        self._ps = poisson_solver

    def solve(self, rhs, tol=1e-8):
        return self._ps.solve(rhs, tol)


class AndersonMixer(DensityMixer):
    """Pure-Python Anderson/Pulay density mixer.

    Implements Anderson mixing with configurable history depth::

        rho_next = rho_in + beta * R - (dX + beta * dR) @ c

    where R = rho_out - rho_in is the residual and c minimizes ||R - dR @ c||.
    """

    def __init__(self, beta: float = 0.3, history_depth: int = 7):
        """
        Args:
            beta: Mixing parameter (0 < beta <= 1).
            history_depth: Number of previous iterations to retain.
        """
        self.beta = beta
        self.history_depth = history_depth
        self._dX = []
        self._dR = []
        self._X_prev = None
        self._R_prev = None

    def mix(self, rho_in, rho_out):
        R = rho_out - rho_in

        if self._X_prev is not None:
            dX = rho_in - self._X_prev
            dR = R - self._R_prev
            if len(self._dX) >= self.history_depth:
                self._dX.pop(0)
                self._dR.pop(0)
            self._dX.append(dX)
            self._dR.append(dR)

        self._X_prev = rho_in.copy()
        self._R_prev = R.copy()

        if len(self._dR) == 0:
            return rho_in + self.beta * R

        # Least-squares: minimize ||R - dR @ c||^2
        dR_mat = np.column_stack(self._dR)
        A = dR_mat.T @ dR_mat
        b = dR_mat.T @ R

        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return rho_in + self.beta * R

        dX_mat = np.column_stack(self._dX)
        return rho_in + self.beta * R - (dX_mat + self.beta * dR_mat) @ c

    def reset(self):
        self._dX.clear()
        self._dR.clear()
        self._X_prev = None
        self._R_prev = None


class FermiDiracOccupation(OccupationFunction):
    """Fermi-Dirac smearing for occupation numbers.

    Pure Python implementation using bisection for Fermi level.
    """

    # Boltzmann constant in Hartree/Kelvin
    _kB = 3.1668114e-6

    def __init__(self, elec_temp: float = 315.77, Nspin: int = 1):
        """
        Args:
            elec_temp: Electronic temperature in Kelvin.
            Nspin: Number of spin channels (1 = no spin, 2 = collinear).
        """
        self.elec_temp = elec_temp
        self.Nspin = Nspin
        self._beta = 1.0 / (self._kB * elec_temp) if elec_temp > 0 else 1e10

    def compute(self, eigenvalues, Nelectron, kpt_weights=None):
        occfac = 2.0 / self.Nspin
        if kpt_weights is None:
            kpt_weights = np.array([1.0])
        Nkpts = len(kpt_weights)

        def count_electrons(Ef):
            Ne = 0.0
            for idx, eigs in enumerate(eigenvalues):
                wk = kpt_weights[idx % Nkpts]
                occ = self._fd_vec(eigs, Ef)
                Ne += occfac * wk * np.sum(occ)
            return Ne

        all_eigs = np.concatenate(eigenvalues)
        Ef_lo = float(all_eigs.min()) - 1.0
        Ef_hi = float(all_eigs.max()) + 1.0

        for _ in range(200):
            Ef_mid = 0.5 * (Ef_lo + Ef_hi)
            Ne = count_electrons(Ef_mid)
            if abs(Ne - Nelectron) < 1e-12:
                break
            if Ne < Nelectron:
                Ef_lo = Ef_mid
            else:
                Ef_hi = Ef_mid

        Ef = 0.5 * (Ef_lo + Ef_hi)

        occupations = []
        for idx, eigs in enumerate(eigenvalues):
            occ = occfac * self._fd_vec(eigs, Ef)
            occupations.append(occ)

        return occupations, Ef

    def _fd_vec(self, eigs, Ef):
        """Vectorized Fermi-Dirac: f = 1 / (1 + exp((e - Ef) * beta))."""
        x = (eigs - Ef) * self._beta
        with np.errstate(over='ignore'):
            return np.where(x > 50, 0.0, np.where(x < -50, 1.0,
                                                    1.0 / (1.0 + np.exp(x))))


class GaussianOccupation(OccupationFunction):
    """Gaussian smearing for occupation numbers.

    Pure Python implementation using bisection for Fermi level.
    """

    _kB = 3.1668114e-6

    def __init__(self, elec_temp: float = 315.77, Nspin: int = 1):
        """
        Args:
            elec_temp: Electronic temperature in Kelvin.
            Nspin: Number of spin channels (1 = no spin, 2 = collinear).
        """
        self.elec_temp = elec_temp
        self.Nspin = Nspin
        self._beta = 1.0 / (self._kB * elec_temp) if elec_temp > 0 else 1e10

    def compute(self, eigenvalues, Nelectron, kpt_weights=None):
        from scipy.special import erfc

        occfac = 2.0 / self.Nspin
        if kpt_weights is None:
            kpt_weights = np.array([1.0])
        Nkpts = len(kpt_weights)

        def gauss_occ(eigs, Ef):
            return 0.5 * erfc(self._beta * (eigs - Ef))

        def count_electrons(Ef):
            Ne = 0.0
            for idx, eigs in enumerate(eigenvalues):
                wk = kpt_weights[idx % Nkpts]
                Ne += occfac * wk * np.sum(gauss_occ(eigs, Ef))
            return Ne

        all_eigs = np.concatenate(eigenvalues)
        Ef_lo = float(all_eigs.min()) - 1.0
        Ef_hi = float(all_eigs.max()) + 1.0

        for _ in range(200):
            Ef_mid = 0.5 * (Ef_lo + Ef_hi)
            Ne = count_electrons(Ef_mid)
            if abs(Ne - Nelectron) < 1e-12:
                break
            if Ne < Nelectron:
                Ef_lo = Ef_mid
            else:
                Ef_hi = Ef_mid

        Ef = 0.5 * (Ef_lo + Ef_hi)

        occupations = []
        for idx, eigs in enumerate(eigenvalues):
            occ = occfac * gauss_occ(eigs, Ef)
            occupations.append(occ)

        return occupations, Ef
