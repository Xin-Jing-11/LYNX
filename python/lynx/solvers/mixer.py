"""Density/potential mixing for SCF convergence."""
from abc import ABC, abstractmethod
import numpy as np


class Mixer(ABC):
    """Abstract base class for density mixers."""

    @abstractmethod
    def mix(self, rho_in, rho_out):
        """Mix input and output densities.

        Args:
            rho_in: input density from this SCF step
            rho_out: output density from solving KS equations

        Returns:
            Mixed density for next iteration
        """
        ...

    @abstractmethod
    def reset(self):
        """Reset mixing history."""
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class PulayMixer(Mixer):
    """Pulay/DIIS mixing with optional Kerker preconditioner (default).

    Wraps the C++ Mixer.
    """

    def __init__(self, beta=0.3, history=7, precondition="kerker"):
        self.beta = beta
        self.history = history
        self.precondition = precondition
        self._core_mixer = None  # set during DFT setup

    def mix(self, rho_in, rho_out):
        if self._core_mixer is None:
            raise RuntimeError("PulayMixer not initialized. Use via lynx.DFT().")
        raise NotImplementedError("Standalone mix not yet implemented")

    def reset(self):
        pass  # C++ mixer handles reset internally

    def __repr__(self):
        return f"PulayMixer(beta={self.beta}, history={self.history})"


class AndersonMixer(Mixer):
    """Pure Python Anderson mixing.

    Simple implementation for testing and educational use.
    """

    def __init__(self, beta=0.3, history=7):
        self.beta = beta
        self.history = history
        self._X = []  # input history
        self._F = []  # residual history

    def mix(self, rho_in, rho_out):
        f = rho_out - rho_in  # residual

        if len(self._X) == 0:
            # Simple linear mixing for first step
            rho_new = rho_in + self.beta * f
        else:
            # Anderson mixing with history
            m = len(self._X)
            dF = np.array([self._F[i] - f for i in range(m)])

            # Solve least-squares: min ||f - dF^T * gamma||
            if m == 1:
                dFf = np.dot(dF[0].ravel(), f.ravel())
                dFdF = np.dot(dF[0].ravel(), dF[0].ravel())
                gamma = np.array([dFf / max(dFdF, 1e-30)])
            else:
                A = np.zeros((m, m))
                b = np.zeros(m)
                for i in range(m):
                    b[i] = np.dot(dF[i].ravel(), f.ravel())
                    for j in range(m):
                        A[i, j] = np.dot(dF[i].ravel(), dF[j].ravel())
                try:
                    gamma = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    gamma = np.linalg.lstsq(A, b, rcond=None)[0]

            rho_new = rho_in + self.beta * f
            for i in range(m):
                dx = self._X[i] - rho_in
                df = self._F[i] - f
                rho_new += gamma[i] * (dx + self.beta * df)

        # Store history
        self._X.append(rho_in.copy())
        self._F.append(f.copy())
        if len(self._X) > self.history:
            self._X.pop(0)
            self._F.pop(0)

        return rho_new

    def reset(self):
        self._X.clear()
        self._F.clear()

    def __repr__(self):
        return f"AndersonMixer(beta={self.beta}, history={self.history})"


class SimpleMixer(Mixer):
    """Simple linear mixing: rho_new = (1-beta)*rho_in + beta*rho_out."""

    def __init__(self, beta=0.3):
        self.beta = beta

    def mix(self, rho_in, rho_out):
        return (1.0 - self.beta) * rho_in + self.beta * rho_out

    def reset(self):
        pass

    def __repr__(self):
        return f"SimpleMixer(beta={self.beta})"
