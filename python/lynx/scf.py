"""Algorithm-agnostic SCF driver.

Composes abstract operator/solver interfaces (from abc.py) to perform the
self-consistent field loop. All heavy computation delegates to the ABC
implementations, which may call into C++.

Supports spin-polarized and k-point calculations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class SCFResult:
    """Result of an SCF calculation."""
    converged: bool
    total_energy: float
    energy_components: dict
    fermi_energy: float
    density: np.ndarray
    wavefunctions: list
    eigenvalues: list
    occupations: list
    n_iterations: int
    history: list = field(default_factory=list)


class SCFDriver:
    """Algorithm-agnostic self-consistent field driver.

    Composes abstract operator/solver interfaces to run the DFT SCF loop.
    The ABCs operate on single-(spin,kpt) channels; SCFDriver manages the
    outer loops and k-point weight integration.

    Example::

        driver = SCFDriver(
            hamiltonian=fd_hamiltonian,
            eigensolver=chefsi,
            poisson=aar_poisson,
            xc=libxc_pbe,
            mixer=anderson,
            occupation=fermi_dirac,
            Nelectron=8, Vloc=Vloc, pseudocharge=b,
            Eself=Eself, Ec=Ec, dV=dV,
        )
        result = driver.run(psi_init, rho_init)
    """

    def __init__(self, hamiltonian, eigensolver, poisson, xc, mixer, occupation,
                 Nelectron, Vloc, pseudocharge, Eself, Ec, dV,
                 Nspin=1, kpt_weights=None,
                 max_iter=100, tol=1e-6, rho_trigger=4, nchefsi=1,
                 on_iteration=None):
        """
        Args:
            hamiltonian: HamiltonianOperator (or list per kpt).
            eigensolver: EigenSolver ABC.
            poisson: PoissonSolver ABC.
            xc: XCFunctional ABC.
            mixer: DensityMixer ABC.
            occupation: OccupationFunction ABC.
            Nelectron: Total number of electrons.
            Vloc: Local pseudopotential, shape (Nd_d,).
            pseudocharge: Pseudocharge density, shape (Nd_d,).
            Eself: Self-energy correction (scalar).
            Ec: Correction energy (scalar).
            dV: Volume element (grid spacing product).
            Nspin: Number of spin channels (1 or 2).
            kpt_weights: K-point weights array, None for gamma-only.
            max_iter: Maximum SCF iterations.
            tol: Convergence tolerance on relative density change.
            rho_trigger: Number of eigensolver passes before first density
                         (warm up wavefunctions from random initial guess).
            nchefsi: Number of eigensolver passes per subsequent iteration.
            on_iteration: Optional callback(iter, error, energy).
        """
        self.hamiltonian = hamiltonian
        self.eigensolver = eigensolver
        self.poisson = poisson
        self.xc = xc
        self.mixer = mixer
        self.occupation = occupation
        self.Nelectron = Nelectron
        self.Vloc = np.asarray(Vloc)
        self.pseudocharge = np.asarray(pseudocharge)
        self.Eself = Eself
        self.Ec = Ec
        self.dV = dV
        self.Nspin = Nspin
        self.kpt_weights = (kpt_weights if kpt_weights is not None
                            else np.array([1.0]))
        self.Nkpts = len(self.kpt_weights)
        self.max_iter = max_iter
        self.tol = tol
        self.rho_trigger = rho_trigger
        self.nchefsi = nchefsi
        self.on_iteration = on_iteration

    def run(self, psi_init, rho_init):
        """Run the SCF loop.

        Args:
            psi_init: List of initial wavefunctions, one (Nd_d, Nband) array
                      per (spin, kpt) channel. Ordered [s0k0, s0k1, ..., s1k0, ...].
            rho_init: Initial density, shape (Nd_d,) for Nspin=1 or
                      (Nspin, Nd_d) for spin-polarized.

        Returns:
            SCFResult with converged energies, density, and wavefunctions.
        """
        Nd_d = self.Vloc.shape[0]
        four_pi = 4.0 * np.pi
        Nspin = self.Nspin
        Nkpts = self.Nkpts
        n_channels = Nspin * Nkpts

        # Normalize density to (Nspin, Nd_d)
        rho = np.asarray(rho_init, dtype=np.float64)
        if rho.ndim == 1:
            rho = rho.reshape(1, -1).copy()
        else:
            rho = rho.reshape(Nspin, -1).copy()

        # Working copies
        psi = [p.copy() for p in psi_init]
        eigenvalues = [np.zeros(p.shape[1]) for p in psi]
        occupations = [np.zeros(p.shape[1]) for p in psi]

        history = []
        converged = False
        Etotal = 0.0
        Ef = 0.0

        self.mixer.reset()

        for scf_iter in range(1, self.max_iter + 1):
            rho_total = np.sum(rho, axis=0) if Nspin > 1 else rho[0]

            # --- 1. Poisson solve ---
            rhs = four_pi * (rho_total + self.pseudocharge)
            phi, _ = self.poisson.solve(rhs)

            # --- 2. XC evaluation per spin channel ---
            Vxc = np.zeros_like(rho)
            exc = np.zeros_like(rho)
            for s in range(Nspin):
                Vxc[s], exc[s] = self.xc.evaluate(
                    rho[s] if Nspin > 1 else rho_total)

            # --- 3. Effective potential per spin ---
            Veff = np.zeros_like(rho)
            for s in range(Nspin):
                Veff[s] = phi + Vxc[s] + self.Vloc

            # --- 4. Eigensolver for each (spin, kpt) channel ---
            n_passes = self.rho_trigger if scf_iter == 1 else self.nchefsi
            for _pass in range(n_passes):
                chan = 0
                for s in range(Nspin):
                    for k in range(Nkpts):
                        H = (self.hamiltonian if not isinstance(self.hamiltonian, list)
                             else self.hamiltonian[k])
                        psi[chan], eigenvalues[chan] = self.eigensolver.solve(
                            H, psi[chan], Veff[s],
                            recompute_bounds=(_pass == 0))
                        chan += 1

            # --- 5. Occupations ---
            occupations, Ef = self.occupation.compute(
                eigenvalues, self.Nelectron, self.kpt_weights)

            # --- 6. New density from wavefunctions ---
            rho_out = np.zeros_like(rho)
            chan = 0
            for s in range(Nspin):
                for k in range(Nkpts):
                    wk = self.kpt_weights[k]
                    occ = occupations[chan]
                    psi_sk = psi[chan]
                    # rho_s += w_k * sum_n occ_n * |psi_n|^2
                    rho_out[s] += wk * np.einsum('n,in->i', occ, psi_sk ** 2)
                    chan += 1

            # --- 7. Energy (using input density, matching C++ convention) ---
            Eband = 0.0
            for chan_idx in range(n_channels):
                wk = self.kpt_weights[chan_idx % Nkpts]
                Eband += wk * np.dot(occupations[chan_idx], eigenvalues[chan_idx])

            Exc_energy = 0.0
            Vxc_rho = 0.0
            for s in range(Nspin):
                rho_s = rho[s] if Nspin > 1 else rho_total
                Exc_energy += np.sum(exc[s] * rho_s) * self.dV
                Vxc_rho += np.sum(Vxc[s] * rho_s) * self.dV

            Ehart = 0.5 * np.sum(phi * rho_total) * self.dV
            Etotal = Eband + Exc_energy - Vxc_rho - Ehart + self.Eself + self.Ec

            # --- 8. Convergence check ---
            rho_total_out = np.sum(rho_out, axis=0) if Nspin > 1 else rho_out[0]
            norm_out = np.linalg.norm(rho_total_out)
            error = (np.linalg.norm(rho_total_out - rho_total)
                     / max(norm_out, 1e-30))

            history.append({
                'iter': scf_iter, 'error': error, 'energy': Etotal})

            if self.on_iteration:
                self.on_iteration(scf_iter, error, Etotal)

            if error < self.tol and scf_iter >= 2:
                converged = True
                rho = rho_out
                break

            # --- 9. Mix density ---
            rho_mixed = self.mixer.mix(rho.ravel(), rho_out.ravel())
            rho = np.maximum(rho_mixed.reshape(rho.shape), 0.0)

        # --- Build result ---
        rho_final = rho[0] if Nspin == 1 else rho

        energy_components = {
            'Eband': Eband,
            'Exc': Exc_energy,
            'Ehart': Ehart,
            'Eself': self.Eself,
            'Ec': self.Ec,
            'Etotal': Etotal,
        }

        return SCFResult(
            converged=converged,
            total_energy=Etotal,
            energy_components=energy_components,
            fermi_energy=Ef,
            density=rho_final,
            wavefunctions=psi,
            eigenvalues=eigenvalues,
            occupations=occupations,
            n_iterations=len(history),
            history=history,
        )
