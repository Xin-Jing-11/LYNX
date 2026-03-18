"""System setup helper — delegates to _core.Calculator, exposes components.

SystemSetup runs the C++ Calculator.setup() to handle the complex initialization
(pseudopotentials, atom influence, electrostatics, operators), then wraps each
component in the appropriate Python ABC implementation.
"""

import numpy as np
from . import _core
from .operators import FDHamiltonian, FDKinetic, FDGradient
from .solvers import (
    CheFSIEigenSolver, AARPoissonSolver, AndersonMixer,
    FermiDiracOccupation, GaussianOccupation,
)
from .xc import LibxcFunctional


class SystemSetup:
    """Set up a DFT system, then expose all components for modular use.

    Delegates the heavy lifting to _core.Calculator.setup(), which handles
    pseudopotential loading, atom influence computation, electrostatics,
    and operator construction. Then wraps everything in Python ABCs.

    Example::

        system = SystemSetup("si2.json")
        ops = system.default_operators()
        driver = SCFDriver(**ops, **system.scf_params())
        result = driver.run(system.random_wavefunctions(), system.atomic_density())
    """

    def __init__(self, config):
        """
        Args:
            config: One of:
                - str/path: JSON config file path
                - DFTConfig: programmatic config (calls _build_system_config())
                - _core.SystemConfig: raw C++ config
        """
        from .config import DFTConfig

        _core._ensure_mpi()
        self._calc = _core.Calculator()
        if isinstance(config, DFTConfig):
            self._calc.set_config(config.system_config)
        elif isinstance(config, _core.SystemConfig):
            self._calc.set_config(config)
        else:
            self._calc.load_config(str(config))

        self._calc.setup()

    # --- Direct access to C++ infrastructure ---

    @property
    def calculator(self):
        """Underlying _core.Calculator (for advanced access)."""
        return self._calc

    @property
    def lattice(self):
        return self._calc.lattice

    @property
    def grid(self):
        return self._calc.grid

    @property
    def stencil(self):
        return self._calc.stencil

    @property
    def domain(self):
        return self._calc.domain

    @property
    def kpoints(self):
        return self._calc.kpoints

    @property
    def halo_exchange(self):
        return self._calc.halo_exchange

    @property
    def crystal(self):
        return self._calc.crystal

    @property
    def electrostatics(self):
        return self._calc.electrostatics

    @property
    def Nelectron(self):
        return self._calc.Nelectron

    @property
    def Nspin(self):
        return self._calc.Nspin

    @property
    def Nd_d(self):
        return self._calc.Nd_d

    @property
    def Vloc(self):
        """Local pseudopotential as numpy array (Nd_d,)."""
        return np.array(self._calc.Vloc)

    @property
    def pseudocharge(self):
        """Pseudocharge density as numpy array (Nd_d,)."""
        return np.array(self._calc.electrostatics.pseudocharge())

    @property
    def Eself(self):
        return self._calc.electrostatics.Eself()

    @property
    def Ec(self):
        return self._calc.electrostatics.Ec()

    @property
    def dV(self):
        return self._calc.grid.dV()

    def default_operators(self, xc_type=None):
        """Return dict of default FD implementations for all ABCs.

        Args:
            xc_type: _core.XCType (e.g. _core.XCType.GGA_PBE).
                     If None, defaults to GGA_PBE.

        Returns:
            Dict with keys: hamiltonian, eigensolver, poisson, xc, mixer,
            occupation, kinetic, gradient.
        """
        calc = self._calc
        Nd_d = calc.Nd_d

        # Hamiltonian (wraps C++ fused operator)
        hamiltonian = FDHamiltonian(calc.hamiltonian_op, calc.halo_exchange, Nd_d)

        # Kinetic (standalone, for diagnostics)
        kinetic = FDKinetic(calc.laplacian_op, calc.halo_exchange, Nd_d)

        # Gradient (for GGA)
        gradient = FDGradient(calc.gradient_op, calc.halo_exchange)

        # Poisson solver (create fresh, set up with C++ infrastructure)
        ps = _core.PoissonSolver()
        ps.setup(calc.laplacian_op, calc.stencil, calc.domain,
                 calc.grid, calc.halo_exchange)
        poisson = AARPoissonSolver(ps)

        # XC functional
        if xc_type is None:
            xc_type = _core.XCType.GGA_PBE
        xc_core = _core.XCFunctional()
        is_gga = xc_type in (_core.XCType.GGA_PBE, _core.XCType.GGA_PBEsol,
                             _core.XCType.GGA_RPBE)
        if is_gga:
            xc_core.setup(xc_type, calc.domain, calc.grid,
                          calc.gradient_op, calc.halo_exchange)
        else:
            xc_core.setup(xc_type, calc.domain, calc.grid)
        xc = LibxcFunctional(xc_core, Nd_d)

        # Eigensolver (create fresh, set up in serial mode)
        wfn = calc.get_wavefunction()
        Nband = wfn.Nband_global()
        es = _core.EigenSolver()
        es.setup_serial(calc.hamiltonian_op, calc.halo_exchange, calc.domain,
                        Nband_global=Nband)
        eigensolver = CheFSIEigenSolver(es)

        # Mixer (pure Python Anderson)
        mixer = AndersonMixer(beta=0.3, history_depth=7)

        # Occupation
        Nspin = calc.Nspin
        occupation = GaussianOccupation(elec_temp=315.77, Nspin=Nspin)

        return {
            'hamiltonian': hamiltonian,
            'eigensolver': eigensolver,
            'poisson': poisson,
            'xc': xc,
            'mixer': mixer,
            'occupation': occupation,
            'kinetic': kinetic,
            'gradient': gradient,
        }

    def scf_params(self):
        """Return dict of SCF parameters suitable for SCFDriver.__init__.

        Combined with default_operators(), provides all arguments needed::

            driver = SCFDriver(**system.default_operators(), **system.scf_params())
        """
        kpts = self._calc.kpoints
        kpt_weights = (np.array(kpts.normalized_weights())
                       if not kpts.is_gamma_only()
                       else np.array([1.0]))

        return {
            'Nelectron': self.Nelectron,
            'Vloc': self.Vloc,
            'pseudocharge': self.pseudocharge,
            'Eself': self.Eself,
            'Ec': self.Ec,
            'dV': self.dV,
            'Nspin': self.Nspin,
            'kpt_weights': kpt_weights,
        }

    def random_wavefunctions(self, Nband=None, seed=42):
        """Generate random initial wavefunctions.

        Returns:
            List of (Nd_d, Nband) arrays, one per (spin, kpt) channel.
        """
        wfn = self._calc.get_wavefunction()
        Nspin = wfn.Nspin()
        Nkpts = wfn.Nkpts()

        for s in range(Nspin):
            for k in range(Nkpts):
                wfn.randomize(s, k, seed + s * 1000 + k)

        psi_list = []
        for s in range(Nspin):
            for k in range(Nkpts):
                # Use Fortran order to match C++ column-major layout
                psi_list.append(np.array(wfn.psi(s, k), order='F'))
        return psi_list

    def atomic_density(self):
        """Return atomic superposition density as numpy array (Nd_d,).

        This is the density used as the initial guess for the SCF loop.
        """
        return np.array(self._calc.atomic_density)
