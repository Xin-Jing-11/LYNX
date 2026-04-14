"""ASE calculator adapter for LYNX.

Usage:
    from ase.build import bulk
    from lynx.ase import LynxCalculator

    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = LynxCalculator(xc="PBE", kpts=[2, 2, 2])

    energy = atoms.get_potential_energy()   # eV
    forces = atoms.get_forces()             # eV/Angstrom
    stress = atoms.get_stress()             # eV/Angstrom^3 (Voigt)
"""

import numpy as np

# Unit conversions
HA_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177249
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
HA_BOHR_TO_EV_ANG = HA_TO_EV / BOHR_TO_ANG      # forces
HA_BOHR3_TO_EV_ANG3 = HA_TO_EV / BOHR_TO_ANG**3  # stress


class LynxCalculator:
    """ASE-compatible calculator wrapping lynx.DFT.

    Handles all unit conversions between ASE (eV, Angstrom) and
    LYNX internal (Hartree, Bohr).
    """

    # ASE calculator interface
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, *, xc="PBE", kpts=(1, 1, 1), kpt_shift=(0, 0, 0),
                 fd_order=12, mesh_spacing=None, grid_shape=None,
                 n_bands=None, max_scf=100, scf_tol=1e-6,
                 mixing_beta=0.3, temperature=315.77,
                 smearing="gaussian", pseudo_files=None, psp_dir=None,
                 device="cpu", verbose=0, **kwargs):
        """
        All parameters are passed to lynx.DFT.
        Units for input: same as DFT (Bohr/Kelvin), except atoms come from ASE in Angstrom.
        """
        self._dft_kwargs = dict(
            xc=xc, kpts=kpts, kpt_shift=kpt_shift,
            fd_order=fd_order, mesh_spacing=mesh_spacing,
            grid_shape=grid_shape, n_bands=n_bands,
            max_scf=max_scf, scf_tol=scf_tol,
            mixing_beta=mixing_beta, temperature=temperature,
            smearing=smearing, device=device, verbose=verbose,
            **kwargs,
        )
        self._pseudo_files = pseudo_files
        self._psp_dir = psp_dir
        self._results = {}
        self._atoms = None
        self._prev_density = None  # converged density for restart

    def calculate(self, atoms=None, properties=None, system_changes=None):
        """Run calculation. Called by ASE."""
        if atoms is not None:
            self._atoms = atoms

        if self._atoms is None:
            raise RuntimeError("No atoms set")

        from lynx.atoms import Atoms as LynxAtoms
        from lynx.dft import DFT

        # Convert ASE Atoms -> LYNX Atoms (Angstrom -> Bohr)
        lynx_atoms = LynxAtoms.from_ase(
            self._atoms,
            pseudo_files=self._pseudo_files,
            psp_dir=self._psp_dir,
        )

        # Run DFT (with density restart if available)
        calc = DFT(**self._dft_kwargs)
        result = calc(lynx_atoms, initial_density=self._prev_density)

        # Save converged density for next step
        if result.density is not None:
            self._prev_density = result.density.copy()

        # Store results in ASE units
        self._results = {
            'energy': result.energy * HA_TO_EV,
            'forces': result.forces * HA_BOHR_TO_EV_ANG if result.forces is not None else None,
        }

        if result.stress is not None:
            # ASE expects stress as (6,) Voigt in eV/Angstrom^3
            # LYNX gives (6,) Voigt in Ha/Bohr^3
            self._results['stress'] = result.stress * HA_BOHR3_TO_EV_ANG3

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Get energy in eV."""
        if atoms is not None or 'energy' not in self._results:
            self.calculate(atoms)
        return self._results['energy']

    def get_forces(self, atoms=None):
        """Get forces in eV/Angstrom."""
        if atoms is not None or 'forces' not in self._results:
            self.calculate(atoms)
        return self._results['forces']

    def get_stress(self, atoms=None):
        """Get stress in eV/Angstrom^3 (Voigt notation)."""
        if atoms is not None or 'stress' not in self._results:
            self.calculate(atoms)
        return self._results['stress']

    def __repr__(self):
        xc = self._dft_kwargs.get('xc', 'PBE')
        kpts = self._dft_kwargs.get('kpts', (1, 1, 1))
        device = self._dft_kwargs.get('device', 'cpu')
        return f"LynxCalculator(xc='{xc}', kpts={kpts}, device='{device}')"


# Try to register as proper ASE calculator if ASE is available
try:
    from ase.calculators.calculator import Calculator as ASECalculator

    _BaseLynxCalculator = LynxCalculator  # capture before replacement

    class _ASELynxCalculator(ASECalculator, _BaseLynxCalculator):
        """Full ASE Calculator subclass (when ASE is installed)."""
        name = 'lynx'
        implemented_properties = ['energy', 'forces', 'stress']

        def __init__(self, **kwargs):
            ASECalculator.__init__(self)
            _BaseLynxCalculator.__init__(self, **kwargs)

        def calculate(self, atoms=None, properties=None, system_changes=None):
            ASECalculator.calculate(self, atoms, properties, system_changes)
            _BaseLynxCalculator.calculate(self, self.atoms, properties, system_changes)
            self.results = self._results

    # Replace the base class with the ASE-aware version
    LynxCalculator = _ASELynxCalculator

except ImportError:
    pass  # ASE not installed — use standalone LynxCalculator
