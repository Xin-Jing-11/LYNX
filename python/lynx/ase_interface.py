"""
ASE Calculator adapter for LYNX.

ASE uses Angstrom/eV units; LYNX uses Bohr/Hartree internally.
This adapter handles all unit conversions automatically.

Usage:
    from ase.build import bulk
    from lynx.ase_interface import LynxCalculator

    atoms = bulk('Si', 'diamond', a=5.43)
    atoms.calc = LynxCalculator(kpts=(4, 4, 4))
    energy = atoms.get_potential_energy()    # eV
    forces = atoms.get_forces()             # eV/Angstrom
    stress = atoms.get_stress()             # eV/Angstrom^3 (Voigt)
"""

import numpy as np

try:
    from ase.calculators.calculator import Calculator as ASECalculator, all_changes
except ImportError:
    raise ImportError(
        "ASE is required for lynx.ase_interface. Install with: pip install ase"
    )

from .config import DFTConfig
from .units import HA_TO_EV, HA_BOHR_TO_EV_ANG, HA_BOHR3_TO_EV_ANG3


class LynxCalculator(ASECalculator):
    """ASE Calculator interface for LYNX DFT.

    All user-facing values are in ASE units (Angstrom, eV).
    Conversion to LYNX internal units (Bohr, Hartree) is automatic.

    Parameters
    ----------
    xc : str
        Exchange-correlation functional (default: 'GGA_PBE').
        Options: 'GGA_PBE', 'LDA_PW', 'LDA_PZ', 'GGA_PBEsol', 'GGA_RPBE'.
    kpts : tuple of 3 ints
        Monkhorst-Pack k-point grid (default: (1,1,1)).
    kpt_shift : tuple of 3 floats
        K-point shift (default: (0,0,0)).
    fd_order : int
        Finite-difference order (default: 12).
    mesh_spacing : float
        Grid spacing in Bohr (default: 0.5).
    Nstates : int
        Number of bands. 0 = auto (Nelectron/2 + 10).
    max_scf_iter : int
        Maximum SCF iterations (default: 100).
    scf_tol : float
        SCF convergence tolerance (default: 1e-6).
    mixing_param : float
        Density mixing parameter (default: 0.3).
    elec_temp : float
        Electronic temperature in Kelvin (default: 300).
    smearing : str
        Smearing type: 'gaussian' or 'fermi-dirac' (default: 'gaussian').
    pseudo_files : dict, optional
        Map element -> pseudopotential file path.
    psp_dir : str, optional
        Directory to search for .psp8 files.
    """

    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters = {
        'xc': 'GGA_PBE',
        'kpts': (1, 1, 1),
        'kpt_shift': (0, 0, 0),
        'fd_order': 12,
        'mesh_spacing': 0.5,
        'Nstates': 0,
        'max_scf_iter': 100,
        'scf_tol': 1e-6,
        'mixing_param': 0.3,
        'elec_temp': 300.0,
        'smearing': 'gaussian',
        'pseudo_files': None,
        'psp_dir': None,
        'use_gpu': False,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        p = self.parameters

        # Build DFTConfig directly from ASE Atoms (Angstrom -> Bohr automatic)
        dft_config = DFTConfig.from_ase(
            self.atoms,
            pseudo_files=p['pseudo_files'],
            psp_dir=p['psp_dir'],
            xc=p['xc'],
            kpts=p['kpts'],
            kpt_shift=p['kpt_shift'],
            fd_order=p['fd_order'],
            mesh_spacing=p['mesh_spacing'],
            Nstates=p['Nstates'],
            elec_temp=p['elec_temp'],
            smearing=p['smearing'],
            max_scf_iter=p['max_scf_iter'],
            scf_tol=p['scf_tol'],
            mixing_param=p['mixing_param'],
        )

        calc = dft_config.create_calculator(auto_run=True, use_gpu=p['use_gpu'])

        # Energy: Hartree -> eV
        self.results['energy'] = calc.total_energy * HA_TO_EV

        # Forces: Ha/Bohr -> eV/Angstrom
        if 'forces' in properties:
            forces_au = calc.compute_forces()
            self.results['forces'] = np.array(forces_au) * HA_BOHR_TO_EV_ANG

        # Stress: Ha/Bohr^3 -> eV/Angstrom^3
        # LYNX Voigt: [xx, xy, xz, yy, yz, zz]
        # ASE Voigt:  [xx, yy, zz, yz, xz, xy]
        if 'stress' in properties:
            s = np.array(calc.compute_stress())
            stress_ase = np.array([
                s[0], s[3], s[5],  # xx, yy, zz
                s[4], s[2], s[1],  # yz, xz, xy
            ]) * HA_BOHR3_TO_EV_ANG3
            self.results['stress'] = stress_ase
