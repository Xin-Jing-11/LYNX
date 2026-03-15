"""
ASE Calculator adapter for LYNX.

Usage:
    from lynx.ase_interface import LynxCalculator
    from ase.build import bulk

    atoms = bulk('Si', 'diamond', a=10.26)
    calc = LynxCalculator(xc='GGA_PBE', kpts=(4,4,4), fd_order=12, mesh_spacing=0.5)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
"""

import json
import os
import tempfile
import numpy as np

try:
    from ase.calculators.calculator import Calculator as ASECalculator, all_changes
except ImportError:
    raise ImportError("ASE is required for lynx.ase_interface. Install with: pip install ase")

from . import Calculator as LynxCalculator_Core

# Conversion factors
BOHR_TO_ANG = 0.529177249
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
HA_TO_EV = 27.211386245988
HA_BOHR_TO_EV_ANG = HA_TO_EV / BOHR_TO_ANG

# Pseudopotential search paths
_PSP_SEARCH_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'psps'),
    os.environ.get('LYNX_PSP_PATH', ''),
]


def _find_psp(element):
    """Find pseudopotential file for an element."""
    for path in _PSP_SEARCH_PATHS:
        if not path or not os.path.isdir(path):
            continue
        for fname in os.listdir(path):
            if fname.endswith('.psp8') and f'_{element}_' in fname:
                return os.path.join(path, fname)
    raise FileNotFoundError(
        f"No pseudopotential found for {element}. "
        f"Set LYNX_PSP_PATH environment variable or place .psp8 files in the psps/ directory."
    )


class LynxCalculator(ASECalculator):
    """ASE Calculator interface for LYNX DFT.

    Parameters
    ----------
    xc : str
        Exchange-correlation functional ('GGA_PBE', 'LDA_PW', etc.)
    kpts : tuple of 3 ints
        Monkhorst-Pack k-point grid
    fd_order : int
        Finite-difference order (default: 12)
    mesh_spacing : float
        Grid spacing in Bohr (default: 0.5). Used to determine Nx, Ny, Nz.
    max_scf_iter : int
        Maximum SCF iterations (default: 100)
    scf_tol : float
        SCF convergence tolerance (default: 1e-6)
    mixing_param : float
        Density mixing parameter (default: 0.3)
    smearing : float
        Electronic temperature in eV (default: 0.1)
    psp_dir : str
        Directory containing pseudopotential files
    """

    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters = {
        'xc': 'GGA_PBE',
        'kpts': (1, 1, 1),
        'fd_order': 12,
        'mesh_spacing': 0.5,
        'max_scf_iter': 100,
        'scf_tol': 1e-6,
        'mixing_param': 0.3,
        'smearing': 0.1,
        'psp_dir': None,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # Build LYNX JSON config from ASE Atoms
        config = self._atoms_to_config(self.atoms)

        # Write temporary JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2)
            json_path = f.name

        try:
            calc = LynxCalculator_Core(json_path, auto_run=True)

            # Energy in eV
            self.results['energy'] = calc.total_energy * HA_TO_EV

            # Forces in eV/Angstrom
            if 'forces' in properties:
                forces = calc.compute_forces()
                self.results['forces'] = np.array(forces) * HA_BOHR_TO_EV_ANG

            # Stress in eV/Angstrom^3 (Voigt notation)
            if 'stress' in properties:
                stress_voigt = calc.compute_stress()
                # LYNX: [xx, xy, xz, yy, yz, zz]
                # ASE:  [xx, yy, zz, yz, xz, xy]
                stress_ase = np.array([
                    stress_voigt[0], stress_voigt[3], stress_voigt[5],
                    stress_voigt[4], stress_voigt[2], stress_voigt[1]
                ]) * (HA_TO_EV / BOHR_TO_ANG**3)
                self.results['stress'] = stress_ase

        finally:
            os.unlink(json_path)

    def _atoms_to_config(self, atoms):
        """Convert ASE Atoms to LYNX JSON config dict."""
        params = self.parameters
        cell = atoms.cell.array * ANG_TO_BOHR  # Convert to Bohr

        # Determine grid size from mesh_spacing
        h = params['mesh_spacing']
        lengths = np.linalg.norm(cell, axis=1)
        Nx = max(1, int(np.ceil(lengths[0] / h)))
        Ny = max(1, int(np.ceil(lengths[1] / h)))
        Nz = max(1, int(np.ceil(lengths[2] / h)))

        # Build atom_types: group by chemical symbol
        symbols = atoms.get_chemical_symbols()
        unique_symbols = list(dict.fromkeys(symbols))  # Preserve order

        psp_dir = params.get('psp_dir')
        atom_types = []
        for sym in unique_symbols:
            indices = [i for i, s in enumerate(symbols) if s == sym]
            frac_coords = atoms.get_scaled_positions()[indices].tolist()

            if psp_dir:
                psp_file = None
                for f in os.listdir(psp_dir):
                    if f.endswith('.psp8') and f'_{sym}_' in f:
                        psp_file = os.path.join(psp_dir, f)
                        break
                if not psp_file:
                    raise FileNotFoundError(f"No .psp8 file for {sym} in {psp_dir}")
            else:
                psp_file = _find_psp(sym)

            atom_types.append({
                'element': sym,
                'pseudo_file': psp_file,
                'coords': frac_coords,
                'fractional': True,
            })

        # Determine cell type
        cell_type = 'Orthogonal'
        off_diag = np.array([cell[0, 1], cell[0, 2], cell[1, 0], cell[1, 2], cell[2, 0], cell[2, 1]])
        if np.any(np.abs(off_diag) > 1e-10):
            cell_type = 'NonOrthogonal'

        kpts = params['kpts']

        config = {
            'latvec': cell.tolist(),
            'cell_type': cell_type,
            'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
            'fd_order': params['fd_order'],
            'bcx': 'Periodic', 'bcy': 'Periodic', 'bcz': 'Periodic',
            'atom_types': atom_types,
            'xc': params['xc'],
            'Kx': kpts[0], 'Ky': kpts[1], 'Kz': kpts[2],
            'max_scf_iter': params['max_scf_iter'],
            'scf_tol': params['scf_tol'],
            'mixing_param': params['mixing_param'],
            'elec_temp': params['smearing'] / HA_TO_EV,  # Convert eV -> Ha
            'print_forces': 'forces' in self.implemented_properties,
            'calc_stress': 'stress' in self.implemented_properties,
        }

        return config
