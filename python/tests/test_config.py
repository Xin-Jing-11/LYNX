"""Test DFTConfig programmatic interface and ASE integration."""

import os
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PSPS_DIR = os.path.join(PROJECT_ROOT, 'psps')


def _find_psp(element):
    for xc_dir in ['ONCVPSP-PBE-PDv0.4', 'ONCVPSP-LDA-PDv0.4']:
        elem_dir = os.path.join(PSPS_DIR, xc_dir, element)
        if os.path.isdir(elem_dir):
            for f in sorted(os.listdir(elem_dir)):
                if f.endswith('.psp8'):
                    return os.path.join(elem_dir, f)
    # Fallback: flat directory search
    if os.path.isdir(PSPS_DIR):
        for f in os.listdir(PSPS_DIR):
            if f.endswith('.psp8') and f'_{element}_' in f:
                return os.path.join(PSPS_DIR, f)
    return None


@pytest.mark.skipif(_find_psp('Si') is None,
                    reason="No Si pseudopotential found")
def test_dft_config_fractional():
    """Create calculator from DFTConfig with fractional coordinates."""
    import lynx
    from lynx.config import DFTConfig
    lynx.init()

    config = DFTConfig(
        cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
        fractional=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        symbols=['Si', 'Si'],
        pseudo_files={'Si': _find_psp('Si')},
        Nx=20, Ny=20, Nz=20,
        Nstates=10,
    )

    calc = config.create_calculator(auto_run=True)
    assert calc.converged
    assert calc.total_energy < 0
    assert calc.Natom == 2
    assert calc.Nelectron == 8


@pytest.mark.skipif(_find_psp('Si') is None,
                    reason="No Si pseudopotential found")
def test_dft_config_cartesian():
    """Create calculator from DFTConfig with Cartesian positions (Bohr)."""
    import lynx
    from lynx.config import DFTConfig
    lynx.init()

    a = 10.26
    config = DFTConfig(
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        positions=[[0, 0, 0], [a*0.25, a*0.25, a*0.25]],
        symbols=['Si', 'Si'],
        pseudo_files={'Si': _find_psp('Si')},
        Nx=20, Ny=20, Nz=20,
        Nstates=10,
    )

    calc = config.create_calculator(auto_run=True)
    assert calc.converged
    assert calc.Natom == 2


@pytest.mark.skipif(_find_psp('Si') is None,
                    reason="No Si pseudopotential found")
def test_dft_config_auto_nstates():
    """Nstates=0 should auto-compute as Nelectron/2 + 10."""
    import lynx
    from lynx.config import DFTConfig
    lynx.init()

    config = DFTConfig(
        cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
        fractional=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        symbols=['Si', 'Si'],
        pseudo_files={'Si': _find_psp('Si')},
        Nx=20, Ny=20, Nz=20,
        Nstates=0,  # auto
        max_scf_iter=5,  # just test setup, don't need convergence
    )

    calc = config.create_calculator(auto_run=False)
    assert calc.is_setup
    assert calc.Nelectron == 8


@pytest.mark.skipif(_find_psp('Si') is None,
                    reason="No Si pseudopotential found")
def test_from_ase():
    """Create DFTConfig from ASE Atoms object."""
    pytest.importorskip('ase')
    from ase import Atoms
    import lynx
    from lynx.config import DFTConfig
    from lynx.units import ANG_TO_BOHR

    lynx.init()

    # ASE Atoms in Angstrom
    a_ang = 5.43
    atoms = Atoms(
        symbols=['Si', 'Si'],
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=[a_ang, a_ang, a_ang],
        pbc=True,
    )

    config = DFTConfig.from_ase(
        atoms,
        pseudo_files={'Si': _find_psp('Si')},
        mesh_spacing=0.5,
        Nstates=10,
        max_scf_iter=50,
        scf_tol=1e-4,
    )

    calc = config.create_calculator(auto_run=True)
    assert calc.converged
    assert calc.Natom == 2
    assert calc.Nelectron == 8
    assert calc.total_energy < 0


@pytest.mark.skipif(_find_psp('Si') is None,
                    reason="No Si pseudopotential found")
def test_ase_calculator():
    """Test the full ASE Calculator interface (LynxCalculator)."""
    pytest.importorskip('ase')
    from ase import Atoms
    from lynx.ase_interface import LynxCalculator
    from lynx.units import HA_TO_EV

    # ASE Atoms in Angstrom
    a_ang = 5.43
    atoms = Atoms(
        symbols=['Si', 'Si'],
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=[a_ang, a_ang, a_ang],
        pbc=True,
    )

    calc = LynxCalculator(
        pseudo_files={'Si': _find_psp('Si')},
        mesh_spacing=0.5,
        Nstates=10,
        max_scf_iter=50,
        scf_tol=1e-4,
    )
    atoms.calc = calc

    # Energy should be in eV
    energy = atoms.get_potential_energy()
    assert energy < 0
    # Si2 energy is ~ -8.18 Ha ~ -222.6 eV
    assert -250 < energy < -200

    # Forces should be (2, 3) in eV/Angstrom
    forces = atoms.get_forces()
    assert forces.shape == (2, 3)


def test_units_module():
    """Verify unit conversion constants."""
    from lynx.units import BOHR_TO_ANG, ANG_TO_BOHR, HA_TO_EV, EV_TO_HA

    assert abs(BOHR_TO_ANG - 0.5291772) < 1e-5
    assert abs(ANG_TO_BOHR * BOHR_TO_ANG - 1.0) < 1e-14
    assert abs(HA_TO_EV - 27.21139) < 1e-3
    assert abs(EV_TO_HA * HA_TO_EV - 1.0) < 1e-14


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
