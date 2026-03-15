"""Test full SCF calculation via Calculator."""

import os
import json
import pytest
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, 'examples')
PSPS_DIR = os.path.join(PROJECT_ROOT, 'psps')


def _find_psp(element):
    """Find pseudopotential file for element in psps/ directory."""
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


def _make_si2_config():
    """Create a minimal Si2 config dict with correct psp paths."""
    psp = _find_psp('Si')
    if psp is None:
        return None
    return {
        "lattice": {
            "vectors": [[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
            "cell_type": "orthogonal"
        },
        "grid": {
            "Nx": 20, "Ny": 20, "Nz": 20,
            "fd_order": 12,
            "boundary_conditions": ["periodic", "periodic", "periodic"]
        },
        "atoms": [{
            "element": "Si",
            "pseudo_file": psp,
            "fractional": True,
            "coordinates": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
        }],
        "electronic": {
            "xc": "GGA_PBE",
            "spin": "none",
            "temperature": 300,
            "smearing": "gaussian",
            "Nstates": 10
        },
        "scf": {
            "max_iter": 50,
            "tolerance": 1e-4,
            "mixing": "density",
            "preconditioner": "kerker",
            "mixing_history": 7,
            "mixing_parameter": 0.3
        }
    }


@pytest.mark.skipif(_find_psp('Si') is None,
                    reason="No Si pseudopotential found in psps/")
def test_calculator_scf():
    """Run full SCF calculation on minimal Si2 system."""
    import lynx
    import tempfile
    lynx.init()

    config = _make_si2_config()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        json_path = f.name

    try:
        calc = lynx.Calculator(json_path, auto_run=True)

        assert calc.is_setup
        assert calc.converged
        assert calc.total_energy < 0

        rho = calc.density
        assert isinstance(rho, np.ndarray)
        assert rho.shape[0] == calc.Nd_d

        E = calc.energy
        assert 'Etotal' in E
        assert 'Eband' in E
        assert 'Exc' in E
    finally:
        os.unlink(json_path)


@pytest.mark.skipif(_find_psp('Si') is None,
                    reason="No Si pseudopotential found in psps/")
def test_calculator_mid_level():
    """Test mid-level access: setup without running."""
    import lynx
    import tempfile
    lynx.init()

    config = _make_si2_config()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        json_path = f.name

    try:
        calc = lynx.Calculator(json_path, auto_run=False)

        assert calc.is_setup
        assert not calc.converged
        assert calc.grid.Nd() > 0
        assert calc.domain.Nd_d() > 0
        assert calc.Nelectron > 0
        assert calc.Natom == 2
    finally:
        os.unlink(json_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
