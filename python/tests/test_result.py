"""Tests for lynx.DFTResult."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import lynx
from lynx.result import DFTResult, EnergyDecomposition
from lynx.units import HA_TO_EV, BOHR_TO_ANG


class TestDFTResult:
    def test_energy_conversion(self):
        r = DFTResult(energy=-1.0)
        np.testing.assert_allclose(r.energy_eV, -HA_TO_EV)

    def test_forces_conversion(self):
        r = DFTResult(forces=np.ones((2, 3)))
        f_eV = r.forces_eV_A
        expected = HA_TO_EV / BOHR_TO_ANG
        np.testing.assert_allclose(f_eV, expected)

    def test_stress_conversion(self):
        r = DFTResult(stress=np.ones(6))
        s_GPa = r.stress_GPa
        assert s_GPa is not None
        assert s_GPa.shape == (6,)

    def test_none_forces(self):
        r = DFTResult()
        assert r.forces_eV_A is None
        assert r.stress_GPa is None

    def test_repr(self):
        r = DFTResult(energy=-7.86, converged=True, n_iterations=15)
        assert "converged" in repr(r)

    def test_summary(self):
        r = DFTResult(
            energy=-7.86,
            converged=True,
            n_iterations=15,
            forces=np.zeros((2, 3)),
            stress=np.zeros(6),
        )
        s = r.summary()
        assert "LYNX" in s
        assert "-7.86" in s


class TestEnergyDecomposition:
    def test_repr(self):
        e = EnergyDecomposition(total=-7.86, band=-3.0, xc=-2.0)
        r = repr(e)
        assert "total" in r
        assert "band" in r
