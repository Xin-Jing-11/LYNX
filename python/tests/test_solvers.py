"""Tests for lynx.solvers module."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import lynx


class TestMixers:
    def test_simple_mixer(self):
        mixer = lynx.solvers.SimpleMixer(beta=0.5)
        rho_in = np.ones(100)
        rho_out = np.ones(100) * 2.0
        rho_new = mixer.mix(rho_in, rho_out)
        np.testing.assert_allclose(rho_new, 1.5)

    def test_anderson_mixer(self):
        # Simple linear mixing test (first step is always linear)
        mixer = lynx.solvers.AndersonMixer(beta=0.5, history=3)
        rho_in = np.zeros(10)
        rho_out = np.ones(10)
        rho_new = mixer.mix(rho_in, rho_out)
        # First step: rho_new = rho_in + beta * (rho_out - rho_in) = 0.5
        np.testing.assert_allclose(rho_new, 0.5)

    def test_mixer_reset(self):
        mixer = lynx.solvers.AndersonMixer(beta=0.3)
        mixer.mix(np.ones(10), np.zeros(10))
        mixer.reset()
        # Should not raise after reset

    def test_repr(self):
        assert "beta=0.3" in repr(lynx.solvers.PulayMixer(beta=0.3))
        assert "SimpleMixer" in repr(lynx.solvers.SimpleMixer())


class TestSolverConstruction:
    def test_chefsi(self):
        es = lynx.solvers.CheFSI(degree=30)
        assert es.degree == 30
        assert "CheFSI" in repr(es)

    def test_aar(self):
        ps = lynx.solvers.AAR(tol=1e-10)
        assert ps.tol == 1e-10

    def test_fermi_dirac(self):
        fd = lynx.solvers.FermiDirac(temperature=500)
        assert fd.temperature == 500

    def test_gaussian(self):
        g = lynx.solvers.Gaussian(temperature=300)
        assert g.temperature == 300
