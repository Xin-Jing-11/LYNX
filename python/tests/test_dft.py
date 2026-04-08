"""Tests for lynx.DFT class (no actual calculations)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lynx


class TestDFTConstruction:
    def test_default(self):
        calc = lynx.DFT()
        assert calc.device == "cpu"
        assert calc.kpts == (1, 1, 1)
        assert calc.xc.name == "PBE"

    def test_xc_string(self):
        calc = lynx.DFT(xc="SCAN")
        assert calc.xc.name == "SCAN"
        assert calc.xc.is_mgga

    def test_xc_object(self):
        calc = lynx.DFT(xc=lynx.xc.HSE06(alpha=0.3))
        assert calc.xc.is_hybrid
        assert abs(calc.xc.exx_fraction - 0.3) < 1e-10

    def test_xc_swap(self):
        calc = lynx.DFT(xc="PBE")
        calc.xc = "SCAN"
        assert calc.xc.name == "SCAN"

    def test_eigensolver_swap(self):
        calc = lynx.DFT()
        calc.eigensolver = lynx.solvers.CheFSI(degree=30)
        assert calc.eigensolver.degree == 30

    def test_to_cpu(self):
        calc = lynx.DFT()
        ret = calc.to("cpu")
        assert ret is calc  # returns self
        assert calc.device == "cpu"

    def test_repr(self):
        calc = lynx.DFT(xc="PBE", kpts=[2, 2, 2])
        r = repr(calc)
        assert "PBE" in r
        assert "(2, 2, 2)" in r

    def test_invalid_xc_type(self):
        import pytest
        with pytest.raises(TypeError):
            lynx.DFT(xc=42)

    def test_invalid_mixer_type(self):
        import pytest
        with pytest.raises(TypeError):
            lynx.DFT(mixing=42)
