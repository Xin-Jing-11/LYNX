"""Tests for lynx.xc module."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lynx


class TestXCRegistry:
    def test_get_pbe(self):
        xc = lynx.xc.get("PBE")
        assert xc.name == "PBE"
        assert xc.is_gga
        assert not xc.is_mgga
        assert not xc.is_hybrid

    def test_get_scan(self):
        xc = lynx.xc.get("SCAN")
        assert xc.is_mgga
        assert not xc.is_hybrid

    def test_get_hse06(self):
        xc = lynx.xc.get("HSE06")
        assert xc.is_hybrid
        assert abs(xc.exx_fraction - 0.25) < 1e-10

    def test_get_pbe0_custom_alpha(self):
        xc = lynx.xc.PBE0(alpha=0.5)
        assert abs(xc.exx_fraction - 0.5) < 1e-10

    def test_get_lda(self):
        xc = lynx.xc.get("LDA_PZ")
        assert not xc.is_gga
        assert not xc.is_mgga

    def test_unknown_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown XC"):
            lynx.xc.get("INVALID_XC")

    def test_case_insensitive(self):
        xc = lynx.xc.get("pbe")
        assert xc.name == "PBE"

    def test_aliases(self):
        assert lynx.xc.get("GGA_PBE").name == "PBE"
        assert lynx.xc.get("HYB_HSE").name == "HSE06"
        assert lynx.xc.get("LDA").name == "LDA_PZ"

    def test_repr(self):
        assert "PBE" in repr(lynx.xc.PBE())
        assert "alpha=0.25" in repr(lynx.xc.HSE06())
