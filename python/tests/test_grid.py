"""Tests for lynx.Grid class."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import lynx


class TestGrid:
    def test_shape(self):
        g = lynx.Grid([10, 10, 10], shape=[20, 20, 20])
        assert g.shape == (20, 20, 20)
        assert g.ndof == 8000
        assert g.Nx == 20

    def test_spacing(self):
        g = lynx.Grid([10, 10, 10], spacing=0.5)
        assert g.Nx == 20
        assert g.Ny == 20
        assert g.Nz == 20

    def test_dV(self):
        g = lynx.Grid([10, 10, 10], shape=[10, 10, 10])
        np.testing.assert_allclose(g.dV, 1.0)

    def test_non_ortho_cell(self):
        cell = [[10, 1, 0], [0, 10, 0], [0, 0, 10]]
        g = lynx.Grid(cell, shape=[20, 20, 20])
        assert g.ndof == 8000

    def test_mutual_exclusion(self):
        import pytest
        with pytest.raises(ValueError):
            lynx.Grid([10, 10, 10], shape=[20, 20, 20], spacing=0.5)

    def test_must_specify_one(self):
        import pytest
        with pytest.raises(ValueError):
            lynx.Grid([10, 10, 10])

    def test_repr(self):
        g = lynx.Grid([10, 10, 10], shape=[20, 20, 20])
        r = repr(g)
        assert "(20, 20, 20)" in r
        assert "Bohr" in r
