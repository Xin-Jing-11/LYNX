"""Tests for lynx.Atoms class."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestAtomsConstruction:
    def test_basic_bohr(self):
        import lynx
        atoms = lynx.Atoms(
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
            positions=[[0, 0, 0], [5, 5, 5]],
            symbols=["Si", "Si"],
            units="bohr",
        )
        assert atoms.n_atoms == 2
        assert atoms.formula == "Si2"
        assert atoms.cell.shape == (3, 3)
        np.testing.assert_allclose(atoms.cell[0, 0], 10.0)
        np.testing.assert_allclose(atoms.positions[1], [5, 5, 5])

    def test_basic_angstrom(self):
        import lynx
        from lynx.units import ANG_TO_BOHR
        atoms = lynx.Atoms(
            cell=[[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
            positions=[[0, 0, 0], [1.3575, 1.3575, 1.3575]],
            symbols=["Si", "Si"],
            units="angstrom",
        )
        # Internal storage is Bohr
        np.testing.assert_allclose(atoms.cell[0, 0], 5.43 * ANG_TO_BOHR, rtol=1e-10)

    def test_fractional(self):
        import lynx
        atoms = lynx.Atoms(
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
            fractional=[[0, 0, 0], [0.5, 0.5, 0.5]],
            symbols=["Si", "Si"],
            units="bohr",
        )
        np.testing.assert_allclose(atoms.positions[1], [5, 5, 5])
        np.testing.assert_allclose(atoms.fractional[1], [0.5, 0.5, 0.5])

    def test_orthorhombic_shorthand(self):
        import lynx
        atoms = lynx.Atoms(
            cell=[10, 10, 10],
            positions=[[0, 0, 0]],
            symbols=["H"],
            units="bohr",
        )
        assert atoms.cell.shape == (3, 3)
        np.testing.assert_allclose(atoms.cell[0, 0], 10.0)
        np.testing.assert_allclose(atoms.cell[0, 1], 0.0)

    def test_volume(self):
        import lynx
        atoms = lynx.Atoms(
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
            positions=[[0, 0, 0]],
            symbols=["H"],
            units="bohr",
        )
        np.testing.assert_allclose(atoms.volume, 1000.0)

    def test_formula(self):
        import lynx
        atoms = lynx.Atoms(
            cell=[10, 10, 10],
            positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            symbols=["Fe", "O", "O"],
            units="bohr",
        )
        assert atoms.formula == "FeO2"

    def test_unique_elements(self):
        import lynx
        atoms = lynx.Atoms(
            cell=[10, 10, 10],
            positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            symbols=["O", "Fe", "O"],
            units="bohr",
        )
        assert atoms.unique_elements == ["Fe", "O"]

    def test_len(self):
        import lynx
        atoms = lynx.Atoms(
            cell=[10, 10, 10],
            positions=[[0, 0, 0], [1, 0, 0]],
            symbols=["Si", "Si"],
            units="bohr",
        )
        assert len(atoms) == 2

    def test_repr(self):
        import lynx
        atoms = lynx.Atoms(
            cell=[10, 10, 10],
            positions=[[0, 0, 0]],
            symbols=["H"],
            units="bohr",
        )
        r = repr(atoms)
        assert "H" in r
        assert "n_atoms=1" in r

    def test_spin(self):
        import lynx
        atoms = lynx.Atoms(
            cell=[10, 10, 10],
            positions=[[0, 0, 0], [5, 5, 5]],
            symbols=["Fe", "Fe"],
            spin=[2.0, -2.0],
            units="bohr",
        )
        np.testing.assert_allclose(atoms.spin, [2.0, -2.0])


class TestAtomsManipulation:
    def test_repeat(self):
        import lynx
        atoms = lynx.Atoms(
            cell=[5, 5, 5],
            positions=[[0, 0, 0]],
            symbols=["H"],
            units="bohr",
        )
        big = atoms.repeat(2)
        assert big.n_atoms == 8
        np.testing.assert_allclose(big.cell[0, 0], 10.0)

    def test_deform(self):
        import lynx
        atoms = lynx.Atoms(
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
            positions=[[0, 0, 0]],
            symbols=["H"],
            units="bohr",
        )
        strain = np.eye(3) * 0.05  # 5% expansion (F = I + strain)
        deformed = atoms.deform(strain)
        np.testing.assert_allclose(deformed.cell[0, 0], 10.5)

    def test_immutability(self):
        import lynx
        atoms = lynx.Atoms(
            cell=[10, 10, 10],
            positions=[[0, 0, 0]],
            symbols=["H"],
            units="bohr",
        )
        big = atoms.repeat(2)
        # Original unchanged
        assert atoms.n_atoms == 1
        assert big.n_atoms == 8
