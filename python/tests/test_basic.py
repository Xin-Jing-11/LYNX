"""Basic import and type tests for pylynx."""

import numpy as np
import pytest


def test_import():
    import lynx
    assert hasattr(lynx, '__version__')
    assert lynx.__version__ == '0.1.0'


def test_enums():
    import lynx
    assert lynx.CellType.Orthogonal.value == 0
    assert lynx.CellType.NonOrthogonal.value == 1
    assert lynx.BCType.Periodic.value == 0
    assert lynx.BCType.Dirichlet.value == 1
    assert lynx.SpinType.NoSpin.value == 0
    assert lynx.XCType.GGA_PBE is not None
    assert lynx.SmearingType.FermiDirac is not None
    assert lynx.MixingVariable.Density.value == 0
    assert lynx.MixingPrecond.Kerker.value == 1


def test_vec3():
    import lynx
    v = lynx.Vec3(1.0, 2.0, 3.0)
    assert v.x == 1.0
    assert v.y == 2.0
    assert v.z == 3.0
    assert abs(v.norm() - np.sqrt(14.0)) < 1e-12
    arr = v.to_numpy()
    np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])


def test_mat3():
    import lynx
    m = lynx.Mat3()
    m.set(0, 0, 10.0)
    m.set(1, 1, 10.0)
    m.set(2, 2, 10.0)
    assert m(0, 0) == 10.0
    assert abs(m.determinant() - 1000.0) < 1e-10
    arr = m.to_numpy()
    assert arr.shape == (3, 3)
    np.testing.assert_allclose(np.diag(arr), [10.0, 10.0, 10.0])

    # Round-trip
    m2 = lynx.Mat3.from_numpy(arr)
    assert abs(m2.determinant() - 1000.0) < 1e-10


def test_lattice():
    import lynx
    lv = np.diag([10.0, 10.0, 10.0])
    lattice = lynx.make_lattice(lv)
    assert lattice.is_orthogonal()
    lens = lattice.lengths()
    assert abs(lens.x - 10.0) < 1e-12
    assert abs(lens.y - 10.0) < 1e-12
    assert abs(lens.z - 10.0) < 1e-12


def test_grid():
    import lynx
    lv = np.diag([10.0, 10.0, 10.0])
    lattice = lynx.make_lattice(lv)
    grid = lynx.FDGrid(24, 24, 24, lattice)
    assert grid.Nx() == 24
    assert grid.Ny() == 24
    assert grid.Nz() == 24
    assert grid.Nd() == 24 * 24 * 24
    assert grid.dV() > 0


def test_domain():
    import lynx
    lv = np.diag([10.0, 10.0, 10.0])
    lattice = lynx.make_lattice(lv)
    grid = lynx.FDGrid(24, 24, 24, lattice)
    domain = lynx.full_domain(grid)
    assert domain.Nd_d() == 24 * 24 * 24
    assert domain.Nx_d() == 24


def test_kpoints():
    import lynx
    lv = np.diag([10.0, 10.0, 10.0])
    lattice = lynx.make_lattice(lv)
    kpts = lynx.KPoints()
    kpts.generate(2, 2, 2, lynx.Vec3(0, 0, 0), lattice)
    assert kpts.Nkpts_full() == 8
    assert kpts.Nkpts() >= 1
    assert not kpts.is_gamma_only()


def test_scf_params():
    import lynx
    p = lynx.SCFParams()
    p.max_iter = 200
    p.tol = 1e-8
    p.mixing_param = 0.5
    assert p.max_iter == 200
    assert p.tol == 1e-8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
