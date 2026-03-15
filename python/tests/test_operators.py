"""Test operator bindings on numpy arrays."""

import numpy as np
import pytest


def test_laplacian_on_numpy():
    """Apply Laplacian to a known function and verify."""
    import lynx

    lynx.init()

    N = 24
    L = 10.0
    lv = np.diag([L, L, L])
    lattice = lynx.make_lattice(lv)
    grid = lynx.FDGrid(N, N, N, lattice)
    stencil = lynx.FDStencil(12, grid, lattice)
    domain = lynx.full_domain(grid)
    halo = lynx.HaloExchange(domain, stencil.FDn())
    lap = lynx.Laplacian(stencil, domain)

    # Create a smooth function: f(x,y,z) = sin(2*pi*x/L) * sin(2*pi*y/L) * sin(2*pi*z/L)
    # Laplacian: -3*(2*pi/L)^2 * f
    dx, dy, dz = grid.dx(), grid.dy(), grid.dz()
    Nd = domain.Nd_d()

    x_arr = np.zeros(Nd)
    lap_exact = np.zeros(Nd)
    k = 2.0 * np.pi / L

    idx = 0
    for kk in range(N):
        for jj in range(N):
            for ii in range(N):
                xi = ii * dx
                yi = jj * dy
                zi = kk * dz
                f = np.sin(k * xi) * np.sin(k * yi) * np.sin(k * zi)
                x_arr[idx] = f
                lap_exact[idx] = -3.0 * k * k * f
                idx += 1

    # Apply Laplacian: y = 1.0 * Lap(x) + 0*x
    y = lap.apply(halo, x_arr, a=1.0, c=0.0)

    # Check relative error (FD approximation won't be exact, but should be small for smooth functions)
    mask = np.abs(lap_exact) > 1e-10
    if np.any(mask):
        rel_err = np.linalg.norm(y[mask] - lap_exact[mask]) / np.linalg.norm(lap_exact[mask])
        assert rel_err < 0.01, f"Laplacian relative error {rel_err:.4e} too large"


def test_gradient_on_numpy():
    """Apply Gradient and verify against known derivative."""
    import lynx

    lynx.init()

    N = 24
    L = 10.0
    lv = np.diag([L, L, L])
    lattice = lynx.make_lattice(lv)
    grid = lynx.FDGrid(N, N, N, lattice)
    stencil = lynx.FDStencil(12, grid, lattice)
    domain = lynx.full_domain(grid)
    halo = lynx.HaloExchange(domain, stencil.FDn())
    grad = lynx.Gradient(stencil, domain)

    dx = grid.dx()
    k = 2.0 * np.pi / L
    Nd = domain.Nd_d()

    # f(x,y,z) = sin(k*x), df/dx = k*cos(k*x)
    x_arr = np.zeros(Nd)
    grad_x_exact = np.zeros(Nd)

    idx = 0
    for kk in range(N):
        for jj in range(N):
            for ii in range(N):
                xi = ii * dx
                x_arr[idx] = np.sin(k * xi)
                grad_x_exact[idx] = k * np.cos(k * xi)
                idx += 1

    y = grad.apply(halo, x_arr, 0)  # direction=0 is x

    mask = np.abs(grad_x_exact) > 1e-10
    if np.any(mask):
        rel_err = np.linalg.norm(y[mask] - grad_x_exact[mask]) / np.linalg.norm(grad_x_exact[mask])
        assert rel_err < 0.01, f"Gradient relative error {rel_err:.4e} too large"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
