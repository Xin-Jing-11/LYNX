"""
Example 5: Use LYNX operators directly on numpy arrays (Level 3).

This is the lowest-level interface — create operators from scratch and apply
them to numpy arrays. All values in Bohr/Hartree.
"""
import numpy as np
import lynx

lynx.init()

# ---- Setup geometry ----
L = 10.0  # Box size in Bohr
N = 48    # Grid points per dimension

lattice = lynx.make_lattice(np.diag([L, L, L]))
grid = lynx.FDGrid(N, N, N, lattice)
stencil = lynx.FDStencil(12, grid, lattice)
domain = lynx.full_domain(grid)
halo = lynx.HaloExchange(domain, stencil.FDn())

print(f"Grid: {N}x{N}x{N} = {grid.Nd()} points")
print(f"Grid spacing: {grid.dx():.4f} Bohr")
print(f"FD order: {stencil.order()}, FDn: {stencil.FDn()}")
print(f"Halo extended: {halo.nx_ex()}x{halo.ny_ex()}x{halo.nz_ex()}")

# ---- Create a smooth test function ----
# f(x,y,z) = sin(2*pi*x/L) * sin(2*pi*y/L) * sin(2*pi*z/L)
# Laplacian(f) = -3 * (2*pi/L)^2 * f
dx, dy, dz = grid.dx(), grid.dy(), grid.dz()
Nd = domain.Nd_d()
k = 2.0 * np.pi / L

f = np.zeros(Nd)
idx = 0
for kk in range(N):
    for jj in range(N):
        for ii in range(N):
            f[idx] = np.sin(k * ii * dx) * np.sin(k * jj * dy) * np.sin(k * kk * dz)
            idx += 1

# ---- Apply Laplacian ----
lap = lynx.Laplacian(stencil, domain)
lap_f = lap.apply(halo, f, a=1.0, c=0.0)  # y = 1.0 * Lap(f) + 0

# Compare with analytical result
lap_exact = -3.0 * k**2 * f
mask = np.abs(lap_exact) > 1e-8
rel_err = np.linalg.norm(lap_f[mask] - lap_exact[mask]) / np.linalg.norm(lap_exact[mask])
print(f"\nLaplacian relative error: {rel_err:.2e}")

# ---- Apply Gradient ----
grad = lynx.Gradient(stencil, domain)
df_dx = grad.apply(halo, f, 0)  # direction 0 = x

# Analytical: df/dx = k * cos(kx) * sin(ky) * sin(kz)
df_dx_exact = np.zeros(Nd)
idx = 0
for kk in range(N):
    for jj in range(N):
        for ii in range(N):
            df_dx_exact[idx] = k * np.cos(k * ii * dx) * np.sin(k * jj * dy) * np.sin(k * kk * dz)
            idx += 1

mask = np.abs(df_dx_exact) > 1e-8
rel_err = np.linalg.norm(df_dx[mask] - df_dx_exact[mask]) / np.linalg.norm(df_dx_exact[mask])
print(f"Gradient(x) relative error: {rel_err:.2e}")

# ---- Apply -0.5 * Laplacian (kinetic energy operator) ----
T_f = lap.apply(halo, f, a=-0.5, c=0.0)  # y = -0.5 * Lap(f)
ke_exact = 0.5 * 3.0 * k**2 * f  # -(-0.5) * 3k^2 * f = 1.5 * k^2 * f
mask = np.abs(ke_exact) > 1e-8
rel_err = np.linalg.norm(T_f[mask] - ke_exact[mask]) / np.linalg.norm(ke_exact[mask])
print(f"Kinetic operator relative error: {rel_err:.2e}")
