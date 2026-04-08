"""Low-level operators on numpy arrays — like torch.nn.functional.

Apply Laplacian, Gradient, etc. directly to numpy arrays.
"""
import numpy as np
import lynx

# Create a grid
grid = lynx.Grid([10, 10, 10], shape=[25, 25, 25])
print(f"Grid: {grid}")
print(f"ndof: {grid.ndof}")
print(f"dV: {grid.dV:.6f}")

# Create operators
lap = lynx.ops.Laplacian(grid)
grad = lynx.ops.Gradient(grid)

print(f"Laplacian: {lap}")
print(f"Gradient: {grad}")

# Apply to a random field
f = np.random.randn(grid.ndof)

# Laplacian: y = nabla^2 f
lap_f = lap(f)
print(f"\n||f|| = {np.linalg.norm(f):.4f}")
print(f"||Lap(f)|| = {np.linalg.norm(lap_f):.4f}")

# Also works with @ syntax
lap_f2 = lap @ f
print(f"||lap @ f|| = {np.linalg.norm(lap_f2):.4f}")

# Gradient in each direction
for d in range(3):
    grad_f = grad(f, direction=d)
    print(f"||grad_{d}(f)|| = {np.linalg.norm(grad_f):.4f}")
