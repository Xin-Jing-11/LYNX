"""GPU acceleration — like .to("cuda") in PyTorch.

Requires LYNX built with -DUSE_CUDA=ON.
"""
import lynx

if not lynx.cuda_available():
    print("GPU not available — LYNX was built without CUDA.")
    print("Rebuild with: cmake .. -DUSE_CUDA=ON")
    raise SystemExit(0)

atoms = lynx.Atoms(
    cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
    fractional=[[0, 0, 0], [0.25, 0.25, 0.25]],
    symbols=["Si", "Si"],
    units="bohr",
    psp_dir="psps",
)

# Method 1: device in constructor
calc = lynx.DFT(xc="LDA_PZ", grid_shape=[25, 25, 25], device="gpu")

# Method 2: .to() chaining (PyTorch-style)
calc = lynx.DFT(xc="LDA_PZ", grid_shape=[25, 25, 25]).to("gpu")

result = calc(atoms)
print(f"GPU energy: {result.energy:.8f} Ha")
print(f"Converged: {result.converged}")
