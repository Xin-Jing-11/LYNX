# LYNX Web UI ↔ C++ Data Interface

## Overview

The C++ LYNX code dumps results to an output directory.
The Python web server reads these files and serves them to the browser.

## Output Directory Structure

After a simulation run, the C++ code writes:

```
<output_dir>/
├── lynx_results.json      # All metadata + scalar results
├── electron_density.bin    # 3D density grid (raw float64)
├── electrostatic_pot.bin   # 3D electrostatic potential (raw float64, optional)
└── scf_progress.jsonl      # One JSON object per line, updated each SCF iteration
```

## 1. lynx_results.json

Written once at the end of the calculation.

```json
{
  "system": {
    "lattice_vectors": [[ax,ay,az],[bx,by,bz],[cx,cy,cz]],
    "cell_type": "orthogonal",
    "grid": {"Nx": 40, "Ny": 40, "Nz": 40},
    "fd_order": 12,
    "boundary_conditions": ["periodic","periodic","periodic"],
    "atoms": [
      {
        "element": "Si",
        "Z": 14,
        "position_frac": [0.0, 0.0, 0.0],
        "position_cart": [0.0, 0.0, 0.0]
      }
    ],
    "n_atoms": 8,
    "n_electrons": 32
  },

  "scf": {
    "converged": true,
    "n_iterations": 25,
    "final_residual": 3.2e-7,
    "history": [
      {"iter": 1, "energy": -31.5, "residual": 1.2e-2, "fermi": -0.15}
    ]
  },

  "energy": {
    "total": -31.512,
    "band": -5.23,
    "xc": -9.87,
    "hartree": 42.1,
    "self": -78.3,
    "correction": 1.23,
    "entropy": -0.001,
    "per_atom": -3.939
  },

  "fermi_energy": -0.152,

  "forces": [
    {"atom": 0, "element": "Si", "fx": 0.001, "fy": -0.002, "fz": 0.0005}
  ],

  "stress": {
    "voigt": [-0.001, 0.0, 0.0, -0.001, 0.0, -0.001],
    "pressure_GPa": 0.5,
    "components": {
      "kinetic":       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "xc":            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "electrostatic": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "nonlocal":      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
  },

  "eigenvalues": {
    "spin_0_kpt_0": {
      "kpoint": [0.0, 0.0, 0.0],
      "weight": 0.125,
      "eigenvalues": [-0.5, -0.3, -0.1, 0.2],
      "occupations": [2.0, 2.0, 2.0, 0.0]
    }
  },

  "density_file": "electron_density.bin",
  "density_format": {
    "dtype": "float64",
    "shape": [40, 40, 40],
    "order": "column_major",
    "units": "electrons/Bohr^3"
  }
}
```

## 2. electron_density.bin

Raw binary dump of the electron density on the real-space grid.

**Format:**
- No header — dimensions come from `lynx_results.json`
- `Nx * Ny * Nz` contiguous `float64` (8 bytes each)
- Column-major order (Fortran/BLAS convention): fastest index = x
- i.e., `rho[ix + Nx * (iy + Ny * iz)]`

**C++ write example:**
```cpp
void dump_density(const NDArray<double>& rho, int Nx, int Ny, int Nz,
                  const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<const char*>(rho.data()),
              Nx * Ny * Nz * sizeof(double));
}
```

**Python read example:**
```python
import numpy as np
shape = (Nx, Ny, Nz)
rho = np.fromfile("electron_density.bin", dtype=np.float64)
rho = rho.reshape(shape, order='F')  # column-major → 3D array
```

## 3. scf_progress.jsonl

JSON Lines format — one JSON object per line, appended each SCF iteration.
The web server can tail this file for live progress.

```jsonl
{"iter":1,"energy":-28.3,"residual":5.2e-2,"fermi":-0.10,"elapsed_sec":1.2}
{"iter":2,"energy":-30.1,"residual":8.1e-3,"fermi":-0.13,"elapsed_sec":2.5}
{"iter":3,"energy":-31.0,"residual":1.5e-3,"fermi":-0.15,"elapsed_sec":3.8}
```

## 4. How the Web Server Uses This

1. User submits a simulation config (JSON) via the web UI
2. Server writes config to `<job_dir>/input.json`
3. Server launches: `mpirun -np N ./lynx input.json --output-dir <job_dir>`
4. Server polls `scf_progress.jsonl` for live SCF updates → pushes to browser via WebSocket
5. On completion, server reads `lynx_results.json` and `electron_density.bin`
6. Browser receives density as a downsampled Float32Array for 3D rendering

## 5. Density Downsampling for Web

For large grids (e.g., 100×100×100 = 8MB float64), the server downsamples
to a manageable size (e.g., 50×50×50) before sending to the browser.
The isosurface quality is still good at 50³ resolution.

The server sends density as a binary ArrayBuffer with a JSON header:
```json
{"shape": [50, 50, 50], "min": 0.0, "max": 0.15, "origin": [0,0,0],
 "cell": [[10.26,0,0],[0,10.26,0],[0,0,10.26]]}
```
Followed by `50*50*50 * 4` bytes of Float32 data (C-order / row-major for JS).
