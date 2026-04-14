---
title: C++ Quickstart
parent: Examples
nav_order: 1
---

# C++ Quickstart

Run LYNX from the command line using a JSON input file.

## Build

```bash
git clone --recurse-submodules https://github.com/Xin-Jing-11/LYNX.git
cd LYNX
mkdir build && cd build
cmake ..
make -j
```

For GPU support:

```bash
cmake -DUSE_CUDA=ON ..
make -j
```

## Run

```bash
# Single process
./build/src/lynx examples/Si8.json

# With MPI (4 ranks)
mpirun -np 4 ./build/src/lynx examples/Si8.json
```

## Input format

The file `examples/Si8.json` from the repository:

```json
{
  "lattice": {
    "vectors": [[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
    "cell_type": "orthogonal"
  },
  "grid": {
    "Nx": 40, "Ny": 40, "Nz": 40,
    "fd_order": 12,
    "boundary_conditions": ["periodic", "periodic", "periodic"]
  },
  "atoms": [
    {
      "element": "Si",
      "pseudo_file": "psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8",
      "fractional": true,
      "coordinates": [
        [0.00, 0.00, 0.00],
        [0.25, 0.25, 0.25],
        [0.50, 0.50, 0.00],
        [0.75, 0.75, 0.25],
        [0.50, 0.00, 0.50],
        [0.75, 0.25, 0.75],
        [0.00, 0.50, 0.50],
        [0.25, 0.75, 0.75]
      ]
    }
  ],
  "electronic": {
    "xc": "GGA_PBE",
    "spin": "none",
    "temperature": 300,
    "smearing": "gaussian",
    "Nstates": 20
  },
  "kpoints": {
    "grid": [2, 2, 2],
    "shift": [0.5, 0.5, 0.5]
  },
  "scf": {
    "max_iter": 100,
    "tolerance": 1e-6,
    "mixing": "density",
    "preconditioner": "kerker",
    "mixing_history": 7,
    "mixing_parameter": 0.3
  },
  "output": {
    "print_forces": true,
    "print_atoms": true
  }
}
```

All lengths are in **Bohr**. Temperature is in **Kelvin**.

## Key fields

| Field | Description |
|-------|-------------|
| `grid.fd_order` | Finite-difference order (6–24) |
| `electronic.xc` | XC functional (`GGA_PBE`, `LDA_PZ`, `SCAN`, …) |
| `electronic.Nstates` | Number of Kohn-Sham states |
| `scf.tolerance` | SCF convergence threshold (Ha/atom) |
| `scf.mixing_history` | Pulay history depth |

**What's next:** [Architecture overview]({{ site.baseurl }}/architecture/) explains how the JSON fields map to internal C++ objects.
