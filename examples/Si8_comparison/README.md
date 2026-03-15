# Si8 Diamond — Three Ways

This example runs the **exact same DFT calculation** (8-atom silicon diamond cell)
using three different interfaces. All three produce identical results.

## System

| Parameter       | Value                      |
|-----------------|----------------------------|
| Structure       | Si diamond, 8 atoms        |
| Lattice         | 10.26 Bohr (5.431 Ang)     |
| Grid            | 24 x 24 x 24              |
| FD order        | 12                         |
| XC functional   | GGA-PBE                    |
| Bands           | 24                         |
| K-points        | Gamma only                 |
| SCF tolerance   | 1e-6                       |

## Running

### 1. C++ executable

```bash
./run_cpp.sh
```

Reads `Si8.json` and runs the LYNX binary. All I/O in Bohr / Hartree.

### 2. Python script (no JSON, no ASE)

```bash
python run_python.py
```

Configures everything programmatically via `DFTConfig`. Units: Bohr / Hartree.

### 3. ASE Calculator

```bash
python run_ase.py
```

Builds the structure with ASE `Atoms` (Angstrom), attaches `LynxCalculator`,
and calls `atoms.get_potential_energy()` / `atoms.get_forces()`.
All ASE-facing values are in eV / Angstrom — conversion is automatic.

## Expected output

All three report the same total energy to full precision:

```
Etotal = -33.3685427384 Ha  (= -907.72 eV)
Eatom  =  -4.1710678423 Ha/atom
```

Forces are near zero by symmetry (~1e-7 Ha/Bohr).

## Units reference

| Interface          | Length   | Energy  | Force       |
|--------------------|----------|---------|-------------|
| C++ / JSON         | Bohr     | Hartree | Ha/Bohr     |
| Python (`DFTConfig`) | Bohr   | Hartree | Ha/Bohr     |
| ASE (`LynxCalculator`) | Angstrom | eV   | eV/Angstrom |

Conversion: 1 Bohr = 0.5292 Ang, 1 Ha = 27.211 eV
