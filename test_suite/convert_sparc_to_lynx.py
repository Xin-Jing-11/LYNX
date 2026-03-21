#!/usr/bin/env python3
"""Convert SPARC test suite systems to LYNX JSON input format."""

import os
import re
import json
import shutil
import numpy as np

SPARC_DIR = "/home/xx/Desktop/coding/Test_Suite/sparc/bulk"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "systems")

HA_TO_K = 315774.65  # 1 Hartree in Kelvin


def parse_inpt(filepath):
    """Parse SPARC .inpt file."""
    params = {}
    latvec_lines = []
    reading_latvec = False

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if reading_latvec:
                latvec_lines.append([float(x) for x in line.split()])
                if len(latvec_lines) == 3:
                    reading_latvec = False
                    params['LATVEC'] = latvec_lines
                continue
            if line.startswith('LATVEC'):
                reading_latvec = True
                rest = line[len('LATVEC'):].strip().strip(':')
                if rest:
                    latvec_lines.append([float(x) for x in rest.split()])
                continue

            # Key: value parsing
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
                params[key] = val

    return params


def parse_ion(filepath):
    """Parse SPARC .ion file."""
    atom_types = []
    current = None

    with open(filepath) as f:
        reading_coords = False
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            # Strip inline comments
            if '#' in line:
                line = line[:line.index('#')].strip()

            if line.startswith('ATOM_TYPE:'):
                if current:
                    atom_types.append(current)
                current = {'element': line.split(':')[1].strip(), 'coords': []}
                reading_coords = False
            elif line.startswith('N_TYPE_ATOM:'):
                current['n_atoms'] = int(line.split(':')[1].strip())
            elif line.startswith('PSEUDO_POT:'):
                current['pseudo_pot'] = line.split(':')[1].strip()
            elif line.startswith('COORD_FRAC'):
                reading_coords = True
                # Check if coords start on same line
                rest = line[len('COORD_FRAC'):].strip().strip(':')
                if rest:
                    vals = [float(x) for x in rest.split()]
                    if len(vals) >= 3:
                        current['coords'].append(vals[:3])
            elif line.startswith('RELAX') or line.startswith('ATOMIC_MASS') or line.startswith('SPIN'):
                reading_coords = False
            elif reading_coords:
                vals = line.split()
                if len(vals) >= 3:
                    try:
                        current['coords'].append([float(vals[0]), float(vals[1]), float(vals[2])])
                    except ValueError:
                        reading_coords = False

    if current:
        atom_types.append(current)
    return atom_types


def parse_out(filepath):
    """Parse SPARC .out file for reference values and computational parameters."""
    ref = {}
    params = {}
    forces = []
    stress = []

    with open(filepath) as f:
        content = f.read()

    # Get computational parameters
    for key in ['FD_GRID', 'FD_ORDER', 'SPIN_TYP', 'NSTATES', 'CHEB_DEGREE',
                'TOL_SCF', 'KPOINT_GRID', 'KPOINT_SHIFT']:
        m = re.search(rf'^{key}:\s*(.+)$', content, re.MULTILINE)
        if m:
            params[key] = m.group(1).strip()

    m = re.search(r'^SMEARING:\s*(.+)$', content, re.MULTILINE)
    if m:
        params['SMEARING'] = float(m.group(1).strip())

    # Get the LAST "Energy and force calculation" block (final SCF result)
    blocks = list(re.finditer(r'Energy and force calculation', content))
    if blocks:
        last_block_start = blocks[-1].start()
        block = content[last_block_start:]

        # Energy
        m = re.search(r'Free energy per atom\s*:\s*([-\d.E+]+)', block)
        if m:
            ref['energy_per_atom'] = float(m.group(1))

        m = re.search(r'Total free energy\s*:\s*([-\d.E+]+)', block)
        if m:
            ref['total_energy'] = float(m.group(1))

        # Forces
        m = re.search(r'Atomic forces \(Ha/Bohr\):\n((?:\s*[-\d.E+]+\s+[-\d.E+]+\s+[-\d.E+]+\s*\n)+)', block)
        if m:
            for line in m.group(1).strip().split('\n'):
                vals = [float(x) for x in line.split()]
                forces.append(vals)
            ref['forces'] = forces

        # Stress
        m = re.search(r'Stress \(GPa\):\s*\n((?:\s*[-\d.E+]+\s+[-\d.E+]+\s+[-\d.E+]+\s*\n){3})', block)
        if m:
            for line in m.group(1).strip().split('\n'):
                vals = [float(x) for x in line.split()]
                stress.append(vals)
            ref['stress'] = stress

        m = re.search(r'Pressure\s*:\s*([-\d.E+]+)', block)
        if m:
            ref['pressure'] = float(m.group(1))

    return ref, params


def is_identity_latvec(latvec):
    """Check if LATVEC is close to identity matrix."""
    identity = np.eye(3)
    return np.allclose(latvec, identity, atol=1e-8)


def convert_system(system_name):
    """Convert a single SPARC system to LYNX JSON format."""
    sys_dir = os.path.join(SPARC_DIR, system_name)
    out_dir = os.path.join(OUTPUT_DIR, system_name)
    os.makedirs(out_dir, exist_ok=True)

    # Parse files
    inpt = parse_inpt(os.path.join(sys_dir, "sprc-calc.inpt"))
    ion_types = parse_ion(os.path.join(sys_dir, "sprc-calc.ion"))
    ref, out_params = parse_out(os.path.join(sys_dir, "sprc-calc.out"))

    # Extract CELL
    cell_vals = [float(x) for x in inpt['CELL'].split()]
    latvec = np.array(inpt['LATVEC'])

    # Compute full lattice vectors = CELL[i] * LATVEC[i]
    full_vectors = []
    for i in range(3):
        full_vectors.append((latvec[i] * cell_vals[i]).tolist())

    # Determine cell type
    cell_type = "orthogonal" if is_identity_latvec(latvec) else "nonorthogonal"

    # Grid dimensions from .out file
    fd_grid = [int(x) for x in out_params['FD_GRID'].split()]
    fd_order = int(out_params.get('FD_ORDER', '12'))

    # Boundary conditions
    bc_str = inpt.get('BC', 'P P P')
    bcs = []
    for b in bc_str.split():
        bcs.append("periodic" if b == 'P' else "dirichlet")

    # K-points
    kgrid = [int(x) for x in out_params.get('KPOINT_GRID', inpt.get('KPOINT_GRID', '1 1 1')).split()]
    kshift_str = out_params.get('KPOINT_SHIFT', inpt.get('KPOINT_SHIFT', '0 0 0'))
    kshift = [float(x) for x in kshift_str.split()]

    # Spin type
    spin_typ = int(out_params.get('SPIN_TYP', '0'))
    if spin_typ == 0:
        spin = "none"
    elif spin_typ == 1:
        spin = "collinear"
    else:
        spin = "noncollinear"

    # Electronic params
    nstates = int(out_params.get('NSTATES', '0'))
    smearing_ha = out_params.get('SMEARING', 0.003674926479)
    temperature = float(smearing_ha) * HA_TO_K
    cheb_degree = int(out_params.get('CHEB_DEGREE', '-1'))
    tol_scf = float(out_params.get('TOL_SCF', '1e-5'))

    # Build atoms list and copy pseudopotentials
    atoms = []
    for at in ion_types:
        # Copy pseudopotential
        pot_name = at['pseudo_pot']
        pot_src = os.path.join(sys_dir, pot_name)
        pot_dst = os.path.join(out_dir, pot_name)
        if os.path.exists(pot_src):
            shutil.copy2(pot_src, pot_dst)

        atoms.append({
            "element": at['element'],
            "pseudo_file": pot_name,
            "fractional": True,
            "coordinates": at['coords']
        })

    # Build LYNX JSON
    lynx_input = {
        "lattice": {
            "vectors": full_vectors,
            "cell_type": cell_type
        },
        "grid": {
            "Nx": fd_grid[0],
            "Ny": fd_grid[1],
            "Nz": fd_grid[2],
            "fd_order": fd_order,
            "boundary_conditions": bcs
        },
        "atoms": atoms,
        "electronic": {
            "xc": "LDA_PZ",
            "spin": spin,
            "temperature": round(temperature, 2),
            "smearing": "gaussian",
            "Nstates": nstates
        },
        "kpoints": {
            "grid": kgrid,
            "shift": kshift
        },
        "scf": {
            "max_iter": 250,
            "tolerance": tol_scf,
            "mixing": "density",
            "preconditioner": "kerker",
            "mixing_history": 7,
            "mixing_parameter": 0.3,
            "cheb_degree": cheb_degree,
            "rho_trigger": 4
        },
        "output": {
            "print_forces": True,
            "calc_stress": True,
            "print_atoms": True
        }
    }

    # Write LYNX input JSON
    with open(os.path.join(out_dir, "input.json"), 'w') as f:
        json.dump(lynx_input, f, indent=2)

    # Write reference data
    with open(os.path.join(out_dir, "reference.json"), 'w') as f:
        json.dump(ref, f, indent=2)

    n_atoms = sum(len(at['coords']) for at in ion_types)
    n_electrons = "?"  # Will be determined by pseudopotentials
    print(f"  {system_name}: {n_atoms} atoms, grid={fd_grid}, kpts={kgrid}, spin={spin}")

    return system_name


def main():
    systems = sorted(os.listdir(SPARC_DIR))
    systems = [s for s in systems if os.path.isdir(os.path.join(SPARC_DIR, s))]

    print(f"Converting {len(systems)} systems from SPARC to LYNX format...")
    print(f"Source: {SPARC_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    converted = []
    for sys_name in systems:
        try:
            convert_system(sys_name)
            converted.append(sys_name)
        except Exception as e:
            print(f"  ERROR converting {sys_name}: {e}")

    print(f"\nSuccessfully converted {len(converted)}/{len(systems)} systems.")

    # Write systems list
    with open(os.path.join(OUTPUT_DIR, "..", "systems_list.json"), 'w') as f:
        json.dump(converted, f, indent=2)


if __name__ == "__main__":
    main()
