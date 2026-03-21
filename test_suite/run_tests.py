#!/usr/bin/env python3
"""Run LYNX on all test suite systems with 1, 4, and 8 MPI cores."""

import os
import sys
import json
import re
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LYNX_BIN = os.path.realpath(os.path.join(SCRIPT_DIR, "..", "build", "src", "lynx"))
SYSTEMS_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, "systems"))
OUTPUT_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, "..", "build", "test_suite", "systems"))
TIMEOUT_SMALL = 1800   # 30 min for small systems
TIMEOUT_MEDIUM = 3600  # 60 min for medium systems
TIMEOUT_LARGE = 7200   # 120 min for large systems

# Systems sorted by size (electrons)
SYSTEM_SIZES = {
    "WC_bulk": 18, "CaN2_bulk": 20, "SbAs_bulk": 20, "BeCu_bulk": 23,
    "Na2S_bulk": 24, "V_bulk": 26, "Bi_bulk": 30, "NbP_bulk": 36,
    "PtAu_bulk": 37, "SnO_bulk": 40, "CdS_bulk": 52, "BeF2_bulk": 54,
    "CaO_bulk": 64, "IrPt3_bulk": 71, "Pd_bulk": 72, "Ag2O_bulk": 88,
    "ZrO2_bulk": 96, "RhS2_bulk": 116, "Au_bulk": 133, "La_bulk": 132,
    "Zn3Cu_bulk": 158, "Zn_bulk": 160, "AsS_bulk": 176, "PSe_bulk": 176,
    "Sb2S3_bulk": 192, "Pt_bulk": 270, "MgS_bulk": 432, "Sc_bulk": 517,
    "Na_bulk": 1080,
}


def get_timeout(system_name):
    ne = SYSTEM_SIZES.get(system_name, 100)
    if ne <= 100:
        return TIMEOUT_SMALL
    elif ne <= 300:
        return TIMEOUT_MEDIUM
    else:
        return TIMEOUT_LARGE


def parse_lynx_output(log_file):
    """Parse LYNX output log for energy, forces, and stress."""
    result = {}

    with open(log_file) as f:
        content = f.read()

    # Check convergence
    result['converged'] = 'SCF CONVERGED' in content

    # Energy
    m = re.search(r'Etotal\s*=\s*([-\d.]+)\s*Ha', content)
    if m:
        result['total_energy'] = float(m.group(1))

    m = re.search(r'Eatom\s*=\s*([-\d.]+)\s*Ha/atom', content)
    if m:
        result['energy_per_atom'] = float(m.group(1))

    # Total forces
    forces = []
    force_block = re.search(r'Total forces \(Ha/Bohr\):\n((?:\s+Atom\s+\d+:.*\n)+)', content)
    if force_block:
        for line in force_block.group(1).strip().split('\n'):
            m = re.match(r'\s*Atom\s+\d+:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', line)
            if m:
                forces.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
    if forces:
        result['forces'] = forces

    # Stress tensor (GPa)
    stress = []
    stress_block = re.search(
        r'Stress tensor \(GPa\):\n'
        r'\s*σ_xx\s*=\s*([-\d.]+)\s+σ_xy\s*=\s*([-\d.]+)\s+σ_xz\s*=\s*([-\d.]+)\n'
        r'\s*σ_yy\s*=\s*([-\d.]+)\s+σ_yz\s*=\s*([-\d.]+)\s+σ_zz\s*=\s*([-\d.]+)',
        content)
    if stress_block:
        sxx, sxy, sxz = float(stress_block.group(1)), float(stress_block.group(2)), float(stress_block.group(3))
        syy, syz, szz = float(stress_block.group(4)), float(stress_block.group(5)), float(stress_block.group(6))
        stress = [[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]]
        result['stress'] = stress

    # Pressure
    m = re.search(r'Pressure:\s*([-\d.]+)\s*GPa', content)
    if m:
        result['pressure'] = float(m.group(1))

    # Wall time (from last SCF iteration timing or total)
    m = re.search(r'LYNX calculation complete', content)
    result['completed'] = m is not None

    return result


def run_system(system_name, nprocs, force_rerun=False):
    """Run LYNX on a system with given number of MPI procs."""
    src_dir = os.path.join(SYSTEMS_DIR, system_name)
    out_dir = os.path.join(OUTPUT_DIR, system_name)
    os.makedirs(out_dir, exist_ok=True)

    input_file = os.path.join(src_dir, "input.json")
    log_file = os.path.join(out_dir, f"output_np{nprocs}.log")
    result_file = os.path.join(out_dir, f"results_np{nprocs}.json")

    if not os.path.exists(input_file):
        return None

    # Skip if already run successfully
    if not force_rerun and os.path.exists(result_file):
        with open(result_file) as f:
            existing = json.load(f)
        if existing.get('converged') or existing.get('completed'):
            print(f"    [SKIP] {system_name} np={nprocs} (already done)")
            return existing

    timeout = get_timeout(system_name)
    cmd = ["mpirun", "--oversubscribe", "-np", str(nprocs), LYNX_BIN, "input.json"]

    print(f"    Running {system_name} np={nprocs} (timeout={timeout}s)...", end=" ", flush=True)
    start_time = time.time()

    try:
        proc = subprocess.run(
            cmd,
            cwd=src_dir,
            stdout=open(log_file, 'w'),
            stderr=subprocess.STDOUT,
            timeout=timeout
        )
        elapsed = time.time() - start_time
        result = parse_lynx_output(log_file)
        result['wall_time'] = round(elapsed, 1)
        result['exit_code'] = proc.returncode
        result['nprocs'] = nprocs

        status = "CONVERGED" if result.get('converged') else "FAILED"
        print(f"{status} ({elapsed:.1f}s)")

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        result = {
            'converged': False,
            'completed': False,
            'wall_time': round(elapsed, 1),
            'exit_code': -1,
            'nprocs': nprocs,
            'error': 'timeout'
        }
        print(f"TIMEOUT ({elapsed:.1f}s)")

    except Exception as e:
        elapsed = time.time() - start_time
        result = {
            'converged': False,
            'completed': False,
            'wall_time': round(elapsed, 1),
            'exit_code': -2,
            'nprocs': nprocs,
            'error': str(e)
        }
        print(f"ERROR: {e}")

    # Save results
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    force_rerun = '--force' in sys.argv
    # Allow specifying systems on command line
    requested = [a for a in sys.argv[1:] if not a.startswith('--')]

    if not os.path.exists(LYNX_BIN):
        print(f"ERROR: LYNX binary not found at {LYNX_BIN}")
        print("Please build LYNX first: cd build && cmake .. && make -j")
        sys.exit(1)

    # Get available systems, sorted by size
    if requested:
        systems = requested
    else:
        systems_list = os.path.join(SCRIPT_DIR, "systems_list.json")
        with open(systems_list) as f:
            all_systems = json.load(f)
        systems = sorted(all_systems, key=lambda s: SYSTEM_SIZES.get(s, 999))

    # Parse --cores argument (default: run 8, 4, 1 with skips for large systems)
    cores_arg = [a for a in sys.argv[1:] if a.startswith('--cores=')]
    if cores_arg:
        core_counts = [int(x) for x in cores_arg[0].split('=')[1].split(',')]
    else:
        core_counts = [8, 4, 1]

    print(f"Running {len(systems)} systems with {core_counts} cores each")
    print(f"LYNX binary: {LYNX_BIN}")
    print(f"Input dir:   {SYSTEMS_DIR}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print(f"{'='*60}")

    for system_name in systems:
        ne = SYSTEM_SIZES.get(system_name, "?")
        ne_val = SYSTEM_SIZES.get(system_name, 100)
        print(f"\n[{system_name}] ({ne} electrons)")
        for np_val in core_counts:
            # Skip 1-core for systems with > 50 electrons
            if np_val == 1 and ne_val > 50:
                print(f"    [SKIP] {system_name} np=1 ({ne_val} electrons)")
                continue
            # Skip 4-core for very large systems (>300 electrons)
            if np_val == 4 and ne_val > 300:
                print(f"    [SKIP] {system_name} np=4 ({ne_val} electrons)")
                continue
            run_system(system_name, np_val, force_rerun)

    print(f"\n{'='*60}")
    print("All runs complete. Use report.py to analyze results.")


if __name__ == "__main__":
    main()
