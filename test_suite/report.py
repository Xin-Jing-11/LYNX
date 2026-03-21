#!/usr/bin/env python3
"""Generate accuracy report comparing LYNX results against SPARC reference."""

import os
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Reference data lives in test_suite/systems/
REF_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, "systems"))
# Results (logs, results JSON) live in build/test_suite/systems/
RESULTS_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, "..", "build", "test_suite", "systems"))

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

CORE_COUNTS = [1, 4, 8]


def load_results(system_name):
    """Load reference (from test_suite/) and LYNX results (from build/test_suite/)."""
    ref_file = os.path.join(REF_DIR, system_name, "reference.json")

    if not os.path.exists(ref_file):
        return None, {}

    with open(ref_file) as f:
        ref = json.load(f)

    results = {}
    for nc in CORE_COUNTS:
        result_file = os.path.join(RESULTS_DIR, system_name, f"results_np{nc}.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                results[nc] = json.load(f)

    return ref, results


def compute_errors(ref, result):
    """Compute error metrics between reference and result."""
    errors = {}

    # Energy error (Ha/atom)
    if 'energy_per_atom' in ref and 'energy_per_atom' in result:
        errors['energy_error'] = abs(result['energy_per_atom'] - ref['energy_per_atom'])

    # Force errors (Ha/Bohr)
    if 'forces' in ref and 'forces' in result:
        ref_f = np.array(ref['forces'])
        res_f = np.array(result['forces'])
        if ref_f.shape == res_f.shape:
            diff = ref_f - res_f
            errors['force_max_error'] = float(np.max(np.abs(diff)))
            errors['force_rms_error'] = float(np.sqrt(np.mean(diff**2)))
        else:
            errors['force_error_note'] = f"Shape mismatch: ref={ref_f.shape} vs result={res_f.shape}"

    # Stress errors (GPa)
    if 'stress' in ref and 'stress' in result:
        ref_s = np.array(ref['stress'])
        res_s = np.array(result['stress'])
        if ref_s.shape == res_s.shape:
            diff = ref_s - res_s
            errors['stress_max_error'] = float(np.max(np.abs(diff)))
            errors['stress_rms_error'] = float(np.sqrt(np.mean(diff**2)))

    # Pressure error (GPa)
    if 'pressure' in ref and 'pressure' in result:
        errors['pressure_error'] = abs(result['pressure'] - ref['pressure'])

    return errors


def generate_report():
    """Generate the full accuracy report."""
    systems_list = os.path.join(SCRIPT_DIR, "systems_list.json")
    with open(systems_list) as f:
        all_systems = json.load(f)

    systems = sorted(all_systems, key=lambda s: SYSTEM_SIZES.get(s, 999))

    print("=" * 120)
    print("LYNX ACCURACY REPORT - Comparison with SPARC Reference (LDA_PZ)")
    print("=" * 120)
    print(f"Reference dir: {REF_DIR}")
    print(f"Results dir:   {RESULTS_DIR}")

    # Summary table
    print(f"\n{'System':<16} {'Ne':>4} {'NP':>3} {'Conv':>5} {'dE/atom (Ha)':>14} "
          f"{'max|dF| (Ha/B)':>16} {'RMS|dF| (Ha/B)':>16} {'max|ds| (GPa)':>15} {'Time(s)':>8}")
    print("-" * 120)

    all_energy_errors = []
    all_force_max_errors = []
    all_force_rms_errors = []
    all_stress_max_errors = []
    convergence_stats = {'converged': 0, 'failed': 0, 'missing': 0}

    # MPI consistency data
    mpi_consistency = {}

    for system_name in systems:
        ref, results = load_results(system_name)
        if ref is None:
            continue

        ne = SYSTEM_SIZES.get(system_name, "?")
        mpi_consistency[system_name] = {}

        for nc in CORE_COUNTS:
            if nc not in results:
                print(f"{system_name:<16} {ne:>4} {nc:>3} {'N/A':>5} {'---':>14} {'---':>16} {'---':>16} {'---':>15} {'---':>8}")
                convergence_stats['missing'] += 1
                continue

            res = results[nc]
            conv = "YES" if res.get('converged') else "NO"
            wtime = res.get('wall_time', 0)

            if res.get('converged'):
                convergence_stats['converged'] += 1
            else:
                convergence_stats['failed'] += 1

            errors = compute_errors(ref, res)

            de = errors.get('energy_error')
            fmax = errors.get('force_max_error')
            frms = errors.get('force_rms_error')
            smax = errors.get('stress_max_error')

            if de is not None:
                all_energy_errors.append(de)
                mpi_consistency[system_name][nc] = res.get('energy_per_atom')

            if fmax is not None:
                all_force_max_errors.append(fmax)
            if frms is not None:
                all_force_rms_errors.append(frms)
            if smax is not None:
                all_stress_max_errors.append(smax)

            de_str = f"{de:.6e}" if de is not None else "---"
            fmax_str = f"{fmax:.6e}" if fmax is not None else "---"
            frms_str = f"{frms:.6e}" if frms is not None else "---"
            smax_str = f"{smax:.6e}" if smax is not None else "---"

            print(f"{system_name:<16} {ne:>4} {nc:>3} {conv:>5} {de_str:>14} "
                  f"{fmax_str:>16} {frms_str:>16} {smax_str:>15} {wtime:>8.1f}")

    # Summary statistics
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)

    print(f"\nConvergence: {convergence_stats['converged']} converged, "
          f"{convergence_stats['failed']} failed, {convergence_stats['missing']} missing")

    if all_energy_errors:
        arr = np.array(all_energy_errors)
        print(f"\nEnergy error (Ha/atom):")
        print(f"  Mean:   {np.mean(arr):.6e}")
        print(f"  Median: {np.median(arr):.6e}")
        print(f"  Max:    {np.max(arr):.6e}")
        print(f"  Min:    {np.min(arr):.6e}")
        n_good = int(np.sum(arr < 1e-4))
        print(f"  Systems with |dE| < 1e-4 Ha/atom: {n_good}/{len(arr)}")

    if all_force_max_errors:
        arr = np.array(all_force_max_errors)
        print(f"\nForce max error (Ha/Bohr):")
        print(f"  Mean:   {np.mean(arr):.6e}")
        print(f"  Median: {np.median(arr):.6e}")
        print(f"  Max:    {np.max(arr):.6e}")
        print(f"  Min:    {np.min(arr):.6e}")
        n_good = int(np.sum(arr < 1e-3))
        print(f"  Systems with max|dF| < 1e-3 Ha/Bohr: {n_good}/{len(arr)}")

    if all_force_rms_errors:
        arr = np.array(all_force_rms_errors)
        print(f"\nForce RMS error (Ha/Bohr):")
        print(f"  Mean:   {np.mean(arr):.6e}")
        print(f"  Median: {np.median(arr):.6e}")
        print(f"  Max:    {np.max(arr):.6e}")
        print(f"  Min:    {np.min(arr):.6e}")

    if all_stress_max_errors:
        arr = np.array(all_stress_max_errors)
        print(f"\nStress max error (GPa):")
        print(f"  Mean:   {np.mean(arr):.6e}")
        print(f"  Median: {np.median(arr):.6e}")
        print(f"  Max:    {np.max(arr):.6e}")
        print(f"  Min:    {np.min(arr):.6e}")

    # MPI consistency check
    print("\n" + "=" * 120)
    print("MPI CONSISTENCY CHECK (energy differences across core counts)")
    print("=" * 120)
    print(f"\n{'System':<16} {'E(np=1)':>18} {'E(np=4)':>18} {'E(np=8)':>18} {'max |diff|':>14}")
    print("-" * 90)

    for system_name in systems:
        if system_name not in mpi_consistency:
            continue
        data = mpi_consistency[system_name]
        if len(data) < 2:
            continue

        vals = list(data.values())
        max_diff = max(vals) - min(vals)

        e1 = f"{data.get(1, 0):.10f}" if 1 in data else "---"
        e4 = f"{data.get(4, 0):.10f}" if 4 in data else "---"
        e8 = f"{data.get(8, 0):.10f}" if 8 in data else "---"

        print(f"{system_name:<16} {e1:>18} {e4:>18} {e8:>18} {max_diff:>14.2e}")

    # Detailed per-system breakdown for systems with large errors
    print("\n" + "=" * 120)
    print("FLAGGED SYSTEMS (energy error > 1e-4 Ha/atom or force error > 1e-3 Ha/Bohr)")
    print("=" * 120)

    flagged = False
    for system_name in systems:
        ref, results = load_results(system_name)
        if ref is None:
            continue

        for nc in CORE_COUNTS:
            if nc not in results:
                continue
            res = results[nc]
            if not res.get('converged'):
                continue

            errors = compute_errors(ref, res)
            de = errors.get('energy_error', 0)
            fmax = errors.get('force_max_error', 0)

            if de > 1e-4 or fmax > 1e-3:
                flagged = True
                print(f"\n  {system_name} (np={nc}):")
                print(f"    Energy error: {de:.6e} Ha/atom")
                if 'energy_per_atom' in ref:
                    print(f"    Ref energy:   {ref['energy_per_atom']:.10f} Ha/atom")
                if 'energy_per_atom' in res:
                    print(f"    LYNX energy:  {res['energy_per_atom']:.10f} Ha/atom")
                if fmax > 0:
                    print(f"    Force max err: {fmax:.6e} Ha/Bohr")
                smax = errors.get('stress_max_error', 0)
                if smax > 0:
                    print(f"    Stress max err: {smax:.6e} GPa")

    if not flagged:
        print("  None -- all converged systems within tolerance!")

    print("\n" + "=" * 120)
    print("Report complete.")


if __name__ == "__main__":
    generate_report()
