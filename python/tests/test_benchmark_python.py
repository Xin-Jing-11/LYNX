"""
Benchmark: verify LYNX Python interface against SPARC reference values.

Runs the 4 standard benchmark systems through lynx.DFT and compares
energy, forces, and stress against known SPARC results.

Usage:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    PYTHONPATH=python python python/tests/test_benchmark_python.py
"""
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lynx

# ── SPARC reference values ──────────────────────────────────────────────

BENCHMARKS = {
    "Si4_scan_gamma": {
        "json": "tests/e2e/data/Si4_scan_gamma.json",
        "sparc_energy": -15.4789705855,
        "sparc_forces": [
            [ 9.7761203e-08, -2.3226440e-07,  2.3343651e-02],
            [ 6.5723061e-04,  6.0597466e-05, -2.3376315e-02],
            [-2.0817305e-07,  1.3096151e-07,  2.3408743e-02],
            [-6.5712020e-04, -6.0496164e-05, -2.3376079e-02],
        ],
        "sparc_stress_GPa": [
            -0.258, -13.335, 0.0,
            0.462, 0.0, -1.389,
        ],  # xx, xy, xz, yy, yz, zz
        "tol_energy": 1e-6,
        "tol_force": 1e-5,
        "tol_stress": 0.01,
    },
    "Si4_scan_kpt": {
        "json": "tests/e2e/data/Si4_scan_kpt.json",
        "sparc_energy": -15.8754574741,
        "sparc_forces": [
            [-2.7676965e-02, -3.1808358e-02, -2.7429107e-02],
            [ 2.7526741e-02,  3.1000410e-02,  2.7308764e-02],
            [-2.2461295e-02, -2.6401277e-02, -2.5239334e-02],
            [ 2.2611519e-02,  2.7209225e-02,  2.5359677e-02],
        ],
        "sparc_stress_GPa": [
            1.5995, -4.0660, -2.4964,
            1.8963, -3.9212, 1.6329,
        ],
        "tol_energy": 1e-6,
        "tol_force": 1e-5,
        "tol_stress": 0.01,
    },
    "Fe2_spin_scan_gamma": {
        "json": "tests/e2e/data/Fe2_spin_scan_gamma.json",
        "sparc_energy": -228.3157354078,
        "sparc_forces": [
            [ 8.0738563e-01,  3.7396715e-01, -3.5795140e-01],
            [-8.0738563e-01, -3.7396715e-01,  3.5795140e-01],
        ],
        "sparc_stress_GPa": [
            -21918.87, 1393.24, -51.02,
            -21975.90, -126.76, -22380.77,
        ],
        "tol_energy": 1e-6,
        "tol_force": 1e-5,
        "tol_stress": 0.1,
    },
    "Fe2_spin_scan_kpt": {
        "json": "tests/e2e/data/Fe2_spin_scan_kpt.json",
        "sparc_energy": -228.1284707525,
        "sparc_forces": [
            [ 6.1172654e-01,  1.9162157e-01, -1.9162157e-01],
            [-6.1172654e-01, -1.9162157e-01,  1.9162157e-01],
        ],
        "sparc_stress_GPa": [
            -22513.54, 16.22, -16.22,
            -22631.81, -36.36, -22631.81,
        ],
        "tol_energy": 1e-6,
        "tol_force": 1e-5,
        "tol_stress": 0.1,
    },
}


def run_from_json(json_path):
    """Run a LYNX calculation using the JSON config via the C++ Calculator
    (the Python DFT class ultimately calls this)."""
    from lynx import _core

    calc = _core.Calculator(json_path, auto_run=True, use_gpu=False)
    energy = calc.total_energy
    forces = np.array(calc.compute_forces())
    stress_voigt = np.array(calc.compute_stress())
    return energy, forces, stress_voigt


def run_benchmark(name, info):
    """Run one benchmark and compare against SPARC reference."""
    json_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        info["json"],
    )
    if not os.path.isfile(json_path):
        # Try relative to CWD
        json_path = info["json"]
    if not os.path.isfile(json_path):
        return None, f"JSON not found: {info['json']}"

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    t0 = time.time()
    energy, forces, stress_ha = run_from_json(json_path)
    dt = time.time() - t0

    # Convert stress Ha/Bohr^3 -> GPa
    HA_BOHR3_TO_GPA = 29421.01569650548
    stress_GPa = stress_ha * HA_BOHR3_TO_GPA

    # Compare energy
    ref_E = info["sparc_energy"]
    err_E = abs(energy - ref_E)

    # Compare forces
    ref_F = np.array(info["sparc_forces"])
    n_atoms = ref_F.shape[0]
    forces_mat = forces.reshape(n_atoms, 3)
    err_F = np.max(np.abs(forces_mat - ref_F))

    # Compare stress (Voigt: xx, xy, xz, yy, yz, zz)
    ref_S = np.array(info["sparc_stress_GPa"])
    err_S = np.max(np.abs(stress_GPa - ref_S))

    # Print results
    print(f"\n  LYNX  Etotal = {energy:.10f} Ha")
    print(f"  SPARC Etotal = {ref_E:.10f} Ha")
    print(f"  Energy error = {err_E:.2e} Ha  (tol: {info['tol_energy']:.0e})")

    print(f"\n  Force error  = {err_F:.2e} Ha/Bohr  (tol: {info['tol_force']:.0e})")
    for i in range(n_atoms):
        print(f"    Atom {i+1}: LYNX [{forces_mat[i,0]:12.8f} {forces_mat[i,1]:12.8f} {forces_mat[i,2]:12.8f}]")
        print(f"           SPARC [{ref_F[i,0]:12.8f} {ref_F[i,1]:12.8f} {ref_F[i,2]:12.8f}]")

    print(f"\n  Stress error = {err_S:.4f} GPa  (tol: {info['tol_stress']})")
    print(f"    LYNX  stress: {stress_GPa}")
    print(f"    SPARC stress: {ref_S}")

    print(f"\n  Wall time: {dt:.1f} s")

    # Pass/fail
    pass_E = err_E < info["tol_energy"]
    pass_F = err_F < info["tol_force"]
    pass_S = err_S < info["tol_stress"]
    status = "PASS" if (pass_E and pass_F and pass_S) else "FAIL"

    print(f"\n  Energy: {'PASS' if pass_E else 'FAIL'}  |  "
          f"Forces: {'PASS' if pass_F else 'FAIL'}  |  "
          f"Stress: {'PASS' if pass_S else 'FAIL'}  |  "
          f"Overall: {status}")

    return {
        "name": name,
        "energy": energy,
        "sparc_energy": ref_E,
        "energy_error": err_E,
        "force_error": err_F,
        "stress_error": err_S,
        "time": dt,
        "pass": status == "PASS",
    }, None


def main():
    print("=" * 60)
    print("  LYNX Python Interface — Benchmark Verification")
    print("  Comparing against SPARC reference values")
    print("=" * 60)

    results = []
    for name, info in BENCHMARKS.items():
        result, err = run_benchmark(name, info)
        if err:
            print(f"\n  SKIP {name}: {err}")
        else:
            results.append(result)

    # Summary table
    print("\n\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  {'Test':<30s} {'ΔE (Ha)':<14s} {'ΔF (Ha/B)':<14s} {'Δσ (GPa)':<14s} {'Time':<8s} {'Status'}")
    print("-" * 80)
    all_pass = True
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        if not r["pass"]:
            all_pass = False
        print(f"  {r['name']:<30s} {r['energy_error']:<14.2e} {r['force_error']:<14.2e} "
              f"{r['stress_error']:<14.4f} {r['time']:<8.1f} {status}")
    print("-" * 80)
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"  {len(results)}/{len(BENCHMARKS)} benchmarks completed")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
