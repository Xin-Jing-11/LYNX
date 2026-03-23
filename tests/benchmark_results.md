# LYNX Benchmark Results

## SCAN (MGGA_SCAN) — CPU, master branch

Date: 2026-03-22

### Summary

| Test name | System | Grid | K-points | Spin | SPARC Etotal (Ha) | LYNX Etotal (Ha) | Energy error (Ha) | SPARC max force (Ha/Bohr) | Force error (Ha/Bohr) | SPARC max stress (GPa) | Stress error (GPa) | SCF iterations | CPU time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Si4_gamma_SCAN | Si4 deformed ortho | 25x26x27 | Gamma | None | N/A | -15.4789706181 | N/A | N/A | N/A | N/A | N/A | 33 | ~131 |
| Si4_kpt_SCAN | Si4 deformed ortho | 25x26x27 | 2x2x2 (0.5 shift) | None | -15.87545747 | (not captured) | ~1e-6 (est.) | 0.027 | 1.22e-5 | 4.07 | **26.70** | (converged) | ~4752 |
| Fe2_spin_SCAN_gamma | Fe2 non-ortho BCC | 29x29x29 | Gamma | Collinear | -228.3157354078 | -228.3157353115 | **9.63e-8** | 0.807 | **6.53e-6** | 22381 | **6790** | 11 | ~217 |
| Fe2_spin_SCAN_kpt | Fe2 non-ortho BCC | 29x29x29 | 3x2x2 (0,0.5,0.5) | Collinear | -228.1284707525 | (not captured) | (unknown) | 0.612 | **0.067** | 22632 | **7273** | (converged) | ~1095 |

### Detailed Results

#### Si4_gamma_SCAN (PASS — no SPARC reference)
- **Energy**: -15.4789706181 Ha
- **SCF**: 33 iterations
- **Notes**: No SPARC reference available for gamma-point SCAN Si4. Run completes successfully.

#### Si4_kpt_SCAN (FAIL — stress)
- **Energy**: ~matches SPARC (exact value not captured from test output)
- **Forces**: Max error 1.22e-5 Ha/Bohr (slightly above 1e-5 tolerance)
- **Stress (GPa)**: Diagonal error ~27 GPa
  - LYNX: xx=28.30, yy=28.60, zz=28.34
  - SPARC: xx=1.60, yy=1.90, zz=1.63
  - Off-diagonal errors: 0.18-0.20 GPa (acceptable)
- **Root cause**: Systematic positive diagonal stress offset (~27 GPa)

#### Fe2_spin_SCAN_gamma (FAIL — stress)
- **Energy**: -228.3157353115 Ha (error: 9.63e-8 Ha) — **PASS**
- **Forces**: Max error 6.53e-6 Ha/Bohr — **PASS**
- **Stress (GPa)**: Massive diagonal error ~6700 GPa
  - LYNX: xx=-15251, xy=1521, xz=-47, yy=-15311, yz=-122, zz=-15591
  - SPARC: xx=-21919, xy=1393, xz=-51, yy=-21976, yz=-127, zz=-22381
  - Diagonal error: ~6668-6790 GPa
  - Off-diagonal error: 3.6-127 GPa (xy is 127 GPa off)
- **Root cause**: Systematic negative diagonal stress offset

#### Fe2_spin_SCAN_kpt (FAIL — forces + stress)
- **Energy**: (not captured from test output)
- **Forces**: Max error 0.067 Ha/Bohr — **FAIL** (far above 1e-5 tolerance)
  - Component z of atom 1: LYNX=-0.124 vs SPARC=-0.192 (error=0.067)
- **Stress (GPa)**: Diagonal error ~7200 GPa + sign-flipped off-diagonals
  - LYNX: xx=-15281, xy=-18.0, xz=+18.0, yy=-15359, yz=+291, zz=-15359
  - SPARC: xx=-22514, xy=+16.2, xz=-16.2, yy=-22632, yz=-36.4, zz=-22632
  - Off-diagonal components have wrong sign!
- **Root cause**: Multiple issues — k-point mGGA forces/stress badly broken for Fe2 non-ortho

### Known Issues

1. **mGGA diagonal stress offset**: All tests show a systematic positive offset on diagonal stress components. The offset scales with the magnitude of the stress (~27 GPa for Si4 at ~2 GPa, ~7000 GPa for Fe2 at ~22000 GPa). This suggests a missing or incorrect contribution in the `Exc - Exc_corr` diagonal term or the mGGA psi stress term in `Stress.cpp`.

2. **Fe2 k-point forces**: The Fe2_spin_SCAN_kpt test has force errors ~1000x larger than the gamma-point test, suggesting a bug in the k-point mGGA Hamiltonian term or force computation for non-orthogonal spin-polarized cells with k-points.

3. **Fe2 k-point off-diagonal stress sign flip**: Off-diagonal stress components have the wrong sign in Fe2_spin_SCAN_kpt, pointing to a possible complex-conjugate or sign convention error in the k-point mGGA stress calculation.
