# PBE0 Benchmark Results

## Summary

All 4 benchmark systems converge in LYNX PBE0. SPARC reference values are **unavailable** because the current SPARC build has a broken EXX implementation (crashes during Fock outer loop for all systems, including the known-working NaCl_PBE0 reference test). The SPARC EXX source files (`exactExchange.c`, `exactExchangeKpt.c`) were modified on 2026-03-22 and the binary was rebuilt, introducing the bug.

### Known Issues
1. **SPARC EXX broken**: Cannot generate reference values with current SPARC build
2. **LYNX MPI bug**: PBE0 crashes with >1 MPI process (`MPI_ERR_TRUNCATE` in Fock loop Allreduce)
3. **Stress offset**: Si2_kpt_PBE0 shows ~200 GPa systematic stress offset vs SPARC reference (EXX stress contribution likely missing/incorrect)
4. **Gamma-only PBE0**: SPARC fails at gamma-only for periodic systems (G=0 divergence in Fock exchange). LYNX handles this via AUXILIARY divergence treatment.

## Existing Test: Si2_kpt_PBE0

| Column | Value |
|--------|-------|
| Test name | EndToEnd.Si2_kpt_PBE0 |
| System | Si2 non-ortho FCC, LATVEC_SCALE=6 |
| Grid | 15x15x15 |
| K-points | 2x2x2 (0.5 shift) |
| Spin | None |
| SPARC Etotal (Ha) | -7.403420694434764 |
| LYNX Etotal (Ha) | -7.4034039587 |
| Energy error (Ha) | 1.67e-5 |
| SPARC max force (Ha/Bohr) | 2.6983e-1 |
| Force error (Ha/Bohr) | 4.38e-5 |
| SPARC max stress (GPa) | 450.56 |
| Stress error (GPa) | ~200 (systematic offset) |
| Fock iterations | 4 |
| CPU time (s) | 27.3 (1 proc) |
| GPU time (s) | N/A |

## Benchmark System Results (LYNX only — no SPARC references available)

### Si4 ortho gamma PBE0

| Column | Value |
|--------|-------|
| Test name | EndToEnd.Si4_PBE0_gamma |
| System | Si4 ortho 10.0x10.26x10.5 Bohr |
| Grid | 25x26x27 |
| K-points | Gamma |
| Spin | None |
| SPARC Etotal (Ha) | N/A (SPARC EXX broken) |
| LYNX Etotal (Ha) | -15.4703016224 |
| Energy error (Ha) | N/A |
| Max force (Ha/Bohr) | 3.23e-2 |
| Force error (Ha/Bohr) | N/A |
| Max stress (GPa) | 39.22 |
| Stress error (GPa) | N/A |
| Fock iterations | 12 |
| CPU time (s) | 84.5 (1 proc) |
| GPU time (s) | N/A |

### Si4 ortho kpt PBE0

| Column | Value |
|--------|-------|
| Test name | EndToEnd.Si4_PBE0_kpt |
| System | Si4 ortho 10.0x10.26x10.5 Bohr |
| Grid | 25x26x27 |
| K-points | 2x2x2 (0.5 shift) |
| Spin | None |
| SPARC Etotal (Ha) | N/A (SPARC EXX broken) |
| LYNX Etotal (Ha) | -15.6525828345 |
| Energy error (Ha) | N/A |
| Max force (Ha/Bohr) | 8.11e-3 |
| Force error (Ha/Bohr) | N/A |
| Max stress (GPa) | 33.00 |
| Stress error (GPa) | N/A |
| Fock iterations | 3 |
| CPU time (s) | 60.4 (1 proc) |
| GPU time (s) | N/A |

### Fe2 spin gamma PBE0

| Column | Value |
|--------|-------|
| Test name | EndToEnd.Fe2_spin_PBE0_gamma |
| System | Fe2 non-ortho sheared BCC 2.84 Bohr |
| Grid | 29x29x29 |
| K-points | Gamma |
| Spin | Collinear |
| SPARC Etotal (Ha) | N/A (SPARC EXX broken) |
| LYNX Etotal (Ha) | -228.3598025816 |
| Energy error (Ha) | N/A |
| Max force (Ha/Bohr) | 7.86e-1 |
| Force error (Ha/Bohr) | N/A |
| Max stress (GPa) | 35218 |
| Stress error (GPa) | N/A |
| Fock iterations | 3 |
| CPU time (s) | 267.1 (1 proc) |
| GPU time (s) | N/A |

### Fe2 spin kpt PBE0

| Column | Value |
|--------|-------|
| Test name | EndToEnd.Fe2_spin_PBE0_kpt |
| System | Fe2 cubic 2.84 Bohr |
| Grid | 29x29x29 |
| K-points | 3x2x2 (0.0, 0.5, 0.5 shift) |
| Spin | Collinear |
| SPARC Etotal (Ha) | N/A (SPARC EXX broken) |
| LYNX Etotal (Ha) | -228.1476439029 |
| Energy error (Ha) | N/A |
| Max force (Ha/Bohr) | 6.29e-1 |
| Force error (Ha/Bohr) | N/A |
| Max stress (GPa) | 34973 |
| Stress error (GPa) | N/A |
| Fock iterations | 4 |
| CPU time (s) | 1717.7 (1 proc) |
| GPU time (s) | N/A |

## Test Settings

All tests use identical settings between SPARC inputs and LYNX JSON configs:
- FD_ORDER: 12
- SCF tolerance: 1e-6
- Mixing: density (Si) / potential (Fe), Kerker preconditioner, beta=0.3, history=7
- Temperature: 315.775131 K, Fermi-Dirac
- PBE0: exx_frac=0.25, spherical divergence, FFT Poisson solver
- PSP: Si-4-2.4_LDA.psp8, Fe_LDA.psp8

## Blocking Issues for Merge

1. **No SPARC references**: Cannot validate accuracy until SPARC EXX is fixed
2. **MPI parallelization bug**: PBE0 crashes with >1 MPI process — must be fixed before production use
3. **Stress accuracy**: ~200 GPa systematic offset in stress (seen on Si2_kpt_PBE0 vs known SPARC reference)
4. **Energy/force accuracy**: Si2_kpt_PBE0 shows 1.67e-5 Ha energy error and 4.38e-5 force error, exceeding the 1e-6 and 1e-5 targets respectively
