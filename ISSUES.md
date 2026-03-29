# LYNX Porting — Issues Tracker

## Status Legend
- **OPEN**: Not started
- **WIP**: Work in progress
- **FIXED**: Fixed and verified
- **BLOCKED**: Waiting on dependency

---

## feature/scan (SCAN CPU)

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| S1 | rSCAN & r2SCAN functional support missing | FIXED | Added enum entries, libxc IDs, InputParser support |
| S2 | No spin-polarized E2E test | FIXED | Fe2 non-orth spin SCAN: energy 9.9e-08 Ha, forces 6.5e-06, stress 0.05 GPa |
| S4 | mGGA stress wrong gradient transform for non-orth cells | FIXED | Used gradT instead of lat_uvec_inv — caused ~2300 GPa error on non-orth cells |
| S3 | Stress not verified against SPARC | FIXED | Two bugs found: GGA grad stress skipped for mGGA + Exc_corr missing ∫τ·vtau. Now 4.3e-05 GPa |

## feature/scan-gpu (SCAN GPU)

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| G1 | No spin-polarized GPU SCAN E2E test | FIXED | Test added, 2.5e-7 Ha energy, 5.9e-7 forces, 1.3e-4 GPa stress |
| G2 | Missing spin density download from GPU | FIXED | Root cause of 4.5 GPa stress error — stale densities used for stress |
| G3 | Missing 0.5 spin factor for tau | FIXED | SPARC applies vscal*=0.5 to per-spin tau |
| G4 | Wrong Vxc buffer for spin download | FIXED | Was Nd_ size, needed 2*Nd_ |
| G5 | Tau/vtau size mismatch GPU→CPU | FIXED | GPU 2*Nd vs CPU 3*Nd layout |
| G6 | No rSCAN/r2SCAN GPU kernels | OPEN | Depends on CPU rSCAN/r2SCAN first |
| G7 | mGGA force/stress on CPU after download | OPEN | Performance issue, not accuracy |

## feature/hybrid (Hybrid Functionals) — MERGED 2026-03-25

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| H1 | EXX stress not implemented | FIXED | 491-line compute_stress() with gradient integrals, spherical truncation, non-orth transform |
| H1b | EXX stress ~90 GPa diagonal offset | FIXED | Non-orth transform used grad_T instead of lat_uvec_inv + transposed indexing (same bug as S4) |
| H1c | GGA gradient stress missing for hybrid XC type | FIXED | is_gga check didn't include HYB_PBE0/HYB_HSE |
| H2 | K-point EXX not validated | FIXED | Si2_kpt_PBE0: energy 8.1e-7 Ha, forces 1.3e-6 Ha/Bohr, stress ~0.02 GPa |
| H3 | No GPU exact exchange kernels | OPEN | CPU-only BLAS, no CUDA kernels |
| H4 | Fock loop convergence not validated | FIXED | 5 PBE0 E2E tests pass (Si2_kpt, Si4_gamma, Si4_kpt, Fe2_spin_gamma, Fe2_spin_kpt) |
| H5 | NDArray ld stride mismatch | FIXED | 6 wfn copy sites: bulk memcpy → band-by-band with correct ld |
| H6 | EXX normalization (4 bugs) | FIXED | Spurious Nd in Xi, 1/dV in apply_Vx, missing sqrt(dV) in energy, wrong kpt weights |

## Cross-cutting

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| X1 | Stress baseline investigation needed | FIXED | Root cause: two bugs in mGGA stress path (see S3) |
| X2 | Per-type pseudocharge cutoff not implemented | OPEN | Uses global rc_max instead of adaptive |
| X3 | BaTiO3_SCF used wrong pseudopotentials | FIXED | ONCVPSP-PBE-PDv0.4 vs SPARC's psp8 — 1.31 Ha error. Fixed to use matching psp |
| X4 | ExchangePoissonSolver fails without MKL | FIXED | mkl_dfti.h included unconditionally. Added #ifdef USE_MKL guard + stubs |

---

## Fixed Issues

| # | Issue | Branch | Date | Notes |
|---|-------|--------|------|-------|
| S1 | rSCAN & r2SCAN functional support | scan | 2026-03-22 | Enum, libxc IDs, InputParser, is_mgga_type() helper |
| S3 | SCAN stress two bugs | scan | 2026-03-22 | GGA grad stress skipped for mGGA + Exc_corr missing ∫τ·vtau dV. Max err 4.3e-05 GPa |
| G1 | GPU spin SCAN test + 4 bugs | scan-gpu | 2026-03-22 | Missing spin density download, 0.5 tau factor, Vxc buffer, tau/vtau layout |
| S2 | Fe2 spin SCAN + gradient bug | scan | 2026-03-22 | mGGA stress used gradT instead of lat_uvec_inv for non-orth cells (2300 GPa error) |
| H1 | EXX stress implementation | hybrid | 2026-03-25 | 491-line compute_stress() + non-orth transform fix (grad_T→lat_uvec_inv) |
| H1c | Hybrid GGA gradient stress | hybrid | 2026-03-25 | is_gga check extended to include HYB_PBE0/HYB_HSE |
| H2 | K-point EXX validation | hybrid | 2026-03-25 | Si2_kpt_PBE0: 8.1e-7 Ha energy, 1.3e-6 forces, ~0.02 GPa stress |
| H4 | Fock loop convergence | hybrid | 2026-03-25 | 5 PBE0 E2E tests pass (gamma+kpt, nonspin+spin) |
