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

## feature/hybrid (Hybrid Functionals)

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| H1 | EXX force/stress not implemented | OPEN | Port from SPARC exactExchangeStress.c (~982 lines) + exactExchangePressure.c (~635 lines) |
| H2 | K-point EXX not validated | OPEN | Kptshift_map partially implemented, needs E2E test |
| H3 | No GPU exact exchange kernels | OPEN | CPU-only BLAS, no CUDA kernels |
| H4 | Fock loop convergence not validated | OPEN | tol_fock=1e-5, untested stability |

## Cross-cutting

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| X1 | Stress baseline investigation needed | FIXED | Root cause: two bugs in mGGA stress path (see S3) |
| X2 | Per-type pseudocharge cutoff not implemented | OPEN | Uses global rc_max instead of adaptive |

---

## Fixed Issues

| # | Issue | Branch | Date | Notes |
|---|-------|--------|------|-------|
| S1 | rSCAN & r2SCAN functional support | scan | 2026-03-22 | Enum, libxc IDs, InputParser, is_mgga_type() helper |
| S3 | SCAN stress two bugs | scan | 2026-03-22 | GGA grad stress skipped for mGGA + Exc_corr missing ∫τ·vtau dV. Max err 4.3e-05 GPa |
| G1 | GPU spin SCAN test + 4 bugs | scan-gpu | 2026-03-22 | Missing spin density download, 0.5 tau factor, Vxc buffer, tau/vtau layout |
| S2 | Fe2 spin SCAN + gradient bug | scan | 2026-03-22 | mGGA stress used gradT instead of lat_uvec_inv for non-orth cells (2300 GPa error) |
