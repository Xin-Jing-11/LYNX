# SOC Debug Status

## Root Cause Found

**The SOC Chi projectors use REAL spherical harmonics but SPARC uses COMPLEX spherical harmonics.**

The L·S operator with Term 1 (Lz·Sz) and Term 2 (L±S∓ ladder) is formulated in the **complex Ylm basis**. Using real Ylm gives wrong coupling coefficients because:
- Real Y_l^m is a linear combination of complex Y_l^m and Y_l^{-m}
- The m-dependent factor in Term 1 and the ladder coefficients in Term 2 only work with complex Ylm

## Fix Required

1. Implement `complex_spherical_harmonic(l, m, x, y, z, r)` in math_utils.hpp
2. Change `Chi_soc_` from `NDArray<double>` to `NDArray<Complex>`
3. Update `setup_soc()` to use complex Ylm
4. Update `apply_soc_kpt()` for complex Chi (the inner product becomes Chi^H * psi, not Chi^T * psi)

## Verification Summary

| Component | max_rel_err | Status |
|-----------|-------------|--------|
| Local (Lap + Veff) | 1.3e-15 | PASS |
| Scalar VNL (real Ylm) | 8.2e-16 | PASS |
| SOC (real Ylm - WRONG) | 1.39 (139%) | FAIL |
