# SCAN metaGGA Functional — CPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port SPARC's hand-coded SCAN functional to LYNX, then switch to libxc. Match old SPARC to machine precision. Support non-spin, spin-polarized, gamma-point, k-point, and band parallelism. No domain parallelism. No stress (SCF energy + forces only).

**Architecture:** Three new capabilities: (1) XCFunctional gains SCAN evaluation (first hand-coded, then libxc), (2) SCF loop computes kinetic energy density tau from wavefunctions, (3) Hamiltonian gains mGGA term `-0.5 ∇·(vxcMGGA3 · ∇ψ)`. The vtau potential (vxcMGGA3) is stored in SCF and passed to Hamiltonian. Tau computation follows the same allreduce pattern as ElectronDensity::compute (band → kpt → spin).

**Tech Stack:** C++17, MPI, libxc, finite-difference gradient operators.

**Key conventions from SPARC:**
- Non-spin: `rho(Nd_d)`, `sigma(Nd_d)`, `tau(Nd_d)`
- Spin: `rho[total|up|dn]`, `sigma[total|up|dn]` (3×Nd_d), `tau[up|dn|total]` (3×Nd_d for spin, reordered)
- SCAN outputs 3 potentials: `v1 = d(nε)/dn`, `v2 = d(nε)/d|∇n| / |∇n|`, `v3 = d(nε)/dτ`
- v2 convention: SPARC divides by |∇n| internally; LYNX XCFunctional uses `Dxcdgrho = 2*vsigma` convention for GGA. For SCAN, we use the same `Dxcdgrho` convention (the divergence correction code is shared with GGA).
- The mGGA Hamiltonian term is: `H_mGGA ψ = -0.5 ∇·(v3 · ∇ψ)` — applied per wavefunction column.

**Parallelism note:** LYNX has NO domain parallelism. All grid points are local. Tau needs allreduce over band, kpt, and spin communicators (same pattern as density). vtau (v3) is computed on the same grid as Vxc and broadcast to all processes.

**Critical normalization note (tau):** LYNX's `ElectronDensity::compute` uses `spin_fac = 2.0` for non-spin (occupations don't include spin degeneracy) and `spin_fac = 1.0` for spin-polarized. Tau computation MUST follow the exact same convention. After the allreduce, divide by `dV` (matching SPARC's `vscal = 1/dV` for non-spin). For spin-polarized, SPARC applies an extra factor of 0.5 — but in LYNX, `spin_fac=1.0` already handles this correctly, so just divide by `dV`.

**Critical spin layout note (SCAN correlation):** `scanc_spin` outputs `vc2(Nd_d)` and `vc3(Nd_d)` — single arrays for total density. When combining with exchange: `vtau_up = vx3_up + vc3`, `vtau_dn = vx3_dn + vc3` (NOT `vc3[Nd_d+i]` — that would be out of bounds).

---

## File Structure

### New files
- `src/xc/SCANFunctional.hpp` — Hand-coded SCAN exchange/correlation (ported from SPARC `mGGAscan.c`)
- `src/xc/SCANFunctional.cpp` — Implementation: `scanx`, `scanc`, `scanx_spin`, `scanc_spin`

### Modified files
- `src/core/types.hpp:20` — Add `MGGA_SCAN` to `XCType` enum
- `src/xc/XCFunctional.hpp` — Add `is_mgga()`, extend `evaluate`/`evaluate_spin` signatures with optional `tau`/`vtau` params
- `src/xc/XCFunctional.cpp` — Add SCAN evaluation path (first hand-coded, then libxc), mGGA divergence correction
- `src/io/InputParser.cpp:40-46` — Parse `"SCAN"` string to `XCType::MGGA_SCAN`
- `src/physics/SCF.hpp` — Add tau array, vtau array, `compute_tau()` method
- `src/physics/SCF.cpp` — Compute tau in SCF loop, pass to XC, store vtau, apply mGGA Hamiltonian term
- `src/operators/Hamiltonian.hpp` — Add `apply_mgga()` / `apply_mgga_kpt()` methods, store vtau pointer
- `src/operators/Hamiltonian.cpp` — Implement mGGA Hamiltonian term

### Test files
- `tests/test_SCANFunctional.cpp` — Unit test: hand-coded SCAN vs known values, then vs libxc

---

## Task 1: Add MGGA_SCAN to XCType enum and input parser

**Files:**
- Modify: `src/core/types.hpp:20`
- Modify: `src/io/InputParser.cpp:40-46`
- Modify: `src/xc/XCFunctional.hpp:39-41`

- [ ] **Step 1: Add enum value**

In `src/core/types.hpp:20`, change:
```cpp
enum class XCType { LDA_PZ, LDA_PW, GGA_PBE, GGA_PBEsol, GGA_RPBE };
```
to:
```cpp
enum class XCType { LDA_PZ, LDA_PW, GGA_PBE, GGA_PBEsol, GGA_RPBE, MGGA_SCAN };
```

- [ ] **Step 2: Add input parser entry**

In `src/io/InputParser.cpp`, in `parse_xc()`, add before the throw:
```cpp
if (s == "SCAN") return XCType::MGGA_SCAN;
```

- [ ] **Step 3: Add is_mgga() to XCFunctional**

In `src/xc/XCFunctional.hpp`, add after `is_gga()`:
```cpp
bool is_mgga() const {
    return type_ == XCType::MGGA_SCAN;
}
```

- [ ] **Step 4: Update get_func_ids default case**

In `src/xc/XCFunctional.cpp:31`, the `default:` case in `get_func_ids` should not be reached for MGGA_SCAN (it uses a different path), but add a case to avoid compiler warnings:
```cpp
case XCType::MGGA_SCAN: xc_id = XC_LDA_X; cc_id = XC_LDA_C_PW; break; // placeholder — mGGA uses separate path
```

- [ ] **Step 5: Commit**

```bash
git add src/core/types.hpp src/io/InputParser.cpp src/xc/XCFunctional.hpp src/xc/XCFunctional.cpp
git commit -m "feat: add MGGA_SCAN to XCType enum and input parser"
```

---

## Task 2: Port SPARC hand-coded SCAN functional

Port the exchange and correlation functions from SPARC `mGGAscan.c` into clean C++ in a new file. Keep the math identical to SPARC for machine-precision matching.

**Files:**
- Create: `src/xc/SCANFunctional.hpp`
- Create: `src/xc/SCANFunctional.cpp`

**Source reference:** `/home/xx/Desktop/dev_SPARC_GPU/src/xc/mgga/mGGAscan.c`

- [ ] **Step 1: Create SCANFunctional.hpp**

```cpp
#pragma once

namespace lynx {

// Hand-coded SCAN metaGGA functional — ported from SPARC mGGAscan.c
// Reference: Sun, Ruzsinszky, Perdew, PRL 115, 036402 (2015)
//
// Conventions (same as SPARC):
//   Non-spin inputs:  rho(Nd_d), sigma(Nd_d) = |∇ρ|², tau(Nd_d)
//   Non-spin outputs: ex/ec(Nd_d), v1(Nd_d) = d(nε)/dn,
//                     v2(Nd_d) = d(nε)/d|∇n| / |∇n|,
//                     v3(Nd_d) = d(nε)/dτ
//
//   Spin inputs:  rho[total|up|dn](3*Nd_d), sigma[total|up|dn](3*Nd_d), tau[total|up|dn](3*Nd_d)
//   Spin exchange outputs: ex(Nd_d), vx1[up|dn](2*Nd_d), vx2[up|dn](2*Nd_d), vx3[up|dn](2*Nd_d)
//   Spin correlation outputs: ec(Nd_d), vc1[up|dn](2*Nd_d), vc2(Nd_d), vc3(Nd_d)

namespace scan {

// Non-spin-polarized
void scanx(int DMnd, const double* rho, const double* sigma, const double* tau,
           double* ex, double* vx, double* v2x, double* v3x);
void scanc(int DMnd, const double* rho, const double* sigma, const double* tau,
           double* ec, double* vc, double* v2c, double* v3c);

// Spin-polarized
// rho: [total(Nd_d) | up(Nd_d) | dn(Nd_d)]
// sigma: [|∇ρ|²(Nd_d) | |∇ρ_up|²(Nd_d) | |∇ρ_dn|²(Nd_d)]
// tau: [total(Nd_d) | up(Nd_d) | dn(Nd_d)]
void scanx_spin(int DMnd, const double* rho, const double* sigma, const double* tau,
                double* ex, double* vx, double* v2x, double* v3x);
void scanc_spin(int DMnd, const double* rho, const double* sigma, const double* tau,
                double* ec, double* vc, double* v2c, double* v3c);

} // namespace scan
} // namespace lynx
```

- [ ] **Step 2: Create SCANFunctional.cpp**

Port all functions from SPARC `mGGAscan.c` (783 lines). The math must be byte-identical. Key functions to port:
- `basic_MGGA_variables()` → internal helper
- `Calculate_scanx()` → internal helper
- `scanx()` → `scan::scanx()`
- `Calculate_scanc()` → internal helper
- `scanc()` → `scan::scanc()`
- `basic_MGSGA_variables_exchange()` → internal helper
- `Calculate_scanx_spin()` → internal helper
- `scanx_spin()` → `scan::scanx_spin()`
- `basic_MGSGA_variables_correlation()` → internal helper
- `Calculate_scanc_spin()` → internal helper
- `scanc_spin()` → `scan::scanc_spin()`

Use `std::vector` instead of `malloc`. Keep all constants, formulas, and variable names identical to SPARC for easy diffing. Make inputs `const` pointers.

- [ ] **Step 3: Add to CMakeLists.txt**

Ensure `src/xc/SCANFunctional.cpp` is included in the build. Check how other `.cpp` files in `src/xc/` are included (likely via glob or explicit list).

- [ ] **Step 4: Build and verify compilation**

```bash
cd build && cmake .. && make -j$(nproc) 2>&1 | tail -20
```

- [ ] **Step 5: Commit**

```bash
git add src/xc/SCANFunctional.hpp src/xc/SCANFunctional.cpp CMakeLists.txt
git commit -m "feat: port SPARC hand-coded SCAN exchange and correlation"
```

---

## Task 3: Write unit test — hand-coded SCAN vs known SPARC values

**Files:**
- Create: `tests/test_SCANFunctional.cpp`

- [ ] **Step 1: Write unit test**

Test the hand-coded SCAN against known reference values from SPARC. Use a small grid (e.g. 10 points) with realistic densities. Test both non-spin and spin-polarized. The test calls `scan::scanx`, `scan::scanc`, `scan::scanx_spin`, `scan::scanc_spin` directly.

Structure:
```cpp
#include <gtest/gtest.h>
#include "xc/SCANFunctional.hpp"
#include <cmath>
#include <vector>

TEST(SCANFunctional, NonSpinExchange) {
    // 5 grid points with realistic density values
    int N = 5;
    std::vector<double> rho = {0.1, 0.5, 1.0, 2.0, 5.0};
    std::vector<double> sigma(N), tau(N);
    for (int i = 0; i < N; i++) {
        double kf = std::pow(3.0*M_PI*M_PI*rho[i], 1.0/3.0);
        sigma[i] = 0.01 * rho[i] * rho[i] * kf * kf;  // small gradient
        tau[i] = 0.3 * std::pow(3.0*M_PI*M_PI, 2.0/3.0) * std::pow(rho[i], 5.0/3.0); // near uniform tau
    }
    std::vector<double> ex(N), vx(N), v2x(N), v3x(N);
    lynx::scan::scanx(N, rho.data(), sigma.data(), tau.data(),
                       ex.data(), vx.data(), v2x.data(), v3x.data());
    // Verify finite and negative exchange energy
    for (int i = 0; i < N; i++) {
        EXPECT_TRUE(std::isfinite(ex[i]));
        EXPECT_LT(ex[i], 0.0);
        EXPECT_TRUE(std::isfinite(vx[i]));
        EXPECT_TRUE(std::isfinite(v2x[i]));
        EXPECT_TRUE(std::isfinite(v3x[i]));
    }
}

TEST(SCANFunctional, NonSpinCorrelation) {
    // Similar to exchange test
    int N = 5;
    std::vector<double> rho = {0.1, 0.5, 1.0, 2.0, 5.0};
    std::vector<double> sigma(N), tau(N);
    for (int i = 0; i < N; i++) {
        double kf = std::pow(3.0*M_PI*M_PI*rho[i], 1.0/3.0);
        sigma[i] = 0.01 * rho[i] * rho[i] * kf * kf;
        tau[i] = 0.3 * std::pow(3.0*M_PI*M_PI, 2.0/3.0) * std::pow(rho[i], 5.0/3.0);
    }
    std::vector<double> ec(N), vc(N), v2c(N), v3c(N);
    lynx::scan::scanc(N, rho.data(), sigma.data(), tau.data(),
                       ec.data(), vc.data(), v2c.data(), v3c.data());
    for (int i = 0; i < N; i++) {
        EXPECT_TRUE(std::isfinite(ec[i]));
        EXPECT_LT(ec[i], 0.0);
        EXPECT_TRUE(std::isfinite(vc[i]));
    }
}

TEST(SCANFunctional, SpinExchange) {
    int N = 5;
    // rho: [total|up|dn], sigma: [|∇ρ|²|σ_up|σ_dn], tau: [total|up|dn]
    std::vector<double> rho(3*N), sigma(3*N), tau(3*N);
    for (int i = 0; i < N; i++) {
        double r = 0.1 * (i+1);
        rho[i] = r;                    // total
        rho[N+i] = 0.6 * r;           // up
        rho[2*N+i] = 0.4 * r;         // dn
        double kf = std::pow(3.0*M_PI*M_PI*r, 1.0/3.0);
        sigma[i] = 0.01 * r * r * kf * kf;
        sigma[N+i] = 0.36 * sigma[i];
        sigma[2*N+i] = 0.16 * sigma[i];
        double tau_unif = 0.3 * std::pow(3.0*M_PI*M_PI, 2.0/3.0) * std::pow(r, 5.0/3.0);
        tau[i] = tau_unif;
        tau[N+i] = 0.6 * tau_unif;
        tau[2*N+i] = 0.4 * tau_unif;
    }
    std::vector<double> ex(N), vx(2*N), v2x(2*N), v3x(2*N);
    lynx::scan::scanx_spin(N, rho.data(), sigma.data(), tau.data(),
                            ex.data(), vx.data(), v2x.data(), v3x.data());
    for (int i = 0; i < N; i++) {
        EXPECT_TRUE(std::isfinite(ex[i]));
        EXPECT_LT(ex[i], 0.0);
    }
}

TEST(SCANFunctional, SpinCorrelation) {
    int N = 5;
    std::vector<double> rho(3*N), sigma(3*N), tau(3*N);
    for (int i = 0; i < N; i++) {
        double r = 0.1 * (i+1);
        rho[i] = r;
        rho[N+i] = 0.6 * r;
        rho[2*N+i] = 0.4 * r;
        double kf = std::pow(3.0*M_PI*M_PI*r, 1.0/3.0);
        sigma[i] = 0.01 * r * r * kf * kf;
        sigma[N+i] = 0.36 * sigma[i];
        sigma[2*N+i] = 0.16 * sigma[i];
        double tau_unif = 0.3 * std::pow(3.0*M_PI*M_PI, 2.0/3.0) * std::pow(r, 5.0/3.0);
        tau[i] = tau_unif;
        tau[N+i] = 0.6 * tau_unif;
        tau[2*N+i] = 0.4 * tau_unif;
    }
    std::vector<double> ec(N), vc(2*N), v2c(N), v3c(N);
    lynx::scan::scanc_spin(N, rho.data(), sigma.data(), tau.data(),
                            ec.data(), vc.data(), v2c.data(), v3c.data());
    for (int i = 0; i < N; i++) {
        EXPECT_TRUE(std::isfinite(ec[i]));
        EXPECT_LT(ec[i], 0.0);
    }
}
```

- [ ] **Step 2: Build and run test**

```bash
cd build && make -j$(nproc) && ctest -R SCANFunctional -V
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_SCANFunctional.cpp
git commit -m "test: add SCAN functional unit tests"
```

---

## Task 4: Add hand-coded SCAN to XCFunctional evaluate/evaluate_spin

Extend `XCFunctional::evaluate()` and `evaluate_spin()` to handle `MGGA_SCAN` using the hand-coded implementation. Add optional `tau` input and `vtau` output parameters.

**Files:**
- Modify: `src/xc/XCFunctional.hpp:22-36`
- Modify: `src/xc/XCFunctional.cpp`

- [ ] **Step 1: Extend evaluate signatures**

In `XCFunctional.hpp`, change the evaluate signatures to accept optional tau and vtau:

```cpp
// Evaluate XC energy density and potential (non-spin-polarized).
// For mGGA: tau and vtau must be provided.
void evaluate(const double* rho, double* Vxc, double* exc, int Nd_d,
              double* Dxcdgrho = nullptr,
              const double* tau = nullptr,
              double* vtau = nullptr) const;

// Evaluate for spin-polarized (collinear)
// For mGGA: tau layout [total|up|dn](3*Nd_d), vtau layout [up|dn](2*Nd_d)
void evaluate_spin(const double* rho, double* Vxc, double* exc, int Nd_d,
                   double* Dxcdgrho = nullptr,
                   const double* tau = nullptr,
                   double* vtau = nullptr) const;
```

- [ ] **Step 2: Implement non-spin SCAN path in evaluate()**

In `XCFunctional.cpp`, inside `evaluate()`, add an `if (is_mgga())` branch before the GGA branch. This path:

1. Computes gradients (same as GGA — reuse the gradient code)
2. Computes sigma = |∇ρ|² (same as GGA)
3. Calls `scan::scanx()` and `scan::scanc()` to get ex, ec, vx1, vc1, vx2, vc2, vx3, vc3
4. Combines: `exc[i] = ex[i] + ec[i]`, `Vxc[i] = vx1[i] + vc1[i]`
5. Computes `Dxcdgrho[i] = vx2[i] + vc2[i]` (already divided by |∇n| from SPARC convention)
6. Applies the GGA divergence correction to Vxc (same code as GGA, using Dxcdgrho as the `v2xc`)
7. Sets `vtau[i] = vx3[i] + vc3[i]`

**Critical:** SPARC's v2 output is `d(nε)/d|∇n| / |∇n|`. The existing GGA code uses `v2xc = 2*vsigma`. These are related:
`v2_sparc = d(nε)/d|∇n| / |∇n|`
`2*vsigma_libxc = d(ρε)/d(σ) * 2 = d(nε)/d(|∇n|²) * 2`
Since `d(nε)/d|∇n| = d(nε)/d(σ) * d(σ)/d|∇n| = d(nε)/d(σ) * 2|∇n|`, we get:
`v2_sparc = 2 * d(nε)/d(σ)` = `2*vsigma` — same convention! So the existing divergence code works unchanged.

- [ ] **Step 3: Implement spin SCAN path in evaluate_spin()**

Similar to non-spin but using `scanx_spin()` and `scanc_spin()`. The input layout matches SPARC:
- rho: [total|up|dn] (3*Nd_d) — already the LYNX convention
- sigma: [|∇ρ_total|²|∇ρ_up|²|∇ρ_dn|²] (3*Nd_d) — need to compute from separate gradients
- tau: [total|up|dn] (3*Nd_d) — passed in

Output combination:
- `exc[i] = ex[i] + ec[i]`
- `Vxc[i] = vx1[i] + vc1[i]` (spin up), `Vxc[Nd_d+i] = vx1[Nd_d+i] + vc1[Nd_d+i]` (spin dn)
- `Dxcdgrho`: [vc2(Nd_d) | vx2_up(Nd_d) | vx2_dn(Nd_d)] — same as GGA spin convention
- **CRITICAL:** `vc3` and `vc2` from `scanc_spin` are single Nd_d arrays (total density derivatives). The exchange `vx3` is 2*Nd_d (per-spin). So: `vtau[i] = vx3[i] + vc3[i]` (up), `vtau[Nd_d+i] = vx3[Nd_d+i] + vc3[i]` (dn — note `vc3[i]` NOT `vc3[Nd_d+i]`)

Apply spin divergence correction (reuse the existing GGA spin divergence code — it works identically for mGGA since the v2 convention is the same).

- [ ] **Step 4: Build and run existing tests (should still pass)**

```bash
cd build && make -j$(nproc) && ctest -V 2>&1 | tail -30
```

- [ ] **Step 5: Commit**

```bash
git add src/xc/XCFunctional.hpp src/xc/XCFunctional.cpp
git commit -m "feat: integrate hand-coded SCAN into XCFunctional evaluate"
```

---

## Task 5: Add libxc SCAN path and validate against hand-coded

**Files:**
- Modify: `src/xc/XCFunctional.cpp`
- Modify: `tests/test_SCANFunctional.cpp`

- [ ] **Step 1: Add libxc SCAN evaluation path**

In `XCFunctional.cpp`, add a second mGGA code path that uses libxc:
- `xc_func_init(&func_x, XC_MGGA_X_SCAN, XC_UNPOLARIZED/XC_POLARIZED)`
- `xc_func_init(&func_c, XC_MGGA_C_SCAN, XC_UNPOLARIZED/XC_POLARIZED)`
- `xc_mgga_exc_vxc(&func_x, np, rho, sigma, lapl, tau, zk_x, vrho_x, vsigma_x, vlapl_x, vtau_x)`
- `xc_mgga_exc_vxc(&func_c, np, rho, sigma, lapl, tau, zk_c, vrho_c, vsigma_c, vlapl_c, vtau_c)`

Note: Pass `lapl = nullptr` (SCAN doesn't use laplacian), set `vlapl` output to a dummy buffer.

Keep both paths available — controlled by a compile-time or runtime flag. Initially use the hand-coded path. The libxc path validates correctness.

**libxc calling conventions (non-spin):**
- Input: `rho[np]`, `sigma[np]`, `lapl[np]` (can be NULL for SCAN), `tau[np]`
- Output: `zk[np]` (energy per particle), `vrho[np]`, `vsigma[np]`, `vlapl[np]`, `vtau[np]`

**libxc calling conventions (spin):**
- Input: `rho[2*np]` interleaved [up0,dn0,up1,dn1,...], `sigma[3*np]` interleaved [σuu,σud,σdd,...], `tau[2*np]` interleaved
- Output: `vrho[2*np]`, `vsigma[3*np]`, `vtau[2*np]` all interleaved

Map between LYNX and libxc layouts (same as GGA does already).

- [ ] **Step 2: Write comparison test — hand-coded vs libxc**

Add a test in `tests/test_SCANFunctional.cpp` that calls both the hand-coded SCAN and libxc SCAN on the same inputs and compares outputs to machine precision (~1e-12 relative error).

```cpp
TEST(SCANFunctional, HandCodedVsLibxc_NonSpin) {
    // Call hand-coded scan::scanx/scanc
    // Call libxc XC_MGGA_X_SCAN + XC_MGGA_C_SCAN via xc_mgga_exc_vxc
    // Compare: exc, vrho, vtau should match to ~1e-12
    // vsigma needs convention mapping: hand-coded v2 = 2*vsigma
}

TEST(SCANFunctional, HandCodedVsLibxc_Spin) {
    // Same comparison for spin-polarized
}
```

- [ ] **Step 3: Build and run**

```bash
cd build && make -j$(nproc) && ctest -R SCANFunctional -V
```

- [ ] **Step 4: Commit**

```bash
git add src/xc/XCFunctional.cpp tests/test_SCANFunctional.cpp
git commit -m "feat: add libxc SCAN path and validate against hand-coded"
```

---

## Task 6: Compute kinetic energy density tau in SCF

Port the tau computation from SPARC `compute_Kinetic_Density_Tau()`. tau = 0.5 * Σ_n f_n |∇ψ_n|² (non-spin) or tau_s = 0.5 * Σ_n f_n |∇ψ_{n,s}|² (spin). In LYNX, there's no domain decomposition, so no D2D transfer needed.

**Files:**
- Modify: `src/physics/SCF.hpp`
- Modify: `src/physics/SCF.cpp`

- [ ] **Step 1: Add tau storage and compute_tau declaration to SCF.hpp**

Add to private section of SCF class:
```cpp
NDArray<double> tau_;       // kinetic energy density: [Nd_d] (non-spin) or [up|dn|total](3*Nd_d) (spin)
NDArray<double> vtau_;      // d(nε)/dτ potential: [Nd_d] (non-spin) or [up|dn](2*Nd_d) (spin)

// Compute kinetic energy density tau from wavefunctions
void compute_tau(const Wavefunction& wfn,
                 const std::vector<double>& kpt_weights,
                 int kpt_start, int band_start);
```

Add public accessor:
```cpp
const double* vtau() const { return vtau_.data(); }
```

- [ ] **Step 2: Implement compute_tau()**

In `SCF.cpp`, implement `compute_tau()`. Follow the SPARC pattern but adapted for LYNX:

```cpp
void SCF::compute_tau(const Wavefunction& wfn,
                       const std::vector<double>& kpt_weights,
                       int kpt_start, int band_start) {
    int Nd_d = domain_->Nd_d();
    int Nband_loc = wfn.Nband();
    int Nband_glob = wfn.Nband_global();
    int Nspin_local = wfn.Nspin();
    int Nkpts = wfn.Nkpts();
    int FDn = gradient_->stencil().FDn();
    int nx = domain_->Nx_d(), ny = domain_->Ny_d(), nz = domain_->Nz_d();
    int nd_ex = (nx + 2*FDn) * (ny + 2*FDn) * (nz + 2*FDn);
    bool is_orth = grid_->lattice().is_orthogonal();
    const Mat3& lapcT = grid_->lattice().lapc_T();

    double spin_fac = (Nspin_global_ == 1) ? 2.0 : 1.0;

    // Allocate tau: for spin, layout is [up(Nd_d)|dn(Nd_d)|total(Nd_d)]
    int tau_size = (Nspin_global_ == 2) ? 3 * Nd_d : Nd_d;
    tau_ = NDArray<double>(tau_size);
    tau_.zero();

    // Gradient buffers
    std::vector<double> psi_ex(nd_ex);
    std::vector<double> Dpsi_x(Nd_d), Dpsi_y(Nd_d), Dpsi_z(Nd_d);

    for (int s = 0; s < Nspin_local; ++s) {
        int s_glob = spin_start_ + s;
        double* tau_s = tau_.data() + s_glob * Nd_d;  // tau for this spin channel

        for (int k = 0; k < Nkpts; ++k) {
            const auto& occ = wfn.occupations(s, k);
            double wk = kpt_weights[kpt_start + k];

            if (wfn.is_complex()) {
                // k-point: complex wavefunctions
                std::vector<Complex> psi_ex_c(nd_ex);
                std::vector<Complex> Dpsi_x_c(Nd_d), Dpsi_y_c(Nd_d), Dpsi_z_c(Nd_d);
                const auto& psi_c = wfn.psi_kpt(s, k);
                int k_glob = kpt_start + k;
                Vec3 kpt = kpoints_->kpts_cart()[k_glob];
                Vec3 cell_lengths = grid_->lattice().lengths();

                for (int n = 0; n < Nband_loc; ++n) {
                    double fn = occ(band_start + n);
                    if (fn < 1e-16) continue;
                    double g_nk = spin_fac * wk * fn;

                    const Complex* col = psi_c.col(n);
                    halo_->execute_kpt(col, psi_ex_c.data(), 1, kpt, cell_lengths);
                    gradient_->apply(psi_ex_c.data(), Dpsi_x_c.data(), 0, 1);
                    gradient_->apply(psi_ex_c.data(), Dpsi_y_c.data(), 1, 1);
                    gradient_->apply(psi_ex_c.data(), Dpsi_z_c.data(), 2, 1);

                    for (int i = 0; i < Nd_d; ++i) {
                        if (is_orth) {
                            tau_s[i] += g_nk * (std::norm(Dpsi_x_c[i]) + std::norm(Dpsi_y_c[i]) + std::norm(Dpsi_z_c[i]));
                        } else {
                            // Non-orthogonal: tau = Σ conj(Dψ_a) * lapcT(a,b) * Dψ_b
                            Complex dx = Dpsi_x_c[i], dy = Dpsi_y_c[i], dz = Dpsi_z_c[i];
                            Complex Lx = lapcT(0,0)*dx + lapcT(0,1)*dy + lapcT(0,2)*dz;
                            Complex Ly = lapcT(1,0)*dx + lapcT(1,1)*dy + lapcT(1,2)*dz;
                            Complex Lz = lapcT(2,0)*dx + lapcT(2,1)*dy + lapcT(2,2)*dz;
                            tau_s[i] += g_nk * std::real(std::conj(dx)*Lx + std::conj(dy)*Ly + std::conj(dz)*Lz);
                        }
                    }
                }
            } else {
                // Gamma-point: real wavefunctions
                const auto& psi = wfn.psi(s, k);
                for (int n = 0; n < Nband_loc; ++n) {
                    double fn = occ(band_start + n);
                    if (fn < 1e-16) continue;
                    double g_nk = spin_fac * wk * fn;

                    const double* col = psi.col(n);
                    halo_->execute(col, psi_ex.data(), 1);
                    gradient_->apply(psi_ex.data(), Dpsi_x.data(), 0, 1);
                    gradient_->apply(psi_ex.data(), Dpsi_y.data(), 1, 1);
                    gradient_->apply(psi_ex.data(), Dpsi_z.data(), 2, 1);

                    for (int i = 0; i < Nd_d; ++i) {
                        if (is_orth) {
                            tau_s[i] += g_nk * (Dpsi_x[i]*Dpsi_x[i] + Dpsi_y[i]*Dpsi_y[i] + Dpsi_z[i]*Dpsi_z[i]);
                        } else {
                            double dx = Dpsi_x[i], dy = Dpsi_y[i], dz = Dpsi_z[i];
                            double Lx = lapcT(0,0)*dx + lapcT(0,1)*dy + lapcT(0,2)*dz;
                            double Ly = lapcT(1,0)*dx + lapcT(1,1)*dy + lapcT(1,2)*dz;
                            double Lz = lapcT(2,0)*dx + lapcT(2,1)*dy + lapcT(2,2)*dz;
                            tau_s[i] += g_nk * (dx*Lx + dy*Ly + dz*Lz);
                        }
                    }
                }
            }
        }
    }

    // Allreduce over band communicator
    if (!bandcomm_->is_null() && bandcomm_->size() > 1) {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start_ + s;
            bandcomm_->allreduce_sum(tau_.data() + s_glob * Nd_d, Nd_d);
        }
    }

    // Allreduce over kpt communicator
    if (!kptcomm_->is_null() && kptcomm_->size() > 1) {
        for (int s = 0; s < Nspin_local; ++s) {
            int s_glob = spin_start_ + s;
            kptcomm_->allreduce_sum(tau_.data() + s_glob * Nd_d, Nd_d);
        }
    }

    // Exchange spin channels across spin communicator
    if (spincomm_ && !spincomm_->is_null() && spincomm_->size() > 1 && Nspin_global_ == 2) {
        int my_spin = spin_start_;
        int other_spin = 1 - my_spin;
        int partner = (spincomm_->rank() == 0) ? 1 : 0;
        MPI_Sendrecv(tau_.data() + my_spin * Nd_d, Nd_d, MPI_DOUBLE, partner, 0,
                     tau_.data() + other_spin * Nd_d, Nd_d, MPI_DOUBLE, partner, 0,
                     spincomm_->comm(), MPI_STATUS_IGNORE);
    }

    // Scale by 1/dV. The spin_fac (2.0 for non-spin, 1.0 for spin) was already
    // applied in g_nk above, matching ElectronDensity::compute exactly.
    // This matches SPARC: vscal=1/dV for non-spin, vscal=0.5/dV for spin
    // (SPARC's 0.5 is because SPARC's spin loop covers both spinors but
    //  LYNX already uses spin_fac=1.0 for spin-polarized).
    double vscal = 1.0 / grid_->dV();
    if (Nspin_global_ == 2) {
        for (int i = 0; i < 2 * Nd_d; ++i) tau_.data()[i] *= vscal;
        // Compute total tau = tau_up + tau_dn
        double* tau_up = tau_.data();
        double* tau_dn = tau_.data() + Nd_d;
        double* tau_tot = tau_.data() + 2 * Nd_d;
        for (int i = 0; i < Nd_d; ++i) tau_tot[i] = tau_up[i] + tau_dn[i];
    } else {
        for (int i = 0; i < Nd_d; ++i) tau_.data()[i] *= vscal;
    }
}
```

**Important: verify the SPARC normalization carefully.** SPARC uses `g_nk = occ[n]` for gamma (no kpt weight) and `g_nk = (kptWts/Nkpts) * occ[n]` for k-points. LYNX's occupations from `Occupation::compute` already follow the same convention. The final `1/dV` scaling matches SPARC.

- [ ] **Step 3: Integrate tau computation into SCF loop**

In `SCF::run()`, after density is computed (step 2 in the loop, around line 672-682), add tau computation for mGGA:

```cpp
// Compute tau for mGGA
if (xc_type_ == XCType::MGGA_SCAN) {
    compute_tau(wfn, kpt_weights, kpt_start_, band_start_);
}
```

Also allocate vtau before the SCF loop starts:
```cpp
if (xc_type_ == XCType::MGGA_SCAN) {
    int vtau_size = (Nspin_global_ == 2) ? 2 * Nd_d : Nd_d;
    vtau_ = NDArray<double>(vtau_size);
}
```

- [ ] **Step 4: Pass tau and vtau to XC evaluation in compute_Veff**

In `SCF::compute_Veff()`, modify the XC evaluation calls to pass tau/vtau when mGGA:

```cpp
const double* tau_ptr = xc.is_mgga() ? tau_.data() : nullptr;
double* vtau_ptr = xc.is_mgga() ? vtau_.data() : nullptr;

// Allocate Dxcdgrho for mGGA too (it has gradient dependence)
if ((xc.is_gga() || xc.is_mgga()) && Dxcdgrho_.size() == 0) {
    Dxcdgrho_ = NDArray<double>(Nd_d * dxc_ncol);
}
double* dxc_ptr = (xc.is_gga() || xc.is_mgga()) ? Dxcdgrho_.data() : nullptr;

// Then pass tau_ptr, vtau_ptr to evaluate/evaluate_spin calls
```

- [ ] **Step 5: Compute tau BEFORE density mixing (initial + each SCF step)**

Tau depends on the current wavefunctions, not the mixed density. Compute it after the CheFSI solve but before density update. Actually, tau is needed for computing Veff from the NEXT iteration. Looking at SPARC's flow:

SPARC computes tau at the start of each SCF iteration (before Veff), using the current wavefunctions. In LYNX, compute tau:
1. Before the first `compute_Veff` (initial tau from initial wavefunctions)
2. After each CheFSI solve, before the next `compute_Veff`

Add initial tau computation before the SCF loop (after wavefunction randomization):
```cpp
if (xc_type_ == XCType::MGGA_SCAN) {
    compute_tau(wfn, kpt_weights, kpt_start_, band_start_);
}
```

And in the loop, compute tau after the CheFSI solve and occupations (before density + Veff update).

- [ ] **Step 6: Build and verify**

```bash
cd build && make -j$(nproc) 2>&1 | tail -20
```

- [ ] **Step 7: Commit**

```bash
git add src/physics/SCF.hpp src/physics/SCF.cpp
git commit -m "feat: compute kinetic energy density tau in SCF loop"
```

---

## Task 7: Add mGGA Hamiltonian term

The mGGA adds a term to the Hamiltonian: `H_mGGA ψ = -0.5 ∇·(vtau · ∇ψ)`. This is applied per wavefunction column. Port from SPARC `mGGAhamiltonianTerm.c`.

**Files:**
- Modify: `src/operators/Hamiltonian.hpp`
- Modify: `src/operators/Hamiltonian.cpp`

- [ ] **Step 1: Add mGGA methods to Hamiltonian.hpp**

```cpp
// Set vtau potential for mGGA Hamiltonian term
void set_vtau(const double* vtau) { vtau_ = vtau; }
const double* vtau() const { return vtau_; }

// Apply mGGA term: H_mGGA ψ = -0.5 ∇·(vtau · ∇ψ), added to y
void apply_mgga(const double* psi, double* y, int ncol) const;
void apply_mgga_kpt(const Complex* psi, Complex* y, int ncol,
                     const Vec3& kpt_cart, const Vec3& cell_lengths) const;
```

Add to private:
```cpp
const double* vtau_ = nullptr;
```

- [ ] **Step 2: Implement apply_mgga (gamma-point)**

```cpp
void Hamiltonian::apply_mgga(const double* psi, double* y, int ncol) const {
    if (!vtau_) return;
    int Nd_d = domain_->Nx_d() * domain_->Ny_d() * domain_->Nz_d();
    int FDn = stencil_->FDn();
    int nx = domain_->Nx_d(), ny = domain_->Ny_d(), nz = domain_->Nz_d();
    int nd_ex = (nx + 2*FDn) * (ny + 2*FDn) * (nz + 2*FDn);
    bool is_orth = domain_->global_grid().lattice().is_orthogonal();
    const Mat3& lapcT = domain_->global_grid().lattice().lapc_T();

    std::vector<double> psi_ex(nd_ex);
    std::vector<double> Dpsi_x(Nd_d), Dpsi_y(Nd_d), Dpsi_z(Nd_d);
    std::vector<double> vD_ex(nd_ex), divVD(Nd_d);

    Gradient grad(*stencil_, *domain_);

    for (int n = 0; n < ncol; ++n) {
        const double* psi_n = psi + n * Nd_d;

        // 1. Compute ∇ψ
        halo_->execute(psi_n, psi_ex.data(), 1);
        grad.apply(psi_ex.data(), Dpsi_x.data(), 0, 1);
        grad.apply(psi_ex.data(), Dpsi_y.data(), 1, 1);
        grad.apply(psi_ex.data(), Dpsi_z.data(), 2, 1);

        // 2. Apply metric tensor for non-orthogonal cells, then multiply by vtau
        if (!is_orth) {
            for (int i = 0; i < Nd_d; ++i) {
                double dx = Dpsi_x[i], dy = Dpsi_y[i], dz = Dpsi_z[i];
                Dpsi_x[i] = (lapcT(0,0)*dx + lapcT(0,1)*dy + lapcT(0,2)*dz) * vtau_[i];
                Dpsi_y[i] = (lapcT(1,0)*dx + lapcT(1,1)*dy + lapcT(1,2)*dz) * vtau_[i];
                Dpsi_z[i] = (lapcT(2,0)*dx + lapcT(2,1)*dy + lapcT(2,2)*dz) * vtau_[i];
            }
        } else {
            for (int i = 0; i < Nd_d; ++i) {
                Dpsi_x[i] *= vtau_[i];
                Dpsi_y[i] *= vtau_[i];
                Dpsi_z[i] *= vtau_[i];
            }
        }

        // 3. Compute ∇·(vtau·∇ψ)
        halo_->execute(Dpsi_x.data(), vD_ex.data(), 1);
        grad.apply(vD_ex.data(), divVD.data(), 0, 1);
        for (int i = 0; i < Nd_d; ++i) y[n*Nd_d + i] -= 0.5 * divVD[i];

        halo_->execute(Dpsi_y.data(), vD_ex.data(), 1);
        grad.apply(vD_ex.data(), divVD.data(), 1, 1);
        for (int i = 0; i < Nd_d; ++i) y[n*Nd_d + i] -= 0.5 * divVD[i];

        halo_->execute(Dpsi_z.data(), vD_ex.data(), 1);
        grad.apply(vD_ex.data(), divVD.data(), 2, 1);
        for (int i = 0; i < Nd_d; ++i) y[n*Nd_d + i] -= 0.5 * divVD[i];
    }
}
```

- [ ] **Step 3: Implement apply_mgga_kpt (k-point, complex)**

Same structure but using complex types and `execute_kpt` / complex gradient.

- [ ] **Step 4: Wire mGGA term into Hamiltonian::apply and apply_kpt**

In `Hamiltonian::apply()`, add after the nonlocal term:
```cpp
if (vtau_) {
    apply_mgga(psi, y, ncol);
}
```

In `Hamiltonian::apply_kpt()`, add after the nonlocal term:
```cpp
if (vtau_) {
    apply_mgga_kpt(psi, y, ncol, kpt_cart, cell_lengths);
}
```

- [ ] **Step 5: Set vtau on Hamiltonian from SCF**

In `SCF::compute_Veff()`, after XC evaluation, set vtau on the Hamiltonian:
```cpp
if (xc_type_ == XCType::MGGA_SCAN) {
    // vtau for the Hamiltonian: use the appropriate spin channel's vtau
    // For non-spin: vtau is just vtau_[Nd_d]
    // For spin: Hamiltonian gets vtau for the current spin channel
    const_cast<Hamiltonian*>(hamiltonian_)->set_vtau(vtau_.data());
}
```

For spin-polarized, the vtau pointer must be set per-spin before each CheFSI solve:
```cpp
// In the spin loop of the eigensolver section:
if (xc_type_ == XCType::MGGA_SCAN) {
    const_cast<Hamiltonian*>(hamiltonian_)->set_vtau(vtau_.data() + s_glob * Nd_d);
}
```

- [ ] **Step 6: Build and verify**

```bash
cd build && make -j$(nproc) 2>&1 | tail -20
```

- [ ] **Step 7: Commit**

```bash
git add src/operators/Hamiltonian.hpp src/operators/Hamiltonian.cpp src/physics/SCF.cpp
git commit -m "feat: add mGGA Hamiltonian term and wire vtau in SCF"
```

---

## Task 8: End-to-end test — compare LYNX SCAN vs SPARC SCAN

**Files:**
- Create test input JSON for a small system (e.g., Si4 with k-points or Si2 gamma-only)
- Compare total energy, eigenvalues, forces against SPARC reference

- [ ] **Step 1: Create a SCAN test input**

Use a small Si system with SCAN functional. Create a JSON config with `"xc": "SCAN"` and appropriate grid/k-point settings matching an existing SPARC test case.

Check SPARC test data at: `/home/xx/Desktop/dev_SPARC_GPU/tests/xc/mgga_tests/`

- [ ] **Step 2: Run LYNX with SCAN and capture output**

```bash
mpirun -np 1 ./lynx Si_SCAN.json 2>&1 | tee scan_output.log
```

- [ ] **Step 3: Compare against SPARC reference**

Compare total energy, eigenvalues, and forces. Target: machine precision match (~1e-10 to 1e-12 relative error for energy, ~1e-8 for forces).

- [ ] **Step 4: Debug any discrepancies**

If there are differences:
1. Check tau normalization (SPARC vs LYNX dV scaling)
2. Check sigma convention (|∇ρ|² vs |∇ρ|)
3. Check v2 convention (divided by |∇ρ| or not)
4. Compare intermediate quantities: tau, sigma, exc, Vxc, vtau at each grid point

- [ ] **Step 5: Commit working test**

```bash
git add tests/ examples/
git commit -m "test: add SCAN end-to-end test matching SPARC reference"
```

---

## Task 9: Switch default to libxc, move hand-coded to backup

**Files:**
- Modify: `src/xc/XCFunctional.cpp` — make libxc the default SCAN path
- Rename: `src/xc/SCANFunctional.cpp` → `src/xc/SCANFunctional.cpp.bak`
- Rename: `src/xc/SCANFunctional.hpp` → `src/xc/SCANFunctional.hpp.bak`

- [ ] **Step 1: Switch evaluate/evaluate_spin to use libxc for SCAN**

Remove the hand-coded call path from evaluate/evaluate_spin. The libxc path becomes the only active path.

- [ ] **Step 2: Rename hand-coded files to .bak**

```bash
mv src/xc/SCANFunctional.cpp src/xc/SCANFunctional.cpp.bak
mv src/xc/SCANFunctional.hpp src/xc/SCANFunctional.hpp.bak
```

Update CMakeLists.txt to remove SCANFunctional.cpp from the build.

- [ ] **Step 3: Run all tests — must still pass**

```bash
cd build && cmake .. && make -j$(nproc) && ctest -V
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: switch SCAN to libxc, move hand-coded to backup"
```

---

## Verification Checklist

- [ ] Hand-coded SCAN x/c matches libxc to ~1e-12 (unit test)
- [ ] Non-spin SCAN SCF converges and matches SPARC energy to ~1e-10
- [ ] Spin-polarized SCAN SCF converges and matches SPARC
- [ ] K-point SCAN SCF converges and matches SPARC
- [ ] Band-parallel SCAN gives identical results to serial
- [ ] Kpt-parallel SCAN gives identical results to serial
- [ ] Spin-parallel SCAN gives identical results to serial
- [ ] Forces match SPARC to ~1e-8
- [ ] After switching to libxc, all above still hold
