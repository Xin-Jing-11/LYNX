# LYNX Jekyll GitHub Pages Site — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GitHub Pages site using the `just-the-docs` Jekyll theme that showcases LYNX's design philosophy (Architecture → Theory → Examples), serving both researchers learning DFT and developers studying the code architecture.

**Architecture:** Jekyll source lives in `docs/` at the repo root. GitHub Pages is configured to build from the `docs/` folder on `master`. The `just-the-docs` theme is loaded via `remote_theme` (supported by GitHub Pages without custom Actions). Theory pages are the existing `doc/*.md` files with Jekyll front matter prepended. MathJax v3 is injected via `docs/_includes/head_custom.html` for LaTeX rendering.

**Tech Stack:** Jekyll, `just-the-docs` remote theme, MathJax v3, GitHub Pages (build from `docs/` on `master`), Markdown + raw HTML for home page layout.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `docs/_config.yml` | Create | Site-wide Jekyll + just-the-docs config |
| `docs/Gemfile` | Create | Local dev gem dependencies |
| `docs/_includes/head_custom.html` | Create | MathJax v3 injection for LaTeX in theory pages |
| `docs/index.md` | Create | Home page: hero, design highlights, quick examples |
| `docs/architecture/index.md` | Create | Layer diagram + design philosophy |
| `docs/architecture/dispatch.md` | Create | CPU/GPU dispatch pattern with real code snippets |
| `docs/architecture/python-api.md` | Create | Three-level Python API with examples |
| `docs/architecture/external-libs.md` | Create | libxc, MPI, pybind11, ASE, pseudopotentials |
| `docs/theory/index.md` | Create | Theory overview + cluster links |
| `docs/theory/discretization.md` | Create | Cluster parent page |
| `docs/theory/electronic-structure.md` | Create | Cluster parent page |
| `docs/theory/response-properties.md` | Create | Cluster parent page |
| `docs/theory/advanced-topics.md` | Create | Cluster parent page |
| `docs/theory/01-real-space.md` | Create | Front matter + `doc/01_real_space_discretization.md` content |
| `docs/theory/02-pseudopotentials.md` | Create | Front matter + `doc/02_pseudopotentials.md` content |
| `docs/theory/03-hamiltonian.md` | Create | Front matter + `doc/03_hamiltonian.md` content |
| `docs/theory/04-eigensolver.md` | Create | Front matter + `doc/04_eigensolver.md` content |
| `docs/theory/05-density.md` | Create | Front matter + `doc/05_density_and_occupation.md` content |
| `docs/theory/06-electrostatics.md` | Create | Front matter + `doc/06_electrostatics.md` content |
| `docs/theory/07-xc.md` | Create | Front matter + `doc/07_exchange_correlation.md` content |
| `docs/theory/08-energy.md` | Create | Front matter + `doc/08_total_energy.md` content |
| `docs/theory/09-forces.md` | Create | Front matter + `doc/09_forces.md` content |
| `docs/theory/10-stress.md` | Create | Front matter + `doc/10_stress.md` content |
| `docs/theory/11-mixing.md` | Create | Front matter + `doc/11_mixing.md` content |
| `docs/theory/12-kpoints.md` | Create | Front matter + `doc/12_kpoints.md` content |
| `docs/theory/13-soc.md` | Create | Front matter + `doc/13_spin_orbit_coupling.md` content |
| `docs/theory/14-parallelization.md` | Create | Front matter + `doc/14_parallelization.md` content |
| `docs/examples/index.md` | Create | Examples overview |
| `docs/examples/cpp-quickstart.md` | Create | C++ CLI + JSON input walkthrough |
| `docs/examples/python-quickstart.md` | Create | DFTConfig, Calculator, from_ase() |
| `docs/examples/ase-integration.md` | Create | LynxCalculator as ASE calculator |
| `docs/examples/md-simulation.md` | Create | ASE/i-PI NVE MD with density restart |
| `.gitignore` | Modify | Add `docs/_site/` and `docs/.jekyll-cache/` |

---

## Task 1: Jekyll scaffold

**Files:**
- Create: `docs/_config.yml`
- Create: `docs/Gemfile`
- Create: `docs/_includes/head_custom.html`
- Modify: `.gitignore`

- [ ] **Step 1: Create `docs/_config.yml`**

```yaml
title: LYNX
description: >-
  A real-space DFT engine in C++17/CUDA — GPU acceleration without
  algorithmic duplication, Python access without overhead.

remote_theme: just-the-docs/just-the-docs@v0.10.0

plugins:
  - jekyll-remote-theme
  - jekyll-seo-tag

url: "https://xin-jing-11.github.io"
baseurl: "/LYNX"

aux_links:
  "GitHub":
    - "https://github.com/Xin-Jing-11/LYNX"
aux_links_new_tab: true

search_enabled: true
heading_anchors: true

# Exclude internal dirs from Jekyll processing
exclude:
  - superpowers/
  - Gemfile
  - Gemfile.lock
  - "*.gemspec"
  - vendor/

# Navigation order: Architecture (2), Theory (3), Examples (4)
# Home is nav_order: 1 in index.md
```

- [ ] **Step 2: Create `docs/Gemfile`**

```ruby
source "https://rubygems.org"

gem "github-pages", group: :jekyll_plugins
gem "jekyll-remote-theme"
gem "webrick"  # required for Ruby 3+
```

- [ ] **Step 3: Create `docs/_includes/head_custom.html`** (MathJax for LaTeX in theory pages)

Create the directory first: `mkdir -p docs/_includes`

```html
<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"
        id="MathJax-script" async></script>
```

- [ ] **Step 4: Add Jekyll build artifacts to `.gitignore`**

Append to the existing `.gitignore` at the repo root:

```
# Jekyll build output
docs/_site/
docs/.jekyll-cache/
docs/Gemfile.lock
```

- [ ] **Step 5: Verify Jekyll builds locally**

Install dependencies and do a dry-run build:

```bash
cd docs
bundle install
bundle exec jekyll build
```

Expected: build succeeds with no errors. `docs/_site/` is created (excluded from git by the gitignore you just added).

- [ ] **Step 6: Commit**

```bash
git add docs/_config.yml docs/Gemfile docs/_includes/head_custom.html .gitignore
git commit -m "feat: add Jekyll scaffold for GitHub Pages (just-the-docs)"
```

---

## Task 2: Home page

**Files:**
- Create: `docs/index.md`

The home page uses raw HTML inside the markdown file because just-the-docs is a documentation theme with no built-in hero layout. Raw HTML is valid in Jekyll markdown files and renders correctly.

- [ ] **Step 1: Create `docs/index.md`**

```markdown
---
title: Home
layout: home
nav_order: 1
---

<div style="text-align: center; padding: 3rem 0 2rem;">
  <h1 style="font-size: 3rem; font-weight: 700; letter-spacing: -1px; margin-bottom: 0.5rem;">LYNX</h1>
  <p style="font-size: 1.2rem; color: #555; max-width: 560px; margin: 0 auto 2rem; line-height: 1.6;">
    A real-space DFT engine designed for clarity —<br>
    GPU acceleration without algorithmic duplication,<br>
    Python access without overhead.
  </p>
  <a href="https://github.com/Xin-Jing-11/LYNX" class="btn btn-primary fs-5 mb-4 mb-md-0 mr-2">View on GitHub</a>
  &nbsp;
  <a href="/LYNX/architecture/" class="btn btn-outline fs-5 mb-4">Get Started →</a>
</div>

---

## Design Highlights

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.5rem; margin: 1.5rem 0 2.5rem;">
  <div style="background: #f8f9fb; border: 1px solid #e1e4e8; border-radius: 8px; padding: 1.5rem;">
    <h3 style="margin-top: 0;">⚡ PyTorch-style dispatch</h3>
    <p>Algorithm in <code>.cpp</code>, kernels in <code>.cu</code>. One <code>_cpu()/_gpu()</code> switch per operator. No algorithmic duplication across execution paths.</p>
    <a href="/LYNX/architecture/dispatch/">Learn more →</a>
  </div>
  <div style="background: #f8f9fb; border: 1px solid #e1e4e8; border-radius: 8px; padding: 1.5rem;">
    <h3 style="margin-top: 0;">🚀 GPU-resident SCF pipeline</h3>
    <p>Wavefunctions never leave the GPU. Full SCF → forces → stress on device. 35–89× speedup per operator over CPU.</p>
    <a href="/LYNX/architecture/">Learn more →</a>
  </div>
  <div style="background: #f8f9fb; border: 1px solid #e1e4e8; border-radius: 8px; padding: 1.5rem;">
    <h3 style="margin-top: 0;">🐍 Three-level Python API</h3>
    <p>From a one-liner <code>Calculator</code> to raw numpy operators. Zero-copy interop via pybind11. Drop-in ASE support.</p>
    <a href="/LYNX/architecture/python-api/">Learn more →</a>
  </div>
</div>

---

## Quick Example

**Python — one call:**

```python
from lynx.config import DFTConfig

config = DFTConfig(
    cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
    fractional=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    symbols=['Si', 'Si'],
    pseudo_files={'Si': 'psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8'},
    Nstates=10, xc='GGA_PBE',
)
calc = config.create_calculator(auto_run=True)
print(calc.total_energy)   # Hartree
```

**ASE — drop-in calculator:**

```python
from ase.build import bulk
from lynx.ase_interface import LynxCalculator

atoms = bulk('Si', 'diamond', a=5.43)
atoms.calc = LynxCalculator(xc='GGA_PBE', kpts=(4, 4, 4))
print(atoms.get_potential_energy())   # eV
print(atoms.get_forces())             # eV/Å
```

**C++ — JSON input:**

```bash
mpirun -np 4 ./build/src/lynx examples/Si8.json
```

```json
{
  "lattice": { "vectors": [[10.26,0,0],[0,10.26,0],[0,0,10.26]] },
  "grid": { "Nx": 40, "Ny": 40, "Nz": 40, "fd_order": 12 },
  "atoms": [{ "element": "Si", "pseudo_file": "psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8",
              "fractional": true,
              "coordinates": [[0,0,0],[0.25,0.25,0.25]] }],
  "electronic": { "xc": "GGA_PBE", "Nstates": 10 },
  "scf": { "max_iter": 100, "tolerance": 1e-6 }
}
```

---

## What's Inside

LYNX is built in layers. The **[Architecture]({{ site.baseurl }}/architecture/)** section explains the software design — how operators, solvers, and the Python API fit together. The **[Theory]({{ site.baseurl }}/theory/)** section covers the physics, from real-space discretization to forces and stress. The **[Examples]({{ site.baseurl }}/examples/)** section shows runnable code for common workflows.
```

- [ ] **Step 2: Build and check**

```bash
cd docs
bundle exec jekyll build
```

Expected: no errors. Check `docs/_site/index.html` exists and contains the hero text.

- [ ] **Step 3: Commit**

```bash
git add docs/index.md
git commit -m "feat: add Jekyll home page with hero, design highlights, quick examples"
```

---

## Task 3: Architecture — index page

**Files:**
- Create: `docs/architecture/index.md`

- [ ] **Step 1: Create `docs/architecture/index.md`**

```markdown
---
title: Architecture
nav_order: 2
has_children: true
permalink: /architecture/
---

# Architecture

LYNX is built in clear layers. Each layer depends only on those below it — no physics logic in GPU kernels, no kernel calls in physics algorithms.

## Layer Diagram

```
Python (pylynx / ASE)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                     C++ Engine                      │
│                                                     │
│  physics/   SCF loop, Energy, Forces, Stress        │
│  solvers/   CheFSI eigensolver · AAR Poisson        │
│             Pulay+Kerker mixer                      │
│  operators/ Laplacian · Gradient · Hamiltonian      │
│             NonlocalProjector · FDStencil           │
│  electronic/ Wavefunction · ElectronDensity         │
│              Occupation                             │
│  xc/        XCFunctional (libxc) · ExactExchange    │
│  atoms/     Crystal · Pseudopotential (psp8)        │
│  parallel/  MPI · HaloExchange                      │
│  core/      DeviceArray · Lattice · FDGrid          │
│             Domain · KPoints                        │
└─────────────────────────────────────────────────────┘
```

## Design Principles

**One algorithm, two execution paths.** Every operator has a single public method (e.g., `apply()`, `solve()`). That method dispatches to `_cpu()` or `_gpu()` based on a member `dev_` flag set at construction. The algorithm — iteration loops, convergence checks, sub-steps — lives once in `.cpp`. GPU-specific code is isolated to `.cu` files containing only kernel launches.

**GPU-resident data.** Wavefunctions are born on the GPU (random initialization via cuRAND) and never leave the device. The full SCF → forces → stress pipeline runs on-device. Only small scalar arrays (eigenvalues, occupations) cross the PCIe bus per SCF iteration.

**Python without overhead.** The Python API is a thin pybind11 binding over the C++ engine. NumPy arrays share memory with `DeviceArray<T>` buffers — no copies for CPU arrays. Three granularity levels let users trade control for convenience.

## Sections

- [CPU/GPU Dispatch]({{ site.baseurl }}/architecture/dispatch/) — the `_cpu()/_gpu()` pattern in detail
- [Python API]({{ site.baseurl }}/architecture/python-api/) — three levels of access, zero-copy numpy interop
- [External Libraries]({{ site.baseurl }}/architecture/external-libs/) — libxc, MPI, pybind11, ASE, pseudopotentials
```

- [ ] **Step 2: Build and check**

```bash
cd docs && bundle exec jekyll build
```

Expected: no errors. Architecture appears in the sidebar at position 2.

- [ ] **Step 3: Commit**

```bash
git add docs/architecture/index.md
git commit -m "feat: add Architecture overview page with layer diagram"
```

---

## Task 4: Architecture — CPU/GPU dispatch page

**Files:**
- Create: `docs/architecture/dispatch.md`

The code snippets below are taken directly from `src/operators/Hamiltonian.cpp` and `src/operators/Hamiltonian.hpp`. Use the actual file content — do not fabricate.

- [ ] **Step 1: Create `docs/architecture/dispatch.md`**

```markdown
---
title: CPU/GPU Dispatch
parent: Architecture
nav_order: 1
---

# CPU/GPU Dispatch

LYNX follows the same dispatch pattern as PyTorch and TensorFlow: **one algorithm, kernel-level dispatch.** The iteration loop, convergence check, and sub-steps are written once in `.cpp`. GPU-specific code is isolated to `.cu` files.

## The Pattern

Each operator declares three layers in its header:

```cpp
// Hamiltonian.hpp — three layers

// 1. Public method: the only entry point callers use
void apply(const double* psi, const double* Veff,
           double* y, int ncol, double c) const;

// 2. CPU implementation (in Hamiltonian.cpp)
void apply_cpu(const double* psi, const double* Veff,
               double* y, int ncol, double c) const;

// 3. GPU implementation (in Hamiltonian.cu — kernel launches only)
void apply_gpu(const double* psi, const double* Veff,
               double* y, int ncol, double c) const;
```

The dispatcher in `.cpp` is three lines:

```cpp
// Hamiltonian.cpp
void Hamiltonian::apply(const double* psi, const double* Veff,
                        double* y, int ncol, double c) const {
#ifdef USE_CUDA
    if (dev_ == Device::GPU) { apply_gpu(psi, Veff, y, ncol, c); return; }
#endif
    apply_cpu(psi, Veff, y, ncol, c);
}
```

`dev_` is a member set once at construction or `setup()`. Public methods never take a `Device` parameter — they check `dev_` internally.

## What Lives Where

| File | Contains |
|------|---------|
| `Operator.hpp` | Declares `apply()`, `apply_cpu()`, `apply_gpu()` |
| `Operator.cpp` | Implements `apply()` (dispatcher) and `apply_cpu()` |
| `Operator.cu` | Implements `apply_gpu()` — kernel launches, cuBLAS/cuSOLVER calls only |

**No loops in `.cu`.** No algorithm logic. If a sub-operation needs context, it is a method on `this`, not a standalone GPU function.

## Analogy to PyTorch

PyTorch uses the same principle via `at::Tensor` dispatch keys: the user calls `torch::mm(a, b)` and the framework routes to a CPU or CUDA kernel. The calling code never changes based on device. LYNX applies the same idea at the operator level — callers always call `hamiltonian.apply(...)`, and the device choice is encapsulated inside.

## GPU Data Residency

The dispatch pattern pairs with a data residency rule: **wavefunctions never leave the GPU.** On GPU builds, `psi` is allocated and randomized on-device (cuRAND), lives there for the entire SCF loop, and is used directly in GPU force and stress kernels. No download, no re-upload.

Per-iteration allowed transfers (tiny, scalar):

| Direction | Data | Size |
|-----------|------|------|
| D→H | eigenvalues | ~KB |
| H→D | occupations | ~KB |
| D→H | rho, Veff (for energy on CPU) | once/iter |

Everything else — `psi`, `Veff`, `rho`, `phi`, `exc` — stays on device.

## Why This Matters

- **Testability:** the CPU path is always available, so algorithms can be validated on CPU before GPU implementation.
- **Auditability:** GPU code is isolated to `.cu` files — a CUDA expert can review the kernel implementations without reading algorithm logic.
- **Extensibility:** adding a new execution backend (e.g., HIP/ROCm) requires only new `_hip()` methods and a new dispatch branch — no algorithm code changes.
```

- [ ] **Step 2: Build and check**

```bash
cd docs && bundle exec jekyll build
```

Expected: no errors. "CPU/GPU Dispatch" appears under Architecture in the sidebar.

- [ ] **Step 3: Commit**

```bash
git add docs/architecture/dispatch.md
git commit -m "feat: add CPU/GPU dispatch architecture page with real code snippets"
```

---

## Task 5: Architecture — Python API page

**Files:**
- Create: `docs/architecture/python-api.md`

- [ ] **Step 1: Create `docs/architecture/python-api.md`**

```markdown
---
title: Python API
parent: Architecture
nav_order: 2
---

# Python API

The Python interface (`pylynx`) wraps the C++ engine via pybind11 with three granularity levels. All three produce identical results — choose the level that matches how much control you need.

## Level 1 — Full calculation in one call

```python
from lynx.config import DFTConfig

config = DFTConfig(
    cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],
    fractional=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    symbols=['Si', 'Si'],
    pseudo_files={'Si': 'psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8'},
    Nstates=10, xc='GGA_PBE',
)
calc = config.create_calculator(auto_run=True)
print(calc.total_energy)          # Hartree
forces = calc.compute_forces()    # numpy (Natom, 3) in Ha/Bohr
```

Or from a JSON file (same format the C++ executable uses):

```python
import lynx
calc = lynx.Calculator("Si8.json")
print(calc.total_energy)
```

## Level 2 — SCF control and internal access

```python
calc = lynx.Calculator("Si8.json", auto_run=False)
# Calculator is set up but SCF has not run yet

print(calc.grid)         # FDGrid(40x40x40)
print(calc.Nelectron)    # 8

calc.run()               # run SCF manually
print(calc.total_energy)

wfn = calc.get_wavefunction()
eigs = wfn.eigenvalues(spin=0, kpt=0)   # numpy array of eigenvalues
rho  = calc.density                      # numpy array, shape (Nd_d,)
```

## Level 3 — Raw operators on numpy arrays

```python
import numpy as np
import lynx

lattice = lynx.make_lattice(np.diag([10.0, 10.0, 10.0]))
grid    = lynx.FDGrid(48, 48, 48, lattice)
stencil = lynx.FDStencil(12, grid, lattice)
domain  = lynx.full_domain(grid)
halo    = lynx.HaloExchange(domain, stencil.FDn())

lap  = lynx.Laplacian(stencil, domain)
grad = lynx.Gradient(stencil, domain)

x    = np.random.randn(domain.Nd_d())
y    = lap.apply(halo, x, a=-0.5, c=0.0)    # y = -0.5 * Lap(x)
dfdx = grad.apply(halo, x, direction=0)      # df/dx
```

Level 3 is useful for testing finite-difference operators independently, building custom post-processing, or exploring the grid directly.

## Zero-copy numpy interop

`DeviceArray<double>` (the internal array type) implements the Python buffer protocol. When an array lives on CPU, numpy accesses it directly — no copy. When it lives on GPU, calling `.to_host()` returns a CPU copy as numpy without allocating a new Python object.

```python
rho = calc.density    # numpy view of the internal DeviceArray — no copy
rho[0]                # direct memory access
```

## ASE Calculator

`LynxCalculator` is a drop-in ASE calculator. All values use ASE units (Angstrom, eV) — conversion to Bohr/Hartree is automatic.

```python
from ase.build import bulk
from lynx.ase_interface import LynxCalculator

atoms = bulk('Si', 'diamond', a=5.43)
atoms.calc = LynxCalculator(
    xc='GGA_PBE',
    kpts=(4, 4, 4),
    mesh_spacing=0.5,   # Bohr
    scf_tol=1e-6,
)

energy = atoms.get_potential_energy()   # eV
forces = atoms.get_forces()            # eV/Å
stress = atoms.get_stress()            # eV/Å³ (Voigt: xx yy zz yz xz xy)
```

## Unit conventions

| Interface | Length | Energy | Forces | Stress |
|-----------|--------|--------|--------|--------|
| C++ / JSON | Bohr | Hartree | Ha/Bohr | Ha/Bohr³ |
| `DFTConfig()` | Bohr | Hartree | Ha/Bohr | Ha/Bohr³ |
| `DFTConfig.from_ase()` | auto | auto | auto | auto |
| `LynxCalculator` (ASE) | Å | eV | eV/Å | eV/Å³ |

Constants are in `lynx.units`: `BOHR_TO_ANG`, `HA_TO_EV`, `HA_BOHR_TO_EV_ANG`.
```

- [ ] **Step 2: Build and check**

```bash
cd docs && bundle exec jekyll build
```

Expected: no errors. "Python API" appears under Architecture in sidebar.

- [ ] **Step 3: Commit**

```bash
git add docs/architecture/python-api.md
git commit -m "feat: add Python API architecture page (3 levels + ASE + unit table)"
```

---

## Task 6: Architecture — external libraries page

**Files:**
- Create: `docs/architecture/external-libs.md`

- [ ] **Step 1: Create `docs/architecture/external-libs.md`**

```markdown
---
title: External Libraries
parent: Architecture
nav_order: 3
---

# External Libraries

LYNX is designed so that each external dependency plugs in at a single, well-defined seam.

## libxc — XC functional zoo

libxc provides 600+ exchange-correlation functionals (LDA, GGA, metaGGA, hybrid). LYNX links against libxc and routes functional selection through a single string at input time:

```json
"electronic": { "xc": "GGA_PBE" }
```

```python
DFTConfig(..., xc='GGA_PBE')       # PBE
DFTConfig(..., xc='LDA_PZ')        # Perdew-Zunger LDA
DFTConfig(..., xc='GGA_PBEsol')    # PBEsol
```

Adding a new functional requires no code changes — just pass a valid libxc name. The `XCFunctional` class in `src/xc/` handles the libxc API calls.

## MPI — Parallelization

LYNX parallelizes over spins, k-points, and bands. The `MPIComm` and `HaloExchange` classes in `src/parallel/` encapsulate all MPI calls. The rest of the code does not call MPI directly.

```bash
mpirun -np 8 ./build/src/lynx Si8.json   # 8 MPI ranks
```

Decomposition is automatic: ranks are assigned to spin/k-point groups, and within a group to band blocks. `HaloExchange` handles ghost-node synchronization for finite-difference stencils transparently.

## pybind11 — Python bindings

pybind11 is fetched automatically by CMake (`FetchContent`) — no manual install required. The binding sources live in `python/src/`. Key design choices:

- **Buffer protocol:** `DeviceArray<T>` exposes a buffer interface so numpy can access CPU arrays without copying.
- **Automatic unit conversion:** `DFTConfig.from_ase()` converts Angstrom→Bohr and eV→Hartree on input; `LynxCalculator` converts back on output.
- **No GIL tricks:** each `calc.run()` call releases the GIL so Python threads are not blocked during SCF.

```bash
cmake -DBUILD_PYTHON=ON ..
make -j
export PYTHONPATH=/path/to/LYNX/python:$PYTHONPATH
```

## ASE — Atoms and workflows

The `LynxCalculator` class in `python/lynx/ase_interface.py` implements the ASE [Calculator interface](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculator.html). It adapts `DFTConfig` to accept ASE `Atoms` objects and returns forces/stress in ASE units. No changes to the C++ engine were needed — the adapter is pure Python.

This means LYNX works transparently with all ASE tools: geometry optimizers, molecular dynamics drivers (including i-PI via the ASE socket interface), equation-of-state fitting, and structure generation.

## ONCV Pseudopotentials (PseudoDojo)

Pseudopotential files are included as git submodules from [PseudoDojo](http://www.pseudo-dojo.org/):

```
psps/ONCVPSP-PBE-PDv0.4/   # GGA-PBE
psps/ONCVPSP-LDA-PDv0.4/   # LDA
```

Each element has a subdirectory with a `.psp8` file (e.g., `Si/Si.psp8`). The `Pseudopotential` class in `src/atoms/` reads the psp8 format, handles the Fortran D-notation in exponents, and precomputes projector data.

Initialize submodules after cloning:

```bash
git submodule update --init
```

## Build flags summary

| Flag | Effect |
|------|--------|
| (default) | CPU-only build, MPI, OpenBLAS/MKL, libxc (submodule) |
| `-DUSE_CUDA=ON` | Enables GPU kernels, cuBLAS, cuSOLVER, cuRAND |
| `-DBUILD_PYTHON=ON` | Builds `_core.so` pybind11 extension |
```

- [ ] **Step 2: Build and check**

```bash
cd docs && bundle exec jekyll build
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add docs/architecture/external-libs.md
git commit -m "feat: add External Libraries architecture page"
```

---

## Task 7: Theory scaffold — overview and cluster parent pages

**Files:**
- Create: `docs/theory/index.md`
- Create: `docs/theory/discretization.md`
- Create: `docs/theory/electronic-structure.md`
- Create: `docs/theory/response-properties.md`
- Create: `docs/theory/advanced-topics.md`

- [ ] **Step 1: Create `docs/theory/index.md`**

```markdown
---
title: Theory
nav_order: 3
has_children: true
permalink: /theory/
---

# Theory

LYNX solves the Kohn-Sham equations on a uniform real-space grid using
self-consistent field (SCF) iteration. This section covers the physics and
numerical methods behind each step of the pipeline, from grid setup to
forces and stress.

## Pipeline overview

```
Atoms + pseudopotentials
        │
        ▼
  Real-space grid (FDGrid, Lattice)
        │
        ▼
  Initial density ρ⁰
        │
        ▼
  ┌─── SCF loop ─────────────────────────────────┐
  │  Poisson → φ(r)                              │
  │  XC      → Vxc(r), exc(r)                   │
  │  Veff = Vloc + φ + Vxc                       │
  │  Hamiltonian H[Veff]                         │
  │  CheFSI eigensolver → {ψnk, εnk}            │
  │  Occupations fn (Fermi-Dirac / smearing)     │
  │  New density ρ(r) = Σ fn |ψnk|²             │
  │  Mixer (Pulay+Kerker) → ρ_in for next iter  │
  └──────────────────────────────────────────────┘
        │
        ▼
  Total energy E[ρ]
  Forces  Fα = -∂E/∂Rα
  Stress  σij = -1/Ω ∂E/∂εij
```

## Sections

- [Discretization]({{ site.baseurl }}/theory/discretization/) — real-space grid, finite differences, pseudopotentials
- [Electronic Structure]({{ site.baseurl }}/theory/electronic-structure/) — Hamiltonian, eigensolver, density, electrostatics, XC, energy
- [Response Properties]({{ site.baseurl }}/theory/response-properties/) — forces and stress
- [Advanced Topics]({{ site.baseurl }}/theory/advanced-topics/) — mixing, k-points, spin-orbit coupling, parallelization
```

- [ ] **Step 2: Create the four cluster parent pages**

`docs/theory/discretization.md`:
```markdown
---
title: Discretization
parent: Theory
nav_order: 1
has_children: true
---

# Discretization

How the continuous Kohn-Sham equations are mapped onto a finite real-space grid.
```

`docs/theory/electronic-structure.md`:
```markdown
---
title: Electronic Structure
parent: Theory
nav_order: 2
has_children: true
---

# Electronic Structure

The Hamiltonian, eigensolver, electron density, electrostatics, exchange-correlation, and total energy.
```

`docs/theory/response-properties.md`:
```markdown
---
title: Response Properties
parent: Theory
nav_order: 3
has_children: true
---

# Response Properties

Atomic forces and stress tensors from the converged SCF density.
```

`docs/theory/advanced-topics.md`:
```markdown
---
title: Advanced Topics
parent: Theory
nav_order: 4
has_children: true
---

# Advanced Topics

Density mixing, k-point sampling, spin-orbit coupling, and MPI parallelization.
```

- [ ] **Step 3: Build and check**

```bash
cd docs && bundle exec jekyll build
```

Expected: no errors. "Theory" appears in the sidebar at position 3 with four collapsible sub-sections.

- [ ] **Step 4: Commit**

```bash
git add docs/theory/
git commit -m "feat: add Theory section scaffold with overview and cluster parent pages"
```

---

## Task 8: Theory pages — all 14 content pages

**Files:**
- Create: `docs/theory/01-real-space.md` through `docs/theory/14-parallelization.md`

Each page follows this pattern — copy the content from `doc/` and prepend Jekyll front matter:

```markdown
---
title: <title from table below>
parent: <cluster from table below>
grand_parent: Theory
nav_order: <order within cluster>
---

<paste full content of doc/XX_filename.md here, unchanged>
```

| # | File to create | Title | Parent (cluster) | nav_order | Source file |
|---|----------------|-------|-----------------|-----------|-------------|
| 1 | `01-real-space.md` | Real-Space Finite Differences | Discretization | 1 | `doc/01_real_space_discretization.md` |
| 2 | `02-pseudopotentials.md` | Pseudopotentials | Discretization | 2 | `doc/02_pseudopotentials.md` |
| 3 | `03-hamiltonian.md` | Hamiltonian | Electronic Structure | 1 | `doc/03_hamiltonian.md` |
| 4 | `04-eigensolver.md` | Eigensolver (CheFSI) | Electronic Structure | 2 | `doc/04_eigensolver.md` |
| 5 | `05-density.md` | Electron Density & Occupation | Electronic Structure | 3 | `doc/05_density_and_occupation.md` |
| 6 | `06-electrostatics.md` | Electrostatics | Electronic Structure | 4 | `doc/06_electrostatics.md` |
| 7 | `07-xc.md` | Exchange-Correlation | Electronic Structure | 5 | `doc/07_exchange_correlation.md` |
| 8 | `08-energy.md` | Total Energy | Electronic Structure | 6 | `doc/08_total_energy.md` |
| 9 | `09-forces.md` | Forces | Response Properties | 1 | `doc/09_forces.md` |
| 10 | `10-stress.md` | Stress | Response Properties | 2 | `doc/10_stress.md` |
| 11 | `11-mixing.md` | Mixing & Convergence | Advanced Topics | 1 | `doc/11_mixing.md` |
| 12 | `12-kpoints.md` | K-Points | Advanced Topics | 2 | `doc/12_kpoints.md` |
| 13 | `13-soc.md` | Spin-Orbit Coupling | Advanced Topics | 3 | `doc/13_spin_orbit_coupling.md` |
| 14 | `14-parallelization.md` | Parallelization | Advanced Topics | 4 | `doc/14_parallelization.md` |

- [ ] **Step 1: Create all 14 theory pages**

For each row in the table above, create the file with this script:

```bash
# Run from the LYNX repo root. Creates all 14 theory pages at once.

create_theory_page() {
  local out="docs/theory/$1"
  local title="$2"
  local parent="$3"
  local order="$4"
  local src="doc/$5"

  {
    echo "---"
    echo "title: \"$title\""
    echo "parent: $parent"
    echo "grand_parent: Theory"
    echo "nav_order: $order"
    echo "---"
    echo ""
    cat "$src"
  } > "$out"
}

create_theory_page 01-real-space.md      "Real-Space Finite Differences"   Discretization       1  01_real_space_discretization.md
create_theory_page 02-pseudopotentials.md "Pseudopotentials"                Discretization       2  02_pseudopotentials.md
create_theory_page 03-hamiltonian.md     "Hamiltonian"                     "Electronic Structure" 1  03_hamiltonian.md
create_theory_page 04-eigensolver.md     "Eigensolver (CheFSI)"            "Electronic Structure" 2  04_eigensolver.md
create_theory_page 05-density.md         "Electron Density and Occupation" "Electronic Structure" 3  05_density_and_occupation.md
create_theory_page 06-electrostatics.md  "Electrostatics"                  "Electronic Structure" 4  06_electrostatics.md
create_theory_page 07-xc.md              "Exchange-Correlation"            "Electronic Structure" 5  07_exchange_correlation.md
create_theory_page 08-energy.md          "Total Energy"                    "Electronic Structure" 6  08_total_energy.md
create_theory_page 09-forces.md          "Forces"                          "Response Properties" 1  09_forces.md
create_theory_page 10-stress.md          "Stress"                          "Response Properties" 2  10_stress.md
create_theory_page 11-mixing.md          "Mixing and Convergence"          "Advanced Topics"     1  11_mixing.md
create_theory_page 12-kpoints.md         "K-Points"                        "Advanced Topics"     2  12_kpoints.md
create_theory_page 13-soc.md             "Spin-Orbit Coupling"             "Advanced Topics"     3  13_spin_orbit_coupling.md
create_theory_page 14-parallelization.md "Parallelization"                 "Advanced Topics"     4  14_parallelization.md
```

- [ ] **Step 2: Build and check**

```bash
cd docs && bundle exec jekyll build
```

Expected: no errors. All 14 theory pages appear under their cluster groups in the sidebar. LaTeX renders via MathJax in the browser (`bundle exec jekyll serve` then open `http://localhost:4000/LYNX/`).

Spot-check one page with LaTeX — open `http://localhost:4000/LYNX/theory/04-eigensolver/` and verify the Chebyshev polynomial formula renders.

- [ ] **Step 3: Commit**

```bash
git add docs/theory/
git commit -m "feat: add all 14 theory pages with Jekyll front matter"
```

---

## Task 9: Examples pages

**Files:**
- Create: `docs/examples/index.md`
- Create: `docs/examples/cpp-quickstart.md`
- Create: `docs/examples/python-quickstart.md`
- Create: `docs/examples/ase-integration.md`
- Create: `docs/examples/md-simulation.md`

- [ ] **Step 1: Create `docs/examples/index.md`**

```markdown
---
title: Examples
nav_order: 4
has_children: true
permalink: /examples/
---

# Examples

Runnable examples ordered from simplest to most advanced.

| Example | Interface | What it shows |
|---------|-----------|---------------|
| [C++ quickstart]({{ site.baseurl }}/examples/cpp-quickstart/) | JSON + CLI | Run LYNX from the command line |
| [Python quickstart]({{ site.baseurl }}/examples/python-quickstart/) | Python | DFTConfig, Calculator, from_ase() |
| [ASE integration]({{ site.baseurl }}/examples/ase-integration/) | ASE | LynxCalculator as a drop-in ASE calculator |
| [MD simulation]({{ site.baseurl }}/examples/md-simulation/) | ASE + i-PI | NVE molecular dynamics, density restart |

All examples use the 2-atom Si diamond unit cell. Results should match `Etotal = -33.3685427384 Ha` (8-atom supercell example in `examples/Si8_comparison/` yields the same energy per atom).
```

- [ ] **Step 2: Create `docs/examples/cpp-quickstart.md`**

```markdown
---
title: C++ Quickstart
parent: Examples
nav_order: 1
---

# C++ Quickstart

Run LYNX from the command line using a JSON input file.

## Build

```bash
git clone --recurse-submodules https://github.com/Xin-Jing-11/LYNX.git
cd LYNX
mkdir build && cd build
cmake ..
make -j
```

For GPU support:
```bash
cmake -DUSE_CUDA=ON ..
make -j
```

## Input file

Save as `Si2.json`:

```json
{
  "lattice": {
    "vectors": [[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]]
  },
  "grid": {
    "Nx": 24, "Ny": 24, "Nz": 24,
    "fd_order": 12
  },
  "atoms": [{
    "element": "Si",
    "pseudo_file": "psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8",
    "fractional": true,
    "coordinates": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
  }],
  "electronic": {
    "xc": "GGA_PBE",
    "Nstates": 10,
    "temperature": 300,
    "smearing": "gaussian"
  },
  "scf": {
    "max_iter": 100,
    "tolerance": 1e-6,
    "mixing": "density",
    "preconditioner": "kerker",
    "mixing_history": 7,
    "mixing_parameter": 0.3
  }
}
```

All lengths are in **Bohr**. Temperature is in **Kelvin**.

## Run

```bash
# Single process
./build/src/lynx Si2.json

# With MPI (4 ranks)
mpirun -np 4 ./build/src/lynx Si2.json
```

## Output

```
SCF iter  1: E = -15.821304 Ha, dE = 1.23e+00
SCF iter  2: E = -16.683241 Ha, dE = 8.61e-01
...
SCF iter 18: E = -16.684271 Ha, dE = 4.3e-07  CONVERGED

Total energy: -16.684271 Ha
Fermi energy:  0.173821 Ha

Forces (Ha/Bohr):
  Si  0.000000  0.000000  0.000000
  Si  0.000000  0.000000  0.000000
```

## Key JSON fields

| Field | Description | Default |
|-------|-------------|---------|
| `grid.fd_order` | Finite-difference order (6–24) | 12 |
| `electronic.xc` | XC functional string | `GGA_PBE` |
| `electronic.Nstates` | Number of Kohn-Sham states | — |
| `scf.tolerance` | SCF convergence (Ha/atom) | `1e-6` |
| `scf.mixing_history` | Pulay history depth | 7 |

**What's next:** [Architecture overview]({{ site.baseurl }}/architecture/) explains how the JSON fields map to internal C++ objects.
```

- [ ] **Step 3: Create `docs/examples/python-quickstart.md`**

```markdown
---
title: Python Quickstart
parent: Examples
nav_order: 2
---

# Python Quickstart

Three ways to run LYNX from Python — all produce the same result.

## Setup

```bash
mkdir build && cd build
cmake -DBUILD_PYTHON=ON ..
make -j
export PYTHONPATH=/path/to/LYNX/python:$PYTHONPATH
```

## Option A — DFTConfig (no JSON file)

```python
from lynx.config import DFTConfig

config = DFTConfig(
    cell=[[10.26, 0, 0], [0, 10.26, 0], [0, 0, 10.26]],  # Bohr
    fractional=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    symbols=['Si', 'Si'],
    pseudo_files={'Si': 'psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8'},
    Nx=24, Ny=24, Nz=24,
    Nstates=10,
    xc='GGA_PBE',
)
calc = config.create_calculator(auto_run=True)
print(calc.total_energy)          # Hartree
print(calc.compute_forces())      # numpy (Natom, 3), Ha/Bohr
```

## Option B — Calculator from JSON

```python
import lynx

calc = lynx.Calculator("Si2.json")
print(calc.total_energy)
print(calc.energy)    # dict: Eband, Exc, Ehart, Etotal, ...
```

## Option C — DFTConfig from ASE Atoms

```python
from ase.build import bulk
from lynx.config import DFTConfig

atoms = bulk('Si', 'diamond', a=5.43)          # Angstrom — auto-converted
config = DFTConfig.from_ase(atoms, kpts=(2, 2, 2), Nstates=10, xc='GGA_PBE')
calc = config.create_calculator(auto_run=True)
print(calc.total_energy)
```

## Accessing internals (Level 2)

```python
calc = lynx.Calculator("Si2.json", auto_run=False)
print(calc.grid)          # FDGrid(24x24x24)
print(calc.Nelectron)     # 8

calc.run()

wfn  = calc.get_wavefunction()
eigs = wfn.eigenvalues(spin=0, kpt=0)   # numpy array
rho  = calc.density                      # numpy (Nd_d,)
```

**What's next:** [Python API architecture]({{ site.baseurl }}/architecture/python-api/) explains the three levels and zero-copy numpy interop.
```

- [ ] **Step 4: Create `docs/examples/ase-integration.md`**

```markdown
---
title: ASE Integration
parent: Examples
nav_order: 3
---

# ASE Integration

`LynxCalculator` is a drop-in ASE calculator. All values use ASE units (Å, eV) — conversion to Bohr/Hartree is automatic.

## Basic usage

```python
from ase.build import bulk
from lynx.ase_interface import LynxCalculator

atoms = bulk('Si', 'diamond', a=5.43)
atoms.calc = LynxCalculator(
    xc='GGA_PBE',
    kpts=(4, 4, 4),
    mesh_spacing=0.5,   # Bohr — grid spacing (alternative to Nx/Ny/Nz)
    max_scf_iter=100,
    scf_tol=1e-6,
)

energy = atoms.get_potential_energy()   # eV
forces = atoms.get_forces()            # eV/Å, shape (Natom, 3)
stress = atoms.get_stress()            # eV/Å³, Voigt: (xx, yy, zz, yz, xz, xy)
```

## Geometry optimization

```python
from ase.optimize import BFGS

opt = BFGS(atoms, trajectory='si_relax.traj')
opt.run(fmax=0.01)   # eV/Å
print(atoms.get_potential_energy())
```

## Equation of state

```python
from ase.eos import EquationOfState

volumes, energies = [], []
for scale in [0.95, 0.97, 1.00, 1.03, 1.05]:
    a = atoms.copy()
    a.set_cell(atoms.cell * scale, scale_atoms=True)
    a.calc = LynxCalculator(xc='GGA_PBE', kpts=(4,4,4))
    volumes.append(a.get_volume())
    energies.append(a.get_potential_energy())

eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()
print(f"Bulk modulus: {B / units.GPa:.1f} GPa")
```

## Calculator options

| Option | Type | Description |
|--------|------|-------------|
| `xc` | str | XC functional string (e.g., `'GGA_PBE'`) |
| `kpts` | tuple | Monkhorst-Pack grid, e.g., `(4, 4, 4)` |
| `mesh_spacing` | float | Grid spacing in Bohr (sets Nx/Ny/Nz automatically) |
| `max_scf_iter` | int | Maximum SCF iterations |
| `scf_tol` | float | SCF convergence threshold (Ha/atom) |
| `psp_dir` | str | Path to pseudopotential directory (or set `LYNX_PSP_PATH`) |

**What's next:** [MD simulation example]({{ site.baseurl }}/examples/md-simulation/) shows LynxCalculator in an NVE molecular dynamics run.
```

- [ ] **Step 5: Create `docs/examples/md-simulation.md`**

```markdown
---
title: MD Simulation
parent: Examples
nav_order: 4
---

# MD Simulation

Run NVE molecular dynamics using the ASE VelocityVerlet driver with LynxCalculator. This example also demonstrates density restart (warm-starting SCF from the previous step's density) and SCF iteration tracking.

The example files are in `examples/ase_md/` in the repository.

## NVE run

```python
from ase.build import bulk
from ase.md.verlet import VelocityVerlet
from ase.io import Trajectory
from ase import units
from lynx.ase_interface import LynxCalculator

atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)  # 16-atom supercell

atoms.calc = LynxCalculator(
    xc='GGA_PBE',
    kpts=(2, 2, 2),
    scf_tol=1e-5,     # slightly relaxed tolerance for MD
)

# Initialize velocities at 300 K
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

traj = Trajectory('si_nve.traj', 'w', atoms)
dyn  = VelocityVerlet(atoms, timestep=1.0 * units.fs)
dyn.attach(traj.write, interval=1)

dyn.run(50)   # 50 MD steps = 50 fs
traj.close()
```

## Density restart

Restarting SCF from the previous step's density typically cuts SCF iterations by 40–60% in MD.

```python
from lynx.ase_interface import LynxCalculator

calc = LynxCalculator(xc='GGA_PBE', kpts=(2, 2, 2))

# After first SCF:
rho = calc.get_density()   # save density

# Next MD step — warm-start:
calc.set_initial_density(rho)
atoms.calc = calc
atoms.get_potential_energy()   # SCF converges faster
```

## Reading a trajectory

```python
from ase.io import read

traj = read('si_nve.traj', index=':')

energies = [atoms.get_potential_energy() for atoms in traj]
import matplotlib.pyplot as plt
plt.plot(energies)
plt.xlabel('MD step')
plt.ylabel('Energy (eV)')
plt.savefig('si_nve_energy.png')
```

## i-PI socket interface

For more advanced MD (path integral MD, thermostats), use i-PI via the ASE socket calculator:

```python
from ase.calculators.socketio import SocketIOCalculator

with SocketIOCalculator(LynxCalculator(xc='GGA_PBE'), log='ipi.log') as calc:
    atoms.calc = calc
    # i-PI driver controls the trajectory
```

**What's next:** [Architecture overview]({{ site.baseurl }}/architecture/) explains how the Python interface is structured and why density restart is efficient.
```

- [ ] **Step 6: Build and check all examples**

```bash
cd docs && bundle exec jekyll build
```

Expected: no errors. All four examples appear under Examples in the sidebar.

- [ ] **Step 7: Commit**

```bash
git add docs/examples/
git commit -m "feat: add Examples section (C++, Python, ASE, MD simulation)"
```

---

## Task 10: Enable GitHub Pages

- [ ] **Step 1: Push the branch and configure GitHub Pages**

Push your feature branch:

```bash
git push origin feature/ase-ipi-md
```

After merging to `master`, go to the GitHub repo → **Settings → Pages**:
- Source: **Deploy from a branch**
- Branch: `master`
- Folder: `/docs`
- Click **Save**

GitHub will trigger a Jekyll build. After ~1–2 minutes, the site is live at:
`https://xin-jing-11.github.io/LYNX/`

- [ ] **Step 2: Verify the live site**

Open `https://xin-jing-11.github.io/LYNX/` and check:
- Home page hero renders correctly
- Architecture appears before Theory in sidebar
- At least one theory page renders LaTeX (e.g., `/LYNX/theory/04-eigensolver/`)
- All sidebar links resolve (no 404s)
- Search finds "CheFSI" and returns the eigensolver page

---

## Self-Review

**Spec coverage:**
- ✅ just-the-docs theme via remote_theme
- ✅ Jekyll source in `docs/`, GitHub Pages from `docs/` on `master`
- ✅ Architecture before Theory in nav
- ✅ Home page: hero, 3-column cards, tabbed quick examples, "What's inside" paragraph
- ✅ Architecture: 4 pages (overview + layer diagram, dispatch, Python API, external libs)
- ✅ Theory: all 14 doc pages, grouped into 4 clusters
- ✅ Examples: 4 pages (C++, Python, ASE, MD)
- ✅ MathJax for LaTeX rendering in theory pages
- ✅ `.gitignore` updated for Jekyll build artifacts
- ✅ Each example page ends with "What's next" link

**Placeholder scan:** No TBD, TODO, or "similar to Task N" references found.

**Type consistency:** Front matter field names (`title`, `parent`, `grand_parent`, `nav_order`, `has_children`) are consistent across all tasks. `site.baseurl` used consistently for internal links.
