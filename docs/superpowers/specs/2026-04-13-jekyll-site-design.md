# LYNX GitHub Pages Site — Design Spec
Date: 2026-04-13

## Goal

A GitHub Pages site (Jekyll, `just-the-docs` theme) that showcases LYNX as:
1. A well-designed scientific software project (architecture-first narrative)
2. A DFT learning resource (all 14 theory pages)
3. A practical tool (Python API + examples)

Primary audience: researchers learning DFT and developers interested in the code design. The site should work for both without forcing either to read the other's content first.

---

## Theme

**`just-the-docs`** — sidebar navigation, hierarchical page grouping, built-in search, code highlighting. Hosted via GitHub Pages at `https://<user>.github.io/LYNX/`.

Jekyll source lives in `docs/` at the repo root. GitHub Pages is configured to build from `docs/` on the `master` branch.

---

## Site Structure

```
docs/
  _config.yml              just-the-docs config, site metadata
  index.md                 Home page
  architecture/
    index.md               Layer diagram + design philosophy
    dispatch.md            CPU/GPU dispatch pattern
    python-api.md          Three-level Python API
    external-libs.md       libxc, MPI, pybind11, ASE, pseudopotentials
  theory/
    index.md               Theory overview (narrative intro to real-space DFT)
    01-real-space.md       ← doc/01_real_space_discretization.md
    02-pseudopotentials.md ← doc/02_pseudopotentials.md
    03-hamiltonian.md      ← doc/03_hamiltonian.md
    04-eigensolver.md      ← doc/04_eigensolver.md
    05-density.md          ← doc/05_density_and_occupation.md
    06-electrostatics.md   ← doc/06_electrostatics.md
    07-xc.md               ← doc/07_exchange_correlation.md
    08-energy.md           ← doc/08_total_energy.md
    09-forces.md           ← doc/09_forces.md
    10-stress.md           ← doc/10_stress.md
    11-mixing.md           ← doc/11_mixing.md
    12-kpoints.md          ← doc/12_kpoints.md
    13-soc.md              ← doc/13_spin_orbit_coupling.md
    14-parallelization.md  ← doc/14_parallelization.md
  examples/
    index.md               Examples overview
    cpp-quickstart.md      C++ CLI + JSON input
    python-quickstart.md   DFTConfig, Calculator, from_ase()
    ase-integration.md     LynxCalculator as ASE calculator
    md-simulation.md       ASE/i-PI MD with density restart
```

---

## Navigation Order

Sidebar sections in this order:
1. Home
2. Architecture
3. Theory
4. Examples

Architecture comes before Theory so the site opens with design philosophy before diving into physics.

---

## Page Designs

### Home (`index.md`)

```
Hero
  Title: LYNX
  Tagline: "A real-space DFT engine designed for clarity —
            GPU acceleration without algorithmic duplication,
            Python access without overhead."
  Buttons: [View on GitHub]  [Get Started →]

Design Highlights  (3-column cards)
  ① PyTorch-style dispatch
      Algorithm in .cpp, kernels in .cu. One _cpu()/_gpu() switch.
      No algorithmic duplication across execution paths.
  ② GPU-resident SCF pipeline
      psi never leaves the device. 35–89× speedup per operator.
      Full SCF → forces → stress on GPU without host transfers.
  ③ Three-level Python API
      From one-liner Calculator to raw numpy operators.
      Zero-copy interop via pybind11. ASE drop-in support.

Quick example  (tabbed code block: Python | ASE | C++ JSON)

What's inside  (short paragraph linking to Architecture and Theory)
```

### Architecture — Overview (`architecture/index.md`)

Layer diagram showing the dependency chain:

```
Python (pylynx / ASE)
        ↓
  C++ Engine
  ├── physics/    SCF, Energy, Forces, Stress
  ├── solvers/    CheFSI eigensolver, AAR Poisson, Pulay+Kerker mixer
  ├── operators/  Laplacian, Gradient, Hamiltonian, NonlocalProjector
  ├── electronic/ Wavefunction, ElectronDensity, Occupation
  ├── xc/         XCFunctional (libxc), ExactExchange
  ├── atoms/      Crystal, Pseudopotential (psp8)
  ├── parallel/   MPI, HaloExchange
  └── core/       DeviceArray, Lattice, FDGrid, Domain, KPoints
```

Key principle stated explicitly: each layer depends only on layers below it. No physics logic in GPU kernels. No kernel calls in physics algorithms.

### Architecture — CPU/GPU Dispatch (`architecture/dispatch.md`)

The design principle with concrete LYNX code snippets:
- The `solve()` / `apply_foo()` pattern in `.cpp`
- The `_cpu()` / `_gpu()` dispatcher (3-line method)
- The `.cu` file containing only kernel launches — no loops, no algorithm

Draws the explicit parallel to PyTorch (`at::Tensor` dispatch) and TensorFlow (`OpKernel`). Explains why this matters: the algorithm is testable on CPU, the GPU path is auditable in isolation, and adding a new backend never touches algorithm code.

Also covers GPU data residency rules: what lives on device, what is allowed to transfer per SCF iteration, and why psi never moves to host.

### Architecture — Python API (`architecture/python-api.md`)

Three granularity levels side by side:

| Level | Entry point | Use case |
|-------|------------|----------|
| 1 | `Calculator("file.json")` or `DFTConfig(...).create_calculator()` | Full calculation in one call |
| 2 | `Calculator(auto_run=False)` + `.run()` | SCF control, access internals |
| 3 | `lynx.FDGrid`, `lynx.Laplacian`, `lynx.Gradient` | Raw operators on numpy arrays |

Explains pybind11 binding strategy: zero-copy numpy interop, how `DeviceArray<T>` exposes a buffer protocol, ASE unit conversion (automatic Bohr↔Angstrom, Hartree↔eV).

### Architecture — External Libraries (`architecture/external-libs.md`)

| Library | Role | Integration |
|---------|------|-------------|
| libxc | XC functional zoo | Drop-in via `xc` string (`GGA_PBE`, `LDA_PZ`, etc.) |
| MPI | Parallelization | Spin, k-point, and band decomposition |
| pybind11 | Python bindings | Fetched automatically by CMake |
| ASE | Atoms + workflow | `LynxCalculator` adapter, automatic unit conversion |
| ONCV PseudoDojo | Pseudopotentials | Git submodules (`psps/ONCVPSP-{LDA,PBE}-PDv0.4/`) |

### Theory — Overview (`theory/index.md`)

Short narrative: "DFT reduces the many-body problem to a self-consistent field equation solved on a real-space grid. Here's how each piece of that pipeline is implemented in LYNX." Links to all four clusters.

### Theory pages — Cluster grouping

| Cluster | Pages |
|---------|-------|
| Discretization | Real-space FD, Pseudopotentials |
| Electronic Structure | Hamiltonian, Eigensolver, Density & occupation, Electrostatics, XC, Total energy |
| Response Properties | Forces, Stress |
| Advanced Topics | Mixing, K-points, Spin-orbit coupling, Parallelization |

Content: existing `doc/*.md` files with Jekyll front matter added. No rewriting.

### Examples pages

| Page | Content |
|------|---------|
| `cpp-quickstart.md` | `mpirun ./lynx Si8.json`, key JSON fields explained |
| `python-quickstart.md` | `DFTConfig`, `Calculator`, `DFTConfig.from_ase()` side by side |
| `ase-integration.md` | `LynxCalculator`, `get_forces()`, `get_stress()`, unit conventions |
| `md-simulation.md` | ASE/i-PI NVE MD, density restart, SCF iteration tracking |

Each page ends with a "What's next" link to the relevant Architecture or Theory page.

---

## Implementation Notes

- Jekyll source in `docs/`, built by GitHub Pages from `master` branch
- `_config.yml` sets `theme: just-the-docs`, `title: LYNX`, search enabled
- Theory pages: copy `doc/*.md` content into `docs/theory/*.md`, prepend front matter (`title`, `parent: Theory`, `grand_parent`, `nav_order`)
- Theory cluster grouping: use just-the-docs `parent`/`has_children` front matter — no plugin required
- Code blocks: use fenced markdown with language tags (`cpp`, `python`, `json`, `bash`)
- No custom CSS required for MVP — just-the-docs defaults are sufficient
- `.gitignore`: add `docs/_site/` and `docs/.jekyll-cache/`

---

## Out of Scope

- Custom CSS / branding beyond theme defaults
- Search index customization
- Versioning or changelog pages
- API reference auto-generation (no Doxygen/Sphinx integration)
