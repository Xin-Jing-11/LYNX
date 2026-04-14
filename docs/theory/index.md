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
