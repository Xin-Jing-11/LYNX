# LYNX Theory Documentation

## Overview

LYNX is a real-space Kohn-Sham density functional theory (DFT) code. It solves the Kohn-Sham equations self-consistently on a uniform finite-difference grid, supporting periodic boundary conditions, non-orthogonal unit cells, k-point sampling, collinear spin polarization, and spin-orbit coupling.

## The Kohn-Sham Framework

The central problem in DFT is solving the single-particle Kohn-Sham equations:

$$
\hat{H} \psi_{n\mathbf{k}s}(\mathbf{r}) = \varepsilon_{n\mathbf{k}s} \psi_{n\mathbf{k}s}(\mathbf{r})
$$

where the Kohn-Sham Hamiltonian is:

$$
\hat{H} = -\frac{1}{2}\nabla^2 + V_{\text{eff}}(\mathbf{r}) + \hat{V}_{\text{nl}}
$$

The effective potential is:

$$
V_{\text{eff}}(\mathbf{r}) = V_{\text{xc}}(\mathbf{r}) + \phi(\mathbf{r}) + V_{\text{loc}}(\mathbf{r})
$$

where $V_{\text{xc}}$ is the exchange-correlation potential, $\phi$ is the electrostatic (Hartree) potential from electrons and pseudocharge, and $V_{\text{loc}}$ is the local pseudopotential correction. $\hat{V}_{\text{nl}}$ is the nonlocal Kleinman-Bylander pseudopotential operator.

## Self-Consistent Field (SCF) Cycle

The Kohn-Sham equations are solved iteratively:

1. **Start** with an initial guess for $\rho(\mathbf{r})$ (atomic superposition)
2. **Construct** $V_{\text{eff}}[\rho]$ from the current density
3. **Solve** the eigenvalue problem $\hat{H}\psi = \varepsilon\psi$ (Chebyshev-filtered subspace iteration)
4. **Compute** the Fermi level $E_f$ and occupations $f_{n\mathbf{k}s}$
5. **Build** the new density $\rho_{\text{out}}(\mathbf{r}) = \sum_{n\mathbf{k}s} w_\mathbf{k} f_{n\mathbf{k}s} |\psi_{n\mathbf{k}s}(\mathbf{r})|^2$
6. **Mix** $\rho_{\text{in}}$ and $\rho_{\text{out}}$ (Anderson/Pulay mixing with Kerker preconditioning)
7. **Check** convergence: $\|\rho_{\text{out}} - \rho_{\text{in}}\| / \|\rho_{\text{out}}\| < \text{tol}$
8. **Repeat** from step 2

## Document Index

| Document | Topic |
|----------|-------|
| [01_real_space_discretization](01_real_space_discretization.md) | Finite-difference grid, stencils, Laplacian, gradient |
| [02_pseudopotentials](02_pseudopotentials.md) | Norm-conserving pseudopotentials, Kleinman-Bylander form |
| [03_hamiltonian](03_hamiltonian.md) | Hamiltonian operator application |
| [04_eigensolver](04_eigensolver.md) | Chebyshev-filtered subspace iteration |
| [05_density_and_occupation](05_density_and_occupation.md) | Electron density, Fermi level, smearing |
| [06_electrostatics](06_electrostatics.md) | Pseudocharge, Poisson equation, electrostatic energy |
| [07_exchange_correlation](07_exchange_correlation.md) | LDA and GGA functionals |
| [08_total_energy](08_total_energy.md) | Total energy expression and components |
| [09_forces](09_forces.md) | Hellmann-Feynman forces |
| [10_stress](10_stress.md) | Stress tensor and pressure |
| [11_mixing](11_mixing.md) | Anderson/Pulay mixing, Kerker preconditioner |
| [12_kpoints](12_kpoints.md) | Brillouin zone sampling |
| [13_spin_orbit_coupling](13_spin_orbit_coupling.md) | Noncollinear spin, 2-component spinors, SOC |
| [14_parallelization](14_parallelization.md) | Band, k-point, and spin parallelism |

## Units

All internal quantities are in **Hartree atomic units**:

| Quantity | Unit |
|----------|------|
| Energy | Hartree (1 Ha = 27.211 eV) |
| Length | Bohr (1 Bohr = 0.529 A) |
| Temperature | Kelvin |
| Force | Ha/Bohr |
| Stress/Pressure | Ha/Bohr^3 (1 Ha/Bohr^3 = 29421 GPa) |
| Boltzmann constant | 3.1668 x 10^-6 Ha/K |
