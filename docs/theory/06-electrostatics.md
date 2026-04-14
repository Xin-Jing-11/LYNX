---
title: "Electrostatics"
parent: Electronic Structure
grand_parent: Theory
nav_order: 4
---

# Electrostatics

## Pseudocharge Formulation

In real-space DFT with norm-conserving pseudopotentials, the nuclear point charges are replaced by smooth **pseudocharge** distributions to avoid the Coulomb singularity.

For each atom $J$ with local potential $V_J(\mathbf{r})$, the pseudocharge is:

$$
b_J(\mathbf{r}) = -\frac{1}{4\pi} \nabla^2 V_J(\mathbf{r})
$$

computed via the FD Laplacian applied to the interpolated atomic potential on the grid.

The total pseudocharge: $b(\mathbf{r}) = \sum_J b_J(\mathbf{r})$.

By construction, $\int b \, dV = -N_e$ (total negative charge of all valence electrons, with opposite sign).

## Reference Potential

To handle the long-range $-Z/r$ tail of the atomic potential, a smooth **reference potential** is introduced:

$$
V_{\text{ref}}(r) = \begin{cases}
-Z/r & r > r_c \\
-Z \left[ a_0 + r^2 \left( a_3 + r^3(a_6 + r \cdot a_7) \right) \right] & r \leq r_c
\end{cases}
$$

The polynomial coefficients ensure smoothness and that $V_{\text{ref}}$ matches $-Z/r$ and its derivatives at $r = r_c$:

$$
a_0 = 2.4/r_c, \quad a_3 = -2.8/r_c^3, \quad a_6 = -6.0/r_c^7, \quad a_7 = 1.8/r_c^8
$$

The reference pseudocharge:
$$
b_{\text{ref},J}(\mathbf{r}) = -\frac{1}{4\pi}\nabla^2 V_{\text{ref},J}(\mathbf{r})
$$

## Correction Potential

The **correction potential** captures the short-range difference:

$$
V_c(\mathbf{r}) = V_{\text{ref}}(\mathbf{r}) - V_J(\mathbf{r})
$$

This is nonzero only within the cutoff radius $r_c$ and is smooth.

## Local Potential

The local potential applied to electrons:

$$
V_{\text{loc}}(\mathbf{r}) = \sum_J \left( V_J(\mathbf{r}) - V_{\text{ref},J}(\mathbf{r}) \right)
$$

This is the sum of correction potentials over all atoms and their periodic images.

## Self Energy

The self energy accounts for the interaction of each atom's pseudocharge with its own reference potential:

$$
E_{\text{self}} = -\frac{1}{2} \Delta V \sum_J \sum_i b_{\text{ref},J}(\mathbf{r}_i) \, V_{\text{ref},J}(\mathbf{r}_i)
$$

This is a constant (independent of the electronic state) that must be included in the total energy.

## Correction Energy

$$
E_c = \frac{1}{2} \int (b + b_{\text{ref}}) V_c \, dV
$$

This accounts for the difference between the actual pseudocharge and the reference.

## Poisson Equation

The electrostatic potential $\phi$ is obtained by solving:

$$
-\nabla^2 \phi(\mathbf{r}) = 4\pi \left[ \rho(\mathbf{r}) + b(\mathbf{r}) \right]
$$

For periodic boundary conditions, the RHS must integrate to zero (charge neutrality). This is enforced by subtracting the mean:

$$
\text{RHS}_i \leftarrow \text{RHS}_i - \frac{1}{N_d}\sum_j \text{RHS}_j
$$

After solving, $\phi$ is gauge-fixed so $\sum_i \phi_i = 0$.

## Poisson Solver: AAR (Alternating Anderson-Richardson)

The Poisson equation is solved iteratively:

**Richardson iteration:**
$$
x^{(k+1)} = x^{(k)} + \omega \, M^{-1} r^{(k)}
$$

where $r^{(k)} = b - Ax^{(k)}$ is the residual and $M$ is the Jacobi preconditioner:

$$
M^{-1} = -\frac{1}{D_{2,x}(0) + D_{2,y}(0) + D_{2,z}(0)}
$$

**Anderson extrapolation** (every $p = 6$ iterations):
$$
x = x_{\text{old}} + \beta f - \sum_{j} \gamma_j (\Delta X_j + \beta \Delta F_j)
$$

where $\gamma$ solves the least-squares problem $F^T F \gamma = F^T f$, and $\beta = 0.6$ is the Anderson mixing parameter.

Convergence: $\|r\| \leq \text{tol} \cdot \|b\|$ with default $\text{tol} = 0.01 \times \text{SCF\_tol}$.

## Atomic Density Superposition

The initial guess for $\rho$ is the superposition of isolated atomic densities $\rho_{\text{atom},J}(r)$:

$$
\rho_0(\mathbf{r}) = \sum_J \rho_{\text{atom},J}(|\mathbf{r} - \mathbf{R}_J|)
$$

The isolated atom density is read from the psp8 file and interpolated to the grid using Hermite cubic splines.
