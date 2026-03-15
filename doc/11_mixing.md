# Density Mixing

## The Mixing Problem

At each SCF iteration, we have an input density $\rho_{\text{in}}$ and an output density $\rho_{\text{out}}$ from solving the KS equations. Simple replacement $\rho_{\text{in}} \leftarrow \rho_{\text{out}}$ leads to instability. Instead, a mixing scheme produces a better input for the next iteration.

## Anderson/Pulay Mixing

LYNX implements **Pulay mixing** (also known as DIIS — Direct Inversion in the Iterative Subspace), a generalization of Anderson mixing to multiple history vectors.

Define the residual: $f_k = \rho_{\text{out}}^{(k)} - \rho_{\text{in}}^{(k)}$

### History Vectors

Maintain differences of the last $m$ iterations (default $m = 7$):

$$
\Delta X_j = \rho_{\text{in}}^{(j)} - \rho_{\text{in}}^{(j-1)}, \quad \Delta F_j = f^{(j)} - f^{(j-1)}
$$

### Optimal Coefficients

Solve the least-squares problem:

$$
(F^T F) \boldsymbol{\gamma} = F^T f_k
$$

where $F = [\Delta F_1, \ldots, \Delta F_m]$ is the matrix of residual differences.

### Update

$$
\rho_{\text{in}}^{(k+1)} = \rho_{\text{avg}} + P f_{\text{avg}}
$$

where:
$$
\rho_{\text{avg}} = \rho_{\text{in}}^{(k)} - \sum_j \gamma_j \Delta X_j
$$
$$
f_{\text{avg}} = f_k - \sum_j \gamma_j \Delta F_j
$$

and $P$ is the preconditioner (identity or Kerker).

## Kerker Preconditioner

The Kerker preconditioner damps long-wavelength charge sloshing, which is the dominant instability in metallic systems. It solves:

$$
(-\nabla^2 + k_{\text{TF}}^2) P f = -(\nabla^2 - \epsilon^{-1} k_{\text{TF}}^2) f
$$

where:
- $k_{\text{TF}} = 1.0$ Bohr$^{-1}$ (Thomas-Fermi screening wavevector)
- $\epsilon^{-1} = 0.1$ (inverse dielectric constant threshold)

In Fourier space, this corresponds to the ratio:

$$
\tilde{P}(q) = \frac{q^2 + \epsilon^{-1} k_{\text{TF}}^2}{q^2 + k_{\text{TF}}^2}
$$

At long wavelengths ($q \to 0$), $\tilde{P} \to \epsilon^{-1}$, damping the slow charge oscillations. At short wavelengths ($q \to \infty$), $\tilde{P} \to 1$, leaving high-frequency components unchanged.

The real-space implementation solves the preconditioner equation using the AAR solver with a Jacobi preconditioner.

## Mixing for Spin-Polarized Systems

### Collinear Spin

The mixing variable is the packed array $[\rho_{\text{total}} \;|\; m]$ where $m = \rho_\uparrow - \rho_\downarrow$ is the magnetization. After mixing, the spin densities are reconstructed:

$$
\rho_\uparrow = \frac{1}{2}(\rho + m), \quad \rho_\downarrow = \frac{1}{2}(\rho - m)
$$

### Noncollinear (SOC)

The mixing variable is the 4-component array $[\rho \;|\; m_x \;|\; m_y \;|\; m_z]$. Each component is mixed with the same Pulay coefficients.

## Post-Mixing Corrections

After mixing:
1. **Clamp** negative densities: $\rho_i \leftarrow \max(\rho_i, 0)$
2. **Renormalize** to preserve electron count: $\rho \leftarrow \rho \cdot N_e / \int\rho\,dV$
