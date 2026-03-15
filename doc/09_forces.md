# Atomic Forces

## Hellmann-Feynman Theorem

The force on atom $J$ is:

$$
\mathbf{F}_J = -\frac{\partial E_{\text{total}}}{\partial \mathbf{R}_J}
$$

In the Kohn-Sham framework with norm-conserving pseudopotentials, this decomposes into three contributions:

$$
\mathbf{F}_J = \mathbf{F}_J^{\text{loc}} + \mathbf{F}_J^{\text{nl}} + \mathbf{F}_J^{\text{xc}}
$$

## Local (Electrostatic) Force

The local force arises from the dependence of the pseudocharge $b_J$ and local potential $V_J$ on the atomic position:

$$
\mathbf{F}_J^{\text{loc}} = -\int b_J(\mathbf{r}) \nabla\phi(\mathbf{r}) \, dV + \frac{1}{2}\int \left[\nabla V_c^J \cdot (b + b_{\text{ref}}) - \nabla V_c \cdot (b_J + b_{J,\text{ref}})\right] dV
$$

where:
- $b_J = -\frac{1}{4\pi}\nabla^2 V_J$ is the pseudocharge of atom $J$
- $V_c^J = V_{\text{ref},J} - V_J$ is the correction potential for atom $J$
- $\phi$ is the electrostatic potential
- $b_{\text{ref}}$ is the reference pseudocharge

The gradient $\nabla\phi$ is computed using finite-difference stencils.

## Nonlocal (KB Projector) Force

The nonlocal force comes from the position-dependence of the KB projectors:

$$
\mathbf{F}_J^{\text{nl}} = -2 \sum_{n,\mathbf{k},s} w_\mathbf{k} f_{n\mathbf{k}s} \sum_{l,p,m} \Gamma_{lp} \, \alpha_{lpm} \, \boldsymbol{\beta}_{lpm}
$$

where:
- $\alpha_{lpm} = \langle \chi_{lpm}^J | \psi_{n\mathbf{k}s} \rangle \Delta V$ (scalar projector overlap)
- $\boldsymbol{\beta}_{lpm} = \langle \chi_{lpm}^J | \nabla\psi_{n\mathbf{k}s} \rangle \Delta V$ (gradient projector overlap, 3-component vector)
- $\Gamma_{lp}$ is the KB energy coefficient

The gradient $\nabla\psi$ is computed by finite differences. The factor of 2 arises from $\text{Re}(\alpha^* \beta + \beta^* \alpha) = 2\text{Re}(\alpha^* \beta)$ for Hermitian operators.

### K-Point Phase Factors

For k-point calculations, the Bloch phases enter:
$$
\alpha = e^{i\theta} \Delta V \, \chi^T \psi, \quad \boldsymbol{\beta} = e^{i\theta} \Delta V \, \chi^T \nabla\psi
$$
$$
\text{force contribution} = -2\text{Re}(\Gamma \cdot \alpha^* \cdot \boldsymbol{\beta})
$$

## NLCC XC Force

When nonlinear core correction is active, the core charge depends on atomic position:

$$
\mathbf{F}_J^{\text{xc}} = \int V_{\text{xc}}(\mathbf{r}) \, \nabla\rho_{c,J}(\mathbf{r}) \, dV
$$

The gradient of the core charge $\nabla\rho_{c,J}$ is computed by applying the FD gradient to the interpolated core charge of atom $J$.

For GGA functionals, additional terms involving $\nabla V_{\text{xc}}$ and $D_{\text{xcdgrho}}$ (the derivative $\partial V_{\text{xc}}/\partial|\nabla\rho|^2$) contribute via the chain rule.

## Coordinate Transformation

For non-orthogonal cells, forces are first computed in non-Cartesian coordinates and then transformed:

$$
\mathbf{F}_{\text{cart}} = (\hat{\mathbf{L}}^{-1})^T \, \mathbf{F}_{\text{nonCart}}
$$

where $\hat{\mathbf{L}}$ is the unit-vector lattice matrix.

## Newton's Third Law

By construction, the total force sums to zero: $\sum_J \mathbf{F}_J = 0$ (to within numerical precision), which is a consequence of translational invariance.
