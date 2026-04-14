---
title: "Stress"
parent: Response Properties
grand_parent: Theory
nav_order: 2
---

# Stress Tensor

## Definition

The stress tensor measures the response of total energy to strain deformation of the unit cell:

$$
\sigma_{\alpha\beta} = \frac{1}{\Omega}\frac{\partial E_{\text{total}}}{\partial \epsilon_{\alpha\beta}}
$$

where $\Omega$ is the cell volume and $\epsilon_{\alpha\beta}$ is the strain tensor component. The stress has units of Ha/Bohr^3 (converted to GPa by multiplying by 29421).

## Stored in Voigt Notation

$$
[\sigma_{xx}, \sigma_{xy}, \sigma_{xz}, \sigma_{yy}, \sigma_{yz}, \sigma_{zz}]
$$

## Decomposition

$$
\sigma_{\alpha\beta} = \sigma_{\alpha\beta}^{\text{kin}} + \sigma_{\alpha\beta}^{\text{nl}} + \sigma_{\alpha\beta}^{\text{xc}} + \sigma_{\alpha\beta}^{\text{el}} + \sigma_{\alpha\beta}^{\text{xc,nlcc}}
$$

## Kinetic + Nonlocal Stress

The combined kinetic and nonlocal stress:

$$
\sigma_{\alpha\beta}^{\text{kin+nl}} = \frac{1}{\Omega} \sum_{n,\mathbf{k},s} w_\mathbf{k} f_{n\mathbf{k}s} \, \operatorname{Re}\left\langle \frac{\partial\psi_n}{\partial r_\alpha} \bigg| \frac{\partial\psi_n}{\partial r_\beta} \right\rangle
$$

with KB projector contributions from $\Gamma$ matrices. The gradient of each wavefunction in 3 Cartesian directions is computed via finite differences.

## Exchange-Correlation Stress

**LDA:**
$$
\sigma_{\alpha\beta}^{\text{xc}} = \frac{\delta_{\alpha\beta}}{\Omega} \int \left(\varepsilon_{\text{xc}} - \rho V_{\text{xc}}\right) dV
$$

Only the isotropic (diagonal) part contributes for LDA.

**GGA:**
$$
\sigma_{\alpha\beta}^{\text{xc}} = \frac{\delta_{\alpha\beta}}{\Omega} \int \left(\varepsilon_{\text{xc}} - \rho V_{\text{xc}}\right) dV - \frac{1}{\Omega}\int v_\sigma \frac{\partial\rho}{\partial r_\alpha}\frac{\partial\rho}{\partial r_\beta}\, dV
$$

where $v_\sigma = \partial(\rho\varepsilon_{\text{xc}})/\partial\sigma$ and $\sigma = |\nabla\rho|^2$. The second term is the anisotropic GGA contribution from the gradient coupling.

## Electrostatic (Local) Stress

$$
\sigma_{\alpha\beta}^{\text{el}} = \frac{1}{2\Omega}\int\left[\frac{\partial\phi}{\partial r_\alpha}\frac{\partial\phi}{\partial r_\beta} + b\left(\frac{\partial V_c}{\partial r_\alpha}\frac{\partial\phi}{\partial r_\beta} + \ldots\right)\right] dV
$$

This involves gradients of the electrostatic potential, correction potential, and pseudocharge.

## NLCC XC Stress

$$
\sigma_{\alpha\beta}^{\text{xc,nlcc}} = \frac{1}{\Omega}\int (r_\alpha - R_{J,\alpha})(r_\beta - R_{J,\beta}) \, \nabla\rho_{c,J} \cdot V_{\text{xc}} \, dV
$$

This contributes only when NLCC is active and accounts for the position-dependence of the core charge.

## Pressure

The isotropic pressure:

$$
P = -\frac{1}{3}\left(\sigma_{xx} + \sigma_{yy} + \sigma_{zz}\right)
$$

Negative pressure indicates the system wants to contract; positive indicates expansion.

## Coordinate Transformation

For non-orthogonal cells, gradient computations use the metric tensor to convert between non-Cartesian and Cartesian derivatives, ensuring the stress tensor is in the Cartesian frame.
