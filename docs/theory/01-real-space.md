---
title: "Real-Space Finite Differences"
parent: Discretization
grand_parent: Theory
nav_order: 1
---

# Real-Space Discretization

## Uniform Finite-Difference Grid

LYNX discretizes all fields on a uniform 3D grid with spacings $h_x, h_y, h_z$:

$$
\mathbf{r}_{ijk} = (i \cdot h_x,\; j \cdot h_y,\; k \cdot h_z), \quad i = 0, \ldots, N_x - 1
$$

The volume element is $\Delta V = h_x h_y h_z$ for orthogonal cells, or $\Delta V = h_x h_y h_z \cdot J$ for non-orthogonal cells where $J = |\det(\mathbf{L})|$ is the Jacobian of the lattice vectors.

The total number of grid points is $N_d = N_x \times N_y \times N_z$.

## Finite-Difference Stencil Coefficients

Derivatives are approximated using central finite differences of order $2n$ (default $n = 6$, i.e., 12th-order stencils). The stencil half-width is FDn = $n$.

### First Derivative Weights

$$
w_1(p) = \frac{(-1)^{p+1}}{p} \cdot \text{fract}(n, p), \quad p = 1, \ldots, n
$$

where

$$
\text{fract}(n, k) = \frac{n!}{(n-k)! \cdot (n+k)!}
$$

The first derivative approximation:

$$
\frac{\partial f}{\partial x}\bigg|_i \approx \sum_{p=1}^{n} \frac{w_1(p)}{h_x} \Big[ f(i+p) - f(i-p) \Big]
$$

### Second Derivative Weights

$$
w_2(0) = -\sum_{p=1}^{n} \frac{2}{p^2}, \qquad w_2(p) = \frac{2(-1)^{p+1}}{p^2} \cdot \text{fract}(n, p)
$$

The second derivative approximation:

$$
\frac{\partial^2 f}{\partial x^2}\bigg|_i \approx \frac{1}{h_x^2}\left[ w_2(0) f(i) + \sum_{p=1}^{n} w_2(p) \Big( f(i+p) + f(i-p) \Big) \right]
$$

## Non-Orthogonal Cells

For a general lattice with vectors $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$ (rows of matrix $\mathbf{L}$), the code works in **non-Cartesian (scaled)** coordinates $\boldsymbol{\xi}$ where $\mathbf{r} = \hat{\mathbf{L}}^T \boldsymbol{\xi}$ and $\hat{\mathbf{L}}$ is the unit-vector lattice:

$$
\hat{L}_{ij} = L_{ij} / |\mathbf{a}_i|
$$

The Laplacian in non-Cartesian coordinates becomes:

$$
\nabla^2 f = \sum_{\alpha\beta} G_{\alpha\beta} \frac{\partial^2 f}{\partial \xi_\alpha \partial \xi_\beta}
$$

where $\mathbf{G} = (\hat{\mathbf{L}}^{-1})^T (\hat{\mathbf{L}}^{-1})$ is the **Laplacian metric tensor** (`lapcT` in the code).

### Scaled Stencil Coefficients

Diagonal terms:
$$
D_{2,x}(p) = G_{xx} \cdot \frac{w_2(p)}{h_x^2}
$$

Mixed derivative terms (for off-diagonal $G_{\alpha\beta} \neq 0$):
$$
D_{2,xy}(p) = 2 G_{xy} \cdot \frac{w_1(p)}{h_x}, \quad D_{1,y}(q) = \frac{w_1(q)}{h_y}
$$

The mixed derivative $\partial^2 f / \partial x \partial y$ is computed as:

$$
\frac{\partial^2 f}{\partial x \partial y}\bigg|_{ij} = \sum_{p=1}^{n} \sum_{q=1}^{n} \frac{w_1(p)}{h_x} \frac{w_1(q)}{h_y} \Big[ f_{i+p,j+q} - f_{i+p,j-q} - f_{i-p,j+q} + f_{i-p,j-q} \Big]
$$

## Halo Exchange (Ghost Zones)

Since finite-difference stencils require neighbor values beyond local domain boundaries, periodic boundary conditions are enforced via halo (ghost) zones of width FDn in each direction. Under MPI domain decomposition these are communicated between neighboring processes.

## Integration

All volume integrals are approximated by:

$$
\int f(\mathbf{r}) \, d\mathbf{r} \approx \Delta V \sum_i f(\mathbf{r}_i)
$$

Inner products use:

$$
\langle f | g \rangle = \Delta V \sum_i f^*(\mathbf{r}_i) \, g(\mathbf{r}_i)
$$
