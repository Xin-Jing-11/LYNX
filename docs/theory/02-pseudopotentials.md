---
title: "Pseudopotentials"
parent: Discretization
grand_parent: Theory
nav_order: 2
---

# Pseudopotentials

## Norm-Conserving Pseudopotentials (psp8 Format)

LYNX uses Optimized Norm-Conserving Vanderbilt (ONCVPSP) pseudopotentials in the psp8 format. Each pseudopotential specifies:

- $Z_{\text{val}}$: number of valence electrons
- $l_{\text{max}}$: maximum angular momentum channel
- $l_{\text{loc}}$: angular momentum channel used for the local potential
- Radial grid $\{r_i\}$ with $N_r$ points

## Local Potential

The **local pseudopotential** $V_{\text{loc}}(r)$ is stored as $r V_{\text{loc}}(r)$ on the radial grid. At $r = 0$, the value is extrapolated: $V_{\text{loc}}(0) = r V_{\text{loc}}(r_1) / r_1$.

## Kleinman-Bylander Nonlocal Projectors

For each angular momentum channel $l \neq l_{\text{loc}}$, the KB projector factorization gives:

$$
\hat{V}_{\text{nl}} = \sum_{l \neq l_{\text{loc}}} \sum_{p=1}^{P_l} \sum_{m=-l}^{l} \Gamma_{lp} |\chi_{lpm}\rangle \langle \chi_{lpm}|
$$

where:
- $P_l$ = number of projectors per channel $l$
- $\Gamma_{lp}$ = KB energy coefficient (stored as `ekb` in psp8)
- $\chi_{lpm}(\mathbf{r}) = u_{lp}(r) Y_l^m(\hat{\mathbf{r}})$ is the projector function

The radial part $u_{lp}(r)$ is obtained from the psp8 file (which stores $r \cdot u_{lp}(r)$; division by $r$ recovers $u_{lp}$). At $r = 0$, the boundary condition $u_{lp}(0) = u_{lp}(r_1)$ is applied.

The total number of projectors per atom:

$$
N_{\text{proj}} = \sum_{l \neq l_{\text{loc}}} P_l (2l + 1)
$$

## Spherical Harmonics

Real spherical harmonics $Y_l^m(\hat{\mathbf{r}})$ are used:

$$
Y_0^0 = \frac{1}{2\sqrt{\pi}}, \quad
Y_1^{-1} = \sqrt{\frac{3}{4\pi}} \frac{y}{r}, \quad
Y_1^0 = \sqrt{\frac{3}{4\pi}} \frac{z}{r}, \quad
Y_1^1 = \sqrt{\frac{3}{4\pi}} \frac{x}{r}
$$

and so on up to $l = 6$ (sufficient for all elements through the actinides).

## Radial Interpolation

Radial functions are interpolated from the pseudopotential grid to the real-space grid using **Hermite cubic splines**. The first derivatives at node points are computed by solving a tridiagonal system (Gauss elimination), giving $C^1$ continuity.

For a uniform radial grid, interval lookup is $O(1)$; for non-uniform grids, binary search is used.

## NLCC (Nonlinear Core Correction)

When the pseudopotential has core charge ($f_{\text{chrg}} > 0$), the core electron density $\rho_c(r)$ is read from the psp8 file. The stored quantity is $4\pi r^2 \rho_c(r)$; division by $4\pi$ recovers $\rho_c$.

The core density is added to the valence density when evaluating the XC functional:

$$
V_{\text{xc}} = V_{\text{xc}}[\rho_{\text{val}} + \rho_c]
$$

This corrects the nonlinearity of the XC functional that arises from replacing core electrons with the pseudopotential.

## Cutoff Radius

Each channel $l$ has a cutoff radius $r_c^l$ beyond which the projector $u_{lp}(r)$ vanishes. Only grid points within $r_c^l$ of an atom contribute to the nonlocal operation (the "influence region").

## Periodic Images

Under periodic boundary conditions, each atom may have multiple periodic images whose projector influence regions overlap with the local domain. These images share the same KB coefficients $\Gamma_{lp}$ and projector alpha is accumulated across images before applying $\Gamma$.
