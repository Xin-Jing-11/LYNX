---
title: "Hamiltonian"
parent: Electronic Structure
grand_parent: Theory
nav_order: 1
---

# Hamiltonian Operator

## Full Hamiltonian

The Kohn-Sham Hamiltonian applied to a wavefunction:

$$
\hat{H}\psi = -\frac{1}{2}\nabla^2 \psi + V_{\text{eff}}(\mathbf{r})\psi + \hat{V}_{\text{nl}}\psi
$$

This is computed in three stages: local (kinetic + diagonal potential), and nonlocal.

## Local Part: Kinetic + Effective Potential

$$
(\hat{H}_{\text{loc}}\psi)(\mathbf{r}_i) = -\frac{1}{2}(\nabla^2\psi)_i + (V_{\text{eff},i} + c)\,\psi_i
$$

where $c$ is an optional constant shift (used during Chebyshev filtering to shift the spectrum).

The Laplacian $\nabla^2\psi$ is computed via the finite-difference stencil (see [01_real_space_discretization](01_real_space_discretization.md)), requiring halo exchange of ghost zone values before stencil application.

## Nonlocal Part: Kleinman-Bylander

The KB nonlocal operation is a three-step algorithm:

**Step 1 — Inner products:**
$$
\alpha_{lpm}^{(J)} = \Delta V \sum_{i \in \text{infl}(J)} \chi_{lpm}^{(J)}(\mathbf{r}_i) \, \psi(\mathbf{r}_i)
$$

where the sum runs only over grid points $i$ within the influence region of atom $J$. For periodic images, $\alpha$ is accumulated across all images of the same physical atom.

**Step 2 — Apply KB energy:**
$$
\alpha_{lpm}^{(J)} \leftarrow \Gamma_{lp} \cdot \alpha_{lpm}^{(J)}
$$

**Step 3 — Scatter back:**
$$
(H\psi)(\mathbf{r}_i) \mathrel{+}= \sum_{J,l,p,m} \chi_{lpm}^{(J)}(\mathbf{r}_i) \, \alpha_{lpm}^{(J)}
$$

## K-Point (Bloch) Treatment

For k-point calculations, the projectors $\chi$ remain **real**, but Bloch phase factors are applied. For an atom image at position $\mathbf{R}_{\text{image}}$ relative to the reference cell:

$$
\theta = -\mathbf{k} \cdot \mathbf{R}_{\text{image}}
$$

**Step 1:** $\alpha = e^{i\theta} \cdot \Delta V \cdot \chi^T \psi$ (gather with Bloch phase)

**Step 3:** $H\psi \mathrel{+}= e^{-i\theta} \cdot \chi \cdot \alpha$ (scatter with conjugate phase)

This avoids storing complex projector arrays — the phase factor is a scalar applied during the matrix operations.

## Spinor Hamiltonian (SOC)

For spin-orbit coupling, $\psi$ is a 2-component spinor $[\psi_\uparrow | \psi_\downarrow]$ and the effective potential is a $2 \times 2$ matrix at each grid point:

$$
\hat{H}_{\text{spinor}}\begin{pmatrix}\psi_\uparrow \\ \psi_\downarrow\end{pmatrix} =
\begin{pmatrix}
-\frac{1}{2}\nabla^2 + V_{\uparrow\uparrow} & V_{\uparrow\downarrow} \\
V_{\uparrow\downarrow}^* & -\frac{1}{2}\nabla^2 + V_{\downarrow\downarrow}
\end{pmatrix}
\begin{pmatrix}\psi_\uparrow \\ \psi_\downarrow\end{pmatrix}
+ \hat{V}_{\text{nl}}^{\text{SR}}\begin{pmatrix}\psi_\uparrow \\ \psi_\downarrow\end{pmatrix}
+ \hat{V}_{\text{SOC}}\begin{pmatrix}\psi_\uparrow \\ \psi_\downarrow\end{pmatrix}
$$

See [13_spin_orbit_coupling](13_spin_orbit_coupling.md) for the full SOC formulation.

## Multiple Bands

In practice, $H$ is applied to $N_{\text{band}}$ wavefunctions simultaneously. The psi array has layout `(Nd_d, Nband)` in column-major order, and the stencil loops over all bands in the inner loop for cache efficiency.
