# Brillouin Zone Sampling

## Bloch's Theorem

In a periodic crystal, the Kohn-Sham wavefunctions can be labeled by a crystal momentum $\mathbf{k}$:

$$
\psi_{n\mathbf{k}}(\mathbf{r} + \mathbf{R}) = e^{i\mathbf{k}\cdot\mathbf{R}} \psi_{n\mathbf{k}}(\mathbf{r})
$$

Physical observables require integration over the Brillouin zone (BZ):

$$
\rho(\mathbf{r}) = \sum_n \frac{1}{\Omega_{\text{BZ}}}\int_{\text{BZ}} f_{n\mathbf{k}} |\psi_{n\mathbf{k}}(\mathbf{r})|^2 \, d\mathbf{k}
$$

This integral is discretized on a finite grid of k-points.

## Monkhorst-Pack Grid

The k-points are generated on a uniform $K_x \times K_y \times K_z$ Monkhorst-Pack grid:

$$
\mathbf{k}_{n_1 n_2 n_3} = \sum_{i=1}^{3} \frac{n_i + s_i}{K_i} \, \mathbf{b}_i
$$

where:
- $n_i$ ranges from $-\lfloor(K_i-1)/2\rfloor$ to $\lceil(K_i-1)/2\rceil$
- $s_i$ is an optional shift (typically 0 or 0.5)
- $\mathbf{b}_i = 2\pi \mathbf{a}_j \times \mathbf{a}_k / (\mathbf{a}_i \cdot \mathbf{a}_j \times \mathbf{a}_k)$ are reciprocal lattice vectors

In LYNX, the k-points are stored in **reduced coordinates** (fractions of the reciprocal lattice vectors) and converted to Cartesian:

$$
\mathbf{k}_{\text{cart}} = \frac{2\pi}{L_i} \mathbf{k}_{\text{red}}
$$

where $L_i$ are the lattice vector lengths.

## Time-Reversal Symmetry

For systems without spin-orbit coupling, time-reversal symmetry implies $\varepsilon_{n\mathbf{k}} = \varepsilon_{n,-\mathbf{k}}$. The k-point set is reduced:

- If $-\mathbf{k}$ is already in the set: double its weight
- Otherwise: keep both $\mathbf{k}$ and $-\mathbf{k}$

This typically halves the number of k-points.

## Gamma-Point Optimization

When only the $\Gamma$-point ($\mathbf{k} = 0$) is used, the wavefunctions are real (no complex phase). This halves memory and doubles the speed of BLAS operations.

LYNX automatically detects gamma-only calculations and uses real arithmetic (`dgemm`, `dsyev`) instead of complex (`zgemm`, `zheev`).

## K-Point Weights

Each k-point has a weight $w_\mathbf{k}$ (unnormalized integer from symmetry reduction). The normalized weights satisfy:

$$
\sum_\mathbf{k} w_\mathbf{k} = 1
$$

## Bloch Phase in Real-Space

In real-space with periodic boundaries, the Bloch condition is enforced via **phase factors in the halo exchange**. When copying ghost zone values across a periodic boundary shifted by lattice vector $\mathbf{a}_i$:

$$
\psi_{\text{ghost}} = e^{i\mathbf{k}\cdot\mathbf{a}_i} \psi_{\text{interior}}
$$

This ensures the finite-difference stencil correctly applies to the periodic part of the Bloch function.

## K-Point Parallelism

Different k-points are distributed across MPI processes. Each process handles $N_{\mathbf{k},\text{local}}$ k-points and solves independent eigenvalue problems for each. Cross-k-point communication is needed only for:
- Fermi level determination (global eigenvalue collection)
- Density reduction (sum over k-points)
