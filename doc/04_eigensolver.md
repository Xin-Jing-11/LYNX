# Eigensolver: Chebyshev-Filtered Subspace Iteration (CheFSI)

## Overview

LYNX solves the Kohn-Sham eigenvalue problem using **Chebyshev-filtered subspace iteration** (CheFSI). This avoids explicit diagonalization of the full Hamiltonian matrix (which is never formed) and instead works with matrix-vector products $H\psi$.

Each SCF iteration performs:
1. **Chebyshev filtering** — amplify the desired eigenspace
2. **Orthogonalization** — Cholesky QR
3. **Subspace projection** — reduce to small dense eigenproblem
4. **Diagonalization** — LAPACK dense eigensolver
5. **Orbital rotation** — rotate to eigenvector basis

## Chebyshev Polynomial Filter

Given spectral bounds $[\lambda_{\min}, \lambda_{\max}]$ and a cutoff $\lambda_c$ (above which eigenvalues are to be damped), the Chebyshev filter of degree $m$ is:

$$
Y = p_m(H) X
$$

where $p_m$ is a Chebyshev polynomial scaled to $[\lambda_c, \lambda_{\max}]$.

**Initialization:**
$$
e = \frac{\lambda_{\max} - \lambda_c}{2}, \quad c = \frac{\lambda_{\max} + \lambda_c}{2}, \quad \sigma_1 = \frac{e}{\lambda_{\min} - c}
$$

**First step ($k = 1$):**
$$
Y = \frac{\sigma_1}{e}(HX - cX)
$$

**Recurrence ($k = 2, \ldots, m$):**
$$
\sigma_{\text{new}} = \frac{1}{2/\sigma_1 - \sigma}, \quad \gamma = \frac{2\sigma_{\text{new}}}{e}
$$
$$
X_{\text{new}} = \gamma(HY - cY) - \sigma \sigma_{\text{new}} X_{\text{old}}
$$

The three-term recurrence avoids storing the full polynomial. The filter exponentially damps components above $\lambda_c$ while preserving those below.

## Spectral Bound Estimation (Lanczos)

Before the first CheFSI pass, the spectral bounds $\lambda_{\min}$ and $\lambda_{\max}$ are estimated using a **Lanczos iteration**:

1. Start with random vector $v_0$, normalize
2. Build tridiagonal matrix $T_j$ via Lanczos recurrence:
   - $v_{j+1} = H v_j - \alpha_j v_j - \beta_{j-1} v_{j-1}$
   - $\alpha_j = \langle v_j | H v_j \rangle$, $\beta_j = \|v_{j+1}\|$
3. Eigenvalues of $T_j$ approximate extremal eigenvalues of $H$
4. Converge when $|\lambda - \lambda_{\text{prev}}| < \text{tol}$ (default $10^{-2}$)

Safety margins: $\lambda_{\max} \leftarrow 1.01 \lambda_{\max}$, $\lambda_{\min} \leftarrow \lambda_{\min} - 0.1$.

After the first SCF iteration, $\lambda_{\min}$ is updated from the smallest computed eigenvalue, and $\lambda_c$ from the largest eigenvalue $+ 0.1$.

## Auto Chebyshev Degree

The polynomial degree is automatically chosen based on the effective mesh spacing $h_{\text{eff}}$:

$$
h_{\text{eff}} = \begin{cases}
h & \text{if } h_x = h_y = h_z = h \\
\sqrt{3/(h_x^{-2} + h_y^{-2} + h_z^{-2})} & \text{otherwise}
\end{cases}
$$

A cubic polynomial fit gives the degree: $m(h) = p_3 h^3 + p_2 h^2 + p_1 h + p_0$, clamped to 14 for $h > 0.7$ Bohr.

## Orthogonalization (Cholesky QR)

After filtering, the vectors $Y$ are orthogonalized via Cholesky QR:

1. Compute overlap: $S = Y^\dagger Y \cdot \Delta V$
2. Cholesky factorization: $S = R^\dagger R$ (LAPACK `dpotrf`/`zpotrf`)
3. Solve: $Y \leftarrow Y R^{-1}$ (LAPACK `dtrsm`/`ztrsm`)

This gives $Y^\dagger Y \cdot \Delta V = I$.

## Subspace Projection

Project the Hamiltonian onto the orthogonal subspace:

$$
H_s = Y^\dagger (HY) \cdot \Delta V
$$

This requires one additional Hamiltonian application. The result is a dense $N_{\text{band}} \times N_{\text{band}}$ Hermitian matrix.

Symmetrization: $H_s \leftarrow \frac{1}{2}(H_s + H_s^\dagger)$ to correct roundoff.

## Subspace Diagonalization

Diagonalize the small dense matrix:

$$
H_s Q = Q \Lambda
$$

using LAPACK `dsyev` (real) or `zheev` (complex). $\Lambda = \text{diag}(\varepsilon_1, \ldots, \varepsilon_N)$ gives the eigenvalues.

## Orbital Rotation

Rotate the filtered vectors into the eigenvector basis:

$$
\psi \leftarrow Y Q
$$

via BLAS `dgemm`/`zgemm`.

## Band Parallelism

When using band parallelism ($n_{\text{pband}} > 1$):
- Each process holds $N_{\text{band,local}}$ columns of $\psi$
- Chebyshev filtering is embarrassingly parallel (each process filters its own bands)
- Subspace operations require `MPI_Allgatherv` to collect the full $X$ and $HX$ matrices
- Overlap/projection matrices are computed redundantly on all processes
- After diagonalization, each process extracts its local columns from the rotated result

## Packing/Unpacking

NDArray may have a leading dimension (ld) different from Nd_d due to alignment padding. The solver packs psi into contiguous Nd_d-strided layout before BLAS calls and unpacks afterward.
