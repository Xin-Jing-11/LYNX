# Parallelization

## Three Levels of Parallelism

LYNX distributes work across MPI processes using three orthogonal levels:

$$
N_{\text{procs}} = n_{\text{pspin}} \times n_{\text{pkpt}} \times n_{\text{pband}}
$$

Each level creates independent groups that can work without communication (except at synchronization points).

## Spin Parallelism ($n_{\text{pspin}}$)

For collinear spin-polarized calculations ($N_{\text{spin}} = 2$):
- Process group 0 handles spin-up
- Process group 1 handles spin-down
- Each group solves an independent eigenvalue problem with its own $V_{\text{eff},\sigma}$
- **Sync points**: density exchange (Sendrecv), Fermi level (Allreduce over spins)

For SOC: disabled ($n_{\text{pspin}} = 1$) since spin channels are coupled.

## K-Point Parallelism ($n_{\text{pkpt}}$)

Different k-points are distributed evenly across process groups:
- Each group handles $N_{\mathbf{k},\text{local}} = \lceil N_\mathbf{k} / n_{\text{pkpt}} \rceil$ k-points
- Eigenvalue problems at different k-points are independent
- **Sync points**: density Allreduce, Fermi level Allreduce

## Band Parallelism ($n_{\text{pband}}$)

The $N_{\text{band}}$ orbitals are distributed across processes:
- Each process holds $N_{\text{band,local}}$ columns of $\psi$
- Chebyshev filtering is embarrassingly parallel
- **Sync points**: subspace operations require Allgatherv of the full $X$ matrix

### Subspace Operations with Band Parallelism

For the overlap matrix $S = X^\dagger X \cdot \Delta V$:
1. `Allgatherv` to collect the full $X$ ($N_d \times N_{\text{band}}$)
2. Compute $S$ redundantly on all processes via BLAS
3. Cholesky factorize $S$ redundantly
4. Apply $R^{-1}$ to full $X$, extract local columns

This "Allgather + redundant LAPACK" approach is communication-optimal for typical DFT band counts ($N_{\text{band}} \lesssim 1000$).

## Communicator Hierarchy

| Communicator | Groups | Purpose |
|---|---|---|
| `spincomm` | $n_{\text{pspin}}$ groups | Processes handling same spin |
| `kptcomm` | $n_{\text{pkpt}}$ groups within each spin group | Processes handling same k-point |
| `bandcomm` | $n_{\text{pband}}$ groups within each kpt group | Processes handling same band subset |
| `kpt_bridge` | connects across k-point groups | Cross-kpt reductions (density, Fermi level) |
| `spin_bridge` | connects across spin groups | Spin density exchange |

## Auto-Detection

If parallelization parameters are not set:
1. If $N_{\text{spin}} = 2$ and $N_{\text{procs}} \geq 2$: use $n_{\text{pspin}} = 2$
2. Remaining procs distributed to k-points: $n_{\text{pkpt}} = \min(N_{\text{procs}}/n_{\text{pspin}}, N_\mathbf{k})$
3. Remaining procs for bands: $n_{\text{pband}} = N_{\text{procs}} / (n_{\text{pspin}} \times n_{\text{pkpt}})$

## What is NOT Parallelized (Domain Decomposition)

The current implementation assumes each MPI process holds the **full spatial domain** ($N_d$ grid points). Spatial domain decomposition (splitting the grid across processes) is not implemented — all stencil operations are local.

## Global vs. Local Quantities

| Quantity | Local | Global |
|---|---|---|
| $\psi$ columns | $N_{\text{band,local}}$ | $N_{\text{band}}$ (= `Nstates`) |
| Eigenvalues | $N_{\text{band}}$ (all) | Same (replicated) |
| Occupations | $N_{\text{band}}$ (all) | Same (replicated) |
| Density $\rho$ | $N_d$ (local spin) | $N_d \times N_{\text{spin}}$ (after exchange) |
| k-points | $N_{\mathbf{k},\text{local}}$ | $N_\mathbf{k}$ |
