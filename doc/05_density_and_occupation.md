# Electron Density and Occupation

## Electron Density

The electron density is constructed from the Kohn-Sham orbitals:

$$
\rho(\mathbf{r}) = \sum_{n,\mathbf{k},s} w_\mathbf{k} \, f_{n\mathbf{k}s} \, |\psi_{n\mathbf{k}s}(\mathbf{r})|^2
$$

where:
- $w_\mathbf{k}$ = normalized k-point weight ($\sum_\mathbf{k} w_\mathbf{k} = 1$)
- $f_{n\mathbf{k}s}$ = occupation number
- The spin factor is implicit: for non-spin-polarized ($N_{\text{spin}} = 1$), each state carries a factor of 2; for collinear spin ($N_{\text{spin}} = 2$), each spin channel has factor 1.

### Collinear Spin

For collinear spin-polarized calculations, separate densities are maintained:

$$
\rho_\uparrow(\mathbf{r}) = \sum_{n,\mathbf{k}} w_\mathbf{k} f_{n\mathbf{k}\uparrow} |\psi_{n\mathbf{k}\uparrow}|^2, \quad
\rho_\downarrow(\mathbf{r}) = \sum_{n,\mathbf{k}} w_\mathbf{k} f_{n\mathbf{k}\downarrow} |\psi_{n\mathbf{k}\downarrow}|^2
$$

Total density: $\rho = \rho_\uparrow + \rho_\downarrow$

Magnetization: $m = \rho_\uparrow - \rho_\downarrow$

### Noncollinear Spin (SOC)

For 2-component spinors $\psi_n = [\psi_\uparrow | \psi_\downarrow]$:

$$
\rho = \sum_n w_\mathbf{k} f_n \left( |\psi_\uparrow|^2 + |\psi_\downarrow|^2 \right)
$$

$$
m_x = \sum_n w_\mathbf{k} f_n \cdot 2 \operatorname{Re}(\psi_\uparrow^* \psi_\downarrow)
$$

$$
m_y = \sum_n w_\mathbf{k} f_n \cdot (-2) \operatorname{Im}(\psi_\uparrow^* \psi_\downarrow)
$$

$$
m_z = \sum_n w_\mathbf{k} f_n \left( |\psi_\uparrow|^2 - |\psi_\downarrow|^2 \right)
$$

The spin factor is 1 (not 2) because both components are present.

## Fermi Level and Occupation

The Fermi level $E_f$ is determined by the constraint:

$$
\sum_{n,\mathbf{k},s} w_\mathbf{k} \, f(\varepsilon_{n\mathbf{k}s}; E_f) = N_e
$$

where $N_e$ is the total number of electrons. This nonlinear equation is solved by **Brent's method** (bracketed root-finding).

### Fermi-Dirac Smearing

$$
f(\varepsilon) = \frac{1}{1 + \exp\left(\beta(\varepsilon - E_f)\right)}
$$

where $\beta = 1/(k_B T)$ and $T$ is the electronic temperature.

### Gaussian Smearing

$$
f(\varepsilon) = \frac{1}{2}\operatorname{erfc}\left(\beta(\varepsilon - E_f)\right)
$$

### Default Temperatures

- Gaussian smearing: 0.2 eV ($\beta = E_H / 0.2$ in atomic units)
- Fermi-Dirac smearing: 0.1 eV ($\beta = E_H / 0.1$ in atomic units)

where $E_H = 27.211$ eV is the Hartree energy.

## Electronic Entropy

The entropy contribution to the free energy depends on the smearing scheme.

### Fermi-Dirac Entropy

$$
S = -k_B \beta \sum_{n,\mathbf{k},s} w_\mathbf{k} \left[ f \ln f + (1-f) \ln(1-f) \right]
$$

### Gaussian Entropy

$$
S = \sum_{n,\mathbf{k},s} w_\mathbf{k} \frac{1}{2\sqrt{\pi}\beta} \exp\left(-\beta^2(\varepsilon_n - E_f)^2\right)
$$

The total free energy includes the entropy: $F = E_{\text{total}} - TS$ (with sign conventions such that the entropy term is added to $E_{\text{total}}$).

## MPI Reductions

In parallel, density contributions from different band processes and k-point processes are summed via `MPI_Allreduce`. For spin parallelism, spin densities are exchanged via `MPI_Sendrecv` between spin-partner processes.
