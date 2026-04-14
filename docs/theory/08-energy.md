---
title: "Total Energy"
parent: Electronic Structure
grand_parent: Theory
nav_order: 6
---

# Total Energy

## Free Energy Expression

The total Kohn-Sham free energy is:

$$
E_{\text{total}} = E_{\text{band}} - E_2 + E_{\text{xc}} - E_3 + E_{\text{hart}} + E_{\text{self}} + E_c + T S_e
$$

where each component is defined below.

## Band Energy

$$
E_{\text{band}} = \sum_{n,\mathbf{k},s} w_\mathbf{k} \, f_{n\mathbf{k}s} \, \varepsilon_{n\mathbf{k}s}
$$

This is the sum of occupied eigenvalues weighted by k-point weights and occupations. For non-spin-polarized calculations, a factor of 2 is included per state.

## Exchange-Correlation Energy

$$
E_{\text{xc}} = \int \rho_{\text{xc}}(\mathbf{r}) \, \varepsilon_{\text{xc}}(\mathbf{r}) \, d\mathbf{r}
$$

where $\rho_{\text{xc}} = \rho + \rho_c$ (with NLCC core charge if present).

## Double-Counting Correction

$$
E_2 = \int \rho(\mathbf{r}) \, V_{\text{xc}}(\mathbf{r}) \, d\mathbf{r}
$$

For spin-polarized:
$$
E_2 = \int \left( \rho_\uparrow V_{\text{xc},\uparrow} + \rho_\downarrow V_{\text{xc},\downarrow} \right) d\mathbf{r}
$$

## Electrostatic Terms

**Density-potential product:**
$$
E_3 = \int \rho(\mathbf{r}) \, \phi(\mathbf{r}) \, d\mathbf{r}
$$

**Hartree energy:**
$$
E_{\text{hart}} = \frac{1}{2} \int \left(\rho(\mathbf{r}) + b(\mathbf{r})\right) \phi(\mathbf{r}) \, d\mathbf{r}
$$

These combine to give the electrostatic energy. Note:
$$
-E_3 + E_{\text{hart}} = -\frac{1}{2}\int\rho\phi\,dV + \frac{1}{2}\int b\,\phi\,dV
$$

## Self Energy and Correction

$$
E_{\text{self}} = -\frac{1}{2} \Delta V \sum_J \sum_i b_{\text{ref},J}(\mathbf{r}_i) V_{\text{ref},J}(\mathbf{r}_i)
$$

$$
E_c = \frac{1}{2}\int (b + b_{\text{ref}}) V_c \, dV
$$

These are constants determined by the pseudopotentials and atomic positions, independent of the electron density.

## Electronic Entropy

**Fermi-Dirac:**
$$
TS_e = -k_B T \cdot \beta \sum_{n,\mathbf{k},s} w_\mathbf{k} \left[ f\ln f + (1-f)\ln(1-f) \right]
$$

**Gaussian:**
$$
TS_e = \sum_{n,\mathbf{k},s} w_\mathbf{k} \frac{1}{2\sqrt{\pi}\beta} \exp\left(-\beta^2(\varepsilon_n - E_f)^2\right)
$$

## Why This Decomposition?

The band energy $E_{\text{band}}$ contains the kinetic energy, electron-nuclear attraction, Hartree energy, and XC energy mixed together via the effective potential. The double-counting corrections $-E_2$ and $-E_3$ remove the XC and electrostatic contributions that were counted once in $E_{\text{band}}$, and $E_{\text{xc}}$ and $E_{\text{hart}}$ add them back with the correct expressions.

This decomposition ensures the total energy is variational with respect to the density and converges faster than the individual components.

## Energy Evaluation Timing

Energy is computed using $\rho_{\text{in}}$ (the density that generated the current potentials), **before** density mixing. This follows the Harris-Foulkes approach and converges faster than evaluating with $\rho_{\text{out}}$.
