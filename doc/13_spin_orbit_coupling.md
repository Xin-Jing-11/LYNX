# Spin-Orbit Coupling

## Physical Origin

Spin-orbit coupling (SOC) arises from the relativistic interaction between an electron's spin and its orbital angular momentum in the potential of the nucleus. In the Kohn-Sham framework, SOC is incorporated through the **fully-relativistic pseudopotential**.

SOC lifts degeneracies that exist in scalar-relativistic calculations (e.g., the j = l +/- 1/2 splitting in p, d, and f states) and is essential for:
- Heavy elements (Pt, Au, Bi, ...)
- Topological insulators
- Rashba/Dresselhaus splitting
- Magnetic anisotropy

## Two-Component Spinor Wavefunctions

With SOC, the Kohn-Sham orbitals become 2-component complex spinors:

$$
\psi_n(\mathbf{r}) = \begin{pmatrix} \psi_\uparrow(\mathbf{r}) \\ \psi_\downarrow(\mathbf{r}) \end{pmatrix}
$$

Each band has $2 N_d$ complex entries (spin-up and spin-down on the full grid). The spin index is no longer a good quantum number; both components are coupled.

### Storage Layout

In LYNX, each band column has layout:
```
psi_band[0 .. Nd_d-1]        = spin-up component
psi_band[Nd_d .. 2*Nd_d-1]   = spin-down component
```

The effective row dimension is `Nd_d_spinor = 2 * Nd_d`.

## Configuration

- `Nspin = 1` (no spin loop; spinor encodes both components)
- `Nspinor = 2`
- `is_kpt = true` always (spinor wavefunctions are always complex)
- `Nstates = N_electron + 20` (no division by 2, since each band holds both spins)
- Spin parallelism disabled (`npspin = 1`)
- `occfac = 1.0` (each spinor band holds 1 electron, not 2)

## SOC Pseudopotential Data

Fully-relativistic psp8 files (with `extension_switch >= 2`) contain additional SOC projectors for $l = 1, \ldots, l_{\max}$ (no SOC for $l = 0$ since $s$ orbitals have zero orbital angular momentum).

### PSP8 File Order

The psp8 format stores sections in this order:
1. Header (6 lines + nprojso line + extension_switch line)
2. Regular channel data (l = 0 .. lmax, each: header + mmax data lines)
3. Local channel (if lloc > lmax)
4. **SOC projector data** (l = 1 .. lmax, each: header + mmax data lines)
5. NLCC core charge (if fchrg > 0)
6. Isolated atom density

Note: SOC data comes **before** NLCC and isolated atom density. The nprojso line (after extension_switch) specifies the number of SOC projectors per l channel.

For each SOC channel $l$, the data consists of:
- $\Gamma_{\text{soc},lp}$: SOC energy coefficients
- $u_{\text{soc},lp}(r)$: SOC radial projector functions

## Complex Spherical Harmonics

**SOC requires complex spherical harmonics**, not real. This is because the angular momentum operators $\hat{L}_\pm = \hat{L}_x \pm i\hat{L}_y$ act as raising/lowering operators on the complex $Y_l^m$:

$$
\hat{L}_+ Y_l^m = \sqrt{l(l+1) - m(m+1)} \, Y_l^{m+1}
$$
$$
\hat{L}_- Y_l^m = \sqrt{l(l+1) - m(m-1)} \, Y_l^{m-1}
$$

These ladder relations do NOT hold for real spherical harmonics (which are linear combinations of $Y_l^m$ and $Y_l^{-m}$).

The complex spherical harmonics use the Condon-Shortley phase convention:
$$
Y_l^m(\hat{\mathbf{r}}) = (-1)^m C_{lm} \frac{(x+iy)^m}{r^l} P_l^m\!\left(\frac{z}{r}\right) \quad (m > 0)
$$
$$
Y_l^{-|m|}(\hat{\mathbf{r}}) = C_{l|m|} \frac{(x-iy)^{|m|}}{r^l} P_l^{|m|}\!\left(\frac{z}{r}\right) \quad (m < 0)
$$

The SOC projectors are built as:
$$
\chi_{\text{soc},lpm}(\mathbf{r}) = u_{\text{soc},lp}(r) \, Y_l^m(\hat{\mathbf{r}}) \quad \in \mathbb{C}
$$

The `Chi_soc` arrays are **complex-valued** (`NDArray<Complex>`), unlike the scalar-relativistic `Chi` which uses real spherical harmonics.

## SOC Hamiltonian

The SOC contribution to the Hamiltonian has the form $\hat{V}_{\text{SOC}} = \hat{\mathbf{L}} \cdot \hat{\mathbf{S}}$ projected through the nonlocal pseudopotential, decomposing into two terms.

### Inner Product Convention

Since $\chi_{\text{soc}}$ is complex, the inner product uses the **conjugate transpose**:

$$
\alpha_\sigma = \Delta V \cdot e^{i\theta} \cdot \chi_{\text{soc}}^\dagger \psi_\sigma = \Delta V \cdot e^{i\theta} \sum_i \bar{\chi}_{\text{soc}}(\mathbf{r}_i) \, \psi_\sigma(\mathbf{r}_i)
$$

The scatter (accumulation) uses the **non-conjugated** projector:

$$
(H\psi)(\mathbf{r}_i) \mathrel{+}= e^{-i\theta} \chi_{\text{soc}}(\mathbf{r}_i) \cdot \alpha
$$

### Term 1: On-Diagonal ($L_z S_z$)

This couples each spin component to itself:

$$
(H\psi)_\uparrow \mathrel{+}= \frac{1}{2}\sum_{l,p,m} m \cdot \Gamma_{\text{soc},lp} \langle\chi_{\text{soc},lpm}|\psi_\uparrow\rangle |\chi_{\text{soc},lpm}\rangle
$$

$$
(H\psi)_\downarrow \mathrel{-}= \frac{1}{2}\sum_{l,p,m} m \cdot \Gamma_{\text{soc},lp} \langle\chi_{\text{soc},lpm}|\psi_\downarrow\rangle |\chi_{\text{soc},lpm}\rangle
$$

The factor $m$ (magnetic quantum number) provides spin-up/down splitting. This term vanishes for $m = 0$.

### Term 2: Off-Diagonal (Ladder Operators $L_+ S_-$ and $L_- S_+$)

This couples spin-up to spin-down:

**$L_+ S_-$ (raises $m$, flips spin down to up):**
$$
(H\psi)_\uparrow \mathrel{+}= \frac{1}{2}\sum_{l,p,m} \sqrt{l(l+1) - m(m+1)} \cdot \Gamma_{\text{soc},lp} \langle\chi_{\text{soc},lp,m+1}|\psi_\downarrow\rangle |\chi_{\text{soc},lpm}\rangle
$$

**$L_- S_+$ (lowers $m$, flips spin up to down):**
$$
(H\psi)_\downarrow \mathrel{+}= \frac{1}{2}\sum_{l,p,m} \sqrt{l(l+1) - m(m-1)} \cdot \Gamma_{\text{soc},lp} \langle\chi_{\text{soc},lp,m-1}|\psi_\uparrow\rangle |\chi_{\text{soc},lpm}\rangle
$$

The ladder coefficients $\sqrt{l(l+1) - m(m \pm 1)}$ are the Clebsch-Gordan factors for the angular momentum raising/lowering operators.

## Spinor Effective Potential

The effective potential is a $2\times 2$ Hermitian matrix at each grid point:

$$
V_{\text{eff}}^{\text{spinor}} = \begin{pmatrix} V_{\uparrow\uparrow} & V_{\uparrow\downarrow} \\ V_{\uparrow\downarrow}^* & V_{\downarrow\downarrow} \end{pmatrix}
$$

### Construction from Noncollinear Density

Given the density $\rho$ and magnetization vector $\mathbf{m} = (m_x, m_y, m_z)$:

1. **Compute XC in collinear frame**: map to effective spin-up/down densities:
$$
\rho_\uparrow^{\text{xc}} = \frac{1}{2}(\rho + |\mathbf{m}|), \quad \rho_\downarrow^{\text{xc}} = \frac{1}{2}(\rho - |\mathbf{m}|)
$$

2. **Evaluate** $V_{\text{xc},\uparrow}$ and $V_{\text{xc},\downarrow}$ from the spin-polarized XC functional

3. **Transform** back to spinor Veff:
$$
V_{\text{avg}} = \frac{1}{2}(V_{\text{xc},\uparrow} + V_{\text{xc},\downarrow}), \quad V_{\text{diff}} = \frac{1}{2}(V_{\text{xc},\uparrow} - V_{\text{xc},\downarrow})
$$

$$
V_{\uparrow\uparrow} = V_{\text{avg}} + V_{\text{diff}} \frac{m_z}{|\mathbf{m}|} + \phi
$$

$$
V_{\downarrow\downarrow} = V_{\text{avg}} - V_{\text{diff}} \frac{m_z}{|\mathbf{m}|} + \phi
$$

$$
V_{\uparrow\downarrow} = V_{\text{diff}} \frac{m_x - i m_y}{|\mathbf{m}|}
$$

When $|\mathbf{m}| = 0$: $V_{\uparrow\uparrow} = V_{\downarrow\downarrow} = V_{\text{avg}} + \phi$ and $V_{\uparrow\downarrow} = 0$.

### Storage Layout

$$
V_{\text{eff}}^{\text{spinor}} = [V_{\uparrow\uparrow}(N_d) \;|\; V_{\downarrow\downarrow}(N_d) \;|\; \operatorname{Re}(V_{\uparrow\downarrow})(N_d) \;|\; \operatorname{Im}(V_{\uparrow\downarrow})(N_d)]
$$

Total size: $4 N_d$ doubles.

## Noncollinear Density and Magnetization

From spinor wavefunctions:

$$
\rho(\mathbf{r}) = \sum_n w_\mathbf{k} f_n \left(|\psi_\uparrow|^2 + |\psi_\downarrow|^2\right)
$$

$$
m_x(\mathbf{r}) = \sum_n w_\mathbf{k} f_n \cdot 2\operatorname{Re}(\psi_\uparrow^* \psi_\downarrow) = \sum_n w_\mathbf{k} f_n \operatorname{Tr}(\sigma_x |\psi\rangle\langle\psi|)
$$

$$
m_y(\mathbf{r}) = -\sum_n w_\mathbf{k} f_n \cdot 2\operatorname{Im}(\psi_\uparrow^* \psi_\downarrow) = \sum_n w_\mathbf{k} f_n \operatorname{Tr}(\sigma_y |\psi\rangle\langle\psi|)
$$

$$
m_z(\mathbf{r}) = \sum_n w_\mathbf{k} f_n \left(|\psi_\uparrow|^2 - |\psi_\downarrow|^2\right) = \sum_n w_\mathbf{k} f_n \operatorname{Tr}(\sigma_z |\psi\rangle\langle\psi|)
$$

where $\sigma_x, \sigma_y, \sigma_z$ are the Pauli matrices. The spin factor is 1 (not 2) since both spin components are explicitly present.

## Occupation and Spin Factor

For SOC, each spinor band holds **one electron** (both spin components are present in the band). Therefore:

$$
\text{occfac} = \frac{2}{N_{\text{spinor}}} = \frac{2}{2} = 1
$$

The total electron count: $N_e = \sum_{n,\mathbf{k}} w_\mathbf{k} \cdot \text{occfac} \cdot f_n = \sum_{n,\mathbf{k}} w_\mathbf{k} \cdot f_n$.

This contrasts with non-spin-polarized calculations where occfac = 2 (each band holds 2 electrons).

## Eigensolver for Spinors

The Chebyshev-filtered subspace iteration works unchanged with the spinor row dimension $2 N_d$:

- All BLAS/LAPACK calls (`zgemm`, `zpotrf`, `zheev`) see a $2N_d \times N_{\text{band}}$ matrix
- Only the Hamiltonian application callback uses the spinor-aware variant
- Orthogonalization naturally enforces $\langle\psi_i|\psi_j\rangle = \delta_{ij}$ across both spin components
- Column normalization before Cholesky QR prevents overflow from Chebyshev filter amplification

## SCF Mixing for Noncollinear

The mixing variable is the 4-component array:

$$
[\rho \;|\; m_x \;|\; m_y \;|\; m_z]
$$

All four components are mixed simultaneously with the same Pulay coefficients. After mixing, the charge density is clamped and renormalized while magnetization components are left unconstrained.

## Relation to Collinear Spin

Collinear spin ($N_{\text{spin}} = 2$, $N_{\text{spinor}} = 1$) is a special case where $m_x = m_y = 0$ and the off-diagonal $V_{\uparrow\downarrow} = 0$. The two spin channels decouple completely and are solved independently. SOC breaks this decoupling.

## Verification

The SOC H*psi implementation has been verified against SPARC (reference C code) by comparing the full spinor Hamiltonian applied to the same input vector:

| Component | Relative Error |
|-----------|---------------|
| Local (kinetic + Veff) | 1.3e-15 (machine precision) |
| Scalar-relativistic nonlocal | 8.2e-16 (machine precision) |
| SOC nonlocal | 4.4e-12 absolute |
| Full spinor H*psi | 5.3e-12 |

The PtAu_SOC benchmark (Au+Pt, non-orthogonal cell, 2x2x1 k-points, GGA_PBE) converges to -253.658 Ha in 12 SCF iterations, matching the SPARC reference of -253.657 Ha to 0.9 mHa.
