# Pseudopotentials

ONCV norm-conserving pseudopotentials from the [PseudoDojo](http://www.pseudo-dojo.org/) project, linked as git submodules.

## Setup

After cloning LYNX, fetch the pseudopotential files:

```bash
git submodule update --init
```

This populates:

| Directory | Functional | Relativistic | Use case |
|-----------|-----------|-------------|----------|
| `psps/ONCVPSP-PBE-PDv0.4/` | GGA-PBE | Scalar (SR) | Standard calculations |
| `psps/ONCVPSP-LDA-PDv0.4/` | LDA | Scalar (SR) | LDA calculations |
| `psps/ONCVPSP-PBE-FR-PDv0.4/` | GGA-PBE | Fully relativistic (FR) | SOC calculations |
| `psps/ONCVPSP-LDA-FR-PDv0.4/` | LDA | Fully relativistic (FR) | SOC + LDA |

## Auto-selection

When `pseudo_file` is omitted from the input JSON, LYNX automatically selects the correct pseudopotential based on the XC functional and spin type:

```json
{
  "atoms": [
    {
      "element": "Au",
      "fractional": true,
      "coordinates": [[0.0, 0.0, 0.0]]
    }
  ],
  "electronic": {
    "xc": "GGA_PBE",
    "spin": "noncollinear"
  }
}
```

This auto-selects `psps/ONCVPSP-PBE-FR-PDv0.4/Au/Au-sp_r.psp8` (FR for SOC).

For non-SOC: `"spin": "none"` selects `psps/ONCVPSP-PBE-PDv0.4/Au/Au-sp.psp8` (SR).

## File layout

Each element has a subdirectory containing `.psp8` files:

```
psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8         # SR PBE for Si
psps/ONCVPSP-PBE-FR-PDv0.4/Si/Si_r.psp8    # FR PBE for Si (SOC)
psps/ONCVPSP-LDA-PDv0.4/Si/Si.psp8         # SR LDA for Si
psps/ONCVPSP-LDA-FR-PDv0.4/Si/Si_r.psp8    # FR LDA for Si (SOC)
```

FR files use the `_r` suffix convention (e.g., `Si_r.psp8`, `Au-sp_r.psp8`).

The recommended pseudopotential for each element is listed in `standard.txt` within each table directory.

## Manual override

You can always specify an explicit path:

```json
"pseudo_file": "/path/to/custom/Pt_fr.psp8"
```

## Reference

> M. van Setten et al., "The PseudoDojo: Training and grading a 85 element
> optimized norm-conserving pseudopotential table",
> Computer Physics Communications 226, 39-54 (2018).
