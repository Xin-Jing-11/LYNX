# Pseudopotentials

ONCV norm-conserving pseudopotentials from the [PseudoDojo](http://www.pseudo-dojo.org/) project, linked as git submodules.

## Setup

After cloning LYNX, fetch the pseudopotential files:

```bash
git submodule update --init
```

This populates:

| Directory   | Functional | Source |
|-------------|-----------|--------|
| `psps/ONCVPSP-PBE-PDv0.4/` | GGA-PBE   | [ONCVPSP-PBE-PDv0.4](https://github.com/PseudoDojo/ONCVPSP-PBE-PDv0.4) |
| `psps/ONCVPSP-LDA-PDv0.4/` | LDA       | [ONCVPSP-LDA-PDv0.4](https://github.com/PseudoDojo/ONCVPSP-LDA-PDv0.4) |

## File layout

Each element has a subdirectory containing `.psp8` files:

```
psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8        # standard PBE pseudopotential for Si
psps/ONCVPSP-PBE-PDv0.4/Si/Si-sp.psp8     # with semicore states
psps/ONCVPSP-LDA-PDv0.4/Si/Si.psp8        # LDA version
```

## Usage

Point LYNX to the appropriate `.psp8` file:

```json
"pseudo_file": "psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8"
```

Or in Python:

```python
from lynx.config import DFTConfig

config = DFTConfig(
    ...,
    pseudo_files={'Si': 'psps/ONCVPSP-PBE-PDv0.4/Si/Si.psp8'},
)
```

## Reference

> M. van Setten et al., "The PseudoDojo: Training and grading a 85 element
> optimized norm-conserving pseudopotential table",
> Computer Physics Communications 226, 39-54 (2018).
